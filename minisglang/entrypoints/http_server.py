"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements HTTP APIs for the inference engine via fastapi.
"""

import asyncio
import dataclasses
import logging
import multiprocessing as multiprocessing
import os
import threading
import time
from http import HTTPStatus
from typing import AsyncIterator, Callable, Optional

from contextlib import asynccontextmanager


# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)


import numpy as np
import orjson
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from minisglang.engine.engine import _launch_subprocesses
from minisglang.utils.io_struct import (
    GenerateReqInput,
)
from minisglang.engine.tokenizer import TokenizerManager


from minisglang.utils.args import ServerArgs
from minisglang.utils import (
    kill_process_tree,
)

from minisglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state

@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    warmup_thread = getattr(fast_api_app, "warmup_thread", None)
    if warmup_thread is not None:
        warmup_thread.start()
    yield
    
# Fast API
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))


##### Native API endpoints #####


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer_path,
    }
    return result


@app.get("/get_server_info")
async def get_server_info():
    internal_states = await _global_state.tokenizer_manager.get_internal_state()
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        **_global_state.scheduler_info,
        **internal_states,
    }





# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route("/generate", methods=["POST", "PUT"])
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error(f"Error: {e}")
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return ret
        except ValueError as e:
            logger.error(f"Error: {e}")
            return _create_error_response(e)




@app.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache():
    """Flush the radix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )



def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    
    tokenizer_manager = _launch_subprocesses(server_args=server_args)
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
        )
    )

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        warmup_thread.join()


def _wait_and_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
    launch_callback: Optional[Callable[[], None]] = None,
):
    headers = {}
    url = server_args.url()

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    model_info = res.json()

    # Send a warmup request
    request_name = "/generate"
    max_new_tokens = 8
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    
    json_data["text"] = "The capital city of France is"

    try:
        res = requests.post(
            url + request_name,
            json=json_data,
            headers=headers,
            timeout=600,
        )
        assert res.status_code == 200, f"{res}"
    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    # Debug print
    logger.info(f"{res.json()=}")

    logger.info("The server is fired up and ready to roll!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")


    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if launch_callback is not None:
        launch_callback()
