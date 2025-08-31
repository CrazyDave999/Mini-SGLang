import dataclasses
from dataclasses import dataclass

from typing import Dict, List, Optional, Union
import uuid

from minisglang.layers.sampler import SamplingParams


@dataclasses.dataclass
class BatchStrOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[dict]
    # The output decoded strings
    output_strs: List[str]
    # The token ids
    output_ids: Optional[List[int]]

@dataclasses.dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    
    stream: bool = False
    
    def normalize_batch_and_args(self):
        # determin batch size
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
        else:
            raise
        
        if self.is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
        else:
            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            elif isinstance(self.sampling_params, dict):
                self.sampling_params = [self.sampling_params] * self.batch_size
            elif not isinstance(self.sampling_params, list):
                raise
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            elif not isinstance(self.rid, list):
                raise
    def __getitem__(self, i):
        
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            rid=self.rid[i],
            sampling_params=self.sampling_params[i] if self.sampling_params is not None else None, 
            stream=self.stream
        )
        
@dataclass
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The sampling parameters
    sampling_params: SamplingParams
    

@dataclass
class FlushCacheReqInput:
    pass


@dataclass
class FlushCacheReqOutput:
    success: bool