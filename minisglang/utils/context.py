from dataclasses import dataclass
import torch


@dataclass
class Context:
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0


_CONTEXT = Context()


def get_context() -> Context:
    """Get the current context."""
    return _CONTEXT


def set_context(
    cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0
) -> None:
    """Set the current context."""
    global _CONTEXT
    _CONTEXT = Context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)

def reset_context() -> None:
    """Reset the current context to default values."""
    global _CONTEXT
    _CONTEXT = Context()