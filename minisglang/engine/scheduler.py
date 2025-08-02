from enum import Enum
class Mode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    
    
class Scheduler:
    """
    For receiving reqs from tokenizer, managing the waiting queue 
    """
    def __init__(
        self,
        tp_rank: int,
    ):
        self.waiting_queue = []
        self.tp_rank = tp_rank