from typing import List
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.functional import cross_entropy
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.functional import cross_entropy

class TransformerLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        step = max(1, self._step_count)
        scale = min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        
        return [(self.d_model ** (-0.5)) * scale for _ in self.base_lrs]