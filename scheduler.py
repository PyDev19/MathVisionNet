from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.functional import cross_entropy

class TransformerScheduler(_LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer,
        dim_embed: int,
        warmup_steps: int,
        last_epoch: int=-1,
        verbose: bool=False
    ) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
    
    def calc_lr(self, step: int) -> float:
        return self.dim_embed**(-0.5) * min(step**(-0.5), step * self.warmup_steps**(-1.5))
        
    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count)
        return [lr] * self.num_param_groups
