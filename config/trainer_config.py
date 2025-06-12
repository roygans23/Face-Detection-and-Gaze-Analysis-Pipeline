from dataclasses import dataclass
from typing import Optional
import torch.optim.lr_scheduler as lr_scheduler_module
import torch

@dataclass
class TrainerConfig:
    device: str = 'cpu'
    num_epochs: int = 20
    criterion: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler: Optional[lr_scheduler_module._LRScheduler] = None
    log_dir: str = 'tensorboard_logs/experiment'
