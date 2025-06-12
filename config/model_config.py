from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    # Mode
    train_mode: bool = True  # or False for 'eval'
    pretrained: bool = True
    copy_batchnorm_stats_from_pretrained : bool = False

    # Architecture
    arch: str = "vgg"
    num_classes: int = 1000
    pretrained: bool = True

    # Checkpoint
    checkpoint_path: Optional[str] = None
    data_parallel_patch : bool = False # If 'feature' layers are wrapped in 'DataParallel' layers for faster execution
    device: str = "cpu"
    
    # Training-specific
    freeze_grads: bool = False
    additional_trainable_keywords: Optional[List[str]] = None

    # Debug / logging
    show_logs: bool = False
