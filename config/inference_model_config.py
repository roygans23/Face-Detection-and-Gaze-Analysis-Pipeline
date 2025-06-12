from dataclasses import dataclass
from typing import Optional
from config.model_config import ModelConfig

@dataclass
class InferenceModelConfig(ModelConfig):
    embedding_layer_name: str = 'last_bn'
    copy_batchnorm_stats_from_pretrained: bool = False