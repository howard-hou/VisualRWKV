from dataclasses import dataclass


@dataclass
class AdapterConfig:
    vit_model: str = "google/vit-base-patch16-224-in21k"
    t5_model: str = "t5-small"
    num_task_embeddings: int = 32
    feature_size: int = 256
    is_freeze_vit: bool = True
    max_length: int = 64


@dataclass
class RWKVConfig:
    model_path: str
    strategy: str = "cpu fp32"


@dataclass
class VisualRWKVConfig:
    model_name: str = "RWKV/rwkv-4-169m-pile"
    max_length: int = 64
