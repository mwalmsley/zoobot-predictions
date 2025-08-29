from dataclasses import dataclass, field
from typing import Generator, Optional, Any

from hydra.core.config_store import ConfigStore

# so timm knows about my custom model
from timm.models import register_model
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer
@register_model
def zoobot_vit_small_patch8_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        # from aug config
        img_size=224,
        # from vit config
        patch_size=8, depth=12, num_heads=6, embed_dim=384, mlp_ratio=4, global_pool='avg', num_classes=0)
    model = _create_vision_transformer('zoobot_vit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))  # type: ignore
    return model


# datasets

@dataclass
class DatasetConfig:
    dataset_name: str
    config_name: str
    max_items: Optional[int] = None

@dataclass
class DatasetGZEuclidConfig(DatasetConfig):
    dataset_name: str = "mwalmsley/gz_euclid"
    config_name: str = "default"
    max_items: Optional[int] = None

@dataclass
class DatasetDebugConfig(DatasetConfig):
    dataset_name: str = "mwalmsley/euclid_q1"
    config_name: str = "tiny-v1-gz_arcsinh_vis_y"
    max_items: Optional[int] = 64

@dataclass
class DatasetQ1Config(DatasetConfig):
    dataset_name: str = "mwalmsley/euclid_q1"
    config_name: str = "v1-gz_arcsinh_vis_y"
    max_items: Optional[int] = None

@dataclass
class DatasetRR2Config(DatasetConfig):
    dataset_name: str = "mwalmsley/euclid_rr2"
    config_name: str = "v2"
    max_items: Optional[int] = None

# models

@dataclass
class ModelConfig:
    model_name: str
    pretrained: bool
    batch_size: dict
    indices_to_use: list
    embed_dim: int  # in principle, could be a list matching indices_to_use

@dataclass
class MAEConfig(ModelConfig):
    model_name: str = "hf_hub:mwalmsley/euclid_encoder_mae_zoobot_vit_small_patch8_224"
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 1024,
        "l4": 1024,
        "t400": 2,
        'rtx2060': 128
    })
    embed_dim: int = 384
    indices_to_use: list = field(default_factory=lambda: [0, 9, 10, 11])  # which layers to extract features from


@dataclass
class LocalMAEConfig(ModelConfig):
    # model_name: str = "/home/walml/Dropbox (The University of Manchester)/euclid/euclid_morphology/models/u11vazbb/checkpoints/model.ckpt"
    model_name: str = "https://huggingface.co/mwalmsley/euclid-rr2-mae-lightning/resolve/main/model.ckpt"  # for datalabs
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 1024,
        "l4": 1024,
        "t400": 2,
        'rtx2060': 128
    })
    embed_dim: int = 384
    indices_to_use: list = field(default_factory=lambda: [0, 9, 10, 11])  # which layers to extract features from


@dataclass
class ConvNextBaseConfig(ModelConfig):
    # uploaded with gz_evo export to hub, must be full timm encoder
    model_name: str = "hf_hub:mwalmsley/baseline-encoder-regression-convnext_base" 
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 512,
        "l4": 512,
        "t400": 2,
        'rtx2060': 128
    })
    embed_dim: int = 1024
    indices_to_use: list = field(default_factory=lambda: [3])

@dataclass
class MaxViTBaseConfig(ModelConfig):
    # uploaded with gz_evo export to hub, must be full timm encoder
    model_name: str = "hf_hub:mwalmsley/zoobot-encoder-evo-maxvit_base" 
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 512,
        "l4": 512,
        "t400": 2,
        'rtx2060': 128
    })
    embed_dim: int = 768
    indices_to_use: list = field(default_factory=lambda: [4])

# hardware

@dataclass
class HardwareConfig:
    accelerator: str
    gpu: str
    num_workers: int  # per device
    prefetch_factor: int  # per device

@dataclass
class OfficeConfig(HardwareConfig):
    accelerator: str = "gpu"
    gpu: str = 't400'
    num_workers: int = 1
    prefetch_factor: int = 1

@dataclass
class HomeConfig(HardwareConfig):
    accelerator: str = "gpu"
    gpu: str = 'rtx2060'
    num_workers: int = 1
    prefetch_factor: int = 1

@dataclass
class DatalabsConfig(HardwareConfig):
    accelerator: str = "gpu"
    gpu: str = 'l4'
    num_workers: int = 1
    prefetch_factor: int = 2

@dataclass
class MyConfig:
    dataset: DatasetConfig
    model: ModelConfig
    hardware: HardwareConfig

# https://hydra.cc/docs/tutorials/structured_config/config_groups/
cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

cs.store(group="dataset", name="debug", node=DatasetDebugConfig)
cs.store(group="dataset", name="gz_euclid", node=DatasetGZEuclidConfig)
cs.store(group="dataset", name="euclid_q1", node=DatasetQ1Config)
cs.store(group="dataset", name="euclid_rr2", node=DatasetRR2Config)

cs.store(group="model", name="mae", node=MAEConfig)
cs.store(group="model", name="local_mae", node=LocalMAEConfig)
cs.store(group="model", name="convnext_base", node=ConvNextBaseConfig)
cs.store(group="model", name="maxvit_base", node=MaxViTBaseConfig)

cs.store(group="hardware", name="office", node=OfficeConfig)
cs.store(group="hardware", name="datalabs", node=DatalabsConfig)
cs.store(group="hardware", name="home", node=HomeConfig)

