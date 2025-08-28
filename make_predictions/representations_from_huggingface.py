# in the closing part of my research, I can assume datasets are all now nicely on huggingface

# general large huggingface upload procedure

# create dataset using pandas and file paths 
# (a little slow checking if files exist, but only need to do once)

# save locally as arrow files
# https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.save_to_disk

# upload to hub using multiprocessing
# https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.push_to_hub

# with some patience, this should do for DR1

# this script then loads from the hub, and makes predictions

# generalised from foundation/export/mae_features.py


# for datalabs
# conda activate gpu
# pip install timm datasets lightning hydra-core scikit-learn
# pip install -e galaxy-datasets/.

# from abc import ABC, abstractmethod
import os
from dataclasses import dataclass, field
from typing import Generator, Optional, Any
import json

import hydra
# from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore

import timm
import torch
import datasets as hf_datasets

from galaxy_datasets.pytorch.galaxy_datamodule import HuggingFaceDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform


# https://hydra.cc/docs/tutorials/structured_config/hierarchical_static_config/

from datasets import load_dataset

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

@dataclass
class MAEConfig(ModelConfig):
    model_name: str = "hf_hub:mwalmsley/euclid_encoder_mae_zoobot_vit_small_patch8_224"
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 1028,
        "l4": 1028,
        "t400": 2
    })
    indices_to_use: list = field(default_factory=lambda: [0, 9, 10, 11])  # which layers to extract features from

@dataclass
class EvoConvNextBaseConfig(ModelConfig):
    # uploaded with gz_evo export to hub, must be full timm encoder
    model_name: str = "hf_hub:mwalmsley/baseline-encoder-regression-convnext_base" 
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 1028,
        "l4": 1028,
        "t400": 2
    })
    indices_to_use: list = field(default_factory=lambda: [3])  # which layers to extract features from

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
cs.store(group="model", name="evo_convnext_base", node=EvoConvNextBaseConfig)
cs.store(group="hardware", name="office", node=OfficeConfig)
cs.store(group="hardware", name="datalabs", node=DatalabsConfig)


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

        

def setup(cfg, split, token):

    model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained)


    dataset_dict: hf_datasets.DatasetDict = hf_datasets.load_dataset(cfg.dataset.dataset_name, cfg.dataset.config_name, token=token)  # type: ignore
    dataset_dict['predict'] = dataset_dict.pop(split)
    if cfg.dataset.max_items is not None:
            # don't exceed max items in dataset
            max_items = min(cfg.dataset.max_items, dataset_dict['predict'].num_rows)
            dataset_dict['predict'] = dataset_dict['predict'].select(range(max_items))
    
    # for now, just assume uses default_view_config
    view_config = default_view_config()
    view_config.output_size = 224
    view_config.erase_iterations = 0
    inference_transform = get_galaxy_transform(cfg=view_config)

    batch_size = cfg.model.batch_size[cfg.hardware.gpu]
    
    datamodule = HuggingFaceDataModule(
        dataset_dict=dataset_dict,
        train_transform=inference_transform, 
        test_transform=inference_transform,
        batch_size=batch_size,
        num_workers=cfg.hardware.num_workers,
        prefetch_factor=cfg.hardware.prefetch_factor
    )

    return model, datamodule


def generate_features(cfg, split, token) -> Generator[dict, None, None]:  # yield type, send type, return type
    model, datamodule = setup(cfg, split, token)
    model.to('cuda')
    model.eval()
    datamodule.setup('predict')
    for batch in datamodule.predict_dataloader():
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        features = get_features(batch, model, indices_to_use=cfg.model.indices_to_use)
        for f in features:
            yield f  # dict for single galaxy


# should work for any timm model
def get_features(batch, model, indices_to_use, to_cpu=True) -> list[dict]:
    images = batch['image']  # [b, 3, h, w]
    with torch.no_grad():
        intermediates = model.forward_intermediates(images, indices=indices_to_use, intermediates_only=True, norm=True)

    # concat to [block_index, batch, hidden], possibly also [patch, patch] for vit etc
    intermediates = torch.stack(intermediates, dim=0)

    if intermediates.ndim == 5:
        # yes, it has patch dims, pool them
        pooled = torch.mean(intermediates, dim=(-2, -1), keepdim=False)
    else:
        # no patch dims, all is well
        pooled = intermediates  # [block_index, batch, hidden]

    if to_cpu:
        pooled = pooled.cpu()

    features = []
    for galaxy_i, id_str in enumerate(batch['id_str']):
        row_features = {'id_str': id_str}
        for block_i, block_name in enumerate(indices_to_use):
            row_features['pooled_features_block_{}'.format(block_name)] = pooled[block_i, galaxy_i]  # type: ignore
        features.append(row_features)
    
    return features 
    # list of dicts, each dict keyed by features, each value for one galaxy
    # standard flat/huggingface format
    

@hydra.main(version_base=None, config_name="config")
def main(cfg):

    on_datalabs = os.path.isdir('/media/home/my_workspace')

    # datalabs only, for token
    if on_datalabs:
        with open('/media/home/my_workspace/_credentials/secrets.txt') as f:
            token = json.load(f)['token']
    else:
        token = None

    # torch.cuda.empty_cache()

    # gen = generate_features(cfg, split='train')
    # instance = next(gen)
    # print(instance)
    # print(instance[0])
    # exit()
    # for k, v in instance.items():
    #     print(k, v, v.dtype)


    from datasets.features import Value, List, Features
    
    features = Features({
        'id_str': Value('string'),
        'pooled_features_block_0': List(feature=Value('float32'), length=384),
        'pooled_features_block_9': List(feature=Value('float32'), length=384),
        'pooled_features_block_10': List(feature=Value('float32'), length=384),
        'pooled_features_block_11': List(feature=Value('float32'), length=384),
    })
    print(features)

    ds_dict = {}
    for split in ['train', 'test']:
        # first split as arg to generator, second split names the output dataset split
        ds = hf_datasets.Dataset.from_generator(
            generate_features, 
            gen_kwargs={'cfg': cfg, 'split': split, 'token': token}, 
            features=features,
            split=split  # type: ignore
        )
        # print(ds)
        ds_dict[split] = ds
    ds_dict = hf_datasets.DatasetDict(ds_dict)
    # print(ds_dict)

    safe_model_name = cfg.model.model_name.replace("hf_hub:mwalmsley/", "").replace("/", "_")
    # safe_dataset_name = cfg.dataset.dataset_name.replace("mwalmsley/", "")
    # config_name = f'{safe_dataset_name}_{cfg.dataset.config_name}_{safe_model_name}'
    config_name = f'{cfg.dataset.config_name}___{safe_model_name}'
    print(config_name)
    exit()

    if on_datalabs:
        ds_dict.save_to_disk(f'datasets/' + config_name)

    
    ds_dict.push_to_hub(
        f'{cfg.dataset.dataset_name}_embeddings',
        config_name=config_name,
        token=token,
        private=False)


if __name__ == '__main__':
    main()

    """
    python make_predictions/representations_from_huggingface.py +dataset=debug +model=mae +hardware=office
    python make_predictions/representations_from_huggingface.py +dataset=debug +model=evo_convnext_base +hardware=office
    """