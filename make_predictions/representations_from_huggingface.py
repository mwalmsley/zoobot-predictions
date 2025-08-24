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

from dataclasses import dataclass, field
from typing import Generator, Optional

import hydra
# from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore

import timm
import torch
import datasets as hf_datasets

from galaxy_datasets.pytorch.galaxy_datamodule import HuggingFaceDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform


@dataclass
class DatasetConfig:
    dataset_name: str = "mwalmsley/gz_euclid"
    split_name: str = "tiny"
    max_items: Optional[int] = None

@dataclass
class ModelConfig:
    model_name: str = "hf_hub:mwalmsley/euclid_encoder_mae_zoobot_vit_small_patch8_224"
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 64
    })
    indices_to_use: list = field(default_factory=lambda: [0, 9, 10, 11])  # which layers to extract features from

@dataclass
class HardwareConfig:
    accelerator: str = "gpu"
    gpu: str = 'a100'
    num_workers: int = 4  # per device
    prefetch_factor: int = 4  # per device

@dataclass
class MyConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

cs = ConfigStore.instance()
cs.store(name="config", node=MyConfig)

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
    model = _create_vision_transformer('zoobot_vit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def setup(cfg, split):

    model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained)

    dataset_dict: hf_datasets.DatasetDict = hf_datasets.load_dataset(cfg.dataset.dataset_name, cfg.dataset.split_name)  # type: ignore

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


def generate_features(cfg, split) -> Generator[dict, None, None]:  # yield type, send type, return type
    model, datamodule = setup(cfg, split)
    datamodule.setup('predict')
    for batch in datamodule.predict_dataloader():
        features = get_features(batch, model, indices_to_use=cfg.model.indices_to_use)
        yield features

# should work for any timm model
def get_features(batch, model, indices_to_use) -> dict:
    images = batch['image']  # [b, 3, h, w]
    with torch.no_grad():
        intermediates = model.forward_intermediates(images, indices=indices_to_use, intermediates_only=True, norm=True)
    features = {}
    for i, intermediate in zip(indices_to_use, intermediates):
        # if pool:  for now, always pool
            # https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
            # intermediate = torch.nn.functional.adaptive_avg_pool2d(intermediate, (1, 1))
            # [b, hidden_d, p, p] -> [b, hidden_d]
        intermediate = torch.mean(intermediate, dim=(-2, -1), keepdim=False)
        features['pooled_features_block_{}'.format(i)] = intermediate  # type: ignore
        features['id_str'] = batch['id_str']
    return features  
    # dict with keys, each key has features or id_str for this batch, stored as arr
    # easy to convert to huggingface dataset

# from hf_datasets.Split import NamedSplits

@hydra.main(version_base=None, config_name="config")
def main(cfg):

    # gen = generate_features(cfg, split='train')
    # instance = next(gen)
    # for k, v in instance.items():
    #     print(k, v, v.dtype)

    ds_dict = {}
    for split in ['train', 'test']:
        # first split as arg to generator, second split names the output dataset split
        ds = hf_datasets.Dataset.from_generator(generate_features, gen_kwargs={'cfg': cfg, 'split': split}, split=split)  # type: ignore
        print(ds)
        ds_dict[split] = ds
    ds_dict = hf_datasets.DatasetDict(ds_dict)
    print(ds_dict)

    ds_dict.save_to_disk('euclid_tiny_representations')

    ds_dict.push_to_hub('mwalmsley/euclid_tiny_representations', private=True)


if __name__ == '__main__':
    main()