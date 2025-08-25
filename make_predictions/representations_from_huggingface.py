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


# https://hydra.cc/docs/tutorials/structured_config/hierarchical_static_config/
@dataclass
class DatasetConfig:
    dataset_name: str = "mwalmsley/gz_euclid"
    config_name: str = "default"
    max_items: Optional[int] = None

@dataclass
class ModelConfig:
    model_name: str = "hf_hub:mwalmsley/euclid_encoder_mae_zoobot_vit_small_patch8_224"
    pretrained: bool = True
    batch_size: dict = field(default_factory=lambda: {
        "a100": 1028,
        "l4": 1028
    })
    indices_to_use: list = field(default_factory=lambda: [0, 9, 10, 11])  # which layers to extract features from

@dataclass
class HardwareConfig:
    accelerator: str = "gpu"
    gpu: str = 'l4'
    num_workers: int = 1  # per device
    prefetch_factor: int = 2  # per device

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

    dataset_dict: hf_datasets.DatasetDict = hf_datasets.load_dataset(cfg.dataset.dataset_name, cfg.dataset.config_name)  # type: ignore

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
    model.to('cuda')
    model.eval()
    datamodule.setup('predict')
    for batch in datamodule.predict_dataloader():
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        features = get_features(batch, model, indices_to_use=cfg.model.indices_to_use)
        for f in features:
            yield f  # dict for single galaxy

# should work for any timm model
def get_features(batch, model, indices_to_use, to_cpu=True) -> dict:
    images = batch['image']  # [b, 3, h, w]
    with torch.no_grad():
        intermediates = model.forward_intermediates(images, indices=indices_to_use, intermediates_only=True, norm=True)

    # concat to [n_int, batch, hidden, patch, patch]
    intermediates = torch.stack(intermediates, dim=0)
    # print(intermediates.shape)

    # [n_int, batch, hidden]
    pooled = torch.mean(intermediates, dim=(-2, -1), keepdim=False)
    # print(pooled.shape)

    if to_cpu:
        pooled = pooled.cpu()
        # id_str = id_str.cpu()

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

    # detect datalabs, get token if so
    if os.path.isdir('/media/home/my_workspace/_credentials'):
        import json
        with open('/media/home/my_workspace/_credentials/secrets.txt') as f:
            token = json.load(f)['token']
    else:
        token = None

    torch.cuda.empty_cache()

    # debugging
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
            gen_kwargs={'cfg': cfg, 'split': split}, 
            features=features,
            split=split
        )
        # print(ds)
        ds_dict[split] = ds
    ds_dict = hf_datasets.DatasetDict(ds_dict)
    # print(ds_dict)

    # ds_dict.save_to_disk(f'datasets/{cfg.dataset.dataset_name}_{cfg.dataset.config_name}')

    
    ds_dict.push_to_hub(
        f'{cfg.dataset.dataset_name}_embeddings',
        config_name=cfg.dataset.config_name,
        token=token,
        private=False)


if __name__ == '__main__':
    main()