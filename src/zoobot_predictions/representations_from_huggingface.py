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
# pip install timm datasets lightning hydra-core scikit-learn wandb lightly
# pip install -e repos/galaxy-datasets/.
# pip install --no-deps -e repos/galaxy-datasets/.

# export HF_HOME=/media/home/team_workspaces/Galaxy-Zoo-Euclid/huggingface

# from abc import ABC, abstractmethod
import os
from dataclasses import dataclass, field
import sys
from typing import Generator, Optional, Any
import json

import hydra
# from hydra.utils import instantiate
# from hydra.core.config_store import ConfigStore

import timm
import torch
import datasets as hf_datasets

from galaxy_datasets.pytorch.galaxy_datamodule import HuggingFaceDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform


# https://hydra.cc/docs/tutorials/structured_config/hierarchical_static_config/

from datasets import load_dataset

# sys.path.append("/home/walml/repos/zoobot-predictions/zoobot_predictions/zoobot_predictions")
#from zoobot_predictions 
import config_options  # to register configs




def create_datamodule(cfg, split, token):
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
    
    return datamodule


def generate_features(cfg, split, token) -> Generator[dict, None, None]:  # yield type, send type, return type

    model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained)
    model.to('cuda')
    model.eval()

    datamodule = create_datamodule(cfg, split, token)
    # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    # with torch.autocast(device_type=device, dtype=torch.float16): TODO if mixed precision needed
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
            # print(row_features['pooled_features_block_{}'.format(block_name)].shape)
        features.append(row_features)
    
    return features 
    # list of dicts, each dict keyed by features, each value for one galaxy
    # standard flat/huggingface format
    

@hydra.main(version_base=None, config_name="config")
def main(cfg):

    on_datalabs = os.path.isdir('/media/home/my_workspace')

    # datalabs only, for token
    if on_datalabs:
        torch.set_float32_matmul_precision('medium')  # L4
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
    features_dict = {
        'id_str': Value('string')
    }
    for index in cfg.model.indices_to_use:
        features_dict[f'pooled_features_block_{index}'] = List(feature=Value('float32'), length=cfg.model.embed_dim)

    features = Features(features_dict)
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

    # if on_datalabs:
    #     ds_dict.save_to_disk(f'datasets/' + config_name)

    
    # ds_dict.push_to_hub(
    #     f'{cfg.dataset.dataset_name}_embeddings',
    #     config_name=config_name,
    #     token=token,
    #     private=False)


if __name__ == '__main__':
    main()


    # import timm
    # model = timm.create_model('hf_hub:mwalmsley/baseline-encoder-regression-convnext_base', pretrained=True)
    # print(model.num_features)

    """
    python make_predictions/representations_from_huggingface.py +dataset=debug +model=mae +hardware=office
    python make_predictions/representations_from_huggingface.py +dataset=debug +model=convnext_base +hardware=office

    python make_predictions/representations_from_huggingface.py +dataset=debug +model=convnext_base +hardware=office ++model.model_name=hf_hub:mwalmsley/zoobot-encoder-euclid-convnext-base
    python _representations_from_huggingface.py +dataset=euclid_q1 +model=convnext_base +hardware=datalabs ++model.model_name=hf_hub:mwalmsley/zoobot-encoder-euclid-convnext-base

    python make_predictions/representations_from_huggingface.py +dataset=debug +model=maxvit_base +hardware=office ++model.model_name=hf_hub:mwalmsley/zoobot-encoder-euclid-maxvit-base
    """

    # regression encoder = maxvit
    # this was retrained using the new timm optimiser and did well
    # local:/share/nas2/walml/repos/gz-evo/results/baselines/regression/maxvit_base_534895718_1753972557/checkpoints/epoch\=23-step\=12432.ckpt
    # and excellent (best) gz euclid finetuned version
    # r8zxu9aj
    # neither are loaded/uploaded yet
    # linear transfer wasn't quite as good as v4 convnext base, but fully finetuned was better

    # regression encoder - convnext base
    # above is the v4 best effort, evo only
    # uploaded using the gz-evo encoder_to_hub script (so full timm encoder)
    # gz euclid finetuned version is
    # /share/nas2/walml/repos/zoobot-foundation/results/finetune/dnb_debug/ty2yh1qv/checkpoints/17.ckpt 
    # not yet loaded/uploaded