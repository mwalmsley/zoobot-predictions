import os
import logging

import torch

from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.training import finetune


def get_zoobot_model_to_use(config):
    if config.cluster.accelerator == 'gpu':
        map_location = None
        torch.set_float32_matmul_precision('medium')  # use Ampere cores on A100, H100 by default
    elif config.cluster.accelerator == 'cpu':
        logging.info('Forcing model to load on CPU')
        map_location = torch.device('cpu')

    logging.info('Returning model from checkpoint: {}'.format(config.model.checkpoint_loc))  # automatically has all the hparams saved e.g. image size

    if config.model.zoobot_class == 'ZoobotTree':
        # can't just load encoder from HF, need full lightning checkpoint
        model = define_model.ZoobotTree.load_from_checkpoint(config.model.checkpoint_loc, map_location=map_location)
    elif config.model.zoobot_class == 'ZoobotEncoder':
        logging.warning('Loading encoder only, for representations')
        from zoobot.pytorch.training import representations 
        model = representations.ZoobotEncoder.load_from_name(config.model.encoder_name)
    # for these two, we need to specify the location of the original encoder checkpoint
    # even though the weights of the original encoder are never used, it means we can remake the same encoder *architecture*
    # if you are making on predictions on the same system you did the finetuning on, 
    # this isn't necessary as pytorch lightning records where the encoder checkpoint is - but best to be explicit
    elif config.model.zoobot_class == 'FinetuneableZoobotClassifier':
        model = finetune.FinetuneableZoobotClassifier.load_from_name(config.model.encoder_name)
    elif config.model.zoobot_class == 'FinetuneableZoobotTree':
        model = finetune.FinetuneableZoobotTree.load_from_name(config.model.encoder_name)
    
    else:
        raise ValueError(config.model.zoobot_class)
    
    return model        
        
# def get_model_name(checkpoint_loc):
#     return os.path.basename(os.path.dirname(os.path.dirname(checkpoint_loc)))
