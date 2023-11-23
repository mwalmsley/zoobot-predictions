import os
import logging

import torch

from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.training import finetune
from zoobot.shared import schemas


def get_zoobot_model_to_use(config):
    if config.cluster.accelerator == 'gpu':
        map_location = None
        torch.set_float32_matmul_precision('medium')  # use Ampere cores on A100, H100 by default
    elif config.cluster.accelerator == 'cpu':
        logging.info('Forcing model to load on CPU')
        map_location = torch.device('cpu')

    logging.info('Returning model from checkpoint: {}'.format(config.model.checkpoint_loc))  # automatically has all the hparams saved e.g. image size

    if config.model.zoobot_class == 'ZoobotTree':
        return define_model.ZoobotTree.load_from_checkpoint(config.model.checkpoint_loc, map_location=map_location)
    
    # for these two, we need to specify the location of the original encoder checkpoint
    # even though the weights of the original encoder are never used, it means we can remake the same encoder *architecture*
    # if you are making on predictions on the same system you did the finetuning on, 
    # this isn't necessary as pytorch lightning records where the encoder checkpoint is - but best to be explicit
    elif config.model.zoobot_class == 'FinetuneableZoobotClassifier':
        assert config.model.encoder_loc is not None
        return finetune.FinetuneableZoobotClassifier.load_from_checkpoint(config.model.checkpoint_loc, checkpoint_loc=config.model.encoder_loc)
    elif config.model.zoobot_class == 'FinetuneableZoobotTree':
        assert config.model.encoder_loc is not None
        schema = getattr(schemas, config.model.schema_name)
        return finetune.FinetuneableZoobotTree.load_from_checkpoint(
            config.model.checkpoint_loc, checkpoint_loc=config.model.encoder_loc, schema=schema)

def get_model_name(checkpoint_loc):
    return os.path.basename(os.path.dirname(os.path.dirname(checkpoint_loc)))
