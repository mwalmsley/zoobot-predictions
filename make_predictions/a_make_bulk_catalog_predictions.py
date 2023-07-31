import os
import glob
import logging

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig

from galaxy_datasets.shared import label_metadata
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.pytorch.estimators import define_model

import utils

"""
Uses the snippets to define subsets of galaxies, and makes predictions with one model on a subset
Easy to distribute across a slurm cluster (see .sh script of the same name)
"""

# logs will appear in .hydra
# https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/
@hydra.main(version_base=None, config_path="../conf")
def main(config: DictConfig):

    if not os.path.isdir(config.galaxies.predictions_dir):
        os.mkdir(config.galaxies.predictions_dir)

    snippet_locs = glob.glob(os.path.join(config.galaxies.snippet_dir, '*.csv'))
    assert snippet_locs
    logging.info('Snippets: {} (e.g. {})'.format(len(snippet_locs), snippet_locs[0]))
    snippet_locs.sort() # deterministic order, acts inplace

    if config.galaxies.subset_loc != '':
        subset_df = pd.read_parquet(config.galaxies.subset_loc, columns=['file_loc'])
        logging.info('Will filter galaxies to those with file_loc in {} ({}, e.g. {})'.format(config.galaxies.subset_loc, len(subset_df), subset_df.iloc[0]['file_loc']))
    else:
        logging.info('No subset loc provided, not filtering galaxies beyond those included in snippets')
        subset_df = None

    logging.info('Loading zoobot model (no finetuning head')
    model = get_zoobot_model_to_use(config.model.checkpoint_loc, accelerator=config.cluster.accelerator)

    predict_on_snippets(
        config,
        snippet_locs,
        subset_df, 
        model
    )

    logging.info('Finished. Enjoy your predictions.')


def predict_on_snippets(
        config,
        snippet_locs,
        subset_df, 
        model
    ):
    logging.info('Using snippet indices: {} to {}'.format(config.cluster.start_snippet_index, config.cluster.end_snippet_index))

    num_snippets_to_load = config.cluster.snippets_per_node_batch  # must be evenly divisible into length of relevant snippets (e.g. 10, 20, 100 for 0-100)

    this_batch_start = config.cluster.start_snippet_index
    this_batch_end = config.cluster.start_snippet_index + num_snippets_to_load

    while this_batch_start <= config.cluster.end_snippet_index:
        logging.info('Prediction batch: {} to {}'.format(this_batch_start, this_batch_end))

        this_batch_snippet_locs = snippet_locs[this_batch_start:this_batch_end]
        logging.info('Snippets in this batch: {}'.format(len(this_batch_snippet_locs)))

        if len(this_batch_snippet_locs) == 0:
            logging.warning('Empty snippet batch {}-{} - hopefully the end of the dataset? Exiting.'.format(this_batch_start, this_batch_end))
            break
        else:
            # saves to e.g. predictions_dir / model_name / 873000_873499_preds_{model_name}.hdf5
            model_name = utils.get_model_name(config.model.checkpoint_loc)  # first dir is 'checkpoints', second is model name
            subfolder = os.path.join(config.galaxies.predictions_dir, model_name)
            if not os.path.isdir(subfolder):
                try:
                    os.mkdir(subfolder)
                except FileExistsError:
                    pass  # race condition where another node tries to make it at the same instant
            save_name = os.path.basename(snippet_locs[this_batch_start]).replace('.csv', f'_preds_{model_name}.hdf5')
            save_loc = os.path.join(subfolder, save_name)

            if os.path.isfile(save_loc) and not config.cluster.overwrite:
                logging.warning('Overwrite=False and predictions exist at {} - skipping'.format(save_loc))
            else:
                # set columns
                df = pd.concat([pd.read_csv(snippet_loc, usecols=['id_str', 'file_loc']) for snippet_loc in this_batch_snippet_locs])

                if subset_df is not None:
                    logging.info('Filtering')
                    df = df[df['file_loc'].isin(subset_df['file_loc'])]
                    logging.info('Snippet galaxies in subset: {}'.format(len(df)))
    
                if len(df) == 0:
                    logging.warning('df to predict is empty, likely due to none in subset - skipping predictions. Check paths line up!')
                else:
                    predict_on_snippet(config, df, save_loc, model)

            this_batch_start += num_snippets_to_load
            this_batch_end += num_snippets_to_load
            if this_batch_end > config.cluster.end_snippet_index:
                this_batch_end = config.cluster.end_snippet_index


def predict_on_snippet(config, df, save_loc, model):

    label_cols = getattr(label_metadata, config.model.label_cols_name)
    logging.info(f'{len(label_cols)} label cols')

    datamodule_kwargs = {
        'greyscale': not config.model.color,
        'resize_after_crop': config.model.resize_after_crop,
        # these are cluster params, model doesn't care but hardware does
        'batch_size': config.cluster.batch_size, 
        'num_workers': config.cluster.num_workers}
    trainer_kwargs = {'devices': config.cluster.devices, 'accelerator': config.cluster.accelerator}

    predict_on_catalog.predict(
        df, model, config.galaxies.n_samples, label_cols, save_loc,
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )

    logging.info('Predictions saved to {}'.format(save_loc))


def get_zoobot_model_to_use(checkpoint_loc, accelerator=None):
    logging.info('Returning model from checkpoint: {}'.format(checkpoint_loc))  # automatically has all the hparams saved e.g. image size
    if accelerator == None:
        map_location = None
    elif accelerator == 'cpu':
        logging.info('Forcing model to load on CPU')
        map_location = torch.device('cpu')
    return define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc, map_location=map_location)


if __name__ == '__main__':

    main()  # config passed by hydra
