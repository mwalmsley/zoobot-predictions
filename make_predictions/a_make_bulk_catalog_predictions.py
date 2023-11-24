import os
import glob
import logging
import time
import datetime

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig

from galaxy_datasets.shared import label_metadata
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.pytorch.datasets import webdatamodule
from zoobot.shared import save_predictions

import webdataset as wds

import model_utils

"""
Uses the snippets to define subsets of galaxies, and makes predictions with one model on a subset
Easy to distribute across a slurm cluster (see .sh script of the same name)
"""

class PredictAbstract():
    def __init__(self, config, shard_extension) -> None:
        self.config = config
            
        if not os.path.isdir(self.config.predictions_dir):
            os.mkdir(self.config.predictions_dir)

        if self.config.galaxies.subset_loc != '':
            self.subset_df = pd.read_parquet(self.config.galaxies.subset_loc, columns=['file_loc'])
            logging.info('Will filter galaxies to those with file_loc in {} ({}, e.g. {})'.format(self.config.galaxies.subset_loc, len(self.subset_df), self.subset_df.iloc[0]['file_loc']))
        else:
            logging.info('No subset loc provided, not filtering galaxies beyond those included in snippets')
            self.subset_df = None

        logging.info('Loading Zoobot model')
        self.model = model_utils.get_zoobot_model_to_use(self.config)

        if 'representations' in self.config.predictions_dir:
            logging.warning('representations mode - ignoring label_cols')
            self.label_cols = None  # Zoobot will fill with [feat_0, feat_1 etc]
        else:
            self.label_cols = getattr(label_metadata, self.config.model.label_cols_name)
            logging.info(f'{len(self.label_cols)} label cols')

        self.datamodule_kwargs = {
            # these are cluster params, model doesn't care but hardware does
            'batch_size': self.config.cluster.batch_size, 
            'num_workers': self.config.cluster.num_workers,
            # these aug params vary by model
            'crop_scale_bounds': self.config.model.crop_scale_bounds,
            'crop_ratio_bounds': self.config.model.crop_ratio_bounds,
            'resize_after_crop': self.config.model.resize_after_crop,
        }
        self.trainer_kwargs = {
            'devices': self.config.cluster.devices,
            'accelerator': self.config.cluster.accelerator,
            'logger': False,
            'precision': self.config.cluster.precision
        }

        self.shard_extension = shard_extension
        shard_locs = glob.glob(os.path.join(self.config.galaxies.shard_dir, f'*.{self.shard_extension}'))
        assert shard_locs
        logging.info('Shards: {} (e.g. {})'.format(len(shard_locs), shard_locs[0]))
        shard_locs.sort() # deterministic order, acts inplace
        self.shard_locs = shard_locs



    def predict_on_shards(self):
        logging.info('Using snippet indices: {} to {}'.format(self.config.cluster.start_shard_index, self.config.cluster.end_shard_index))

        max_snippets_to_load = self.config.cluster.shards_per_node_batch
        this_batch_start = self.config.cluster.start_shard_index
        while this_batch_start <= self.config.cluster.end_shard_index:

            # select the snippets to load
            # either simply get this index + max snippets to load, or,
            # stop at end_shard_index
            this_batch_end = min(self.config.cluster.end_shard_index, this_batch_start + max_snippets_to_load)

            logging.info('Prediction batch: {} to {}'.format(this_batch_start, this_batch_end))

            this_batch_shard_locs = self.shard_locs[this_batch_start:this_batch_end]
            logging.info('Shards in this batch: {}'.format(len(this_batch_shard_locs)))

            if len(this_batch_shard_locs) == 0:
                logging.warning('Empty shard batch {}-{} - hopefully the end of the dataset? Exiting.'.format(this_batch_start, this_batch_end))
                break
            else:
                # saves to e.g. predictions_dir / model_name / 873000_873499_preds_{model_name}.hdf5
                # model_name = utils.get_model_name(self.config.model.checkpoint_loc)  # first dir is 'checkpoints', second is model name
                subfolder = os.path.join(self.config.predictions_dir, self.config.model.model_name)
                # subfolder = self.config.predictions_dir
                if not os.path.isdir(subfolder):
                    try:
                        os.mkdir(subfolder)
                    except FileExistsError:
                        pass  # race condition where another node tries to make it at the same instant
                # will be named like the FIRST shard in this shard batch
                # all galaxies in the shard batch will be included, not just the first - don't worry!
                save_name = os.path.basename(this_batch_shard_locs[0]).replace(f'.{self.shard_extension}', f'_preds.hdf5')
                save_loc = os.path.join(subfolder, save_name)

                if os.path.isfile(save_loc) and not self.config.cluster.overwrite:
                    logging.warning('Overwrite=False and predictions exist at {} - skipping'.format(save_loc))
                else:
                    # load galaxy details
                    self.load_shard_and_predict(this_batch_shard_locs, save_loc)

                this_batch_start = this_batch_end


    def load_shard_and_predict(self, this_batch_shard_locs, save_loc):
        raise NotImplementedError


class PredictSnippets(PredictAbstract):
    def __init__(self, config) -> None:
        super().__init__(config, shard_extension = 'csv')
        self.datamodule_kwargs.update({
            'greyscale': not self.config.model.color
        })
        

    def load_shard_and_predict(self, this_batch_shard_locs, save_loc):
        df = pd.concat([pd.read_csv(snippet_loc, usecols=['id_str', 'file_loc']) for snippet_loc in this_batch_shard_locs])

        if self.subset_df is not None:
            logging.info('Filtering')
            df = df[df['file_loc'].isin(self.subset_df['file_loc'])]
            logging.info('Snippet galaxies in subset: {}'.format(len(df)))
        
        if len(df) == 0:
            logging.warning('df to predict is empty, likely due to none in subset - skipping predictions. Check paths line up!')
        else:
            self.predict_on_snippet(df, save_loc)


    def predict_on_snippet(self, df, save_loc):
        # assert config.cluster.shards_per_node_batch >= config.cluster.batch_size  # otherwise we 
        predict_on_catalog.predict(
            df, self.model, self.config.galaxies.n_samples, self.label_cols, save_loc,
            datamodule_kwargs=self.datamodule_kwargs,
            trainer_kwargs=self.trainer_kwargs
        )
        logging.info('Predictions saved to {}'.format(save_loc))


class PredictWDS(PredictAbstract):
    def __init__(self, config) -> None:
        super().__init__(config, shard_extension = 'tar')
        self.datamodule_kwargs.update({
            'color': self.config.model.color
        })

    def load_shard_and_predict(self, this_batch_shard_locs, save_loc):

        id_str_tuples = list(wds.WebDataset(this_batch_shard_locs, shardshuffle=False).to_tuple('__key__'))
        # these are (id_str,) format, pick first element
        image_id_strs = [x[0] for x in id_str_tuples]
        # print(image_id_strs)
        # exit()
        
        import pytorch_lightning as pl
        # from zoobot.pytorch.dataset
        trainer = pl.Trainer(max_epochs=-1, inference_mode=True, **self.trainer_kwargs)
        
        datamodule = webdatamodule.WebDataModule(
            train_urls =this_batch_shard_locs,
            predict_urls=this_batch_shard_locs,
            label_cols=self.label_cols,
            **self.datamodule_kwargs
        )
        # datamodule.setup('train')
        # for (im, label) in datamodule.train_dataloader():
        datamodule.setup('predict')
        # for (im,) in datamodule.predict_dataloader():
        #     print(type(im))        
        #     print(len(im))
        #     print(im[0].shape)
        #     exit()

        # copied from zoobot.shared.predict_on_catalog
        # this might eventually move there TODO
        logging.info('Beginning predictions')
        start = datetime.datetime.fromtimestamp(time.time())
        logging.info('Starting at: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S')))
        predictions = torch.stack(
            [   
                # trainer.predict gives [(galaxy, answer), ...] list, batchwise
                # concat batches
                torch.concat(trainer.predict(self.model, datamodule), dim=0)
                for _ in range(self.config.galaxies.n_samples)
            ], 
            dim=-1).numpy()  # now stack on final dim for (galaxy, answer, dropout) shape
        logging.info('Predictions complete - {}'.format(predictions.shape))

        logging.info(f'Saving predictions to {save_loc}')
        save_predictions.predictions_to_hdf5(predictions, image_id_strs, self.label_cols, save_loc)


# logs will appear in .hydra
# https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/
@hydra.main(version_base=None, config_path="../conf", config_name='default')
def main(config: DictConfig):

    if config.galaxies.format == 'jpg':
        predicter = PredictSnippets(config)
    elif config.galaxies.format == 'wds':
        predicter = PredictWDS(config)      
    else:
        raise ValueError(config.galaxies.format)  

    predicter.predict_on_shards()

    logging.info('Finished. Enjoy your predictions.')



if __name__ == '__main__':

    main()  # config passed by hydra
