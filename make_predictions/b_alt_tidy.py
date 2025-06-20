import os
import glob
import logging

import pandas as pd
from omegaconf import DictConfig
import hydra

# for now, assumes one model only

@hydra.main(version_base=None, config_path="../conf", config_name='default')
def main(config: DictConfig):

    # model_name = utils.get_model_name(config.model.checkpoint_loc)
    search_str = os.path.join(config.predictions_dir, config.model.model_name, '*_preds.csv')
    logging.info('Searching for predictions in {}'.format(search_str))
    pred_locs = glob.glob(search_str)
    assert pred_locs
    pred_locs.sort()

    question_suffix = config.aggregation.question_suffix 

    # load
    for pred_loc in pred_locs:
        logging.info('Loading predictions from {}'.format(pred_loc))

        # rewrite and save, without the extra columns
        pred_df = pd.read_csv(pred_loc)
        
        if question_suffix is not None:
            pred_df = pred_df[['id_str'] + [col for col in pred_df.columns if question_suffix in col]]

    logging.info('All predictions loaded, saving')

    save_loc = os.path.join(config.predictions_dir, config.model.model_name, 'grouped.csv')

    # save
    pred_df.to_csv(save_loc, index=False)
    pred_df.to_parquet(save_loc.replace('.csv', '.parquet'), index=False)

    logging.info('Predictions aggregated to {} - exiting gracefully'.format(save_loc))

if __name__ == '__main__':

    # each model produces many prediction hdf5 files
    # these are converted to many parquet files (assuming single dropout forward pass)
    # this script then concat's those parquet files into one file

    logging.basicConfig(level=logging.INFO)

    main()
