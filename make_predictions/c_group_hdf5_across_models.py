import logging
import argparse
import os
import glob

import hydra
from omegaconf import DictConfig
import numpy as np

from zoobot.shared import save_predictions, load_predictions


@hydra.main(version_base=None, config_path="../conf")
def main(config: DictConfig):

    glob_str = os.path.join(config.predictions_dir, '*/grouped.hdf5')  # * for predictions from any model
    
    logging.info('Searching for {}'.format(glob_str))
    pred_locs = glob.glob(glob_str)
    assert pred_locs
    logging.info('Found {} (hopefully aggregated) prediction hdf5 snippets to load: {}'.format(len(pred_locs), pred_locs))

    # load individually, because they need to be stacked in new dimension

    all_predictions = []

    for loc_n, loc in enumerate(pred_locs):
        logging.info('Loading from {}'.format(loc))
    
        model_galaxy_id_df, model_predictions, model_label_cols = load_predictions.load_hdf5s(loc)
        # df: rows of id_str, hdf5_loc
        # predictions: model predictions, usually dirichlet concentrations, like (galaxy, answer, forward pass)
        assert not any(model_galaxy_id_df['id_str'].duplicated())  # we don't expect any duplicate predictions
        assert len(model_galaxy_id_df) == len(model_predictions)

        if loc_n == 0:
            # use as canonical example
            id_strs = model_galaxy_id_df['id_str'].values
            label_cols = model_label_cols
        else:
            # check against canonical example
            if not all(model_galaxy_id_df['id_str'].values == id_strs):  # elementwise as np arrays
                logging.critical(model_galaxy_id_df['id_str'].values[:10])
                logging.critical(id_strs[:10])
                raise ValueError('id strs dont match')
            if not np.all(model_label_cols == label_cols):
                logging.critical(model_label_cols)
                logging.critical(label_cols)
                raise ValueError('label cols dont match')

        # each model_predictions has shape (question, answer, pass)
        all_predictions.append(model_predictions)  # will stack this

    all_predictions = np.stack(all_predictions, axis=2)  # [(galaxy, question, pass), ...] to (galaxy, question, model, pass)
    logging.info('Stacked prediction shape: {}'.format(all_predictions.shape))


    # write to hdf5, re-using the save_predictions func
    save_loc = config.predictions_dir + '/grouped_across_models.hdf5'
    save_predictions.predictions_to_hdf5(all_predictions, id_strs, label_cols, save_loc)

    logging.info('Aggregated predictions stacked across models to {} - exiting gracefully'.format(save_loc))



if __name__ == '__main__':

    # each model produces many prediction hdf5 files
    # these are converted to many parquet files (assuming single dropout forward pass)
    # this script then concat's those parquet files into one file

    logging.basicConfig(level=logging.INFO)

    # parser = argparse.ArgumentParser(description='stack aggregated hdf5 predictions from different models')
    # parser.add_argument('--glob', dest='glob_str', type=str, default='/Users/user/repos/zoobot-predictions/example/predictions/evo_py_co_vittiny_224_STAR_grouped.hdf5')  # search for matches to this glob
    # parser.add_argument('--save-loc', dest='save_loc', type=str)
    # args = parser.parse_args()

    # main(args.glob_str, args.save_loc)

    main()

    """
    python make_predictions/c_group_hdf5_across_models.py \
        --glob /Users/user/repos/zoobot-predictions/example/predictions/evo_py_co_vittiny_224_STAR_grouped.hdf5 \
        --save-loc /Users/user/repos/zoobot-predictions/example/predictions/evo_py_co_vittiny_224_all.hdf5
    """