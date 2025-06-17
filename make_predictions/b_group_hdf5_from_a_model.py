import logging
import os
import glob

import hydra
from omegaconf import DictConfig

from zoobot.shared import save_predictions, load_predictions


@hydra.main(version_base=None, config_path="../conf", config_name='default')
def main(config: DictConfig):

    # model_name = utils.get_model_name(config.model.checkpoint_loc)
    pred_locs = glob.glob(os.path.join(config.predictions_dir, config.model.model_name, '*_preds.hdf5'))
    assert pred_locs
    pred_locs.sort()
    logging.info('Found {} prediction hdf5 snippets to load e.g. {}'.format(len(pred_locs), pred_locs[0]))
    
    galaxy_id_df, predictions, label_cols = load_predictions.load_hdf5s(pred_locs)
    # from galaxy_datasets.shared import label_metadata
    # label_cols = label_metadata.decals_all_campaigns_ortho_label_cols # TODO temp
    # df: rows of id_str, hdf5_loc
    # predictions: model predictions, usually dirichlet concentrations, like (galaxy, answer, forward pass)
    # will have been concatenated across locs
    assert not any(galaxy_id_df['id_str'].duplicated()), galaxy_id_df['id_str'].value_counts()  # we don't expect any duplicate predictions
    assert len(galaxy_id_df) == len(predictions), (len(galaxy_id_df), len(predictions))
    assert len(label_cols) == predictions.shape[1], (len(label_cols), predictions.shape[1])
    logging.info('All predictions loaded')

    # write to hdf5, re-using the save_predictions func


    question_suffix = config.aggregation.question_suffix 
    if question_suffix is not None:
        if any([question_suffix in col for col in label_cols]):
            logging.info(f'survey suffix {question_suffix} detected in label_cols')
            # rewriting variables, feels clearer
            logging.info(f'Saving only survey suffix {question_suffix} predictions, for clarity and storage space')
            question_has_target_suffix = [question_suffix in col for col in label_cols]
            predictions = predictions[:, question_has_target_suffix]
            # label_cols = label_cols[question_has_target_suffix]
            label_cols = [col for col in label_cols if question_suffix in col]

    save_loc = os.path.join(config.predictions_dir, config.model.model_name, 'grouped.hdf5')

    # all predictions must fit in memory
    save_predictions.predictions_to_hdf5(predictions, galaxy_id_df['id_str'].values, label_cols, save_loc)

    logging.info('Predictions aggregated to {} - exiting gracefully'.format(save_loc))



if __name__ == '__main__':

    # each model produces many prediction hdf5 files
    # these are converted to many parquet files (assuming single dropout forward pass)
    # this script then concat's those parquet files into one file

    logging.basicConfig(level=logging.INFO)

    main()

    # parser = argparse.ArgumentParser(description='aggregate hdf5 predictions across snippets from one model')
    # parser.add_argument('--checkpoint-name', dest='model_name', type=str, default='_desi_pytorch_v5_hpv2_train_all_notest_m1.hdf5')
    # parser.add_argument('--predictions-dir', dest='predictions_dir', type=str, default='/share/nas2/walml/galaxy_zoo/decals/dr8/predictions')
    # parser.add_argument('--save-loc', dest='save_loc', type=str)
    # args = parser.parse_args()

    # main(args.model_name, args.predictions_dir, args.save_loc)
