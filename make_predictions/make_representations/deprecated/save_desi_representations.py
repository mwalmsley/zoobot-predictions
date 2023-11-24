
import os
import logging
import glob
import argparse

import torch
import pandas as pd

from zoobot.pytorch.training import representations
from zoobot.pytorch.estimators import define_model

from make_predictions import a_make_bulk_catalog_predictions


def main(checkpoint_loc, color, predictions_dir, start_snippet_index, end_snippet_index, overwrite, subset_loc):

    base_dir = '/share/nas2/walml/galaxy_zoo/decals/dr8'
    assert os.path.isdir(base_dir)
    checks_dir = os.path.join(base_dir, 'checks')
    assert os.path.isdir(checks_dir)

    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    snippet_locs = glob.glob(os.path.join(checks_dir, '*.parquet'))
    assert snippet_locs
    logging.info('Snippets: {} (e.g. {})'.format(len(snippet_locs), snippet_locs[0]))
    snippet_locs.sort() # deterministic order, acts inplace

    # resize_size = 224
    greyscale = not color

    label_cols = ['feat_' + str(n) for n in range(REPRESENTATION_DIM)]

    if subset_loc != '':
        subset_df = pd.read_parquet(subset_loc, columns=['file_loc'])
        logging.info('Will filter galaxies to those with file_loc in {} ({}, e.g. {})'.format(subset_loc, len(subset_df), subset_df.iloc[0]['file_loc']))
    else:
        logging.info('No subset loc provided, not filtering galaxies beyond those included in snippets')
        subset_df = None

    pytorch_model = define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc).encoder
    model = representations.ZoobotEncoder(pytorch_model, pyramid=False)

    a_make_bulk_catalog_predictions.predict_on_snippets(checkpoint_loc, predictions_dir, start_snippet_index, end_snippet_index, overwrite, snippet_locs, greyscale, label_cols, subset_df, model, n_samples=1)

    logging.info('Finished. Enjoy your predictions.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='predict by slice')
    parser.add_argument('--checkpoint-loc', dest='checkpoint_loc', type=str)
    parser.add_argument('--color', dest='color', default=False, action='store_true')
    parser.add_argument('--predictions-dir', dest='predictions_dir', type=str, default='/share/nas2/galaxy_zoo/decals/dr8/predictions')
    parser.add_argument('--start-snippet', dest='start_snippet_index', default=0, type=int)
    parser.add_argument('--end-snippet', dest='end_snippet_index', default=None, type=int)
    parser.add_argument('--task-id', dest='task_id', default=0, type=int)
    parser.add_argument('--overwrite', dest='overwrite', default=False, action='store_true')
    parser.add_argument('--subset-loc', dest='subset_loc', type=str, default='')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    if 'effnet' in args.checkpoint_loc:
        REPRESENTATION_DIM = 1280
    else:
        REPRESENTATION_DIM = 512

    main(args.checkpoint_loc, args.color, args.predictions_dir, args.start_snippet_index, args.end_snippet_index, args.overwrite, args.subset_loc)
