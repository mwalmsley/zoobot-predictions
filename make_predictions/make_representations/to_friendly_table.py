import argparse
import logging

from zoobot.shared import load_predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='preds to table')
    parser.add_argument('--hdf5-loc', dest='hdf5_loc', type=str)
    parser.add_argument('--save-loc', dest='save_loc', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


    df = load_predictions.single_forward_pass_hdf5s_to_df([args.hdf5_loc], drop_extra_dims=True)
    df.to_parquet(args.save_loc, index=False)
    logging.info(f'Saved to {args.save_loc}')

    logging.warning('Exiting gracefully')
