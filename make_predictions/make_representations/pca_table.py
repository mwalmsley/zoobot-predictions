import argparse
import logging

import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='table to pca')
    parser.add_argument('--parquet-loc', dest='parquet_loc', type=str)
    parser.add_argument('--save-loc', dest='save_loc', type=str)
    parser.add_argument('--components', dest='components', type=int, default=100)
    args = parser.parse_args()


    if 'effnet' in args.parquet_loc:
        REPRESENTATION_DIM = 1280
    else:
        REPRESENTATION_DIM = 512

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    full_df = pd.read_parquet(args.parquet_loc)
    # debug mode
    # full_df = full_df.sample(5000)

    pca = IncrementalPCA(n_components=args.components, batch_size=10000)
    
    X_subset = full_df.values[:1000000, :REPRESENTATION_DIM]  # 1 mil max for fitting
    pca.fit(X_subset)
    X_pca = pca.transform(full_df.values[:, :REPRESENTATION_DIM])  # now do everything

    variance_info = dict(zip(range(args.components), pca.explained_variance_ratio_))
    with np.printoptions(precision=3):
        # doesn't seem to work with logging
        logging.info('Explained variance: \n{}\n'.format(variance_info))
        logging.info('Total percentage variance explained: {}'.format(pca.explained_variance_ratio_.sum()))

    pca_save_loc = args.save_loc.replace('.parquet', '_pca.npy')
    logging.info('Saving component vectors to {}'.format(pca_save_loc))
    with open(pca_save_loc, 'w') as f:
        np.savetxt(f, pca.components_)

    pca_df = pd.DataFrame(data=X_pca, columns=['feat_pca_' + str(n) for n in range(args.components)])
    logging.debug(pca_df)
    df = pd.concat([full_df[[col for col in full_df.columns.values if 'feat' not in col]], pca_df], axis=1)

    df.to_parquet(args.save_loc, index=False)
    logging.info(f'Saved to {args.save_loc}')

    logging.warning('Exiting gracefully')
