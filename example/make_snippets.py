import os

import numpy as np
import pandas as pd

# set of CSVs, each with file_loc and id_str columns

if __name__ == '__main__':

    # using Cosmic Dawn uploaded subjects as an example
    # import will only work for GZ people
    from galaxy_datasets import gz_cosmic_dawn
    download = False
    master_catalog, _  = gz_cosmic_dawn(root='/Users/user/repos/zoobot-predictions/example/data', download=download, train=True)
    # TODO replace with your galaxy catalog
    # master_catalog = pd.read_parquet(...)

    # standard column names
    assert 'file_loc' in master_catalog.columns.values
    assert 'id_str' in master_catalog.columns.values
    # this can be slow - optionally skip, if you already checked the catalog
    # is_valid_file = master_catalog['file_loc'].apply(os.path.isfile)
    # if not all(is_valid_file):
        # raise FileNotFoundError(master_catalog[~is_valid_file][0]['file_loc'])
    
    master_catalog = master_catalog.sort_values('id_str')

    # one has a dodgy image
    master_catalog = master_catalog[~master_catalog['file_loc'].str.endswith('aae47c8-1a00-477c-9eb2-5b7825e2d8b3.png')]

    galaxy_start_index = 0
    snippet_size = 10000
    while galaxy_start_index < len(master_catalog):
        galaxy_end_index = galaxy_start_index + snippet_size
        snippet = master_catalog[galaxy_start_index:galaxy_end_index][['id_str', 'file_loc']]
        # TODO for demo purposes, very small snippets
        snippet = snippet[:32]
        snippet.to_csv(f'example/snippets/snippet_{galaxy_start_index}_{galaxy_end_index}.csv', index=False)
        galaxy_start_index = galaxy_end_index
