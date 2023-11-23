import os

import numpy as np
import pandas as pd

# set of CSVs, each with file_loc and id_str columns

if __name__ == '__main__':

    from galaxy_datasets import demo_rings
    download = True
    master_catalog, _  = demo_rings(root='data/example_jpg/download_root', download=download, train=True)
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

    galaxy_start_index = 0
    snippet_size = 16  # TODO for demo purposes, very small snippets
    while galaxy_start_index < len(master_catalog):
        galaxy_end_index = galaxy_start_index + snippet_size
        snippet = master_catalog[galaxy_start_index:galaxy_end_index][['id_str', 'file_loc']]
        snippet.to_csv(f'data/example_jpg/snippets/snippet_{galaxy_start_index}_{galaxy_end_index}.csv', index=False)
        galaxy_start_index = galaxy_end_index
