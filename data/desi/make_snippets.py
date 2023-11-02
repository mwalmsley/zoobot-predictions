import os
import glob

import pandas as pd

# set of CSVs, each with file_loc and id_str columns

if __name__ == '__main__':

    # standard column names
    df = pd.read_parquet('data/desi/master_all_file_index_passes_file_checks.parquet', columns=['dr8_id', 'jpg_loc'])
    df['file_loc'] = df['jpg_loc']
    df['id_str'] = df['dr8_id']
    assert 'file_loc' in df.columns.values
    assert 'id_str' in df.columns.values
    
    df = df.sort_values('id_str')

    galaxy_start_index = 0
    snippet_size = 4096
    while galaxy_start_index < len(df):
        galaxy_end_index = galaxy_start_index + snippet_size
        snippet = df[galaxy_start_index:galaxy_end_index][['id_str', 'file_loc']]
        snippet.to_csv(f'data/desi/snippets/snippet_{galaxy_start_index}_{galaxy_end_index}.csv', index=False)
        galaxy_start_index = galaxy_end_index
