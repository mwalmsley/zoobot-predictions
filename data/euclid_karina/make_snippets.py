import os
import glob

import pandas as pd

# set of CSVs, each with file_loc and id_str columns

if __name__ == '__main__':

    # standard column names
    df = pd.read_csv('/home/walml/repos/euclid-morphology/data/combined_karina_strong_lens_classifications_with_cutouts.csv', usecols=['id_str', 'file_loc'])

    df = df.sort_values('id_str')

    galaxy_start_index = 0
    snippet_size = 4096
    while galaxy_start_index < len(df):
        galaxy_end_index = galaxy_start_index + snippet_size
        snippet = df[galaxy_start_index:galaxy_end_index][['id_str', 'file_loc']]
        snippet.to_csv(f'data/euclid_karina/snippets/snippet_{galaxy_start_index}_{galaxy_end_index}.csv', index=False)
        galaxy_start_index = galaxy_end_index
