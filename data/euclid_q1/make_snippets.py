import os
import glob
import logging

import tqdm
import pandas as pd

# set of CSVs, each with file_loc and id_str columns

def get_id_str(galaxy):
    # TODO NEG part, like pipeline
    return f'{galaxy["release_name"]}_{galaxy["tile_index"]}_{galaxy["object_id"]}'


if __name__ == '__main__':

    # we start with the per-tile catalogs as snippets, and set the 'file_loc' column to the colour image location
    # not adapting for relative folders as this runs on datalabs

    # MUST ADJUST MANUALLY
    catalog_dir = '/media/team_workspaces/Euclid-Consortium/data/galaxy_zoo_euclid/q1_v6/q1_v6/catalogs'
    snippet_dir = '/media/team_workspaces/Galaxy-Zoo-Euclid/data/zoobot/predictions/q1_v6/snippets'


    assert os.path.exists(catalog_dir), f'{catalog_dir} does not exist'
    assert os.path.exists(snippet_dir), f'{snippet_dir} does not exist'

    tile_catalog_locs = glob.glob(catalog_dir + '/*_mer_catalog.csv')
    for loc in tqdm.tqdm(tile_catalog_locs):
        df = pd.read_csv(loc, usecols=['release_name', 'tile_index', 'object_id', 'jpg_loc_generic'])
        if len(df) == 0:
            logging.warning(f'Empty file {loc}')
            continue
        df['file_loc'] = df['jpg_loc_generic'].str.replace('generic', 'gz_arcsinh_vis_y') # .apply(lambda x: f'{image_dir}/{x}.png')
        df['id_str'] = df.apply(get_id_str, axis=1)
        df['file_exists'] = df['file_loc'].apply(os.path.exists)
        print(df['file_loc'].iloc[0])
        df = df.query('file_exists')
        if len(df) == 0:
            logging.warning(f'No files found for {loc}')
            continue
        df = df.sort_values('id_str')
        df.to_csv(os.path.join(snippet_dir, os.path.basename(loc)), index=False)
