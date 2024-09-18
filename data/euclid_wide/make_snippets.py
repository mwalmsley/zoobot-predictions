import os
import glob

import pandas as pd

# set of CSVs, each with file_loc and id_str columns

def get_id_str(galaxy):
    return f'{galaxy["release_name"]}_{galaxy["tile_id"]}_{galaxy["object_id"]}'


if __name__ == '__main__':

    # we start with the per-tile catalogs as snippets, and set the 'file_loc' column to the colour image location
    # not adapting for relative folders as this runs on datalabs

    catalog_dir = '/media/team_workspaces/Galaxy-Zoo-Euclid/data/pipeline_runs/v4_post_euclid_challenge/catalogs'
    snippet_dir = '/media/team_workspaces/Galaxy-Zoo-Euclid/data/zoobot/predictions/v4_post_euclid_challenge/snippets'

    assert os.path.exists(catalog_dir), f'{catalog_dir} does not exist'
    assert os.path.exists(snippet_dir), f'{snippet_dir} does not exist'

    tile_catalog_locs = glob.glob(catalog_dir + '/*_mer_catalog.csv')
    for loc in tile_catalog_locs:
        df = pd.read_csv(loc, usecols=['release_name', 'tile_id', 'object_id', 'jpg_loc_composite'])
        df['file_loc'] = df['jpg_loc_composite'] # .apply(lambda x: f'{image_dir}/{x}.png')
        df['id_str'] = df.apply(get_id_str, axis=1)
        df = df.sort_values('id_str')
        df.to_csv(os.path.join(snippet_dir, os.path.basename(loc)), index=False)
