import os
import math
import pandas as pd
import numpy as np
import glob


if __name__ == '__main__':

    jpg_dir = '/home/walml/repos/download_DECaLS_images/data/sga/images/jpg'
    jpg_locs = glob.glob(f'{jpg_dir}/*/*.jpg')
    assert jpg_locs
    id_strs = [os.path.basename(loc.replace('.jpg', '')) for loc in jpg_locs]


    df = pd.DataFrame({'id_str': id_strs, 'file_loc': jpg_locs})

    snippet_size = 1000
    snippets = np.array_split(df, math.ceil(len(df) // snippet_size))
    for i, snippet in enumerate(snippets):
        snippet.to_csv(f'/home/walml/repos/zoobot-predictions/data/sga2020/snippets/snippet_{i}.csv', index=False)
        print(f'Saved snippet_{i}.csv')