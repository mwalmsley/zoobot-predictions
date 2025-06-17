import os
import math
import pandas as pd
import numpy as np
import glob


if __name__ == '__main__':

    if os.path.isdir('/media/walml/ssd'):
        print('detected office ssd')
        jpg_dir = '/media/walml/ssd/sga/images/jpg'
        snippet_dir = '/home/walml/repos/zoobot-predictions/data/sga2020/snippets'
    else:
        print('assuming galahad')
        assert os.path.isdir('/share/nas2')
        jpg_dir = '/share/nas2/walml/data/sga/images/jpg'
        snippet_dir = '/share/nas2/walml/repos/zoobot-predictions/data/sga2020/snippets'

    jpg_locs = glob.glob(f'{jpg_dir}/*/*.jpg')
    assert jpg_locs
    print(f'Found {len(jpg_locs)} jpg files in {jpg_dir}')
    id_strs = [os.path.basename(loc.replace('.jpg', '')) for loc in jpg_locs]

    df = pd.DataFrame({'id_str': id_strs, 'file_loc': jpg_locs})

    snippet_size = 1000
    snippets = np.array_split(df, math.ceil(len(df) // snippet_size))
    for i, snippet in enumerate(snippets):
        snippet.to_csv(snippet_dir + f'/snippet_{i}.csv', index=False)
        print(f'Saved snippet_{i}.csv')