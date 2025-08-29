import os
import io

import pandas as pd
from tqdm import tqdm
import datasets as hf_datasets
from PIL import Image

# export HF_HOME=/media/home/team_workspaces/Galaxy-Zoo-Euclid/huggingface

def main():

    save_dir = 'gz_euclid_demo'

    im_ds_dict = hf_datasets.load_dataset("mwalmsley/gz_euclid", "tiny")  # 1% subset of images for testing
    em_ds_dict = hf_datasets.load_dataset("mwalmsley/gz_euclid_embeddings")  # the full embeddings dataset (not large enough to need a subset)

    for split in ['train', 'test']: 

        im_ds = im_ds_dict[split]
        em_ds = em_ds_dict[split]

        # filter em_ds to only include rows where id_str is in im_ds
        # and vica versa
        im_ids = set(im_ds['id_str'])
        em_ids = set(em_ds['id_str'])
        joint_ids = im_ids.intersection(em_ids)
        print(f'Number of image ids: {len(im_ids)}')
        print(f'Number of embedding ids: {len(em_ids)}')
        print(f'Number of joint ids: {len(joint_ids)}')

        # much faster to use a specific column, yay parquet
        em_ds_filtered = em_ds.filter(lambda x: x in joint_ids, input_columns=['id_str'])
        im_ds_filtered = im_ds.filter(lambda x: x in joint_ids, input_columns=['id_str'])

        # slice to apply the format
        em_ds_filtered: pd.DataFrame = em_ds_filtered.with_format(type='pandas')[:] 
        im_ds_filtered: pd.DataFrame = im_ds_filtered.with_format(type='pandas')[:]

        # join into a nice friendly pandas dataframe
        df = pd.merge(im_ds_filtered, em_ds_filtered, on='id_str', how='inner')
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f'{save_dir}/{split}_dataset.csv', index=False)

        # images are saved under 'image' key as a PIL object
        # optionally, write images out for 'normal' loading
        os.makedirs(f'{save_dir}/images/{split}', exist_ok=True)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            id_str = row['id_str']
            image = row['image'] # PIL Image
            image.save(f'{save_dir}/images/{split}/{id_str}.jpg')



if __name__ == "__main__":
    main()