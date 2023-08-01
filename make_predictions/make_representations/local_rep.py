import logging
import os
import pandas as pd

from zoobot.pytorch.training import finetune, representations
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.shared import load_predictions


def main(catalog, checkpoint_loc, save_dir):

    assert all([os.path.isfile(x) for x in catalog['file_loc']])
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # can load from either ZoobotTree (if trained from scratch) or FinetuneableZoobotTree (if finetuned)
    # encoder = finetune.FinetuneableZoobotTree.load_from_checkpoint(checkpoint_loc).encoder
    encoder = define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc).encoder

    # convert to simple pytorch lightning model
    model = representations.ZoobotEncoder(encoder=encoder, pyramid=False)

    label_cols = [f'feat_{n}' for n in range(1280)]
    save_loc = os.path.join(save_dir, 'representations.hdf5')

    accelerator = 'cpu'  # or 'gpu' if available
    batch_size = 32
    resize_after_crop = 224

    datamodule_kwargs = {'batch_size': batch_size, 'resize_after_crop': resize_after_crop}
    trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
    predict_on_catalog.predict(
        catalog,
        model,
        n_samples=1,
        label_cols=label_cols,
        save_loc=save_loc,
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )

    return save_loc


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)

    # load the gz evo model for representations
    checkpoint_loc = '/Users/user/repos/gz-decals-classifiers/results/benchmarks/pytorch/evo/uploaded/effnetb0_greyscale_224px.ckpt'

    catalog = pd.DataFrame([{'id_str': n, 'file_loc': '/Users/user/repos/download_DECaLS_images/risa_gz.jpg'} for n in range(10)] )

    # save the representations here
    # TODO change this to wherever you'd like
    save_dir = os.path.join('/Users/user/repos/desi-predictions/risa')

    save_loc = main(catalog, checkpoint_loc, save_dir)

    representations_loc = main(catalog, checkpoint_loc, save_dir)
    rep_df = load_predictions.single_forward_pass_hdf5s_to_df(representations_loc)
    print(rep_df)
    rep_df.iloc[0].to_parquet('risa.parquet')
