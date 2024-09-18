# zoobot-predictions
Make arbitrary-scale (100k+ galaxies) predictions (inc. representations) from Zoobot.

## Introduction

Predictions are simple enough with lightning and galaxy-datasets, but it's fiddly to make predictions on very large numbers of galaxies distributed across a cluster. There's also some further aggregation to ensemble together the multiple forward passes from one model and then the multiple models, and create nice neat tables.

The approach is:

- Divide the galaxy catalog into ordered subsets (snippets), each with relatively few galaxies (not crucial, say 10k). See `make_hsc_snippets.py`
- Assign each node to make single-model predictions on a few of those snippets. This is done by using a slurm task array to provide each node with different snippet indexes (e.g. snippets 0-5, snippets 6-10, etc) to make predictions on.  See `a_make_bulk_catalog_predictions.sh`
- Predictions by one model on each snippet are saved as hdf5 arrays, with shape (galaxy_in_snippet, question, forward pass). See `a_make_bulk_catalog_predictions.py`
- Group all the predictions by one model on every snippet into a single hdf5 file with shape (galaxy, question, forward pass). See `b_group_hdf5_from_a_model.py`
- Group those single hdf5 files across models. Simply stack them for a single hdf5 array of shape (galaxy, question, model, forward pass). See `c_group_hdf5_across_models.py`
- Take that final (potentially quite large, you may need high mem node) hdf5 file and apply some formatting and statistics to create final output tables of astronomer-friendly predictions. See `d_all_predictions_to_friendly_table.py`.

To manage the configuration (paths, model options, etc) I use [hydra](https://hydra.cc/docs/intro/). This lets you write yaml config files which can then be mixed-and-matched. See the example below. You will want to make new config files under the `conf` subfolders.

## Install

Requires Zoobot (any version, but must match the version for which the model was trained) and hydra/webdataset. `pip install -r requirements.txt`.

`zoobot-predictions` is not itself a package, just some folders with code.

### Example Predictions Locally

I've included a working example with 1000 ringed galaxies. Here's the [pretrained model](https://dl.dropboxusercontent.com/s/epam7u354zzx62n/binary_ring_resnet_greyscale.ckpt?dl=0) and the [encoder architecture](https://dl.dropboxusercontent.com/s/hvyfw6avwhep4qqpg4wwg/resnet50_greyscale_224px.ckpt?rlkey=hconeglfnsxt73ot2gq8xg6p3&dl=0).

    python data/example_jpg/make_snippets.py 

    python make_predictions/a_make_bulk_catalog_predictions.py +predictions_dir=data/example_jpg/predictions +cluster=local_debug +galaxies=example_jpg +model=effnet_rings_dirichlet

    python make_predictions/b_group_hdf5_from_a_model.py +predictions_dir=data/example_jpg/predictions +model=effnet_rings_dirichlet +aggregation=example_jpg

    python make_predictions/c_group_hdf5_across_models.py +predictions_dir=data/example_jpg/predictions +model=effnet_rings_dirichlet  +aggregation=example_jpg

    python make_predictions/d_all_predictions_to_friendly_table.py +predictions_dir=data/example_jpg/predictions +model=effnet_rings_dirichlet +aggregation=example_jpg

### Example for WDS

    python make_predictions/a_make_bulk_catalog_predictions.py +predictions_dir=data/example_wds/predictions +cluster=local_debug +galaxies=example_wds +model=effnet_rings_dirichlet  ++galaxies.n_samples=1

    python make_predictions/b_group_hdf5_from_a_model.py +predictions_dir=data/example_wds/predictions +model=effnet_rings_dirichlet +aggregation=example

    python make_predictions/c_group_hdf5_across_models.py +predictions_dir=data/example_wds/predictions +model=effnet_rings_dirichlet  +aggregation=example

    python make_predictions/d_all_predictions_to_friendly_table.py +predictions_dir=data/example_wds/predictions +model=effnet_rings_dirichlet +aggregation=example

### Example for WDS Representations

    python make_predictions/a_make_bulk_catalog_predictions.py +predictions_dir=data/example_wds/representations +cluster=local_debug +galaxies=example_wds +model=effnet_rings_dirichlet ++galaxies.n_samples=1

    python make_predictions/b_group_hdf5_from_a_model.py +predictions_dir=data/example_wds/representations +model=effnet_rings_dirichlet +aggregation=representations

Saves to {predictions_dir}/{model_name}/grouped.hdf5. You might want to make into a parquet with

    python make_predictions/make_representations/to_friendly_table.py \
    --hdf5-loc data/example_wds/representations/effnet_rings_dirichlet/grouped.hdf5 \
    --save-loc data/example_wds/representations/effnet_rings_dirichlet/representations.parquet

Doesn't make sense to combine across models etc.

### Example Predictions on Cluster

    See /data/example/slurm for the slurm scripts. These construct slurm jobs that look just like those above, except for also passing start/end snippet indices so each node can make predictions on different snippets
    python data/example/make_snippets.py 
    
    python make_predictions/c_group_hdf5_across_models.py +predictions_dir=data/example/predictions +model=effnet_rings_dirichlet  +aggregation=manchester

    python make_predictions/d_all_predictions_to_friendly_table.py +predictions_dir=data/example/predictions +model=effnet_rings_dirichlet +aggregation=manchester

## Real Use

### Ring Predictions

    TODO

### HSC Predictions

    python data/hsc_pdr3/make_snippets.py 

    python make_predictions/a_make_bulk_catalog_predictions.py +predictions_dir=data/hsc_pdr3/predictions +cluster=local_debug +galaxies=hsc_pdr3 +model=effnet_rings_dirichlet

    python make_predictions/b_group_hdf5_from_a_model.py +predictions_dir=data/hsc_pdr3/predictions +model=effnet_rings_dirichlet +aggregation=local_debug

    python make_predictions/c_group_hdf5_across_models.py +predictions_dir=data/hsc_pdr3/predictions +model=effnet_rings_dirichlet  +aggregation=local_debug

    python make_predictions/d_all_predictions_to_friendly_table.py +predictions_dir=data/hsc_pdr3/predictions +model=effnet_rings_dirichlet +aggregation=local_debug


### WDS DESI Predictions

    # run on slurm, see a_make_bulk_catalog_predictions.sh
    python make_predictions/a_make_bulk_catalog_predictions.py +predictions_dir=data/desi_wds/predictions +cluster=beluga +galaxies=desi_wds +model=effnetb0_f128_desi_wds

    python make_predictions/b_group_hdf5_from_a_model.py +predictions_dir=data/desi_wds/predictions +model=effnetb0_f128_desi_wds +aggregation=desi

    python make_predictions/c_group_hdf5_across_models.py +predictions_dir=data/desi_wds/predictions +model=effnetb0_f128_desi_wds  +aggregation=desi

    python make_predictions/d_all_predictions_to_friendly_table.py +predictions_dir=data/desi_wds/predictions +model=effnetb0_f128_desi_wds +aggregation=desi


# DESI WDS Representations

    python make_predictions/b_group_hdf5_from_a_model.py +predictions_dir=data/desi_wds/representations +model=effnetb0_f128_desi_wds +aggregation=representations

    python make_predictions/make_representations/to_friendly_table.py \
        --hdf5-loc data/desi_wds/representations/desi_300px_f128_1gpu/grouped.hdf5 \
        --save-loc data/desi_wds/representations/desi_300px_f128_1gpu/representations.parquet \
        --subset-frac 0.5

### Euclid Predictions

    python data/euclid_wide/make_snippets.py 

    conda activate pytorch
    conda activate /usr/miniforge3/envs/pytorch

    python make_predictions/a_make_bulk_catalog_predictions.py +predictions_dir=/media/team_workspaces/Galaxy-Zoo-Euclid/data/zoobot/predictions/v4_post_euclid_challenge/predictions +cluster=datalabs_l4 +galaxies=euclid_wide +model=convnext_nano_euclid

    python make_predictions/b_group_hdf5_from_a_model.py +predictions_dir=/media/team_workspaces/Galaxy-Zoo-Euclid/data/zoobot/predictions/v4_post_euclid_challenge/predictions +model=convnext_nano_euclid +aggregation=euclid_wide

    python make_predictions/c_group_hdf5_across_models.py +predictions_dir=/media/team_workspaces/Galaxy-Zoo-Euclid/data/zoobot/predictions/v4_post_euclid_challenge/predictions +model=convnext_nano_euclid +aggregation=euclid_wide

    python make_predictions/d_all_predictions_to_friendly_table.py +predictions_dir=/media/team_workspaces/Galaxy-Zoo-Euclid/data/zoobot/predictions/v4_post_euclid_challenge/predictions +model=convnext_nano_euclid +aggregation=euclid_wide
