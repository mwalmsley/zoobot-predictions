#!/bin/bash
#SBATCH --job-name=prd-tab                    # Job name
#SBATCH --output=prd-tab_%A.log 
#SBATCH --mem=0gb                                     # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive  # might as well, only need 20 nodes

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions

# HDF5_LOC=$PREDICTIONS_DIR/_desi_pytorch_v5_hpv2_train_all_notest_all.hdf5
# SAVE_LOC=$PREDICTIONS_DIR/_desi_pytorch_v5_hpv2_train_all_notest_ml_catalog_x2.parquet

$PYTHON /share/nas2/walml/repos/gz-decals-classifiers/make_predictions/d_all_predictions_to_friendly_table.py

#  \
#     --hdf5-loc $HDF5_LOC \
#     --save-loc $SAVE_LOC
    
    #  \
    # --debug
