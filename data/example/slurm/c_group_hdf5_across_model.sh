#!/bin/bash
#SBATCH --job-name=grp-mod                    # Job name
#SBATCH --output=grp-mod_%A.log 
#SBATCH --mem=0gb                                     # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive  # might as well, only need 20 nodes
#SBATCH --dependency=afterok:76677

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions

# STAR not *, will swap in python - or bash gets confused
# GLOB_STR=$PREDICTIONS_DIR/_desi_pytorch_v5_hpv2_train_all_notest_mSTAR_grouped.hdf5
# SAVE_LOC=$PREDICTIONS_DIR/_desi_pytorch_v5_hpv2_train_all_notest_all.hdf5

$PYTHON /share/nas2/walml/repos/gz-decals-classifiers/make_predictions/c_group_hdf5_across_models.py

#  \
#     --glob $GLOB_STR \
#     --save-loc $SAVE_LOC
