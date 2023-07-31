#!/bin/bash
#SBATCH --job-name=groupmod                    # Job name
#SBATCH --output=groupmod_%A.log 
#SBATCH --mem=0gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# m1 done
# m2 done
# m3 done
# m4 done
# m5 done
# CHECKPOINT_NAME=_desi_pytorch_v5_hpv2_train_all_notest_m3
# PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions
# SAVE_LOC=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions/${CHECKPOINT_NAME}_grouped.hdf5

$PYTHON /share/nas2/walml/repos/gz-decals-classifiers/make_predictions/b_group_hdf5_from_a_model.py

#  \
#     --checkpoint-name $CHECKPOINT_NAME \
#     --predictions-dir $PREDICTIONS_DIR \
#     --save-loc $SAVE_LOC
