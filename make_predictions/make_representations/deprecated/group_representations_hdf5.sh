#!/bin/bash
#SBATCH --job-name=groupmod                    # Job name
#SBATCH --output=groupmod_%A.log 
#SBATCH --mem=100gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# CHECKPOINT_NAME=evo_py_gr_eff_224_814
# CHECKPOINT_NAME=evo_py_gr_vittiny_224_19505
CHECKPOINT_NAME=evo_py_co_vittiny_224_7325
PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/representations
SAVE_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}_grouped.hdf5

$PYTHON /share/nas2/walml/repos/desi-predictions/make_predictions/b_group_hdf5_from_a_model.py \
    --checkpoint-name $CHECKPOINT_NAME \
    --predictions-dir $PREDICTIONS_DIR \
    --save-loc $SAVE_LOC
