#!/bin/bash
#SBATCH --job-name=pca                    # Job name
#SBATCH --output=pca_%A.log 
#SBATCH --mem=100gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

COMPONENTS=30

CHECKPOINT_NAME=evo_py_gr_eff_224_814
# CHECKPOINT_NAME=evo_py_gr_vittiny_224_19505
# CHECKPOINT_NAME=evo_py_co_vittiny_224_7325

PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/representations
PARQUET_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}_representations.parquet
SAVE_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}_representations_pca_${COMPONENTS}.parquet

$PYTHON /share/nas2/walml/repos/desi-predictions/make_predictions/representations/pca_table.py \
    --parquet-loc $PARQUET_LOC \
    --save-loc $SAVE_LOC
