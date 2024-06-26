#!/bin/bash
#SBATCH --job-name=pca                    # Job name
#SBATCH --output=pca_%A.log 
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem 170G  # crank

pwd; hostname; date

# PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python
PYTHON=/home/walml/envs/zoobot39_dev/bin/python

# COMPONENTS=40
COMPONENTS=10

# CHECKPOINT_NAME=evo_py_gr_eff_224_814
# CHECKPOINT_NAME=evo_py_gr_vittiny_224_19505
# CHECKPOINT_NAME=evo_py_co_vittiny_224_7325
CHECKPOINT_NAME=convnext_nano_evo

REPO_DIR=/project/def-bovy/walml/repos/zoobot-predictions

PREDICTIONS_DIR=$REPO_DIR/data/desi_wds/representations

PARQUET_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}/representations.parquet
SAVE_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}/representations_pca_${COMPONENTS}.parquet


$PYTHON $REPO_DIR/make_predictions/make_representations/pca_table.py \
    --parquet-loc $PARQUET_LOC \
    --save-loc $SAVE_LOC \
    --components $COMPONENTS
