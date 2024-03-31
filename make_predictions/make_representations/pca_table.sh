#!/bin/bash
#SBATCH --job-name=pca                    # Job name
#SBATCH --output=pca_%A.log 
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu 10G

pwd; hostname; date

# PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python
PYTHON=/home/walml/envs/zoobot39_dev/bin/python

COMPONENTS=30

# CHECKPOINT_NAME=evo_py_gr_eff_224_814
# CHECKPOINT_NAME=evo_py_gr_vittiny_224_19505
# CHECKPOINT_NAME=evo_py_co_vittiny_224_7325
CHECKPOINT_NAME=convnext_nano_evo

# PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/representations
PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/representations

PARQUET_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}_representations.parquet
SAVE_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}_representations_pca_${COMPONENTS}.parquet

REPO_DIR=/project/def-bovy/walml/repos/zoobot-predictions

$PYTHON $REPO_DIR/make_predictions/make_representations/pca_table.py \
    --parquet-loc $PARQUET_LOC \
    --save-loc $SAVE_LOC
