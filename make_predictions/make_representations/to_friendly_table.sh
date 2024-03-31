#!/bin/bash
#SBATCH --job-name=tb                    # Job name
#SBATCH --output=tb_%A.log                                     # Do not resubmit a failed job
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu 10G

pwd; hostname; date

# PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python
PYTHON=/home/walml/envs/zoobot39_dev/bin/python

# CHECKPOINT_NAME=evo_py_gr_eff_224_814
# CHECKPOINT_NAME=evo_py_gr_vittiny_224_19505
# CHECKPOINT_NAME=evo_py_co_vittiny_224_7325
# CHECKPOINT_NAME=desi_300px_f128_1gpu
CHECKPOINT_NAME=convnext_nano_evo


PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/representations
HDF5_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}/grouped.hdf5
SAVE_LOC=${PREDICTIONS_DIR}/${CHECKPOINT_NAME}/representations.parquet

$PYTHON /project/def-bovy/walml/repos/zoobot-predictions/make_predictions/make_representations/to_friendly_table.py \
    --hdf5-loc $HDF5_LOC \
    --save-loc $SAVE_LOC
