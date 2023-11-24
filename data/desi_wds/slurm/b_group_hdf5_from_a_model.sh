#!/bin/bash
#SBATCH --job-name=desi-pred                    # Job name
#SBATCH --output=desi-pred_%A_%a.log 
#SBATCH --time=02:30:0  
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu 10G
# no gpu etc

pwd; hostname; date

nvidia-smi

PYTHON=/home/walml/envs/zoobot39_dev/bin/python

export NCCL_BLOCKING_WAIT=1

PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/predictions
MODEL=maxvit_tiny_desi_wds
GALAXIES=desi_wds

AGGREGATION=desi

srun $PYTHON /share/nas2/walml/repos/zoobot-predictions/make_predictions/b_group_hdf5_from_a_model.py \
    +predictions_dir=$PREDICTIONS_DIR \
    +cluster=beluga \
    +galaxies=$GALAXIES \
    +model=$MODEL \
    +aggregation=$AGGREGATION
