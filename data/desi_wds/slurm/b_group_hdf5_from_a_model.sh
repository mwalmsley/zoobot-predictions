#!/bin/bash
#SBATCH --job-name=desi-pred                    # Job name
#SBATCH --output=desi-pred_%A_%a.log 
#SBATCH --time=02:30:0  
#SBATCH --cpus-per-task=5
#SBATCH --mem 10G
# no gpu etc

pwd; hostname; date

PYTHON=/home/walml/envs/zoobot39_dev/bin/python


MODEL=convnext_nano_evo
GALAXIES=desi_wds

# AGGREGATION=desi
# PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/predictions

AGGREGATION=representations
PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/representations

srun $PYTHON /project/def-bovy/walml/repos/zoobot-predictions/make_predictions/b_group_hdf5_from_a_model.py \
    +predictions_dir=$PREDICTIONS_DIR \
    +cluster=beluga \
    +galaxies=$GALAXIES \
    +model=$MODEL \
    +aggregation=$AGGREGATION
