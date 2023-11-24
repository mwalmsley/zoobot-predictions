#!/bin/bash
#SBATCH --job-name=desi-pred                    # Job name
#SBATCH --output=desi-pred_%A_%a.log 
#SBATCH --time=02:30:0  
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu 10G
# no gpu etc

pwd; hostname; date

PYTHON=/home/walml/envs/zoobot39_dev/bin/python

PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/predictions
MODEL=effnetb0_f128_desi_wds
GALAXIES=desi_wds
AGGREGATION=desi

srun $PYTHON /project/def-bovy/walml/repos/zoobot-predictions/make_predictions/c_group_hdf5_across_models.py \
    +predictions_dir=$PREDICTIONS_DIR \
    +model=$MODEL \
    +aggregation=$AGGREGATION
