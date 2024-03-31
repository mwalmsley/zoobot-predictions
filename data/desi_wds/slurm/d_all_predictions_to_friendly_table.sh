#!/bin/bash
#SBATCH --job-name=desi-pred                    # Job name
#SBATCH --output=desi-pred_%A_%a.log 
#SBATCH --time=02:30:0  
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu 10G
# no gpu etc

pwd; hostname; date

PYTHON=/home/walml/envs/zoobot39_dev/bin/python

MODEL=convnext_nano_evo
GALAXIES=desi_wds

AGGREGATION=desi
PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/predictions

# not appropriate for representations - see make_predictions/make_representations/to_friendly_table.sh and then pca_table.sh

srun $PYTHON /project/def-bovy/walml/repos/zoobot-predictions/make_predictions/d_all_predictions_to_friendly_table.py \
    +predictions_dir=$PREDICTIONS_DIR \
    +model=$MODEL \
    +aggregation=$AGGREGATION
