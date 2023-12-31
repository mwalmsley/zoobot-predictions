#!/bin/bash
#SBATCH --job-name=prd-tab                    # Job name
#SBATCH --output=prd-tab_%A.log 
#SBATCH --mem=0gb                                     # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive  # might as well, only need 20 nodes

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python


PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions

# PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions
# MODEL=maxvit_tiny_evo
# GALAXIES=desi

# or rings
PREDICTIONS_DIR=data/desi/predictions/rings
MODEL=effnet_rings_dirichlet
GALAXIES=desi
AGGREGATION=example  # fine here, no suffix

srun $PYTHON /share/nas2/walml/repos/zoobot-predictions/make_predictions/d_all_predictions_to_friendly_table.py \
    +predictions_dir=$PREDICTIONS_DIR \
    +model=effnet_rings_dirichlet  \
    +aggregation=$AGGREGATION
