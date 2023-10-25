#!/bin/bash
#SBATCH --job-name=grp-mod                    # Job name
#SBATCH --output=grp-mod_%A.log 
#SBATCH --mem=0gb                                     # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive  # might as well, only need 20 nodes
#SBATCH --dependency=afterok:76677

pwd; hostname; date

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions

# MODEL=maxvit_tiny_evo
# GALAXIES=desi
# AGGREGATION=desi

# or rings
MODEL=effnet_rings_dirichlet
GALAXIES=desi
AGGREGATION=example  # fine here, no suffix

srun $PYTHON /share/nas2/walml/repos/zoobot-predictions/make_predictions/c_group_hdf5_across_models.py \
    +predictions_dir=$PREDICTIONS_DIR \
    +model=effnet_rings_dirichlet  \
    +aggregation=$AGGREGATION
