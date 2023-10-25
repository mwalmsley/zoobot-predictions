#!/bin/bash
#SBATCH --job-name=desi-pred                    # Job name
#SBATCH --output=desi-pred_%A_%a.log 
#SBATCH --mem=0                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00                         # Time limit hrs:min:sec
#SBATCH --constraint=A100
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH --array=1-4%2

pwd; hostname; date

nvidia-smi

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot  # be careful zoobot is up to date
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# start snippets [0 to 3] (slurm task id is 1-indexed, so the -1 is never used)
# end snippets [1 to 4]
POSSIBLE_START_SNIPPETS=( $(seq -1 2 3 ) )
POSSIBLE_END_SNIPPETS=( $(seq 0 2 4 ) )

START_SNIPPET_INDEX=${POSSIBLE_START_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using start snippet $START_SNIPPET_INDEX

END_SNIPPET_INDEX=${POSSIBLE_END_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using end snippet $END_SNIPPET_INDEX

PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/debug_predictions  # saves here/{checkpoint_name}/*.hdf5

srun $PYTHON /share/nas2/walml/repos/gz-decals-classifiers/make_predictions/a_make_bulk_catalog_predictions.py \
    +cluster.start_snippet_index=$START_SNIPPET_INDEX \
    +cluster.end_snippet_index=$END_SNIPPET_INDEX \
    +cluster.task_id=$SLURM_ARRAY_TASK_ID \
    +predictions_dir=$PREDICTIONS_DIR \
    +galaxies=example \
    +model=effnet_rings_dirichlet
