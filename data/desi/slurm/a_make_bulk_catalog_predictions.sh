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
#SBATCH --array=1-3%3

# snippets have 4096 galaxies each
# 9M total galaxies, so 2200 snippets total
# 20 snippet batches per node, so 110 nodes required


pwd; hostname; date

nvidia-smi

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot  # be careful zoobot is up to date
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# the 20 MUST match cluster.snippets_per_node
POSSIBLE_START_SNIPPETS=( $(seq -20 20 2180 ) )
POSSIBLE_END_SNIPPETS=( $(seq 0 20 2200 ) )

START_SNIPPET_INDEX=${POSSIBLE_START_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using start snippet $START_SNIPPET_INDEX

END_SNIPPET_INDEX=${POSSIBLE_END_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using end snippet $END_SNIPPET_INDEX

PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions

# MODEL=maxvit_tiny_evo
# GALAXIES=desi

# or rings
MODEL=effnet_rings_dirichlet
GALAXIES=desi

srun $PYTHON /share/nas2/walml/repos/zoobot-predictions/make_predictions/a_make_bulk_catalog_predictions.py \
    +cluster.start_snippet_index=$START_SNIPPET_INDEX \
    +cluster.end_snippet_index=$END_SNIPPET_INDEX \
    +cluster.task_id=$SLURM_ARRAY_TASK_ID \
    +predictions_dir=$PREDICTIONS_DIR \
    +cluster=manchester \
    +galaxies=$GALAXIES \
    +model=$MODEL
