#!/bin/bash
#SBATCH --job-name=desi-pred                    # Job name
#SBATCH --output=desi-pred_%A_%a.log 
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:30:0  
#SBATCH --array=0-0

# SBATCH --time=01:30:0  
# SBATCH --array=0-12%10

pwd; hostname; date

nvidia-smi

PYTHON=/home/walml/envs/zoobot39_dev/bin/python
# source ~/envs/zoobot39_dev/bin/activate

export NCCL_BLOCKING_WAIT=1

# POSSIBLE_START_SNIPPETS=( $(seq 0 400 3600 ) )
# POSSIBLE_END_SNIPPETS=( $(seq 400 400 4000 ) )

# START_SNIPPET_INDEX=${POSSIBLE_START_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
# echo Using start snippet $START_SNIPPET_INDEX

# END_SNIPPET_INDEX=${POSSIBLE_END_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
# echo Using end snippet $END_SNIPPET_INDEX

START_SNIPPET_INDEX=0
END_SNIPPET_INDEX=400

# predictions or representations
PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/representations
MODEL=convnext_nano_evo
GALAXIES=desi_wds

srun $PYTHON /project/def-bovy/walml/repos/zoobot-predictions/make_predictions/a_make_bulk_catalog_predictions.py \
    +cluster.start_shard_index=$START_SNIPPET_INDEX \
    +cluster.end_shard_index=$END_SNIPPET_INDEX \
    +cluster.task_id=$SLURM_ARRAY_TASK_ID \
    +predictions_dir=$PREDICTIONS_DIR \
    +cluster=beluga \
    +galaxies=$GALAXIES \
    +model=$MODEL \
    ++cluster.batch_size=64
