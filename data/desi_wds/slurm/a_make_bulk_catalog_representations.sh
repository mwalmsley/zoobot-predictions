#!/bin/bash
#SBATCH --job-name=desi-rep                    # Job name
#SBATCH --output=desi-rep_%A_%a.log 
#SBATCH --time=01:30:0  
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-12%10

pwd; hostname; date

nvidia-smi

PYTHON=/home/walml/envs/zoobot39_dev/bin/python
# source ~/envs/zoobot39_dev/bin/activate

export NCCL_BLOCKING_WAIT=1

POSSIBLE_START_SNIPPETS=( $(seq -400 400 3600 ) )
POSSIBLE_END_SNIPPETS=( $(seq 0 400 4000 ) )

START_SNIPPET_INDEX=${POSSIBLE_START_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using start snippet $START_SNIPPET_INDEX

END_SNIPPET_INDEX=${POSSIBLE_END_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using end snippet $END_SNIPPET_INDEX

# just rename predictions_dir to representations
PREDICTIONS_DIR=/project/def-bovy/walml/repos/zoobot-predictions/data/desi_wds/representations
MODEL=maxvit_tiny_desi_wds
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
