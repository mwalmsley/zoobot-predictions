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
#SBATCH --array=1-40%20  # currently only 35 needed, last snippet 17468  # --array=[1-1]

pwd; hostname; date

nvidia-smi

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot  # be careful zoobot is up to date
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# 2000 snippets per million galaxies (set --array[1-9] above)
# so this is enough for 10M galaxies (plenty, there are about 8.7M in practice)
# lines up with the "checks" snippet - load that for filename, predict, save back with same name (but potentially fewer rows)
POSSIBLE_START_SNIPPETS=( $(seq -500 500 19500 ) )
POSSIBLE_END_SNIPPETS=( $(seq 0 500 20000 ) )

START_SNIPPET_INDEX=${POSSIBLE_START_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using start snippet $START_SNIPPET_INDEX

END_SNIPPET_INDEX=${POSSIBLE_END_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using end snippet $END_SNIPPET_INDEX

# # morphology
# CHECKPOINT_FOLDER=/share/nas2/walml/repos/gz-decals-classifiers/results/pytorch/desi

# # ls -1 /share/nas2/walml/galaxy_zoo/decals/dr8/predictions/_desi_pytorch_v5_hpv2_train_all_notest_m1 | wc -l

# # CHECKPOINT_NAME=_desi_pytorch_v5_hpv2_train_all_notest_m1/checkpoints/epoch=66-step=49915.ckpt  # DONE 874
# # CHECKPOINT_NAME=_desi_pytorch_v5_hpv2_train_all_notest_m2/checkpoints/epoch=65-step=49170.ckpt  # DONE 874
# CHECKPOINT_NAME=_desi_pytorch_v5_hpv2_train_all_notest_m3/checkpoints/epoch=49-step=37250.ckpt # DONE 874
# # CHECKPOINT_NAME=_desi_pytorch_v5_hpv2_train_all_notest_m4/checkpoints/epoch=42-step=32035.ckpt # DONE 874
# # CHECKPOINT_NAME=_desi_pytorch_v5_hpv2_train_all_notest_m5/checkpoints/epoch=46-step=35015.ckpt # DONE 874

# CHECKPOINT_LOC=${CHECKPOINT_FOLDER}/${CHECKPOINT_NAME}
# PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/predictions  # saves here/{checkpoint_name}/*.hdf5

srun $PYTHON /share/nas2/walml/repos/gz-decals-classifiers/make_predictions/a_make_bulk_catalog_predictions.py \
    cluster.start_snippet_index $START_SNIPPET_INDEX \
    cluster.end_snippet_index $END_SNIPPET_INDEX \
    cluster.task_id $SLURM_ARRAY_TASK_ID

    # can override other hydra params if needed
    # model.checkpoint_loc=$CHECKPOINT_LOC \



    # --predictions-dir $PREDICTIONS_DIR \
    
    #  \
    # --subset-loc $SUBSET_LOC
    
    #  \
    # --overwrite \
    
    # --color
    
    #  \
    # --rings
