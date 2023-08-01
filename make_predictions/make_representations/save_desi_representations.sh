#!/bin/bash
#SBATCH --job-name=desi-rep                    # Job name
#SBATCH --output=desi-rep_%A_%a.log 
#SBATCH --mem=80gb                                      # Job memory request
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=128:00:00                         # Time limit hrs:min:sec
#SBATCH --constraint=A100
#SBATCH -N 1   # 1 node
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH --array=1-40%8  # currently only 35 needed, last snippet 17468  # --array=[1-1]

pwd; hostname; date

nvidia-smi

ZOOBOT_DIR=/share/nas2/walml/repos/zoobot  # be careful zoobot is up to date
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot38_torch/bin/python

# 2000 snippets per million galaxies (set --array[1-9] above)
# so this is enough for 10M galaxies (plenty, there are about 8.7M in practice)
# lines up with the "checks" snippet - load that for filename, predict, save back with same name (but potentially fewer rows)
POSSIBLE_START_SNIPPETS=( $(seq -500 500 19500 ) )
POSSIBLE_END_SNIPPETS=( $(seq 0 500 20000 ) )

START_SNIPPET=${POSSIBLE_START_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using start snippet $START_SNIPPET

END_SNIPPET=${POSSIBLE_END_SNIPPETS[$SLURM_ARRAY_TASK_ID]}
echo Using end snippet $END_SNIPPET

CHECKPOINT_FOLDER=/share/nas2/walml/repos/gz-decals-classifiers/results/benchmarks/pytorch/evo
# CHECKPOINT_NAME=evo_py_gr_eff_224_814/checkpoints/epoch=53-step=31158.ckpt  # effnetb0, with dodgy alpha (though basically fine)
# CHECKPOINT_NAME=evo_py_gr_vittiny_224_19505/checkpoints/epoch=66-step=77251.ckpt  # vittiny greyscale, fixed alpha
# TODO need to change to use color channels
CHECKPOINT_NAME=evo_py_co_vittiny_224_7325/checkpoints/epoch=62-step=72639.ckpt  # as above, in color
CHECKPOINT_LOC=${CHECKPOINT_FOLDER}/${CHECKPOINT_NAME}

PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/representations
# PREDICTIONS_DIR=/share/nas2/walml/galaxy_zoo/decals/dr8/representations_0p1  # saves here/{checkpoint_name}/*.hdf5

# SUBSET_LOC='/share/nas2/walml/repos/desi-predictions/z_below_0p1.parquet'

srun $PYTHON /share/nas2/walml/repos/desi-predictions/make_predictions/representations/save_desi_representations.py \
    --checkpoint-loc $CHECKPOINT_LOC \
    --predictions-dir $PREDICTIONS_DIR \
    --start-snippet $START_SNIPPET \
    --end-snippet $END_SNIPPET \
    --task-id $SLURM_ARRAY_TASK_ID \
    --color
    
    #  \
    # --subset-loc $SUBSET_LOC \
    # --overwrite
    


