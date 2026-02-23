#!/bin/bash
# Single training run. For hyperparameter sweeps, see:
#   euler/launch_sweep.sh  (generates & submits SLURM array jobs)

#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --job-name="WWSim-Pose-Orientation-Sim2Real-Direct-v2"
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# --- CONFIGURATION ---
TASK_NAME="WWSim-Pose-Orientation-Sim2Real-Direct-v2"
# UPDATE THIS PATH to where you uploaded your .sif file
SIF_PATH="/cluster/scratch/$USER/isaac_euler_salziegl.sif"

PROJECT_PATH=$(pwd)
PROJECT_NAME=$(basename "$PROJECT_PATH")

# WandB API Key
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/.wandb_key)
else
    echo "Error: ~/.wandb_key not found!"
    exit 1
fi

# --- CACHE SETUP ---
# Creates writable scratch folders for Isaac Sim caches
JOB_CACHE="/cluster/scratch/$USER/isaac_cache/$SLURM_JOB_ID"

mkdir -p $JOB_CACHE/kit_cache $JOB_CACHE/kit_data $JOB_CACHE/ov $JOB_CACHE/pip \
         $JOB_CACHE/glcache $JOB_CACHE/computecache $JOB_CACHE/logs $JOB_CACHE/data \
         $JOB_CACHE/documents $JOB_CACHE/warp $JOB_CACHE/local_lib \
         $JOB_CACHE/wandb_cache $JOB_CACHE/wandb_config $JOB_CACHE/wandb_data

# Load Proxy (Required for internet access on compute nodes)
module load eth_proxy

echo "----------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Project: $PROJECT_NAME"
echo "Container: $SIF_PATH"
echo "----------------------------------------------------------------"

# --- EXECUTION ---
apptainer exec --nv \
    -B $JOB_CACHE/kit_cache:/isaac-sim/kit/cache:rw \
    -B $JOB_CACHE/kit_data:/isaac-sim/kit/data:rw \
    -B $JOB_CACHE/ov:$HOME/.cache/ov:rw \
    -B $JOB_CACHE/pip:$HOME/.cache/pip:rw \
    -B $JOB_CACHE/warp:$HOME/.cache/warp:rw \
    -B $JOB_CACHE/local_lib:$HOME/.local:rw \
    -B $JOB_CACHE/glcache:$HOME/.cache/nvidia/GLCache:rw \
    -B $JOB_CACHE/computecache:$HOME/.nv/ComputeCache:rw \
    -B $JOB_CACHE/logs:$HOME/.nvidia-omniverse/logs:rw \
    -B $JOB_CACHE/data:$HOME/.local/share/ov/data:rw \
    -B $JOB_CACHE/documents:$HOME/Documents:rw \
    -B $JOB_CACHE/wandb_cache:$HOME/.cache/wandb:rw \
    -B $JOB_CACHE/wandb_config:$HOME/.config/wandb:rw \
    -B $JOB_CACHE/wandb_data:$PROJECT_PATH/wandb:rw \
    -B $PROJECT_PATH:/workspace/isaaclab/$PROJECT_NAME:rw \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=$PROJECT_PATH \
    --env WANDB_CACHE_DIR=$HOME/.cache/wandb \
    --env WANDB_CONFIG_DIR=$HOME/.config/wandb \
    $SIF_PATH \
    bash -c "
        # 1. Install Project in Editable Mode
        echo 'Installing Project...'
        /isaac-sim/python.sh -m pip install --user -e /workspace/isaaclab/$PROJECT_NAME/source/$PROJECT_NAME

        # 2. Run Training
        echo 'Starting Training...'
        /isaac-sim/python.sh /workspace/isaaclab/$PROJECT_NAME/scripts/rsl_rl/train.py \
            --task $TASK_NAME\
            --headless \
            agent.algorithm.entropy_coef=0.04 \
            agent.algorithm.desired_kl=0.013 \
    "

# Cleanup Cache
rm -rf $JOB_CACHE
