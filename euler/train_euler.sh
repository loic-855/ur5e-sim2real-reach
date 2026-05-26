#!/bin/bash
# Single training run. For hyperparameter sweeps, see:
#   euler/launch_sweep.sh  (generates & submits SLURM array jobs)

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=3000
#SBATCH --job-name="WWSim-Pose-Orientation-Sim2Real-Screwdriver-Direct-v1"
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# --- CONFIGURATION ---
TASK_NAME="WWSim-Pose-Orientation-Sim2Real-Direct-v1"  # Must match a task_name in your config files (e.g. source/wwsim/configs/pose_orientation_sim2real_direct.yaml)
# UPDATE THIS PATH to where you uploaded your .sif file
SIF_PATH="/cluster/scratch/$USER/isaac_euler_salziegl.sif"

PROJECT_PATH=$(pwd)
# Keep the internal project/package name stable even if the repo folder was renamed.
PROJECT_NAME="Woodworking_Simulation"

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
         $JOB_CACHE/wandb_cache $JOB_CACHE/wandb_config $JOB_CACHE/wandb_data \
         $JOB_CACHE/tmp

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
    -B $JOB_CACHE/tmp:/tmp:rw \
    -B $PROJECT_PATH:/workspace/isaaclab/$PROJECT_NAME:rw \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=$PROJECT_PATH \
    --env WANDB_CACHE_DIR=$HOME/.cache/wandb \
    --env WANDB_CONFIG_DIR=$HOME/.config/wandb \
    --env WANDB_START_METHOD=thread \
    --env WANDB__SERVICE_WAIT=10 \
    $SIF_PATH \
    bash -c "
        # 1. Upgrade W&B to avoid known SDK 0.24.0 upload issues
        echo 'Upgrading Weights & Biases (wandb)...'
        /isaac-sim/python.sh -m pip install --user --upgrade 'wandb>=0.24.1'

        # 2. Install Project in Editable Mode
        echo 'Installing Project...'
        /isaac-sim/python.sh -m pip install --user -e /workspace/isaaclab/$PROJECT_NAME/source/$PROJECT_NAME

        # 3. Run Training
        echo 'Starting Training'
        /isaac-sim/python.sh /workspace/isaaclab/$PROJECT_NAME/scripts/rsl_rl/train.py \
            --task=$TASK_NAME \
            --headless \
            --run_name=reduced_obs_opti_contacts\
            agent.max_iterations=1500 \
            env.debug=False \
            env.domain_rand.enable_actuator_rand=True \
            env.domain_rand.enable_mass_com_rand=False \
            env.domain_rand.enable_noise=False \
            env.domain_rand.enable_delay=True \
            env.domain_rand.action_delay_range=[1,2] \
            env.domain_rand.obs_delay_range=[0,1]
    "

# Cleanup Cache
rm -rf $JOB_CACHE
