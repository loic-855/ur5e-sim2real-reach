#!/bin/bash
# ============================================================================
# SLURM Array Job for Hyperparameter Sweeps on Euler
# ============================================================================
# Each array task runs SEQUENTIAL_PER_JOB training runs sequentially.
# Array index selects which chunk of sweep_runs.txt to execute.
#
# Usage (auto):  bash euler/launch_sweep.sh       (computes array range)
# Usage (manual): sbatch --array=0-2 euler/sweep_euler.sh     (3 jobs for 12 runs)
# ============================================================================

#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=07:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --job-name="ur5e_sweep"
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# --- CONFIGURATION ---
TASK_NAME="Template-Pose-Orientation-Sim2Real-Direct-v1-ext"
SEQUENTIAL_PER_JOB=4
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWEEP_FILE="$SCRIPT_DIR/sweep_runs.txt"

# UPDATE THIS PATH to where you uploaded your .sif file
SIF_PATH="/cluster/scratch/$USER/isaac_euler_salziegl.sif"

PROJECT_PATH=$(pwd)
PROJECT_NAME=$(basename "$PROJECT_PATH")

# --- WandB API Key ---
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/.wandb_key)
else
    echo "Error: ~/.wandb_key not found!"
    exit 1
fi

# --- Validate sweep file ---
if [ ! -f "$SWEEP_FILE" ]; then
    echo "Error: $SWEEP_FILE not found. Run: python euler/generate_sweep.py"
    exit 1
fi

TOTAL_RUNS=$(wc -l < "$SWEEP_FILE")
START_IDX=$((SLURM_ARRAY_TASK_ID * SEQUENTIAL_PER_JOB))
END_IDX=$((START_IDX + SEQUENTIAL_PER_JOB - 1))

# --- CACHE SETUP ---
JOB_CACHE="/cluster/scratch/$USER/isaac_cache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p $JOB_CACHE/kit_cache $JOB_CACHE/kit_data $JOB_CACHE/ov $JOB_CACHE/pip \
         $JOB_CACHE/glcache $JOB_CACHE/computecache $JOB_CACHE/logs $JOB_CACHE/data \
         $JOB_CACHE/documents $JOB_CACHE/warp $JOB_CACHE/local_lib \
         $JOB_CACHE/wandb_cache $JOB_CACHE/wandb_config $JOB_CACHE/wandb_data

# Load Proxy (Required for internet access on compute nodes)
module load eth_proxy

echo "================================================================"
echo "SWEEP Job $SLURM_ARRAY_TASK_ID  (Array Job $SLURM_ARRAY_JOB_ID)"
echo "Project:    $PROJECT_NAME"
echo "Container:  $SIF_PATH"
echo "Runs:       $START_IDX to $END_IDX  (of $TOTAL_RUNS total)"
echo "Started:    $(date)"
echo "================================================================"

# --- INSTALL PROJECT (once per job) ---
echo ""
echo "[Setup] Installing project in editable mode..."
apptainer exec --nv \
    -B $JOB_CACHE/pip:$HOME/.cache/pip:rw \
    -B $JOB_CACHE/local_lib:$HOME/.local:rw \
    -B $PROJECT_PATH:/workspace/isaaclab/$PROJECT_NAME:rw \
    $SIF_PATH \
    bash -c "/isaac-sim/python.sh -m pip install --user -e /workspace/isaaclab/$PROJECT_NAME/source/$PROJECT_NAME"

INSTALL_EXIT=$?
if [ $INSTALL_EXIT -ne 0 ]; then
    echo "[Setup] FATAL: pip install failed with exit code $INSTALL_EXIT"
    rm -rf $JOB_CACHE
    exit 1
fi
echo "[Setup] Installation complete."

# --- SEQUENTIAL RUNS ---
PASS=0
FAIL=0

for RUN_IDX in $(seq $START_IDX $END_IDX); do
    # Skip if beyond total runs
    if [ $RUN_IDX -ge $TOTAL_RUNS ]; then
        echo ""
        echo "[Run $RUN_IDX] Skipping (index >= $TOTAL_RUNS)"
        continue
    fi

    # Read run config from sweep_runs.txt  (format: run_name|hydra_overrides)
    LINE=$(sed -n "$((RUN_IDX + 1))p" "$SWEEP_FILE")
    RUN_NAME=$(echo "$LINE" | cut -d'|' -f1)
    HYDRA_OVERRIDES=$(echo "$LINE" | cut -d'|' -f2-)

    echo ""
    echo "----------------------------------------------------------------"
    echo "[Run $RUN_IDX] $RUN_NAME"
    echo "  Overrides: $HYDRA_OVERRIDES"
    echo "  Started:   $(date)"
    echo "----------------------------------------------------------------"

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
            echo 'Starting Training: $RUN_NAME'
            /isaac-sim/python.sh /workspace/isaaclab/$PROJECT_NAME/scripts/rsl_rl/train.py \
                --task=$TASK_NAME \
                --headless \
                --run_name=$RUN_NAME \
                $HYDRA_OVERRIDES
        "

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[Run $RUN_IDX] FAILED (exit code $EXIT_CODE) at $(date)"
        echo "[Run $RUN_IDX] Continuing to next run..."
        FAIL=$((FAIL + 1))
    else
        echo "[Run $RUN_IDX] PASSED at $(date)"
        PASS=$((PASS + 1))
    fi
done

echo ""
echo "================================================================"
echo "SWEEP Job $SLURM_ARRAY_TASK_ID complete at $(date)"
echo "  Passed: $PASS   Failed: $FAIL"
echo "================================================================"

# Cleanup Cache
rm -rf $JOB_CACHE
