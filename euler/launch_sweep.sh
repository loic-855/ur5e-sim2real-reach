#!/bin/bash
# ============================================================================
# Launch a hyperparameter sweep on Euler
# ============================================================================
# 1. Generates sweep_runs.txt from sweep_config.yaml
# 2. Computes the SLURM array range
# 3. Submits sweep_euler.sh with the correct --array spec
#
# Usage:  cd /path/to/Woodworking_Simulation && bash euler/launch_sweep.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/sweep_config.yaml"
SWEEP_SCRIPT="$SCRIPT_DIR/sweep_euler.sh"

# --- Generate sweep_runs.txt ---
echo "=== Generating sweep configurations ==="
python3 "$SCRIPT_DIR/generate_sweep.py" --config "$CONFIG"

SWEEP_FILE="$SCRIPT_DIR/sweep_runs.txt"
if [ ! -f "$SWEEP_FILE" ]; then
    echo "Error: sweep_runs.txt was not generated."
    exit 1
fi

TOTAL_RUNS=$(wc -l < "$SWEEP_FILE")

# Read sequential_per_job from the YAML config
SEQ_PER_JOB=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('sequential_per_job', 3))
")

NUM_JOBS=$(( (TOTAL_RUNS + SEQ_PER_JOB - 1) / SEQ_PER_JOB ))
ARRAY_SPEC="0-$((NUM_JOBS - 1))"

# Ensure logs directory exists
mkdir -p logs

echo ""
echo "=== Submitting SLURM array job ==="
echo "  Array range:      --array=$ARRAY_SPEC"
echo "  Total runs:       $TOTAL_RUNS"
echo "  Parallel jobs:    $NUM_JOBS"
echo "  Sequential/job:   $SEQ_PER_JOB"
echo ""

sbatch --array=$ARRAY_SPEC "$SWEEP_SCRIPT"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs:          logs/sweep_<JOB_ID>_<ARRAY_ID>.out"
echo "WandB:         https://wandb.ai  (project: isaaclab_euler)"
