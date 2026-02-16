#!/bin/bash
# ============================================================================
# Launch inference from a trained sweep run using saved configs
# ============================================================================
# Usage:
#   bash euler/play_from_run.sh actuators-high_domain_rand-current_network-ext4
#   bash euler/play_from_run.sh 2026-02-15_10-20-30_actuators-high  (partial match)
#   bash euler/play_from_run.sh logs/rsl_rl/pose_orientation_sim2real/2026-02-15_...  (full path)
#
# Finds the run, loads env.yaml + agent.yaml, converts them to Hydra overrides,
# and launches play.py with all saved configuration automatically.
# ============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash euler/play_from_run.sh <run_id_or_pattern>"
    echo ""
    echo "Examples:"
    echo "  bash euler/play_from_run.sh actuators-high_domain_rand-current_network-ext4"
    echo "  bash euler/play_from_run.sh 2026-02-15_10-20-30"
    echo "  bash euler/play_from_run.sh logs/rsl_rl/pose_orientation_sim2real/2026-02-15_..."
    exit 1
fi

PATTERN="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs/rsl_rl/pose_orientation_sim2real"

# --- Find run directory ---
# If pattern is a full path, use it directly
if [[ "$PATTERN" == /* ]] || [[ "$PATTERN" == logs/* ]]; then
    RUN_DIR="$PATTERN"
    if [[ "$RUN_DIR" != /* ]]; then
        RUN_DIR="$PROJECT_ROOT/$RUN_DIR"
    fi
else
    # Otherwise, search for matching run in logs/
    RUN_DIR=$(find "$LOGS_DIR" -maxdepth 2 -type d -name "*$PATTERN*" | head -1)
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: No run found matching '$PATTERN'"
    echo "  Searched in: $LOGS_DIR"
    exit 1
fi

ENV_YAML="$RUN_DIR/params/env.yaml"
AGENT_YAML="$RUN_DIR/params/agent.yaml"

if [ ! -f "$ENV_YAML" ] || [ ! -f "$AGENT_YAML" ]; then
    echo "Error: Missing params files in $RUN_DIR"
    echo "  env.yaml: $([ -f "$ENV_YAML" ] && echo 'OK' || echo 'NOT FOUND')"
    echo "  agent.yaml: $([ -f "$AGENT_YAML" ] && echo 'OK' || echo 'NOT FOUND')"
    exit 1
fi

# Find the checkpoint in the same directory
CHECKPOINT=$(find "$RUN_DIR" -name "model_*.pt" | head -1)
if [ -z "$CHECKPOINT" ]; then
    echo "Warning: No model checkpoint (.pt file) found in $RUN_DIR"
    echo "  Will attempt to load from default checkpoint locations"
fi

echo "================================================================"
echo "Playing from run:"
echo "  Run directory: $RUN_DIR"
echo "  Env config:    $ENV_YAML"
echo "  Agent config:  $AGENT_YAML"
[ -n "$CHECKPOINT" ] && echo "  Checkpoint:    $CHECKPOINT"
echo "================================================================"
echo ""

# --- Build Hydra overrides from saved YAML files ---
# Convert simple YAML key-value pairs to Hydra dot-notation.
# This is a basic parser for flat YAML (works for most env/agent configs).
build_overrides() {
    local yaml_file="$1"
    local prefix="$2"
    
    while IFS= read -r line; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Parse "key: value" pattern at the top level (no indentation)
        if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*)[[:space:]]*:[[:space:]]*(.+)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            
            # Trim trailing comments and whitespace
            value=$(echo "$value" | sed 's/[[:space:]]*#.*$//' | xargs)
            
            # Skip complex structures (lists starting with -, nested dicts)
            [[ "$value" =~ ^- ]] && continue
            [[ "$value" =~ ^{.*}$ ]] && continue
            [[ -z "$value" ]] && continue
            
            echo "${prefix}.${key}=${value}"
        fi
    done < "$yaml_file"
}

ENV_OVERRIDES=$(build_overrides "$ENV_YAML" "env")
AGENT_OVERRIDES=$(build_overrides "$AGENT_YAML" "agent")

# Combine all overrides (base + env + agent + checkpoint)
HYDRA_ARGS=""
if [ -n "$ENV_OVERRIDES" ]; then
    HYDRA_ARGS="$HYDRA_ARGS $ENV_OVERRIDES"
fi
if [ -n "$AGENT_OVERRIDES" ]; then
    HYDRA_ARGS="$HYDRA_ARGS $AGENT_OVERRIDES"
fi
if [ -n "$CHECKPOINT" ]; then
    # For play.py, use --load_run and --checkpoint to locate the model
    RUN_NAME=$(basename "$RUN_DIR")
    CHECKPOINT_NAME=$(basename "$CHECKPOINT" .pt | sed 's/model_//')
    HYDRA_ARGS="$HYDRA_ARGS --load_run=$RUN_NAME --checkpoint=$CHECKPOINT_NAME"
fi

echo "[Setup] Built Hydra overrides:"
echo "$HYDRA_ARGS" | tr ' ' '\n' | sed 's/^/  /'
echo ""

# --- Extract task name from run dir or use default ---
TASK_NAME="Template-Pose-Orientation-Sim2Real-Direct-v1-ext"

# --- Launch play.py ---
echo "[Launch] Starting play.py..."
cd "$PROJECT_ROOT"
python scripts/rsl_rl/play.py \
    --task="$TASK_NAME" \
    $HYDRA_ARGS
