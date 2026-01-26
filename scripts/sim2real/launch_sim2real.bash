#!/bin/bash
# =============================================================================
# Sim2Real Launch Script for UR5e Pose Control
# =============================================================================
# This script sources the ROS2 workspace and launches the sim2real node.
#
# Usage:
#   ./launch_sim2real.bash                    # Default: gripper robot, default model
#   ./launch_sim2real.bash --robot gripper    # Specify robot
#   ./launch_sim2real.bash --model /path/to/model.pt  # Custom model
#   ./launch_sim2real.bash --help             # Show all options
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# ROS2 Environment Setup
# =============================================================================
echo "=== Setting up ROS2 environment ==="

# Source ROS2 Humble

#if [ -f "/opt/ros/humble/setup.bash" ]; then
    #source /opt/ros/humble/setup.bash
    #echo "Sourced ROS2 Humble"
#else
#    echo "ERROR: ROS2 Humble not found at /opt/ros/humble/setup.bash"
#    exit 1
#fi

# Source workspace
WORKSPACE_SETUP="$HOME/wwro_ws/install/local_setup.bash"
if [ -f "$WORKSPACE_SETUP" ]; then
    source "$WORKSPACE_SETUP"
    echo "Sourced workspace: $WORKSPACE_SETUP"
else
    echo "ERROR: Workspace setup not found at $WORKSPACE_SETUP"
    exit 1
fi

# Set ROS2 environment variables
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"

# =============================================================================
# Default arguments
# =============================================================================
ROBOT="gripper"
MODEL_PATH="$REPO_ROOT/logs/rsl_rl/pose_orientation_no_gripper/2026-01-25_16-39-48/exported/policy.pt"
RATE="60.0"
DEVICE="cuda"

# =============================================================================
# Parse arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot)
            ROBOT="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --rate)
            RATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --robot ROBOT     Robot prefix: gripper or screwdriver (default: gripper)"
            echo "  --model PATH      Path to TorchScript policy .pt file"
            echo "  --rate HZ         Control loop rate (default: 60.0)"
            echo "  --device DEVICE   Inference device: cpu or cuda (default: cpu)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Verify model exists
# =============================================================================
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo ""
    echo "Available models in pretrained_models/:"
    ls -la "$REPO_ROOT/pretrained_models/"* 2>/dev/null || echo "  (none)"
    exit 1
fi
# =============================================================================
# Put robot at home position before starting
# =============================================================================
ros2 action send_goal /gripper_scaled_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [gripper_shoulder_pan_joint, gripper_shoulder_lift_joint, gripper_elbow_joint, gripper_wrist_1_joint, gripper_wrist_2_joint, gripper_wrist_3_joint],
    points: [{positions: [0.0, -1.57, -1.57, -1.57, 0.0, 0.0], time_from_start: {sec: 3, nanosec: 0}}]
  }
}"
# =============================================================================
# Launch node
# =============================================================================
echo ""
echo "=== Launching Sim2Real Node ==="
echo "Robot:  $ROBOT"
echo "Model:  $MODEL_PATH"
echo "Rate:   ${RATE}Hz"
echo "Device: $DEVICE"
echo ""



cd "$SCRIPT_DIR"

python3 sim2real_node.py \
    --robot "$ROBOT" \
    --model "$MODEL_PATH" \
    --rate "$RATE" \
    --device "$DEVICE"
