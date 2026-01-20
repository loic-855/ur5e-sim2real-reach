#!/bin/bash
# Launch script for gripper robot reach policy on dual-arm setup
#
# Usage:
#   ./launch_gripper_policy.sh
#
# The script uses the default pretrained model and env config.
# You can override by passing arguments:
#   ./launch_gripper_policy.sh --policy <path> --env <path>

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Default paths
POLICY_PATH="$REPO_ROOT/pretrained_models/pose_orientation_reach_v1/2026-01-16_14-25-14_final/exported/policy.pt"
ENV_PATH="$REPO_ROOT/pretrained_models/pose_orientation_reach_v1/2026-01-16_14-25-14_final/params/env.yaml"

echo "================================================"
echo "  Gripper Robot Reach Policy Launch Script"
echo "================================================"
echo ""
echo "Policy: $POLICY_PATH"
echo "Config: $ENV_PATH"
echo ""
echo "Setting up ROS2 and conda environment..."
echo ""

# Source ROS2 Jazzy
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo "✓ ROS2 Jazzy sourced"
else
    echo "✗ ERROR: ROS2 Jazzy not found at /opt/ros/jazzy/setup.bash"
    exit 1
fi

# Source workspace
if [ -f ~/wwro_ws/install/local_setup.bash ]; then
    source ~/wwro_ws/install/local_setup.bash
    echo "✓ Workspace sourced"
else
    echo "✗ ERROR: Workspace not found at ~/wwro_ws/install/local_setup.bash"
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate conda environment and run the script
echo "✓ Activating conda environment: sim2real"
conda activate sim2real

if [ $? -ne 0 ]; then
    echo "✗ ERROR: Failed to activate conda environment 'sim2real'"
    exit 1
fi

echo ""
echo "Starting ROS2 node..."
echo ""

ros2 action send_goal /gripper_scaled_joint_trajectory_controller/follow_joint_trajectory \
control_msgs/action/FollowJointTrajectory "{
  trajectory: {
    joint_names: [gripper_shoulder_pan_joint, gripper_shoulder_lift_joint, gripper_elbow_joint, gripper_wrist_1_joint, gripper_wrist_2_joint, gripper_wrist_3_joint],
    points: [{positions: [0.0,-1.57, -1.57, -1.57, 0.0, 0.0], time_from_start: {sec: 4, nanosec: 0}}]
  }
}"

python3 "$SCRIPT_DIR/run_task_ur5e_reach.py" \
    --policy "$POLICY_PATH" \
    --env "$ENV_PATH" \
    "$@"
