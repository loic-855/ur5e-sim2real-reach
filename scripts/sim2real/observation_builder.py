"""
Observation builder for sim2real transfer.
Replicates exact normalization from IsaacSim pose_orientation_no_gripper task.

Observation vector (19 dims):
  - pos_error[3]: (goal_pos - ee_pos) / MAX_REACH
  - quat_error[4]: quaternion difference (goal_quat * ee_quat^-1)
  - joint_pos_norm[6]: 2 * (pos - lower) / (upper - lower) - 1
  - joint_vel_norm[6]: vel / MAX_JOINT_VEL
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# Normalization constants (must match IsaacSim task exactly!)
# ============================================================================

MAX_REACH = 0.85  # UR5e reach ~850mm
MAX_JOINT_VEL = 3.14  # ~180°/s

# Joint limits for real robot (elbow has cable constraint)
# Order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
JOINT_LIMITS_LOWER = np.array([
    -6.283185307179586,  # shoulder_pan_joint: -2π
    -6.283185307179586,  # shoulder_lift_joint: -2π
    -3.141592653589793,  # elbow_joint: -π (cable constraint!)
    -6.283185307179586,  # wrist_1_joint: -2π
    -6.283185307179586,  # wrist_2_joint: -2π
    -6.283185307179586,  # wrist_3_joint: -2π
], dtype=np.float32)

JOINT_LIMITS_UPPER = np.array([
    6.283185307179586,   # shoulder_pan_joint: 2π
    6.283185307179586,   # shoulder_lift_joint: 2π
    3.141592653589793,   # elbow_joint: π (cable constraint!)
    6.283185307179586,   # wrist_1_joint: 2π
    6.283185307179586,   # wrist_2_joint: 2π
    6.283185307179586,   # wrist_3_joint: 2π
], dtype=np.float32)

# Joint names in SIMULATION order (must match IsaacSim!)
JOINT_NAMES_SIM = [
    "shoulder_pan_joint",
    "shoulder_lift_joint", 
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Joint names as they appear in ROS2 /joint_states (different order!)
# ROS2 order: elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3
# This mapping is used to reorder from ROS2 to simulation order
JOINT_NAME_SUFFIXES = {
    "shoulder_pan_joint": 0,   # sim index 0
    "shoulder_lift_joint": 1,  # sim index 1
    "elbow_joint": 2,          # sim index 2
    "wrist_1_joint": 3,        # sim index 3
    "wrist_2_joint": 4,        # sim index 4
    "wrist_3_joint": 5,        # sim index 5
}


@dataclass
class RobotState:
    """Current robot state from ROS2 topics."""
    joint_positions: np.ndarray  # [6] radians
    joint_velocities: np.ndarray  # [6] rad/s
    ee_position: np.ndarray  # [3] meters (in table frame)
    ee_quaternion: np.ndarray  # [4] (w, x, y, z)


@dataclass
class GoalState:
    """Goal pose for the end-effector."""
    position: np.ndarray  # [3] meters (in table frame)
    quaternion: np.ndarray  # [4] (w, x, y, z)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternion).
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Inverted quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions.
    
    Args:
        q1, q2: Quaternions [w, x, y, z]
        
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_diff(q_goal: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Compute quaternion difference (q_goal * q_current^-1).
    
    This matches the _quat_diff method in IsaacSim task.
    
    Args:
        q_goal: Goal quaternion [w, x, y, z]
        q_current: Current quaternion [w, x, y, z]
        
    Returns:
        Error quaternion [w, x, y, z]
    """
    q_current_inv = quat_inverse(q_current)
    return quat_multiply(q_goal, q_current_inv)


def normalize_joint_positions(joint_pos: np.ndarray) -> np.ndarray:
    """Normalize joint positions to [-1, 1] range.
    
    Formula: 2 * (pos - lower) / (upper - lower) - 1
    
    Args:
        joint_pos: Raw joint positions [6] in radians
        
    Returns:
        Normalized positions [6] in [-1, 1]
    """
    return 2.0 * (joint_pos - JOINT_LIMITS_LOWER) / (JOINT_LIMITS_UPPER - JOINT_LIMITS_LOWER) - 1.0


def normalize_joint_velocities(joint_vel: np.ndarray) -> np.ndarray:
    """Normalize joint velocities.
    
    Formula: vel / MAX_JOINT_VEL
    
    Args:
        joint_vel: Raw joint velocities [6] in rad/s
        
    Returns:
        Normalized velocities [6] (approximately [-1, 1])
    """
    return joint_vel / MAX_JOINT_VEL


def build_observation(robot_state: RobotState, goal_state: GoalState) -> np.ndarray:
    """Build the full 19-dim observation vector.
    
    Observation structure (must match IsaacSim exactly):
      [0:3]   pos_error: (goal_pos - ee_pos) / MAX_REACH
      [3:7]   quat_error: quaternion difference
      [7:13]  joint_pos_norm: normalized joint positions
      [13:19] joint_vel_norm: normalized joint velocities
    
    Args:
        robot_state: Current robot state
        goal_state: Goal pose
        
    Returns:
        Observation vector [19] as float32
    """
    # Position error (normalized)
    pos_error = (goal_state.position - robot_state.ee_position) / MAX_REACH
    
    # Quaternion error
    quat_error = quat_diff(goal_state.quaternion, robot_state.ee_quaternion)
    
    # Joint positions (normalized to [-1, 1])
    joint_pos_norm = normalize_joint_positions(robot_state.joint_positions)
    
    # Joint velocities (normalized)
    joint_vel_norm = normalize_joint_velocities(robot_state.joint_velocities)
    
    # Concatenate all components
    obs = np.concatenate([
        pos_error.astype(np.float32),      # [3]
        quat_error.astype(np.float32),     # [4]
        joint_pos_norm.astype(np.float32), # [6]
        joint_vel_norm.astype(np.float32), # [6]
    ])
    
    assert obs.shape == (19,), f"Observation shape mismatch: {obs.shape}"
    return obs


def reorder_joints_from_ros(
    joint_names: list,
    positions: list,
    velocities: list,
    robot_prefix: str = "gripper"
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder joint data from ROS message to SIMULATION order.
    
    ROS2 /joint_states contains joints for multiple robots (gripper and screwdriver).
    This function:
    1. Filters for only the specified robot_prefix (e.g., "gripper_")
    2. Maps from ROS order to SIMULATION order
    
    ROS2 /joint_states has joints in alphabetical order with prefix:
        gripper_elbow_joint, gripper_shoulder_lift_joint, gripper_shoulder_pan_joint,
        gripper_wrist_1_joint, gripper_wrist_2_joint, gripper_wrist_3_joint
        screwdriver_* (ignored if robot_prefix="gripper")
    
    Simulation expects:
        shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    
    Args:
        joint_names: Joint names from ROS message
        positions: Joint positions from ROS message
        velocities: Joint velocities from ROS message
        robot_prefix: Robot prefix to filter by (e.g., "gripper" or "screwdriver")
        
    Returns:
        Tuple of (positions, velocities) arrays in SIMULATION order [6]
    """
    pos_ordered = np.zeros(6, dtype=np.float32)
    vel_ordered = np.zeros(6, dtype=np.float32)
    
    # Build mapping from ROS index to simulation index
    # First, filter for the robot prefix and match joint suffixes
    ros_to_sim_mapping = {}
    
    for ros_idx, ros_name in enumerate(joint_names):
        # Only consider joints with the correct prefix
        if not ros_name.startswith(robot_prefix + "_"):
            continue
        
        # Check each known joint suffix
        for suffix, sim_idx in JOINT_NAME_SUFFIXES.items():
            if ros_name.endswith(suffix):
                ros_to_sim_mapping[ros_idx] = sim_idx
                break
    
    # Verify we found all 6 joints for the specified robot
    if len(ros_to_sim_mapping) != 6:
        found_joints = [joint_names[i] for i in ros_to_sim_mapping.keys()]
        available_for_prefix = [name for name in joint_names if name.startswith(robot_prefix + "_")]
        raise ValueError(
            f"Expected 6 {robot_prefix} UR5e joints, found {len(ros_to_sim_mapping)}. "
            f"Found: {found_joints}, "
            f"Available for prefix '{robot_prefix}': {available_for_prefix}, "
            f"All available: {joint_names}"
        )
    
    # Reorder to simulation order
    for ros_idx, sim_idx in ros_to_sim_mapping.items():
        pos_ordered[sim_idx] = positions[ros_idx]
        vel_ordered[sim_idx] = velocities[ros_idx] if ros_idx < len(velocities) else 0.0
    
    return pos_ordered, vel_ordered


# ============================================================================
# Action denormalization (policy outputs -> joint targets)
# ============================================================================

# Action scaling - REDUCED for real robot safety (IsaacSim uses 7.5)
# Start with lower value and increase gradually if needed
ACTION_SCALE = 5.0  # Original: 7.5, reduced to avoid overshooting
DOF_VELOCITY_SCALE = 0.1


def compute_dof_targets(
    current_targets: np.ndarray,
    actions: np.ndarray,
    dt: float = 1/60,  # 60Hz policy
    action_scale: float = ACTION_SCALE  # Can be overridden
) -> np.ndarray:
    """Compute new joint position targets from policy actions.
    
    Formula (from IsaacSim):
        inc = dt * dof_velocity_scale * action_scale * actions
        targets = clamp(targets + inc, lower, upper)
    
    Args:
        current_targets: Current joint position targets [6]
        actions: Policy output actions [6] in [-1, 1]
        dt: Timestep (default 1/60 for 60Hz)
        action_scale: Scaling factor (default ACTION_SCALE, can reduce for safety)
        
    Returns:
        New joint position targets [6] (clamped to limits)
    """
    # Clamp actions to [-1, 1]
    actions = np.clip(actions, -1.0, 1.0)
    
    # Compute increment (use provided action_scale)
    inc = dt * DOF_VELOCITY_SCALE * action_scale * actions
    
    # Update targets
    new_targets = current_targets + inc
    
    # Clamp to joint limits
    new_targets = np.clip(new_targets, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    
    return new_targets.astype(np.float32)
