"""
Observation builder for sim2real transfer.
Matches the exact observation layout from the IsaacSim
``pose_orientation_sim2real`` task (26 dims, **no normalisation**).

Observation vector (26 dims):
  - ee_pos_source  [3]: EE position in table-centre frame
  - ee_quat_source [4]: EE orientation quaternion (w, x, y, z)
  - joint_pos      [6]: raw joint positions (rad)
  - joint_vel      [6]: raw joint velocities (rad/s)
  - goal_pos       [3]: goal position in table-centre frame
  - goal_quat      [4]: goal orientation quaternion (w, x, y, z)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# Joint constants
# ============================================================================

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
JOINT_NAME_SUFFIXES = {
    "shoulder_pan_joint": 0,   # sim index 0
    "shoulder_lift_joint": 1,  # sim index 1
    "elbow_joint": 2,          # sim index 2
    "wrist_1_joint": 3,        # sim index 3
    "wrist_2_joint": 4,        # sim index 4
    "wrist_3_joint": 5,        # sim index 5
}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class RobotState:
    """Current robot state."""
    joint_positions: np.ndarray  # [6] radians
    joint_velocities: np.ndarray  # [6] rad/s
    ee_position: np.ndarray  # [3] meters (in table-centre frame)
    ee_quaternion: np.ndarray  # [4] (w, x, y, z)


@dataclass
class GoalState:
    """Goal pose for the end-effector."""
    position: np.ndarray  # [3] meters (in table-centre frame)
    quaternion: np.ndarray  # [4] (w, x, y, z)


# ============================================================================
# Observation builder  (26-dim, raw – matches simulation exactly)
# ============================================================================

def build_observation(robot_state: RobotState, goal_state: GoalState) -> np.ndarray:
    """Build the 26-dim observation vector (no normalisation).

    Layout (must match IsaacSim ``_get_observations`` exactly):
      [0:3]   ee_pos_source   – EE position
      [3:7]   ee_quat_source  – EE quaternion
      [7:13]  joint_pos       – raw joint positions
      [13:19] joint_vel       – raw joint velocities
      [19:22] goal_pos        – goal position
      [22:26] goal_quat       – goal quaternion

    Args:
        robot_state: Current robot state
        goal_state: Goal pose

    Returns:
        Observation vector [26] as float32
    """
    obs = np.concatenate([
        robot_state.ee_position.astype(np.float32),       # [3]
        robot_state.ee_quaternion.astype(np.float32),      # [4]
        robot_state.joint_positions.astype(np.float32),    # [6]
        robot_state.joint_velocities.astype(np.float32),   # [6]
        goal_state.position.astype(np.float32),            # [3]
        goal_state.quaternion.astype(np.float32),          # [4]
    ])

    assert obs.shape == (26,), f"Observation shape mismatch: expected (26,), got {obs.shape}"
    return obs


# ============================================================================
# Joint reordering  (ROS → simulation order)
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
