"""
Observation builder for sim2real transfer – **V2** (normalised, error-based).

Matches the exact observation layout from the IsaacSim
``pose_orientation_sim2real_v2`` task (24 dims, normalised).

Observation vector (24 dims):
  - to_target_norm       [3]: (goal_pos - tcp_pos) / MAX_REACH
  - orientation_error_norm [3]: quat_box_minus(goal_quat, tcp_quat) / π
  - joint_pos_norm       [6]: 2*(q - lower)/(upper - lower) - 1
  - joint_vel_norm       [6]: qd / MAX_JOINT_VEL
  - tcp_linear_vel_norm  [3]: v_tcp / TCP_MAX_SPEED
  - tcp_angular_vel_norm [3]: ω_tcp / π
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# Constants (must match robot_configs.py)
# ============================================================================

MAX_REACH: float = 0.85           # UR5e reach ~850 mm
MAX_JOINT_VEL: float = 3.14       # ~180°/s
TCP_MAX_SPEED: float = 2.0        # m/s  (cfg.tcp_max_speed in sim)

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

# Joint names in SIMULATION order
JOINT_NAMES_SIM = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ============================================================================
# Quaternion / axis-angle helpers  (numpy, (w, x, y, z) convention)
# ============================================================================

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a unit quaternion (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def axis_angle_from_quat(q: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to axis-angle (3,).

    Exact replica of IsaacLab ``axis_angle_from_quat``.
    """
    # Ensure w >= 0 (canonical form)
    if q[0] < 0.0:
        q = -q

    xyz = q[1:4]
    mag = np.linalg.norm(xyz)
    half_angle = math.atan2(mag, q[0])
    angle = 2.0 * half_angle

    if abs(angle) > eps:
        sin_half_over_angle = math.sin(half_angle) / angle
    else:
        sin_half_over_angle = 0.5 - angle * angle / 48.0

    return (xyz / sin_half_over_angle).astype(np.float32)


def quat_box_minus(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Box-minus operator: log(q1 * q2^{-1}).  Returns axis-angle (3,).

    Matches ``isaaclab.utils.math.quat_box_minus`` exactly.
    Quaternions are in (w, x, y, z) convention.
    """
    quat_diff = quat_multiply(q1, quat_conjugate(q2))
    return axis_angle_from_quat(quat_diff)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by the inverse of quaternion *q* (w, x, y, z).

    Equivalent to ``q^{-1} * v * q`` for a unit quaternion.
    """
    q_conj = quat_conjugate(q)
    # encode v as pure quaternion (0, vx, vy, vz)
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    result = quat_multiply(quat_multiply(q_conj, v_quat), q)
    return result[1:4]


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class RobotState:
    """Current robot state."""
    joint_positions: np.ndarray    # [6] radians
    joint_velocities: np.ndarray   # [6] rad/s
    ee_position: np.ndarray        # [3] meters  (in source/table-centre frame)
    ee_quaternion: np.ndarray      # [4] (w, x, y, z) (in source/table-centre frame)
    tcp_linear_vel: np.ndarray     # [3] m/s  (in source frame)
    tcp_angular_vel: np.ndarray    # [3] rad/s (in source frame)


@dataclass
class GoalState:
    """Goal pose for the end-effector."""
    position: np.ndarray   # [3] meters  (in source/table-centre frame)
    quaternion: np.ndarray  # [4] (w, x, y, z)


# ============================================================================
# Observation builder  (24-dim, normalised – matches simulation exactly)
# ============================================================================

def build_observation(robot_state: RobotState, goal_state: GoalState) -> np.ndarray:
    """Build the 24-dim **normalised** observation vector.

    Layout (must match IsaacSim ``_get_observations`` in v2 exactly):
      [0:3]   to_target_norm        – (goal - tcp) / MAX_REACH
      [3:6]   orientation_error_norm – quat_box_minus(goal_q, tcp_q) / π
      [6:12]  joint_pos_norm        – 2*(q - lower)/(upper - lower) - 1
      [12:18] joint_vel_norm        – qd / MAX_JOINT_VEL
      [18:21] tcp_linear_vel_norm   – v / TCP_MAX_SPEED
      [21:24] tcp_angular_vel_norm  – ω / π

    Args:
        robot_state: Current robot state (all quantities in source frame).
        goal_state:  Goal pose (source frame).

    Returns:
        Observation vector [24] as float32.
    """
    # 1. Position error (normalised)
    to_target_norm = (goal_state.position - robot_state.ee_position) / MAX_REACH  # (3,)

    # 2. Orientation error (normalised)
    orientation_error = quat_box_minus(goal_state.quaternion, robot_state.ee_quaternion)
    orientation_error_norm = orientation_error / np.pi  # (3,)

    # 3. Joint positions (normalised to [-1, 1])
    joint_pos_norm = (
        2.0 * (robot_state.joint_positions - JOINT_LIMITS_LOWER)
        / (JOINT_LIMITS_UPPER - JOINT_LIMITS_LOWER)
        - 1.0
    )

    # 4. Joint velocities (normalised)
    joint_vel_norm = robot_state.joint_velocities / MAX_JOINT_VEL

    # 5. TCP linear velocity (normalised)
    tcp_linear_vel_norm = robot_state.tcp_linear_vel / TCP_MAX_SPEED

    # 6. TCP angular velocity (normalised)
    tcp_angular_vel_norm = robot_state.tcp_angular_vel / np.pi

    obs = np.concatenate([
        to_target_norm.astype(np.float32),         # [3]
        orientation_error_norm.astype(np.float32),  # [3]
        joint_pos_norm.astype(np.float32),          # [6]
        joint_vel_norm.astype(np.float32),          # [6]
        tcp_linear_vel_norm.astype(np.float32),     # [3]
        tcp_angular_vel_norm.astype(np.float32),    # [3]
    ])

    assert obs.shape == (24,), f"Observation shape mismatch: expected (24,), got {obs.shape}"
    return obs


# ============================================================================
# Action → DOF targets  (matches v2 _pre_physics_step)
# ============================================================================


def compute_dof_targets(
    current_targets: np.ndarray,
    actions: np.ndarray,
    dt: float = 1 / 60,
    action_scale: float = 3.0,
) -> np.ndarray:
    """Compute new joint position targets from policy actions.

    Formula (from IsaacSim v2):
        inc = speed_scales * dt * dof_velocity_scale * actions * action_scale
        targets = clamp(targets + inc, lower, upper)

    Note: v2 uses uniform speed_scales = 1.0 (no per-joint scaling).

    Args:
        current_targets: Current joint position targets [6]
        actions: Policy output actions [6] in [-1, 1]
        dt: Timestep (default 1/60 for 60 Hz)
        action_scale: Scaling factor (default 3.0 as in sim)

    Returns:
        New joint position targets [6] (clamped to limits)
    """
    actions = np.clip(actions, -1.0, 1.0)

    inc = dt * action_scale * actions  # speed_scales=1
    new_targets = current_targets + inc
    new_targets = np.clip(new_targets, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    return new_targets.astype(np.float32)
