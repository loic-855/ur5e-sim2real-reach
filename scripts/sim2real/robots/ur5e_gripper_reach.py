#!/usr/bin/env python3
"""
ur5e_gripper_reach.py
----------------------
Policy wrapper for UR5e gripper robot (6 arm joints + finger control).

OBSERVATION (29D):
    [0:8]   - 8 scaled joint positions [-1,1]: 6 arm + 2 fingers
    [8:16]  - 8 scaled joint velocities
    [16:19] - 3D vector to target (goal - grasp)
    [19:22] - 3D goal position
    [22:29] - 7 last actions

ACTION (7D):
    [0:6]   - 6 arm joint velocity deltas (scaled)
    [6]     - finger action: -1→closed (0mm), +1→open (20mm per finger = 40mm total)

SCALING:
    - Joint positions: scaled to [-1,1] via (pos - lower) / (upper - lower) * 2 - 1
    - Joint velocities: scaled by dof_velocity_scale factor
    - Arm actions: delta = speed_scale * dt * dof_vel_scale * action_scale * action
    - Finger action: directly maps [-1,1] → [0mm, 40mm] gripper opening
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from controllers.policy_controller import PolicyController
from utils.config_loader import get_task_properties
from obs_logger import ObsLogger

# Joint limits: 6 arm (rad) + 2 finger (m)
JOINT_LIMITS = np.array([
    [-2*np.pi, 2*np.pi],  # shoulder_pan
    [-2*np.pi, 2*np.pi],  # shoulder_lift
    [-np.pi, np.pi],  # elbow, corrected from IsaacSim default
    [-2*np.pi, 2*np.pi],  # wrist_1
    [-2*np.pi, 2*np.pi],  # wrist_2
    [-2*np.pi, 2*np.pi],  # wrist_3
    [0.0, 0.02],          # left_finger (0-20mm)
    [0.0, 0.02],          # right_finger (mirrored)
], dtype=np.float64)

# Gripper mapping constants
FINGER_MAX_M = 0.02       # Max finger position = 20mm
GRIPPER_MAX_MM = 40.0     # Total gripper opening = 40mm (both fingers)

# Goal sampling bounds
TABLE_DEPTH, TABLE_WIDTH = 0.8, 1.2


class UR5eGripperReachPolicy(PolicyController):
    """UR5e reach policy: 29D observation, 7D action (6 arm + 1 finger).
    
    Gripper handling:
    - Receives 7 joint states: 6 arm + 1 finger (single command for both fingers)
    - Computes finger velocity from position delta / dt
    - In observation: duplicates finger position negatively for right_finger (mirrored)
    """

    def __init__(self, policy_path: Optional[Path] = None, env_path: Optional[Path] = None) -> None:
        super().__init__()

        # Joint names must match training order (without gripper_ prefix)
        self.dof_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "left_finger_joint", "right_finger_joint",
        ]

        # Load policy
        repo_root = Path(__file__).resolve().parents[3]
        run_root = repo_root / "pretrained_models/pose_orientation_reach_v1/2026-01-16_14-25-14_final"
        policy_path = policy_path or (run_root / "exported/policy.pt")
        env_path = env_path or (run_root / "params/env.yaml")
        self.load_policy(policy_path, env_path)

        # Scaling factors from env config (policy_env_params is set by load_policy)
        assert self.policy_env_params is not None
        self.action_scale, self.dof_velocity_scale = 0.6, 0.3 #get_task_properties(self.policy_env_params)

        # Joint limits
        self.joint_lower = JOINT_LIMITS[:, 0].copy()
        self.joint_upper = JOINT_LIMITS[:, 1].copy()
        
        # Speed scales (fingers move slower)
        self.speed_scales = np.ones(8, dtype=np.float64)
        self.speed_scales[6:8] = 0.5

        # State tracking (8 joints for obs, but only 7 from ROS2: 6 arm + 1 finger)
        self.joint_targets = np.array(self.default_pos, dtype=np.float64)
        self.joint_positions = np.array(self.default_pos, dtype=np.float64)
        self.joint_velocities = np.zeros(8, dtype=np.float64)
        self.last_actions = np.zeros(7, dtype=np.float64)
        
        self._policy_counter = 0
        self._has_joint_data = False

        # Observation logger
        self._obs_logger: Optional[ObsLogger] = None
        try:
            self._obs_logger = ObsLogger(out_dir=repo_root / "logs/sim2real", base_name="obs", queue_max_size=10000)
        except Exception as e:
            print(f"Warning: ObsLogger init failed: {e}")

    def update_joint_state(self, arm_pos: np.ndarray, arm_vel: np.ndarray, 
                           finger_pos_m: float, finger_vel_m: float) -> None:
        """Update state from real robot.
        
        Args:
            arm_pos: 6 arm joint positions (rad) in policy order
            arm_vel: 6 arm joint velocities (rad/s) in policy order
            finger_pos_m: Single finger position in meters (0-0.02m = 0-20mm opening)
            finger_vel_m: Single finger velocity in m/s (computed from position delta / dt)
        """
        # Update arm state
        self.joint_positions[:6] = arm_pos
        self.joint_velocities[:6] = arm_vel
        
        # Update finger state (single joint internally)
        # - joint_positions[6]: left finger position
        # - joint_positions[7]: will be computed in _compute_observation (negative duplicate)
        #self.joint_positions[6] = np.clip(finger_pos_m, 0.0, FINGER_MAX_M)
        #self.joint_velocities[6] = finger_vel_m
        #attempt to stabilize the gripper observation
        self.joint_positions[6] = np.random.normal(1, 0.001)
        self.joint_velocities[6] = np.random.normal(0, 0.05)
        
        # CRITICAL: On first state update, sync targets with actual robot position
        # to avoid teleportation on first forward() call
        if not self._has_joint_data:
            self.joint_targets[:6] = arm_pos.copy()
            self.joint_targets[6] = self.joint_positions[6]
        
        self._has_joint_data = True

    def _compute_observation(self, goal_pos: np.ndarray, grasp_pos: np.ndarray) -> np.ndarray:
        """Build 29D observation vector.
        
        Gripper representation:
        - joint_positions[6]: left_finger (positive)
        - joint_positions[7]: -joint_positions[6] (right_finger, negative duplicate for mirrored policy)
        """
        # Create full 8-joint position array with mirrored gripper
        pos_full = self.joint_positions.copy()
        pos_full[7] = pos_full[6]  # Right finger= left finger
        
        # Create full 8-joint velocity array with mirrored gripper
        vel_full = self.joint_velocities.copy()
        vel_full[7] = -vel_full[6]  # Right finger velocity = negative of left finger velocity
        
        # Scale positions and velocities
        pos_scaled = 2.0 * (pos_full - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0
        vel_scaled = vel_full * self.dof_velocity_scale
        
        obs = np.concatenate([
            pos_scaled,             # 8: joint positions [-1,1]
            vel_scaled,             # 8: scaled velocities
            goal_pos - grasp_pos,   # 3: to-target vector
            goal_pos,               # 3: goal position
            self.last_actions       # 7: last actions
        ]).astype(np.float32)
        
        return np.clip(obs, -5.0, 5.0)

    def forward(self, dt: float, goal_pos: np.ndarray, grasp_pos: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Compute next targets from policy.
        
        Returns:
            (arm_targets, gripper_opening_mm): 6 arm targets (rad), gripper opening (0-40mm)
        """
        if not self._has_joint_data:
            return None, None

        obs = self._compute_observation(goal_pos, grasp_pos)

        # Run policy at decimated rate
        if self._policy_counter % self._decimation == 0:
            action = self._compute_action(obs)
            self.last_actions = action.copy()
        else:
            action = self.last_actions

        # Log observation + action
        if self._obs_logger:
            self._obs_logger.push(obs, action)

        # --- ARM JOINTS (action[0:6]) ---
        # delta = speed_scale * dt * dof_vel_scale * action_scale * action
        arm_deltas = self.speed_scales[:6] * dt * self.dof_velocity_scale * self.action_scale * action[:6]
        arm_targets = np.clip(self.joint_targets[:6] + arm_deltas, self.joint_lower[:6], self.joint_upper[:6])
        self.joint_targets[:6] = arm_targets

        # --- FINGER (action[6]) ---
        # Return raw action [-1, 1] - scaling happens in _command_gripper()
        finger_action = action[6]
        
        # Update internal finger target for observation consistency
        # (scaling will happen in _command_gripper for actual command)
        gripper_opening_mm = (finger_action + 1.0) / 2.0 * GRIPPER_MAX_MM  # Map [-1,1] → [0,40]mm
        finger_pos_m = gripper_opening_mm / 2.0 / 1000.0  # Per-finger in meters
        self.joint_targets[6] = np.clip(finger_pos_m, 0.0, FINGER_MAX_M)
        
        self._policy_counter += 1
        return arm_targets.copy(), finger_action

    def reset(self, home_position: Optional[np.ndarray] = None) -> None:
        """Reset internal state for periodic resets during deployment.
        
        Args:
            home_position: 6D array of arm joint positions to initialize targets.
                          If None, uses default_pos from training config.
        """
        # IMPORTANT: Always sync joint_targets with joint_positions (robot's actual position)
        # to avoid sudden jumps in the next forward() call
        self.joint_targets[:6] = self.joint_positions[:6].copy()
        self.joint_targets[6] = self.joint_positions[6]  # Only single finger joint
        
        # Reset state
        self.joint_velocities = np.zeros(8, dtype=np.float64)
        self.last_actions = np.zeros(7, dtype=np.float64)
        self._policy_counter = 0

    def close_logger(self) -> None:
        """Close observation logger."""
        if self._obs_logger:
            self._obs_logger.close()
            self._obs_logger = None

    @staticmethod
    def sample_goal(base_pos: np.ndarray) -> np.ndarray:
        """Sample random goal position above table."""
        offsets = np.array([
            np.random.uniform(-TABLE_DEPTH/2, TABLE_DEPTH/2),
            np.random.uniform(-TABLE_WIDTH/2, TABLE_WIDTH/2),
            np.random.uniform(0.05, 0.6)
        ], dtype=np.float32)
        return np.array(base_pos, dtype=np.float32) + offsets
