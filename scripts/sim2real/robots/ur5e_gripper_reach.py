#!/usr/bin/env python3
"""
ur5e_gripper_reach.py
----------------------

Policy wrapper to run the pose-orientation reach policy trained in Isaac Lab
on a dual-arm robot setup. Adapted for gripper robot with prefix "gripper_".
Only controls the 6 arm joints (gripper stays open).
"""

import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Add scripts directory to path to allow imports from sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from controllers.policy_controller import PolicyController
from utils.config_loader import get_task_properties
from obs_logger import ObsLogger

# Gripper robot joint limits (rad) and gripper finger limits (m)
# Note: right_finger_joint is mimicked from left_finger_joint in sim
_JOINT_LIMITS = np.array(
    [
        [-2.0 * np.pi, 2.0 * np.pi],  # gripper_shoulder_pan_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # gripper_shoulder_lift_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # gripper_elbow_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # gripper_wrist_1_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # gripper_wrist_2_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # gripper_wrist_3_joint
        [0.0, 0.04],  # left_finger_joint
        [0.0, 0.04],  # right_finger_joint (mimicked from left in sim)
    ],
    dtype=np.float64,
)

# Physics constants
BLOCK_THICKNESS = 0.015
BLOCK_MARGIN = 0.003
TABLE_DEPTH = 0.8
TABLE_WIDTH = 1.2 
TABLE_HEIGHT = 0.842

class UR5eGripperReachPolicy(PolicyController):
    """Dual-arm gripper robot reach policy wrapper (29D obs, 7D actions).
    
    Controls only the 6 gripper arm joints. Gripper stays open.
    """

    def __init__(
        self,
        policy_path: Optional[Path] = None,
        env_path: Optional[Path] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        super().__init__()

        # Joint ordering must match training (8 joints for observation)
        # Note: Observation order matches simulation without gripper_ prefix
        self.dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "left_finger_joint",
            "right_finger_joint",  # mimicked from left, but needed for obs
        ]
        # Real robot joint names with gripper_ prefix
        self.robot_joint_names = [
            "gripper_shoulder_pan_joint",
            "gripper_shoulder_lift_joint",
            "gripper_elbow_joint",
            "gripper_wrist_1_joint",
            "gripper_wrist_2_joint",
            "gripper_wrist_3_joint",
        ]
        # Action indices for 6 arm joints (gripper stays open)
        self.action_joint_indices = list(range(6))  # indices 0-5

        # Resolve default model/env paths if not provided
        repo_root = Path(__file__).resolve().parents[3]
        if run_root is None:
            run_root = repo_root / "pretrained_models" / "pose_orientation_reach_v1" / "2026-01-16_14-25-14_final"
        
        policy_path = policy_path or (run_root / "exported" / "policy.pt")
        env_path = env_path or (run_root / "params" / "env.yaml")
        
        if not policy_path.exists() or not env_path.exists():
            raise FileNotFoundError(f"Policy or env file not found: {policy_path}, {env_path}")

        self.load_policy(policy_path, env_path)

        # Ensure policy_env_params was loaded
        if self.policy_env_params is None:
            raise RuntimeError("policy_env_params not loaded - check env.yaml path")

        # Task scales from env config
        self.action_scale, self.dof_velocity_scale = get_task_properties(self.policy_env_params)
        # Note: policy_dt is now a property from the base class

        # Joint limits + speed scales
        self.joint_lower = _JOINT_LIMITS[:, 0].copy()
        self.joint_upper = _JOINT_LIMITS[:, 1].copy()
        self.speed_scales = np.ones_like(self.joint_lower)
        self.speed_scales[6] = 0.5  # left_finger_joint
        self.speed_scales[7] = 0.5  # right_finger_joint (mimicked)

        # Use defaults from base class load_policy (already parsed)
        self.joint_targets = np.array(self.default_pos, dtype=np.float64)
        self.current_joint_positions = np.array(self.default_pos, dtype=np.float64)
        self.current_joint_velocities = np.array(self.default_vel, dtype=np.float64)

        # Actions are 7D but we only actuate 6 arm joints (gripper stays open at default)
        self.num_actions = 7
        self.last_actions = np.zeros(self.num_actions, dtype=np.float64)
        self._policy_counter = 0
        self.has_joint_data = False

        # Obs logger (records obs + action in a single binary file, non-blocking)
        # Annotate as Optional for static checkers
        self._obs_logger: Optional[ObsLogger] = None
        try:
            self._obs_logger = ObsLogger(out_dir=(repo_root / "logs" / "sim2real"), base_name="obs", queue_max_size=10000)
        except Exception as e:
            print(f"Warning: Failed to create ObsLogger: {e}")
            self._obs_logger = None
        
        # Print joint mapping for debugging
        print("\n=== JOINT MAPPING ===")
        print("Policy expects (training order):")
        for i, name in enumerate(self.dof_names):
            print(f"  [{i}] {name}")
        print("\nRobot publishes (ROS2 order):")
        for i, name in enumerate(self.robot_joint_names):
            print(f"  [{i}] {name}")
        print("=====================\n")

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def update_joint_state(self, positions: Iterable[float], velocities: Iterable[float]) -> None:
        """Update joint state from real robot.
        
        Expects 6 joint values (arm only). Adds finger joints at default values
        and mirrors left_finger to right_finger for observation.
        """
        pos_arr = np.array(list(positions), dtype=np.float64)
        vel_arr = np.array(list(velocities), dtype=np.float64)

        # Real robot reports 6 arm joints, we need 8 for observation (6 arm + 2 fingers)
        if len(pos_arr) == 6:
            # Add finger joints at default (open position)
            finger_pos = self.default_pos[6] if len(self.default_pos) > 6 else 0.0
            pos_arr = np.append(pos_arr, [finger_pos, finger_pos])  # left and right fingers
        if len(vel_arr) == 6:
            vel_arr = np.append(vel_arr, [0.0, 0.0])  # finger velocities = 0

        n = min(len(self.dof_names), len(pos_arr))
        self.current_joint_positions[:n] = pos_arr[:n]
        self.current_joint_velocities[:n] = vel_arr[:n] if len(vel_arr) >= n else 0.0
        self.has_joint_data = True

    # ------------------------------------------------------------------
    # Observation / Action
    # ------------------------------------------------------------------
    def _compute_observation(self, goal_pos: np.ndarray, grasp_pos: np.ndarray) -> Optional[np.ndarray]:
        if not self.has_joint_data:
            return None

        # Scale joint positions to [-1, 1]
        dof_pos_scaled = 2.0 * (self.current_joint_positions - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0
        joint_vel_scaled = self.current_joint_velocities * self.dof_velocity_scale
        to_target = goal_pos - grasp_pos

        # Observation: [8 joint_pos, 8 joint_vel, 3 to_target, 3 goal_pos, 7 last_actions] = 29D
        obs = np.concatenate(
            [
                dof_pos_scaled,
                joint_vel_scaled,
                to_target,
                goal_pos,
                self.last_actions,
            ]
        ).astype(np.float32)
        
        
        return obs

    def forward(self, dt: float, goal_pos: np.ndarray, grasp_pos: np.ndarray) -> Optional[np.ndarray]:
        """Compute next joint targets from policy.
        
        Returns 6D targets (arm joints only, gripper stays open) for the controller.
        """
        if not self.has_joint_data:
            return None

        # Compute observation once and reuse
        obs = self._compute_observation(goal_pos, grasp_pos)
        if obs is None:
            return None

        if self._policy_counter % self._decimation == 0:
            action = self._compute_action(obs)
            action = np.clip(action, -1.0, 1.0)
            self.last_actions = action.copy()
        else:
            action = self.last_actions

        # Log obs+action (non-blocking). Stored record = [29 obs, 7 action] = 36 float32.
        if self._obs_logger is not None:
            try:
                self._obs_logger.push(obs, action)
            except Exception as e:
                print(f"ObsLogger push failed: {e}")

        # Action is 7D (6 arm + 1 finger), but we only actuate the 6 arm joints
        # Extend action to 8D for internal state tracking by mirroring finger
        action_8d = np.append(action, action[6])  # mirror finger action

        deltas = (
            self.speed_scales
            * dt
            * self.dof_velocity_scale
            * self.action_scale
            * action_8d
        )
        new_targets = np.clip(self.joint_targets + deltas, self.joint_lower, self.joint_upper)
        
        # Debug: Log target computation
        if self._policy_counter % 120 == 0:
            print(f"Deltas (8D): {deltas[:6]}")
            print(f"Old targets (6D): {self.joint_targets[:6]}")
            print(f"New targets (6D): {new_targets[:6]}")
            print("========================\n")
        
        self.joint_targets = new_targets
        self._policy_counter += 1
        
        # Return only 6 arm joint targets for the real robot (gripper stays open)
        return self.joint_targets[:6].copy()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def close_logger(self) -> None:
        """Close the ObsLogger if present (flush and write metadata)."""
        if self._obs_logger is not None:
            try:
                self._obs_logger.close()
            except Exception as e:
                print(f"ObsLogger close failed: {e}")
            self._obs_logger = None

    @staticmethod
    def sample_goal(base_pos) -> np.ndarray:
        """Sample a goal like the sim task (above the table, z down).
        
        Args:
            base_pos: Base position as tuple or array-like (x, y, z).
        """
        offsets = np.empty(3, dtype=np.float32)
        offsets[0] = np.random.uniform(-TABLE_DEPTH/2, TABLE_DEPTH/2)
        offsets[1] = np.random.uniform(-TABLE_WIDTH/2, TABLE_WIDTH/2)
        offsets[2] = np.random.uniform(0.05, 0.6)
        return np.array(base_pos, dtype=np.float32) + offsets
