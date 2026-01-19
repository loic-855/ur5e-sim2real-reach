#!/usr/bin/env python3
"""
ur5e_gripper_reach.py
----------------------

Policy wrapper to run the pose-orientation reach policy (27 obs / 7 actions)
trained in Isaac Lab on the UR5e + OnRobot gripper. It mirrors the sim-side
observation and action preprocessing so the TorchScript policy can run on the
real robot.
"""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from controllers.policy_controller import PolicyController
from utils.config_loader import get_task_properties

# UR5e hard joint limits (rad) and gripper stroke (m)
# Note: right_finger_joint is mimicked from left_finger_joint in sim,
# but we need it in observations to match the 29D policy input.
_JOINT_LIMITS = np.array(
    [
        [-2.0 * np.pi, 2.0 * np.pi],  # shoulder_pan_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # shoulder_lift_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # elbow_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # wrist_1_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # wrist_2_joint
        [-2.0 * np.pi, 2.0 * np.pi],  # wrist_3_joint
        [0.0, 0.04],  # left_finger_joint
        [0.0, 0.04],  # right_finger_joint (mimicked from left in sim)
    ],
    dtype=np.float64,
)


class UR5eGripperReachPolicy(PolicyController):
    """UR5e reach policy wrapper (27D obs, 7D actions)."""

    def __init__(
        self,
        policy_path: Optional[Path] = None,
        env_path: Optional[Path] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        super().__init__()

        # Joint ordering must match training (8 joints for observation)
        # Note: right_finger_joint is mimicked in sim but included in observations
        self.dof_names = [
            "elbow_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "left_finger_joint",
            "right_finger_joint",  # mimicked from left, but needed for obs
        ]
        # Action indices exclude right_finger (only 7 actuated joints)
        self.action_joint_indices = list(range(7))  # indices 0-6

        # Resolve default model/env paths if not provided
        repo_root = Path(__file__).resolve().parents[3]
        if run_root is None:
            run_root = repo_root / "logs" / "rsl_rl" / "pose_orientation_reach_v1"
        run_dir = None
        if policy_path is None or env_path is None:
            if run_root.exists():
                candidates = sorted([p for p in run_root.iterdir() if p.is_dir()])
                if candidates:
                    run_dir = candidates[-1]
        policy_path = policy_path or (run_dir / "model_300.pt" if run_dir else None)
        env_path = env_path or (run_dir / "params" / "env.yaml" if run_dir else None)
        if policy_path is None or env_path is None:
            raise FileNotFoundError("Could not resolve default policy/env paths. Provide them explicitly.")

        self.load_policy(policy_path, env_path)

        # Ensure policy_env_params was loaded
        if self.policy_env_params is None:
            raise RuntimeError("policy_env_params not loaded - check env.yaml path")

        # Task scales from env config
        self.action_scale, self.dof_velocity_scale = get_task_properties(self.policy_env_params)
        # Note: policy_dt is now a property from the base class

        # Joint limits + speed scales (fingers slowed like in sim)
        self.joint_lower = _JOINT_LIMITS[:, 0].copy()
        self.joint_upper = _JOINT_LIMITS[:, 1].copy()
        self.speed_scales = np.ones_like(self.joint_lower)
        self.speed_scales[6] = 0.5  # left_finger_joint
        self.speed_scales[7] = 0.5  # right_finger_joint (mimicked)

        # Use defaults from base class load_policy (already parsed)
        self.joint_targets = np.array(self.default_pos, dtype=np.float64)
        self.current_joint_positions = np.array(self.default_pos, dtype=np.float64)
        self.current_joint_velocities = np.array(self.default_vel, dtype=np.float64)

        # Actions are 7D (excludes mimicked right_finger_joint)
        self.num_actions = 7
        self.last_actions = np.zeros(self.num_actions, dtype=np.float64)
        self._policy_counter = 0
        self.has_joint_data = False

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def update_joint_state(self, positions: Iterable[float], velocities: Iterable[float]) -> None:
        """Update joint state from real robot.
        
        The real robot may only report 7 joints (without mimicked right_finger).
        We mirror left_finger_joint to right_finger_joint for observation.
        """
        pos_arr = np.array(list(positions), dtype=np.float64)
        vel_arr = np.array(list(velocities), dtype=np.float64)

        # Real robot has 7 joints, we need 8 for observation
        if len(pos_arr) == 7:
            # Mirror left_finger (index 6) to right_finger (index 7)
            pos_arr = np.append(pos_arr, pos_arr[6])
        if len(vel_arr) == 7:
            vel_arr = np.append(vel_arr, vel_arr[6])

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



        dof_pos_scaled = 2.0 * (self.current_joint_positions - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0
        joint_vel_scaled = self.current_joint_velocities * self.dof_velocity_scale
        to_target = goal_pos - grasp_pos

                # DEBUG: Vérifier les entrées
        print(f"[OBS] Joint positions: {self.current_joint_positions}")
        print(f"[OBS] Joint velocities: {self.current_joint_velocities}")
        print(f"[OBS] Goal: {goal_pos}")

        obs = np.concatenate(
            [
                dof_pos_scaled,
                joint_vel_scaled,
                to_target,
                goal_pos,
                self.last_actions,
            ]
        ).astype(np.float32)
        print(f"[OBS] Shape: {obs.shape}, Values: {obs}[:5]...") 
        return obs

    def forward(self, dt: float, goal_pos: np.ndarray, grasp_pos: np.ndarray) -> Optional[np.ndarray]:
        """Compute next joint targets from policy.
        
        Returns 7D targets (excluding mimicked right_finger_joint) for the controller.
        """
        if not self.has_joint_data:
            return None

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(goal_pos, grasp_pos)
            if obs is None:
                return None
            action = self._compute_action(obs)
            action = np.clip(action, -1.0, 1.0)
            self.last_actions = action.copy()
        else:
            action = self.last_actions

        # Action is 7D, but we have 8 joint targets (with mimicked right_finger)
        # Extend action to 8D by mirroring left_finger action to right_finger
        action_8d = np.append(action, action[6])  # mirror finger action

        deltas = (
            self.speed_scales
            * dt
            * self.dof_velocity_scale
            * self.action_scale
            * action_8d
        )
        new_targets = np.clip(self.joint_targets + deltas, self.joint_lower, self.joint_upper)
        self.joint_targets = new_targets
        self._policy_counter += 1
        
        # Return only 7 targets for the real robot controller (exclude mimicked right_finger)
        return self.joint_targets[:7].copy()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def sample_goal(base_pos) -> np.ndarray:
        """Sample a goal like the sim task (above the table, z down).
        
        Args:
            base_pos: Base position as tuple or array-like (x, y, z).
        """

        offsets = np.empty(3, dtype=np.float32)
        offsets[0] = np.random.uniform(0.15, 1.2 - 0.23)
        offsets[1] = np.random.uniform(0.15, 0.8 - 0.25)
        offsets[2] = np.random.uniform(0.05, 0.6)
        return np.array(base_pos, dtype=np.float32) + offsets
