# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# launch for good experiement name: python .\scripts\rsl_rl\train.py --task Template-Pose-Orientation-Gripper-Robot-Direct-v1 --num_envs 20 --headless agent.experiment_name=pose_orientation_reach_v1

from __future__ import annotations
from pathlib import Path

import torch
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom


import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz

#path constants
REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"
TABLE_ASSET_PATH = (REPO_ROOT/"USD_files"/"woodworking_table.usd")

# Physics constants
BLOCK_THICKNESS = 0.015
BLOCK_MARGIN = 0.003
TABLE_DEPTH = 0.8
TABLE_WIDTH = 1.2 
TABLE_HEIGHT = 0.842

#Offset to place the env origin at table center
ENV_ORIGIN_OFFSET = torch.tensor([TABLE_DEPTH / 2.0, TABLE_WIDTH / 2.0, TABLE_HEIGHT])

"""
This module contains the versioned (v1) copy of the pose + orientation gripper
environment. It is intentionally a duplicate of the original V0 implementation so
you can iterate on v1 independently.
"""


@configclass
class PoseOrientationGripperRobotV1Cfg(DirectRLEnvCfg):
    """Configuration simplifiée et documentée."""
    
    # === Environment ===
    episode_length_s: float = 4.0
    decimation: int = 2
    
    # === Spaces ===
    action_space: int = 7   # 6 arm joints + 1 gripper
    observation_space: int = 29  # 8 joint_pos + 8 joint_vel + 3 to_target + 3 goal_pos + 7 actions
    state_space: int = 0
    
    # === Action scaling ===
    action_scale: float = 3.0           # Amplitude max des deltas
    dof_velocity_scale: float = 0.3     # Normalisation vitesses
    action_smoothing: float = 0.3       # EMA smoothing [0=no smooth, 1=instant]
    max_action_rate: float = 0.5        # Max action change per step
    
    # === Rewards (all scales should sum to ~1 for stability) ===
    dist_reward_scale: float = 0.5
    dist_penalty_scale: float = 0.2      # Pénalité linéaire sur la distance
    orientation_reward_scale: float = 0.3
    action_penalty_scale: float = 0.012   # Pénalité sur magnitude des actions
    speed_penalty_scale: float = 0.005   # Pénalité sur vitesse des joints
    jerk_penalty_scale: float = 0.01     # Pénalise les à-coups (smoothness)
    
    # === Success criteria ===
    success_dist_threshold: float = 0.02      # 2cm
    success_ori_threshold: float = 0.948       # cos(~10°)
    
    # === Sim2Real ===
    observation_noise_std: float = 0.01
    action_delay_steps: int = 1


    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4092, env_spacing=3.0, replicate_physics=True
    )

    # robot
    gripper_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e_gripper_tcp_small",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_FILES_DIR / "ur5e_gripper_tcp_small.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
                "left_finger_joint": 0.0,
                "right_finger_joint": 0.0,
            },
            pos=(0.08, 0.08, 0.842),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "shoulder_action": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                ],
                damping=60, stiffness=800),
            "wrist_action": ImplicitActuatorCfg(
                joint_names_expr=[
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                ],
                damping=35, stiffness=350),
            "gripper_action": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_finger_joint", "right_finger_joint",
                ],
                damping=5, stiffness=1200),
        }
    )

    # Table asset placement: Width = 1.2m, Depth = 0.8m, Height = 0.842m
    table = sim_utils.UsdFileCfg(
        usd_path=str(TABLE_ASSET_PATH)
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    # marker for debug
    goal_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.03, 0.03, 0.03),
            ),
        },
    )

    robot_grasp_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/robot_grasp_markers",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.02, 0.02, 0.02),
            ),
        },
    )

    # Sim2Real: observation noise
    joint_pos_noise_std: float = 0.01      # ~0.5° encoder noise
    joint_vel_noise_std: float = 0.05      # velocity estimation noise
    position_noise_std: float = 0.005      # 5mm tracking noise


class PoseOrientationGripperRobotV1(DirectRLEnv):

    cfg: PoseOrientationGripperRobotV1Cfg

    def __init__(self, cfg: PoseOrientationGripperRobotV1Cfg, render_mode: str | None = None, **kwargs):
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)
        self._robot_grasp_markers = VisualizationMarkers(cfg.robot_grasp_marker)

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs
        self.env_origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32) + ENV_ORIGIN_OFFSET.to(device=self.device, dtype=torch.float32)

        # Joint bookkeeping (exclude mimicked right finger)
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self._left_finger_joint_idx = self._robot.find_joints("left_finger_joint")[0][0]
        self._right_finger_joint_idx = self._robot.find_joints("right_finger_joint")[0][0]
        self.control_joint_indices = torch.tensor(
            [i for i in range(self._robot.num_joints) if i != self._right_finger_joint_idx],
            device=self.device,
            dtype=torch.long,
        )
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._left_finger_joint_idx] = 0.5
        self.robot_dof_speed_scales[self._right_finger_joint_idx] = 0.5

        self.robot_dof_targets = self._robot.data.joint_pos.clone()

        # Link indices
        self.hand_link_idx = self._robot.find_bodies("wrist_3_link")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("left_finger_link")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("right_finger_link")[0][0]

        stage = get_current_stage()
        # Base pose (env-local) to place goals above the table
        robot_base_pos_orient = self._get_env_local_pose(
            self.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/ur5e/base_link")),
            self.device,
        )
        self.robot_base_pos = robot_base_pos_orient[:3].to(device=self.device)

        # Pre-compute local grasp offsets from USD (finger tips + TCP midpoint)
        hand_pose = self._get_env_local_pose(
            self.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/ur5e/wrist_3_link")),
            self.device,
        )
        lfinger_pose = self._get_env_local_pose(
            self.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/left_finger_link")),
            self.device,
        )
        rfinger_pose = self._get_env_local_pose(
            self.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/right_finger_link")),
            self.device,
        )

        lfinger_inv_rot, lfinger_inv_pos = tf_inverse(lfinger_pose[3:7], lfinger_pose[0:3])
        lfinger_grasp_rot, lfinger_grasp_pos = tf_combine(
            lfinger_inv_rot, lfinger_inv_pos, lfinger_pose[3:7], lfinger_pose[0:3]
        )
        lfinger_grasp_pos += torch.tensor([0.015, 0.03, 0.035], device=self.device)

        rfinger_inv_rot, rfinger_inv_pos = tf_inverse(rfinger_pose[3:7], rfinger_pose[0:3])
        rfinger_grasp_rot, rfinger_grasp_pos = tf_combine(
            rfinger_inv_rot, rfinger_inv_pos, rfinger_pose[3:7], rfinger_pose[0:3]
        )
        rfinger_grasp_pos += torch.tensor([0.015, 0.03, 0.035], device=self.device)

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        g_robot_local_grasp_rot, g_robot_local_grasp_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        g_robot_local_grasp_pos += torch.tensor([0.012, 0.03, 0.035], device=self.device)

        # Cache local grasp offsets for all envs
        self.lfinger_local_grasp_rot = lfinger_grasp_rot.repeat((self.num_envs, 1))
        self.lfinger_local_grasp_pos = lfinger_grasp_pos.repeat((self.num_envs, 1))
        self.rfinger_local_grasp_rot = rfinger_grasp_rot.repeat((self.num_envs, 1))
        self.rfinger_local_grasp_pos = rfinger_grasp_pos.repeat((self.num_envs, 1))
        self.g_robot_local_grasp_rot = g_robot_local_grasp_rot.repeat((self.num_envs, 1))
        self.g_robot_local_grasp_pos = g_robot_local_grasp_pos.repeat((self.num_envs, 1))

        # Alignment axes (x->x, z->z)
        self.gripper_align_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.goal_align_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.goal_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        # World grasp pose buffers
        self.w_robot_lfinger_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.w_robot_lfinger_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.w_robot_rfinger_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.w_robot_rfinger_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.w_robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.w_robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Goal buffers (TCP midpoint target)
        self.goal_pos_local = torch.zeros((self._num_envs, 3), device=self.device, dtype=torch.float32)
        self.goal_quat = torch.zeros((self._num_envs, 4), device=self.device, dtype=torch.float32)

        self.actions = torch.zeros((self._num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self.smoothed_actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)  # NEW: for jerk penalty

        # Initial goal sampling and grasp pose computation
        self._sample_goal()
        self._compute_intermediate_values()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.gripper_robot)
        self.scene.articulations["gripper_robot"] = self._robot

        self.cfg.table.func(
            "/World/envs/env_0/WoodworkingTable",
            self.cfg.table,
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        spawn_ground_plane(prim_path=self.cfg.terrain.prim_path, cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # create dummy initial marker
        init_pos = torch.zeros((1, 3), device=self.device)
        init_ori = torch.tensor([[0, 0, 0, 1]], device=self.device)
        self.goal_marker.visualize(init_pos, init_ori, marker_indices=torch.tensor([0]))

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device).clamp(-1.0, 1.0)
        
        # Exponential moving average smoothing
        alpha = self.cfg.action_smoothing
        self.smoothed_actions = alpha * actions + (1 - alpha) * self.smoothed_actions
        self.actions = self.smoothed_actions.clone()

        lower = self.robot_dof_lower_limits[self.control_joint_indices]
        upper = self.robot_dof_upper_limits[self.control_joint_indices]
        deltas = (
            self.robot_dof_speed_scales[self.control_joint_indices]
            * self.dt
            * self.cfg.dof_velocity_scale
            * self.actions
            * self.cfg.action_scale
        )
        new_targets = self.robot_dof_targets[:, self.control_joint_indices] + deltas
        self.robot_dof_targets[:, self.control_joint_indices] = torch.clamp(new_targets, lower, upper)

        # Keep the marker always at the goal pose (world frame)
        marker_idx = torch.zeros(self.goal_pos_local.shape[0], dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(self.goal_pos_local + self.env_origins, self.goal_quat, marker_indices=marker_idx)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)


    def _get_observations(self):  # type: ignore
        self._compute_intermediate_values()

        # Position scaling to [-1, 1]
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        # Add noise for sim2real robustness
        if self.cfg.joint_pos_noise_std > 0:
            dof_pos_scaled = dof_pos_scaled + torch.randn_like(dof_pos_scaled) * self.cfg.joint_pos_noise_std
        
        joint_vel = self._robot.data.joint_vel * self.cfg.dof_velocity_scale
        if self.cfg.joint_vel_noise_std > 0:
            joint_vel = joint_vel + torch.randn_like(joint_vel) * self.cfg.joint_vel_noise_std

        goal_pos_w = self.goal_pos_local + self.env_origins
        to_target = goal_pos_w - self.w_robot_grasp_pos
        
        if self.cfg.position_noise_std > 0:
            to_target = to_target + torch.randn_like(to_target) * self.cfg.position_noise_std

        obs = torch.cat(
            (
                dof_pos_scaled,
                joint_vel,
                to_target,
                self.goal_pos_local,
                self.actions,
            ),
            dim=-1,
        )
        
        # Visualize grasp marker for first envs
        marker_envs = min(10, self.num_envs)
        marker_idx = torch.arange(marker_envs, device=self.device, dtype=torch.int64)
        self._robot_grasp_markers.visualize(self.w_robot_grasp_pos[:marker_envs], self.w_robot_grasp_rot[:marker_envs], marker_indices=marker_idx)

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        if "log" not in self.extras:
            self.extras["log"] = {}

        goal_pos_w = self.goal_pos_local + self.env_origins
        d = torch.norm(goal_pos_w - self.w_robot_grasp_pos, p=2, dim=-1)
        std = 0.02
        dist_reward = self.cfg.dist_reward_scale * (1.0 - torch.tanh(d / std)) - self.cfg.dist_penalty_scale * d

        axis1 = tf_vector(self.w_robot_grasp_rot, self.gripper_align_axis)
        axis2 = tf_vector(self.goal_quat, self.goal_align_axis)
        axis3 = tf_vector(self.w_robot_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.goal_quat, self.goal_up_axis)

        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(self.num_envs, 1, 3), axis4.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        orientation_reward = 0.5 * (dot1 * dot1 + dot2 * dot2)

        action_penalty = torch.sum(self.actions**2, dim=-1)
        speed_penalty = torch.sum((self._robot.data.joint_vel[:, self.control_joint_indices])**2, dim=-1)
        
        # NEW: Jerk penalty (smoothness)
        jerk_penalty = torch.sum((self.actions - self.prev_actions)**2, dim=-1)
        self.prev_actions = self.actions.clone()

        rewards = (
            dist_reward
            + self.cfg.orientation_reward_scale * orientation_reward
            - self.cfg.action_penalty_scale * action_penalty
            - self.cfg.speed_penalty_scale * speed_penalty
            - self.cfg.jerk_penalty_scale * jerk_penalty  # NEW
        )

        self.extras["log"].update({
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(),
            "dist_reward": (dist_reward).mean(),
            "distance to block": d.mean(),
            "orientation_reward": (self.cfg.orientation_reward_scale * orientation_reward).mean(),
            "action_penalty": (-self.cfg.action_penalty_scale * action_penalty).mean(),
            "speed_penalty": (-self.cfg.speed_penalty_scale * speed_penalty).mean(),
            "jerk_penalty": (-self.cfg.jerk_penalty_scale * jerk_penalty).mean(),  # NEW
            "episode_length": (self.episode_length_buf.float() * self.dt).mean()
        })

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        goal_pos_w = self.goal_pos_local + self.env_origins
        d = torch.norm(goal_pos_w - self.w_robot_grasp_pos, p=2, dim=-1)

        axis1 = tf_vector(self.w_robot_grasp_rot, self.gripper_align_axis)
        axis2 = tf_vector(self.goal_quat, self.goal_align_axis)
        axis3 = tf_vector(self.w_robot_grasp_rot, self.gripper_up_axis)
        axis4 = tf_vector(self.goal_quat, self.goal_up_axis)

        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(self.num_envs, 1, 3), axis4.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        success = (d < self.cfg.success_dist_threshold) & (dot1 > self.cfg.success_ori_threshold) & (dot2 > self.cfg.success_ori_threshold)
        terminated = success
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore
        env_ids = env_ids.to(self.device, dtype=torch.long)

        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)  # type: ignore
        self._robot.set_joint_position_target(self.robot_dof_targets)

        self.actions[env_ids] = 0.0
        self.smoothed_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0  # NEW
        
        self._sample_goal(env_ids)
        self._compute_intermediate_values(env_ids)

        # Move marker to the goal
        marker_idx = torch.zeros(len(env_ids), dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(self.goal_pos_local[env_ids] + self.env_origins[env_ids], self.goal_quat[env_ids], marker_indices=marker_idx)

    def _sample_goal(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        num = env_ids.shape[0]
        offsets = torch.empty((num, 3), device=self.device)
        offsets[:, 0].uniform_(-TABLE_DEPTH/2, TABLE_DEPTH/2)
        offsets[:, 1].uniform_(-TABLE_WIDTH/2, TABLE_WIDTH/2)
        offsets[:, 2].uniform_(0.05, 0.5)

        self.goal_pos_local[env_ids] = offsets

        # Orientation: z-axis points down, with tilt up to horizontal and free yaw
        roll_offset = torch.empty((num,), device=self.device).uniform_(-torch.pi / 2, torch.pi / 2)
        pitch_offset = torch.empty((num,), device=self.device).uniform_(-torch.pi / 2, torch.pi / 2)
        yaw = torch.empty((num,), device=self.device).uniform_(-torch.pi, torch.pi)
        eulers = torch.stack((torch.pi + roll_offset, pitch_offset, yaw), dim=1)
        self.goal_quat[env_ids] = quat_from_euler_xyz(eulers[:, 0], eulers[:, 1], eulers[:, 2])


    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        lfinger_rot = self._robot.data.body_quat_w[env_ids, self.left_finger_link_idx]
        lfinger_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_link_idx]
        rfinger_rot = self._robot.data.body_quat_w[env_ids, self.right_finger_link_idx]
        rfinger_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]

        (
            self.w_robot_lfinger_grasp_rot[env_ids],
            self.w_robot_lfinger_grasp_pos[env_ids],
            self.w_robot_rfinger_grasp_rot[env_ids],
            self.w_robot_rfinger_grasp_pos[env_ids],
            self.w_robot_grasp_rot[env_ids],
            self.w_robot_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            lfinger_rot,
            lfinger_pos,
            rfinger_rot,
            rfinger_pos,
            self.lfinger_local_grasp_rot[env_ids],
            self.lfinger_local_grasp_pos[env_ids],
            self.rfinger_local_grasp_rot[env_ids],
            self.rfinger_local_grasp_pos[env_ids],
            hand_rot,
            hand_pos,
            self.g_robot_local_grasp_rot[env_ids],
            self.g_robot_local_grasp_pos[env_ids],
        )


    def _compute_grasp_transforms(
        self,
        lfinger_rot,
        lfinger_pos,
        rfinger_rot,
        rfinger_pos,
        lfinger_local_grasp_rot,
        lfinger_local_grasp_pos,
        rfinger_local_grasp_rot,
        rfinger_local_grasp_pos,
        hand_rot,
        hand_pos,
        ur5e_local_grasp_rot,
        ur5e_local_grasp_pos,
    ):
        w_lfinger_grasp_rot, w_lfinger_grasp_pos = tf_combine(
            lfinger_rot, lfinger_pos, lfinger_local_grasp_rot, lfinger_local_grasp_pos
        )
        w_rfinger_grasp_rot, w_rfinger_grasp_pos = tf_combine(
            rfinger_rot, rfinger_pos, rfinger_local_grasp_rot, rfinger_local_grasp_pos
        )
        w_ur5e_grasp_rot, w_ur5e_grasp_pos = tf_combine(
            hand_rot, hand_pos, ur5e_local_grasp_rot, ur5e_local_grasp_pos
        )

        return (
            w_lfinger_grasp_rot,
            w_lfinger_grasp_pos,
            w_rfinger_grasp_rot,
            w_rfinger_grasp_pos,
            w_ur5e_grasp_rot,
            w_ur5e_grasp_pos,
        )

    def print_tensor_values(self, env_num=0, interval_s: float = 2.0, force: bool = False, **kwargs):
        """Print tensor values on one line, extracting only numeric values."""
        
        dt = getattr(self, "dt", self.cfg.sim.dt * getattr(self.cfg, "decimation", 1))
        step_interval = max(1, int(round(interval_s / dt))) if interval_s is not None else 1

        if not hasattr(self, "_print_step_counter"):
            self._print_step_counter = 0
        self._print_step_counter += 1

        if not force and (self._print_step_counter % step_interval != 0):
            return

        values_str = "\n ".join([f"{name}={tensor[env_num].item():.4f}" for name, tensor in kwargs.items()])
        print(f"[DEBUG] env ({env_num}): \n {values_str}")

    @staticmethod
    def _get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
        """Compute pose in env-local coordinates"""
        world_transform = xformable.ComputeLocalToWorldTransform(0)
        world_pos = world_transform.ExtractTranslation()
        world_quat = world_transform.ExtractRotationQuat()

        px = world_pos[0] - env_pos[0]
        py = world_pos[1] - env_pos[1]
        pz = world_pos[2] - env_pos[2]
        qx = world_quat.imaginary[0]
        qy = world_quat.imaginary[1]
        qz = world_quat.imaginary[2]
        qw = world_quat.real

        return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float32)
