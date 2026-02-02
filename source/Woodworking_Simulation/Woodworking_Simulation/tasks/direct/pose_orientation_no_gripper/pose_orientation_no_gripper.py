# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Simple pose/orientation control for UR5e (6 joints, no gripper)."""


from __future__ import annotations
from pathlib import Path
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform


# Import shared configurations
from Woodworking_Simulation.common.robot_configs import (
    get_robot_cfg,
    get_table_cfg,
    get_terrain_cfg,
    get_goal_marker_cfg,
    get_origin_marker_cfg,
    get_camera_pole_cfg,
    setup_dome_light,
    RobotType,
    TABLE_DEPTH,
    TABLE_WIDTH,
    TABLE_HEIGHT,
    MAX_REACH,
    MAX_JOINT_VEL,
    JOINT_LIMITS,
    ENV_ORIGIN_OFFSET,
)


@configclass
class PoseOrientationNoGripper(DirectRLEnvCfg):
    episode_length_s = 8.0
    decimation = 2
    action_space = 6  # 6 joints UR5e
    observation_space = 19  # pos_error(3), quat_error(4), joint_pos(6), joint_vel(6)
    state_space = 0  # Not used in this task

    # Local simulation and scene configurations
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    
    # Robot configuration using shared factory function
    robot = get_robot_cfg(RobotType.NO_GRIPPER, "/World/envs/env_.*/ur5e")
    
    # Scene assets using shared configurations
    table = get_table_cfg()
    terrain = get_terrain_cfg()
    goal_marker = get_goal_marker_cfg()
    origin_marker = get_origin_marker_cfg()
    
    # Camera pole configuration (generalized)
    camera_pole_spawn_cfg = get_camera_pole_cfg()

    action_scale = 7.5
    dof_velocity_scale = 0.1
    reward_position = -0.2
    reward_orientation = -0.1
    reset_frac = 1
    action_penalty_scale = -0.05


class PoseOrientationNoGripperV0(DirectRLEnv):
    cfg: PoseOrientationNoGripper

    def __init__(self, cfg: PoseOrientationNoGripper, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation  # 1/60s = 60Hz policy
        
        # Cached constants on device - use shared origin offset
        self.env_origins = self.scene.env_origins + ENV_ORIGIN_OFFSET.to(device=self.device)
        
        self.ee_idx = self._robot.body_names.index("wrist_3_link")

        # Reset fraction for stochastic resets (not always the same environments)
        self.reset_frac = self.cfg.reset_frac
        self.num_envs_to_reset = int(self.num_envs * self.reset_frac)
        
        # Apply real robot joint limits (already as floats)
        self.dof_lower = torch.tensor(
            [JOINT_LIMITS[name][0] for name in self._robot.joint_names], device=self.device
        )
        self.dof_upper = torch.tensor(
            [JOINT_LIMITS[name][1] for name in self._robot.joint_names], device=self.device
        )
        self.dof_targets = self._robot.data.joint_pos.clone()
        
        # Cached tensors for _sample_goal
        self.robot_base_local = torch.tensor([-0.36, -0.54, 0.0], device=self.device)
        self.identity_quat = torch.tensor([1, 0, 0, 0], device=self.device)
        
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)
        self.origin_marker = VisualizationMarkers(cfg.origin_marker)
        self._sample_goal()
        print("Init of V0 complete.")
                


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.table.func("/World/envs/env_0/Table", self.cfg.table)

        # Camera poles: left and right positions
        self.camera_left_pole = self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraLeftPole", self.cfg.camera_pole_spawn_cfg,
            translation=(TABLE_DEPTH/2 + 0.365, TABLE_WIDTH/2 - 0.535, TABLE_HEIGHT + 0.37),
        )
        self.camera_right_pole = self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraRightPole", self.cfg.camera_pole_spawn_cfg,
            translation=(TABLE_DEPTH/2 - 0.365, TABLE_WIDTH/2 + 0.535, TABLE_HEIGHT + 0.37),
        )
        spawn_ground_plane(self.cfg.terrain.prim_path, GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        setup_dome_light(intensity=2000.0)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clamp(-1.0, 1.0)
        inc = self.dt * self.cfg.dof_velocity_scale * self.cfg.action_scale * actions
        self.dof_targets = torch.clamp(self.dof_targets + inc, self.dof_lower, self.dof_upper)
        # Update marker (world frame)
        marker_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(self.goal_pos + self.env_origins, self.goal_quat, marker_indices=marker_idx)
        self.origin_marker.visualize(self.env_origins, self.identity_quat.unsqueeze(0).expand(self.num_envs, -1), marker_indices=marker_idx)
    def _apply_action(self):
        self._robot.set_joint_position_target(self.dof_targets)

    def _get_observations(self):
        # EE position in local frame (table center = origin)
        ee_pos = self._robot.data.body_pos_w[:, self.ee_idx] - self.env_origins
        ee_quat = self._robot.data.body_quat_w[:, self.ee_idx]
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        
        # Normalized observations
        pos_error = (self.goal_pos - ee_pos) / MAX_REACH  # [-1, 1] approx
        quat_error = self._quat_diff(self.goal_quat, ee_quat)  # already unit norm
        joint_pos_norm = 2.0 * (joint_pos - self.dof_lower) / (self.dof_upper - self.dof_lower) - 1.0  # [-1, 1]
        joint_vel_norm = joint_vel / MAX_JOINT_VEL  # [-1, 1] approx
        
        obs = torch.cat([pos_error, quat_error, joint_pos_norm, joint_vel_norm], dim=1)
        return {"policy": obs}

    @staticmethod
    def _quat_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion difference (q1 * q2^-1)."""
        q2_inv = q2 * torch.tensor([1, -1, -1, -1], device=q2.device)
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2_inv[:, 0], q2_inv[:, 1], q2_inv[:, 2], q2_inv[:, 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=1)

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self._robot.data.body_pos_w[:, self.ee_idx] - self.env_origins
        ee_quat = self._robot.data.body_quat_w[:, self.ee_idx]
        pos_err = torch.norm(self.goal_pos - ee_pos, dim=1)
        ori_err = 1.0 - torch.abs(torch.sum(self.goal_quat * ee_quat, dim=1))
        action_penalty = torch.sum(self.actions **2, dim=-1)
        return self.cfg.reward_position * pos_err + self.cfg.reward_orientation * ori_err + self.cfg.action_penalty_scale * action_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        
        # Randomly select which environments to reset (shuffle every time)
        # This ensures diversity in joint states across environments during training
        if self.num_envs_to_reset > 0:
            # Shuffle env_ids and pick the first num_envs_to_reset
            shuffled_ids = env_ids[torch.randperm(len(env_ids), device=self.device)]
            envs_to_reset = shuffled_ids[:min(self.num_envs_to_reset, len(env_ids))]
            
            if len(envs_to_reset) > 0:
                joint_pos = self._robot.data.default_joint_pos[envs_to_reset] + sample_uniform(
                    -0.5, 0.5, (len(envs_to_reset), self._robot.num_joints), self.device
                )
                self.dof_targets[envs_to_reset] = joint_pos
                self._robot.data.joint_pos[envs_to_reset] = joint_pos
                self._robot.data.joint_vel[envs_to_reset] = 0.0
                self._robot.set_joint_position_target(self.dof_targets[envs_to_reset], env_ids=envs_to_reset)
        
        # Sample goals for ALL environments
        self._sample_goal(env_ids)

    def _sample_goal(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)
        
        # 360° around robot base (cylindrical sampling)
        # Random angle and radius for full 360° coverage
        angle = torch.empty(n, device=self.device).uniform_(-torch.pi, torch.pi)
        radius = torch.empty(n, device=self.device).uniform_(0.3, 0.75)  # within reach
        height = torch.empty(n, device=self.device).uniform_(0.1, 0.6)   # above table
        
        # Cylindrical to Cartesian (relative to robot base)
        self.goal_pos[env_ids, 0] = self.robot_base_local[0] + radius * torch.cos(angle)
        self.goal_pos[env_ids, 1] = self.robot_base_local[1] + radius * torch.sin(angle)
        self.goal_pos[env_ids, 2] = height
        
        # Random orientation (full SO(3))
        q = torch.randn(n, 4, device=self.device)
        self.goal_quat[env_ids] = q / q.norm(dim=1, keepdim=True)




@configclass
class PoseOrientationNoGripperV1Cfg(PoseOrientationNoGripper):

    robot = get_robot_cfg(RobotType.GRIPPER_TCP_NO_ACTUATION, "/World/envs/env_.*/ur5e") # dans la config c'est le robot sans gripper
    
    # Gains for impedance control (per joint)
    kp = [120.0, 160.0, 160.0, 60.0, 40.0, 20.0]  # Stiffness
    kd = [22.0, 25.0, 25.0, 20.0, 18.0, 15.0]  # augmenté pour wrist

class PoseOrientationNoGripperV1(PoseOrientationNoGripperV0):
    cfg: PoseOrientationNoGripperV1Cfg

    def __init__(self, cfg: PoseOrientationNoGripperV1Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Pre-compute gain tensors (6 joints)
        self.kp_tensor = torch.tensor(self.cfg.kp, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        self.kd_tensor = torch.tensor(self.cfg.kd, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        
        # # Debug counter
        # self.debug_step = 0
        # print(f"[V1 Init] kp: {self.cfg.kp}")
        # print(f"[V1 Init] kd: {self.cfg.kd}")
        # print(f"[V1 Init] robot num_joints: {self._robot.num_joints}")
        # print(f"[V1 Init] dof_targets shape: {self.dof_targets.shape}")

    def _pre_physics_step(self, actions: torch.Tensor):
        # Convert actions to desired joint position increments
        actions = actions.clamp(-1.0, 1.0)
        inc = self.dt * self.cfg.dof_velocity_scale * self.cfg.action_scale * actions
        self.dof_targets = torch.clamp(self.dof_targets + inc, self.dof_lower, self.dof_upper)
        
        # Current joint states
        current_pos = self._robot.data.joint_pos
        current_vel = self._robot.data.joint_vel
        
        # Position error
        pos_error = self.dof_targets - current_pos
        
        # Impedance control: τ = Kp * (q_des - q) - Kd * q_dot
        self.joint_torques = self.kp_tensor * pos_error - self.kd_tensor * current_vel
        
        # Debug logging (every 60 steps ≈ 1 second)
        # self.debug_step += 1
        # if self.debug_step % 60 == 0:
        #     env_idx = 0
        #     print(f"\n[Step {self.debug_step}] === DEBUG V1 ===")
        #     print(f"  actions[0]:       {actions[env_idx].cpu().numpy().round(3)}")
        #     print(f"  dof_targets[0]:   {self.dof_targets[env_idx].cpu().numpy().round(3)}")
        #     print(f"  current_pos[0]:   {current_pos[env_idx].cpu().numpy().round(3)}")
        #     print(f"  current_vel[0]:   {current_vel[env_idx].cpu().numpy().round(3)}")
        #     print(f"  pos_error[0]:     {pos_error[env_idx].cpu().numpy().round(3)}")
        #     print(f"  joint_torques[0]: {self.joint_torques[env_idx].cpu().numpy().round(2)}")
        #     print(f"  torque min/max:   {self.joint_torques.min().item():.2f} / {self.joint_torques.max().item():.2f}")
        
        # Update markers
        marker_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(self.goal_pos + self.env_origins, self.goal_quat, marker_indices=marker_idx)
        self.origin_marker.visualize(self.env_origins, self.identity_quat.unsqueeze(0).expand(self.num_envs, -1), marker_indices=marker_idx)

    def _apply_action(self):
        # Apply computed torques
        self._robot.set_joint_effort_target(self.joint_torques)
