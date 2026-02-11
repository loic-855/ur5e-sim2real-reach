# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from pathlib import Path

import torch
from isaacsim.core.utils.stage import get_current_stage #type: ignore
from pxr import UsdGeom


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.sim import PhysxCfg

# Import shared configurations
from Woodworking_Simulation.common.robot_configs import (
    get_robot_cfg,
    get_table_cfg,
    get_terrain_cfg,
    get_goal_marker_cfg,
    get_origin_marker_cfg,
    get_robot_grasp_marker_cfg, 
    get_camera_pole_cfg,
    setup_dome_light,
    RobotType,
    TABLE_DEPTH,
    TABLE_WIDTH,
    TABLE_HEIGHT,
    MOUNT_HEIGHT,
    MAX_REACH,
    MAX_JOINT_VEL,
    JOINT_LIMITS,
    ENV_ORIGIN_OFFSET,
)

"""
The script implemented a pose and orientation control task with the gripper arm.
The architecture is a centralized policy controlling the arm.
The controller uses the joint space to command the arm.
"""

@configclass
class TestCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    decimation = 2
    # space definition
    action_space = 6
    observation_space = 36 #includes EE pose, orientation quat, EE lin and ang vel, goal pos and quat, joint pos and vel.
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4092, env_spacing=3.0, replicate_physics=True
    )
    # Robot configuration using shared factory function
    robot = get_robot_cfg(RobotType.NO_GRIPPER, "/World/envs/env_.*/ur5e")
    
    # Scene assets using shared configurations
    table = get_table_cfg()
    terrain = get_terrain_cfg()
    goal_marker = get_goal_marker_cfg()
    origin_marker = get_origin_marker_cfg()
    ee_marker = get_robot_grasp_marker_cfg()

    # Frame transformer to compute TCP pose relative to table center
    frame_transformer = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/ur5e/base_link",
        #prim_path="/World/envs/env_.*/Table/woodworking_table",
        #the offset points to the table center.
        source_frame_offset=OffsetCfg(
            pos=(- (TABLE_WIDTH / 2 - 0.08), TABLE_DEPTH / 2 - 0.08, - MOUNT_HEIGHT),
            rot=(0.7071, 0.0, 0.0, 0.7071),    
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ur5e/wrist_3_link",

                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.15),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
        ],
    )
    
    # Camera pole configuration (generalized)
    camera_pole_spawn_cfg = get_camera_pole_cfg()
    
    action_scale = 7.5
    dof_velocity_scale = 0.1

    #reward scale from UR10tuto
    ee_position_tracking = -0.2
    ee_orientation_tracking = -0.1
    action_penalty = -0.0001


class TestV0(DirectRLEnv):
    cfg: TestCfg

    def __init__(self, cfg: TestCfg, render_mode: str | None = None, **kwargs):
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs
        # include global origin offset used elsewhere
        self.env_origins = (self.scene.env_origins + ENV_ORIGIN_OFFSET.to(device=self.device)).to(device=self.device, dtype=torch.float32)

        # frame transformer (created in _setup_scene during super().__init__)
        self._frame_transformer = self.scene.sensors.get("frame_transformer")
        if self._frame_transformer is None:
            raise RuntimeError("Frame transformer not found in scene.sensors")

        self._ee_frame_idx = self._frame_transformer.data.target_frame_names.index("ee_tcp")

        # robot and joint limits / targets
        print("Available body names:", self._robot.body_names)
        # body index for velocities (wrist_3_link used as EE body in frame transformer)
        self.ee_body_idx = self._robot.body_names.index("wrist_3_link")

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = self._robot.data.joint_pos.clone()
        self.robot_default_dof_pos = self.robot_dof_targets.clone()

        stage = get_current_stage()
        robot_base_pos_orient = self._get_env_local_pose(
            self.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e/base_link")),
            self.device,
        )
        self.robot_base_pos = robot_base_pos_orient[:3].to(device=self.device)

        # goals are expressed in the "source" (table-relative / env-local) frame
        self.goal_pos_local = torch.zeros((self._num_envs, 3), device=self.device, dtype=torch.float32)
        self.goal_quat = torch.zeros((self._num_envs, 4), device=self.device, dtype=torch.float32)

        self.actions = torch.zeros((self._num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)

        self._sample_goal()

        # Instanz erstellen
        physx_cfg = PhysxCfg()

        # Alle Standardwerte anzeigen
        print(physx_cfg)

        # Gezielte Abfrage einzelner Werte
        print(f"Solver Type: {physx_cfg.solver_type}") # Default: 1 (TGS)
        print(f"External Forces: {physx_cfg.enable_external_forces_every_iteration}")

    def _setup_scene(self):
        # create robot articulation from config
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # frame transformer sensor (taken from pose_orientation_no_gripper)
        self._frame_transformer = FrameTransformer(self.cfg.frame_transformer)
        self.scene.sensors["frame_transformer"] = self._frame_transformer

        self.cfg.table.func(
            "/World/envs/env_0/Table",
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

        #create dummy initial marker
        init_pos = torch.zeros((1, 3), device=self.device)
        init_ori = torch.tensor([[0, 0, 0, 1]], device=self.device)
        self.goal_marker.visualize(init_pos, init_ori, marker_indices=torch.tensor([0]))

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        self.actions = actions.clone().clamp(-1.0, 1.0)

        increments = (
            self.robot_dof_speed_scales.unsqueeze(0)
            * self.dt
            * self.cfg.dof_velocity_scale
            * self.actions
            * self.cfg.action_scale
        )
        targets = self.robot_dof_targets + increments
        self.robot_dof_targets[:] = torch.clamp(
            targets,
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
        )
         # Keep the marker always at the goal pose
        marker_idx = torch.zeros(self.goal_pos_local.shape[0], dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(self.goal_pos_local + self.env_origins, self.goal_quat, marker_indices=marker_idx)
        

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self): # type: ignore
        # Use frame transformer for EE pose in source (local) and world frames
        frame_data = self._frame_transformer.data
        ee_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        ee_pos_w = frame_data.target_pos_w[:, self._ee_frame_idx, :]
        ee_quat_w = frame_data.target_quat_w[:, self._ee_frame_idx, :]

        # velocities from robot articulation (world frame)
        ee_lin_vel = self._robot.data.body_lin_vel_w[:, self.ee_body_idx]
        ee_ang_vel = self._robot.data.body_ang_vel_w[:, self.ee_body_idx]

        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        # observations: keep positional values in source/local frame (consistent with goal_pos_local)
        obs = torch.cat(
            [
                ee_pos_source,
                ee_quat_w,
                ee_lin_vel,
                ee_ang_vel,
                joint_pos,
                joint_vel,
                self.goal_pos_local,
                self.goal_quat,
            ],
            dim=1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Use frame transformer data so positions are in the same "source" frame as goals
        frame_data = self._frame_transformer.data
        ee_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        ee_quat_w = frame_data.target_quat_w[:, self._ee_frame_idx, :]

        position_error = torch.norm(self.goal_pos_local - ee_pos_source, dim=1)
        quat_dot = torch.sum(self.goal_quat * ee_quat_w, dim=1)
        orientation_error = 1.0 - torch.abs(quat_dot)
        action_cost = torch.sum(self.actions * self.actions, dim=1)

        reward = (
            self.cfg.ee_position_tracking * position_error
            + self.cfg.ee_orientation_tracking * orientation_error
            + self.cfg.action_penalty * action_cost
        )
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update({
            "action_penalty": (self.cfg.action_penalty * action_cost).mean(),
            "ori_reward": (self.cfg.ee_orientation_tracking * orientation_error).mean(),
            "pos_reward": (self.cfg.ee_position_tracking * position_error).mean(),
            "rewards_mean": reward.mean(),
            "rewards_std": reward.std(),
        })
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor): # type: ignore
        super()._reset_idx(env_ids) # type: ignore
        env_ids = env_ids.to(self.device, dtype=torch.long)
         # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.data.joint_pos[env_ids] = joint_pos
        self._robot.data.joint_vel[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        self._robot.set_joint_position_target(self.robot_dof_targets)
        self._sample_goal(env_ids)

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
        offsets[:, 0].uniform_(-0.08, 0.72)
        offsets[:, 1].uniform_(-0.08, 1.12)
        offsets[:, 2].uniform_(0.0, 1.0)

        self.goal_pos_local[env_ids] = self.robot_base_pos.to(device=self.device) + offsets
        delta_quat = torch.randn(num, 4, device=self.device)
        delta_quat = delta_quat / torch.norm(delta_quat, dim=1, keepdim=True)



        self.goal_quat[env_ids] = delta_quat


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


