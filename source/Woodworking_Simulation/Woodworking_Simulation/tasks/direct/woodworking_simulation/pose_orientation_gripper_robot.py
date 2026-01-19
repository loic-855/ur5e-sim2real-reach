# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from pathlib import Path

import torch
from isaacsim.core.utils.stage import get_current_stage
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
from isaaclab.utils.math import sample_uniform

#path constants
REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"

"""
The script implemented a pose and orientation control task with the gripper arm.
The architecture is a centralized policy controlling the arm.
The controller uses the joint space to command the arm.
"""

@configclass
class PoseOrientationGripperRobot(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    decimation = 2
    # space definition
    action_space = 8
    observation_space = 36 #includes EE pose, orientation quat, EE lin and ang vel, goal pos and quat, joint pos and vel.
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4092, env_spacing=3.0, replicate_physics=True
    )

    # robot
    gripper_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e_gripper_tcp",
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(USD_FILES_DIR / "ur5e_gripper_tcp.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
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
                joint_names_expr = [
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    ],
                damping=50, stiffness=700),
            "wrist_action": ImplicitActuatorCfg(    
                joint_names_expr = [
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                    ],
                    damping=30, stiffness=300),
            "gripper_action": ImplicitActuatorCfg(
                joint_names_expr = [
                    "left_finger_joint", "right_finger_joint",
                    ],
                damping=14, stiffness=80),
        }
    )
    # Table asset placement: Width = 1.2m, Depth = 0.8m, Height = 0.842m
    table = sim_utils.UsdFileCfg(
        usd_path=str(USD_FILES_DIR / "woodworking_table.usd")
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )

    #marker for debug
    goal_marker =  VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
            ),
        },
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    #reward scale from UR10tuto
    ee_position_tracking = -0.2
    ee_orientation_tracking = -0.1
    action_penalty = -0.0001


class PoseOrientationGripperRobotV0(DirectRLEnv):
    cfg: PoseOrientationGripperRobot

    def __init__(self, cfg: PoseOrientationGripperRobot, render_mode: str | None = None, **kwargs):
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)

        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs
        self.env_origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32)
        print("Available body names:", self._robot.body_names)
        self.ee_link_idx = self._robot.body_names.index("base_link_0")
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = self._robot.data.joint_pos.clone()
        self.robot_default_dof_pos = self.robot_dof_targets.clone()

        stage = get_current_stage()

        robot_base_pos_orient = self._get_env_local_pose(
            self.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp/ur5e/base_link")),
            self.device) # type: ignore
        self.robot_base_pos = robot_base_pos_orient[:3].to(device=self.device)
        self.goal_pos_local = torch.zeros((self._num_envs, 3), device=self.device, dtype=torch.float32)
        self.goal_quat = torch.zeros((self._num_envs, 4), device=self.device, dtype=torch.float32)

        self.actions = torch.zeros((self._num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32) # type: ignore
        
        self._sample_goal()

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
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_link_idx]
        ee_quat = self._robot.data.body_quat_w[:, self.ee_link_idx]
        ee_lin_vel = self._robot.data.body_lin_vel_w[:, self.ee_link_idx]
        ee_ang_vel = self._robot.data.body_ang_vel_w[:, self.ee_link_idx]
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        ee_pos_local = ee_pos_w - self.env_origins

        obs = torch.cat(
            [
                ee_pos_local,
                ee_quat,
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
        ee_pos_local = self._robot.data.body_pos_w[:, self.ee_link_idx] - self.env_origins
        ee_quat = self._robot.data.body_quat_w[:, self.ee_link_idx]

        position_error = torch.norm(self.goal_pos_local - ee_pos_local, dim=1)
        quat_dot = torch.sum(self.goal_quat * ee_quat, dim=1)
        orientation_error = 1.0 - torch.abs(quat_dot)
        action_cost = torch.sum(self.actions * self.actions, dim=1)

        reward = (
            self.cfg.ee_position_tracking * position_error
            + self.cfg.ee_orientation_tracking * orientation_error
            + self.cfg.action_penalty * action_cost
        )
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
        """
        yaw = (torch.rand(num, device=self.device)*torch.pi - torch.pi)
        cos = torch.cos(yaw * 0.5)
        sin = torch.sin(yaw * 0.5)
        delta_quat = torch.stack(
            (cos, torch.zeros_like(cos), torch.zeros_like(cos), sin),
            dim=1,
        )
        """ 
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


