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
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg, ContactSensor, ContactSensorCfg
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
from Woodworking_Simulation.common.domain_randomization import (
    DomainRandomizationCfg,
    ActionBuffer,
    ObservationBuffer,
    ActuatorRandomizer,
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
    observation_space = 32 #includes EE pose, orientation quat, EE lin and ang vel, goal pos and quat, joint pos and vel.
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
            dt=1 / 120, 
            render_interval=decimation,
            physx=PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True),
        )

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
        pos=(-(TABLE_WIDTH / 2 - 0.08), TABLE_DEPTH / 2 - 0.08, -MOUNT_HEIGHT),
        rot=(0.0, 0.0, 0.0, 1.0), 
),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ur5e/wrist_3_link",

                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.15),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        ],
    )
    
    # Camera pole configuration (generalized)
    camera_pole_spawn_cfg = get_camera_pole_cfg()
    
    action_scale = 7.0
    dof_velocity_scale = 0.1

    #reward scale from UR10tuto
    ee_position_penalty = -0.25
    ee_position_reward = 0.3
    tanh_scaling = 0.2 # distance at which the tanh reward activate
    ee_orientation_penalty = -0.15
    action_penalty = -0.0008
    action_rate_penalty = -0.005
    # contact penalty defaults (unused by TestV0, used by TestV1)
    contact_penalty_scale = -0.01
    contact_force_threshold_penalty = 5.0
    contact_force_threshold_done = 10.0


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

        # body index for velocities (wrist_3_link used as EE body in frame transformer)
        self.ee_body_idx = self._robot.body_names.index("wrist_3_link")

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # Per-joint speed tuning: slow down shoulder/elbow, speed up wrist
        joint_names = self._robot.joint_names
        for i, name in enumerate(joint_names):
            if name in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"):
                self.robot_dof_speed_scales[i] = 0.8
            elif name in ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
                self.robot_dof_speed_scales[i] = 1.5

        self.robot_dof_targets = self._robot.data.joint_pos.clone()

        # goals are expressed in the source frame (table-relative)
        # use explicit names to avoid confusion
        self.robot_base_local = torch.tensor([-0.52, 0.32, 0.0], device=self.device)
        self.goal_pos_source = torch.zeros((self._num_envs, 3), device=self.device, dtype=torch.float32)
        self.goal_quat_source = torch.zeros((self._num_envs, 4), device=self.device, dtype=torch.float32)

        self.actions = torch.zeros((self._num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros_like(self.actions)

        self.goal_marker = VisualizationMarkers(cfg.goal_marker)
        self.origin_marker = VisualizationMarkers(cfg.origin_marker)
        self.ee_marker = VisualizationMarkers(cfg.ee_marker)
        # Marker to visualize the FrameTransformer source_frame_offset in world frame
        self.source_marker = VisualizationMarkers(cfg.origin_marker)

        self._sample_goal()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._frame_transformer = FrameTransformer(self.cfg.frame_transformer)
        self.scene.sensors["frame_transformer"] = self._frame_transformer

        self.cfg.table.func(
            "/World/envs/env_0/Table", self.cfg.table,
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )

        self.camera_left_pole = self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraLeftPole", self.cfg.camera_pole_spawn_cfg,
            translation= (- TABLE_WIDTH + 0.07, 0.05, TABLE_HEIGHT + 0.37),
            orientation= (0.7071068, 0.0, 0.0, 0.7071068)
        )
        self.camera_right_pole = self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraRightPole", self.cfg.camera_pole_spawn_cfg,
            translation=(- 0.05, TABLE_DEPTH - 0.05, TABLE_HEIGHT + 0.37),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068)
        )


        spawn_ground_plane(self.cfg.terrain.prim_path, GroundPlaneCfg())
        setup_dome_light(intensity=2000.0)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])


    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        self.prev_actions[:] = self.actions
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


    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)


    def _get_observations(self): # type: ignore
        # Use frame transformer for EE pose in source (local) and world frames
        frame_data = self._frame_transformer.data
        ee_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        ee_pos_w = frame_data.target_pos_w[:, self._ee_frame_idx, :]
        ee_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        # compute EE point velocities (accounts for FrameTransformer offset)
        ee_lin_vel, ee_ang_vel = self._compute_ee_point_velocity(
            frame_data, self._robot, self._ee_frame_idx, self.ee_body_idx
        )
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        # observations: keep positional and orientation values in source frame
        obs = torch.cat(
            [
                ee_pos_source,
                ee_quat_source,
                ee_lin_vel,
                ee_ang_vel,
                joint_pos,
                joint_vel,
                self.goal_pos_source,
                self.goal_quat_source,
            ],
            dim=1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Use frame transformer data so positions are in the same "source" frame as goals
        frame_data = self._frame_transformer.data
        ee_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        ee_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        # compare in source frame (positions and orientations)
        position_error = torch.norm(self.goal_pos_source - ee_pos_source, dim=1)
        position_error_tanh = 1 - torch.tanh(position_error/self.cfg.tanh_scaling)
        quat_dot = torch.sum(self.goal_quat_source * ee_quat_source, dim=1)
        orientation_error = 1.0 - torch.abs(quat_dot)
        action_cost = torch.sum(self.actions * self.actions, dim=1)
        action_rate_cost = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)

        self.ee_marker.visualize(frame_data.target_pos_w[:, self._ee_frame_idx, :], frame_data.target_quat_w[:, self._ee_frame_idx, :])

        reward = (
            self.cfg.ee_position_penalty * position_error
          #  + self.cfg.ee_position_reward * position_error_tanh
            + self.cfg.ee_orientation_penalty * orientation_error
            + self.cfg.action_penalty * action_cost
            + self.cfg.action_rate_penalty * action_rate_cost
        )
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update({
            "action_penalty": (self.cfg.action_penalty * action_cost).mean(),
            "action_rate_penalty": (self.cfg.action_rate_penalty * action_rate_cost).mean(),
            "ori_error": orientation_error.mean(),
            "ori_reward": (self.cfg.ee_orientation_penalty * orientation_error).mean(),
            "pos_error": position_error.mean(),
            "pos_reward": (self.cfg.ee_position_penalty * position_error).mean(),
          #  "pos_reward_tanh": (self.cfg.ee_position_reward * position_error_tanh).mean(),
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
        self.prev_actions[env_ids] = 0.0

        self._robot.set_joint_position_target(self.robot_dof_targets)
        self._sample_goal(env_ids)

        # Move marker to the goal (convert source->world for visualization)
        marker_idx = torch.zeros(len(env_ids), dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(self.goal_pos_source[env_ids] + self.env_origins[env_ids], self.goal_quat_source[env_ids], marker_indices=marker_idx)
        self.origin_marker.visualize(self.env_origins, marker_indices=marker_idx)
        # Visualize FrameTransformer source_frame_offset (anchor in base_link frame)
        try:
            base_idx = self._robot.body_names.index("base_link")
            # env-local offset (convert tuple to tensor)
            src_off = torch.tensor(self.cfg.frame_transformer.source_frame_offset.pos, device=self.device, dtype=torch.float32)
            src_off = src_off.unsqueeze(0).expand(len(env_ids), -1)
            src_rot = torch.tensor(self.cfg.frame_transformer.source_frame_offset.rot, device=self.device, dtype=torch.float32)
            # base link poses
            body_pos_w = self._robot.data.body_pos_w[env_ids, base_idx]
            body_quat_w = self._robot.data.body_quat_w[env_ids, base_idx]
            # rotate offset into world: v' = v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)
            q_w = body_quat_w[:, 0:1]
            q_vec = body_quat_w[:, 1:4]
            rotated = src_off + 2.0 * torch.cross(q_vec, torch.cross(q_vec, src_off) + q_w * src_off, dim=1)
            src_world = body_pos_w + rotated
            self.source_marker.visualize(src_world, src_rot, marker_indices=marker_idx)
        except Exception:
            # best-effort visualization; don't fail reset if something goes wrong
            pass

    def _sample_goal(self, env_ids: torch.Tensor | None = None):
        """
        Define goal positions and orientations for the end-effector within a cylindrical volume around the robot base.
        The goals are expressed in the source frame (table-relative).
        """

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        # 360° around robot base (cylindrical sampling)
        angle = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        radius = torch.empty(len(env_ids), device=self.device).uniform_(0.3, 0.75)  # within reach
        height = torch.empty(len(env_ids), device=self.device).uniform_(0.1, 0.6)   # above table

        # sample relative to robot base in source frame
        self.goal_pos_source[env_ids, 0] = self.robot_base_local[0] + radius * torch.cos(angle)
        self.goal_pos_source[env_ids, 1] = self.robot_base_local[1] + radius * torch.sin(angle)
        self.goal_pos_source[env_ids, 2] = height

        # sample random orientations in source frame (normalize)
        delta_quat = torch.randn(len(env_ids), 4, device=self.device)
        delta_quat = delta_quat / torch.norm(delta_quat, dim=1, keepdim=True)
        self.goal_quat_source[env_ids] = delta_quat


    @staticmethod
    def _compute_ee_point_velocity(frame_data: object, robot: Articulation, ee_frame_idx: int, ee_body_idx: int):
        """Compute linear and angular velocity of the EE TCP point (accounts for FrameTransformer offset).

        Returns velocities in world frame.

        Args:
            frame_data: the FrameTransformer data object (must provide target_pos_w)
            robot: the `Articulation` instance to read body positions/velocities from
            ee_frame_idx: index of the target frame in the frame_transformer
            ee_body_idx: index of the robot body corresponding to the frame's base

        Returns:
            ee_lin_vel, ee_ang_vel  (both tensors shaped (N,3) in world frame)
        """
        # world-space EE point position (includes offset)
        ee_pos_w = frame_data.target_pos_w[:, ee_frame_idx, :]

        # body origin position and velocities (wrist_3_link)
        body_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        body_lin_vel_w = robot.data.body_lin_vel_w[:, ee_body_idx]
        body_ang_vel_w = robot.data.body_ang_vel_w[:, ee_body_idx]

        # lever arm from body origin to EE point (world)
        r_world = ee_pos_w - body_pos_w

        # velocity at point: v_point = v_body + omega x r
        ee_lin_vel_w_point = body_lin_vel_w + torch.cross(body_ang_vel_w, r_world, dim=1)

        return ee_lin_vel_w_point, body_ang_vel_w


@configclass
class TestV1Cfg(TestCfg):
    # Contact sensor on all robot links.
    # net_forces_w reports the total normal contact force on each body (from ALL contacts).
    # force_matrix_w is only available when filter_prim_paths_expr is non-empty (one-to-many filtering).
    # For our use case (penalize ANY collision), net_forces_w is the correct field.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/ur5e/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False, 
        filter_prim_paths_expr=[],  # Empty = no force_matrix_w, but net_forces_w still works
    )
    # Debug: print contact info every N env steps (0 = disabled)
    contact_debug_interval: int = 0

    episode_length_s = 5.0

    # Domain randomisation (action delay/noise, observation noise, actuator randomisation)
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()


class TestV1(TestV0):
    cfg: TestV1Cfg

    def __init__(self, cfg: TestV1Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._debug_step_count = 0

        # --- Domain randomisation helpers (all tensors on self.device) ---
        dr = cfg.domain_randomization
        self._action_buffer = ActionBuffer(
            num_envs=self._num_envs,
            action_dim=self.cfg.action_space,
            cfg=dr,
            device=self.device,
        )
        self._obs_buffer = ObservationBuffer(
            num_envs=self._num_envs,
            obs_dim=self.cfg.observation_space,
            num_joints=self._robot.num_joints,
            cfg=dr,
            device=self.device,
        )
        self._actuator_randomizer = ActuatorRandomizer(
            robot=self._robot,
            cfg=dr,
            device=self.device,
        )
  

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._frame_transformer = FrameTransformer(self.cfg.frame_transformer)
        self.scene.sensors["frame_transformer"] = self._frame_transformer

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.table.func(
            "/World/envs/env_0/Table", self.cfg.table,
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )

        self.camera_left_pole = self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraLeftPole", self.cfg.camera_pole_spawn_cfg,
            translation= (- TABLE_WIDTH + 0.07, 0.05, TABLE_HEIGHT + 0.37),
            orientation= (0.7071068, 0.0, 0.0, 0.7071068)
        )
        self.camera_right_pole = self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraRightPole", self.cfg.camera_pole_spawn_cfg,
            translation=(- 0.05, TABLE_DEPTH - 0.05, TABLE_HEIGHT + 0.37),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068)
        )

        spawn_ground_plane(self.cfg.terrain.prim_path, GroundPlaneCfg())
        setup_dome_light(intensity=2000.0)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    # -- Domain-randomised overrides -----------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # Push through action buffer (applies delay, noise, packet-loss)
        effective_actions = self._action_buffer.push(self.actions)

        increments = (
            self.robot_dof_speed_scales.unsqueeze(0)
            * self.dt
            * self.cfg.dof_velocity_scale
            * effective_actions
            * self.cfg.action_scale
        )
        targets = self.robot_dof_targets + increments
        self.robot_dof_targets[:] = torch.clamp(
            targets,
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self):  # type: ignore
        obs_dict = super()._get_observations()
        # Apply observation delay + noise
        obs_dict["policy"] = self._obs_buffer.append_and_get(obs_dict["policy"])
        return obs_dict

    def _reset_idx(self, env_ids: torch.Tensor):  # type: ignore
        super()._reset_idx(env_ids)
        env_ids = env_ids.to(self.device, dtype=torch.long)
        # Reset DR buffers for these envs
        self._action_buffer.reset(env_ids)
        self._obs_buffer.reset(env_ids)
        # Randomise actuator dynamics (stiffness, damping, friction)
        self._actuator_randomizer.sample_and_apply(env_ids)

    def _compute_contact_metric(self) -> torch.Tensor:
        """Return max normal contact force per environment (N).

        Uses the contact sensor's ``net_forces_w`` which has shape (num_envs, num_bodies, 3).
        ``force_matrix_w`` is only available when ``filter_prim_paths_expr`` is non-empty.
        If data is not yet available, returns zeros.
        """
        if self._contact_sensor is None:
            return torch.zeros(self._num_envs, device=self.device)

        data = self._contact_sensor.data
        if data is None or data.net_forces_w is None:
            return torch.zeros(self._num_envs, device=self.device)

        # net_forces_w: (num_envs, num_bodies, 3)
        forces = data.net_forces_w
        # magnitude per body
        magnitudes = torch.norm(forces, dim=2)  # (num_envs, num_bodies)
        # max per env
        max_per_env, max_body_idx = magnitudes.max(dim=1)

        # --- Debug logging (only every N steps, env 0 only) ---
        if self.cfg.contact_debug_interval > 0 and self._debug_step_count % self.cfg.contact_debug_interval == 0:
            env0_mags = magnitudes[0]  # per-body forces for env 0
            body_names = self._contact_sensor.body_names
            parts = [f"{name}: {env0_mags[i].item():.2f}N" for i, name in enumerate(body_names)]
            top_body = body_names[max_body_idx[0].item()] if body_names else "?"
            print(
                f"[ContactDebug step={self._debug_step_count}] "
                f"env0 max={max_per_env[0].item():.2f}N ({top_body}) | "
                f"all_envs max={max_per_env.max().item():.2f}N mean={max_per_env.mean().item():.2f}N\n"
                f"  per-body: {', '.join(parts)}"
            )

        return max_per_env

    def _get_rewards(self) -> torch.Tensor:
        base = super()._get_rewards()
        self._debug_step_count += 1
        contact_forces = self._compute_contact_metric()
        # small penalty when contact force exceeds threshold (only amount above threshold)
        excess = torch.relu(contact_forces - self.cfg.contact_force_threshold_penalty)
        contact_penalty = self.cfg.contact_penalty_scale * excess
        #small penalty for action rate change

        
        reward = base + contact_penalty

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update({
            "contact_force_max": contact_forces.max(),
            "contact_force_mean": contact_forces.mean(),
            "contact_penalty_mean": contact_penalty.mean(),
        })
        return reward

    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     terminated, truncated = super()._get_dones()
    #     contact_forces = self._compute_contact_metric()
    #     terminated = terminated | (contact_forces > self.cfg.contact_force_threshold_done)
    #     return terminated, truncated
    
