# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pose-and-orientation reaching task for a UR5e arm (sim-to-real variant).

Observations (26-dim):
    ee_pos_source (3), ee_quat_source (4), joint_pos (6), joint_vel (6),
    goal_pos_source (3), goal_quat_source (4).

Actions (6-dim): joint-position increments.

Compared to the *test* task this variant:
  * removes EE linear / angular velocity from the observation vector,
  * adds a ``debug_visualization`` toggle (default **off**) for markers and
    FrameTransformer source-frame debug,
  * includes domain-randomisation (V1) and contact sensing.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

# Shared project helpers
from Woodworking_Simulation.common.robot_configs import (
    ENV_ORIGIN_OFFSET,
    MOUNT_HEIGHT,
    TABLE_DEPTH,
    TABLE_HEIGHT,
    TABLE_WIDTH,
    RobotType,
    get_camera_pole_cfg,
    get_goal_marker_cfg,
    get_origin_marker_cfg,
    get_robot_cfg,
    get_robot_grasp_marker_cfg,
    get_table_cfg,
    get_terrain_cfg,
    setup_dome_light,
)
from Woodworking_Simulation.common.domain_randomization import (
    ActionBuffer,
    ActuatorRandomizer,
    DomainRandomizationCfg,
    ObservationBuffer,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class PoseOrientationSim2RealCfg(DirectRLEnvCfg):
    """Base configuration (used by V0)."""

    # env timing
    episode_length_s = 8.3333
    decimation = 2

    # spaces  (no EE velocities → 26 obs instead of 32)
    action_space = 6
    observation_space = 26  # ee_pos(3) + ee_quat(4) + jpos(6) + jvel(6) + goal_pos(3) + goal_quat(4)
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

    # robot
    robot = get_robot_cfg(RobotType.NO_GRIPPER, "/World/envs/env_.*/ur5e")

    # scene assets
    table = get_table_cfg()
    terrain = get_terrain_cfg()
    goal_marker = get_goal_marker_cfg()
    origin_marker = get_origin_marker_cfg()
    ee_marker = get_robot_grasp_marker_cfg()

    # frame transformer – TCP pose relative to table centre
    frame_transformer = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/ur5e/base_link",
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

    # camera poles
    camera_pole_spawn_cfg = get_camera_pole_cfg()

    # action scaling
    action_scale = 7.0
    dof_velocity_scale = 0.1

    # reward weights
    ee_position_penalty = -0.25
    ee_position_reward = 0.3
    tanh_scaling = 0.2
    ee_orientation_penalty = -0.15
    action_penalty = -0.0008
    action_rate_penalty = -0.005

    # contact penalties (unused in V0, active in V1)
    contact_penalty_scale = -0.01
    contact_force_threshold_penalty = 5.0
    contact_force_threshold_done = 10.0

    # --- Debug / visualisation toggle (set True to see markers) -----------
    debug_visualization: bool = True


# ---------------------------------------------------------------------------
# V0 – base environment
# ---------------------------------------------------------------------------


class PoseOrientationSim2RealV0(DirectRLEnv):
    """Pose + orientation reaching – no domain randomisation, no contact sensor."""

    cfg: PoseOrientationSim2RealCfg

    def __init__(self, cfg: PoseOrientationSim2RealCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs

        # env origins with global offset
        self.env_origins = (
            self.scene.env_origins + ENV_ORIGIN_OFFSET.to(device=self.device)
        ).to(device=self.device, dtype=torch.float32)

        # frame transformer
        self._frame_transformer = self.scene.sensors["frame_transformer"]
        self._ee_frame_idx = self._frame_transformer.data.target_frame_names.index("ee_tcp")

        # joint limits & speed scales
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        for i, name in enumerate(self._robot.joint_names):
            if name in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"):
                self.robot_dof_speed_scales[i] = 0.8
            elif name in ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
                self.robot_dof_speed_scales[i] = 1.5

        self.robot_dof_targets = self._robot.data.joint_pos.clone()

        # goal tensors (source frame = table-relative)
        self.robot_base_local = torch.tensor([-0.52, 0.32, 0.0], device=self.device)
        self.goal_pos_source = torch.zeros((self._num_envs, 3), device=self.device, dtype=torch.float32)
        self.goal_quat_source = torch.zeros((self._num_envs, 4), device=self.device, dtype=torch.float32)

        # action buffers
        self.actions = torch.zeros((self._num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros_like(self.actions)

        # visualisation markers (only used when debug_visualization is True)
        if self.cfg.debug_visualization:
            self.goal_marker = VisualizationMarkers(cfg.goal_marker)
            self.origin_marker = VisualizationMarkers(cfg.origin_marker)
            self.ee_marker = VisualizationMarkers(cfg.ee_marker)
            self.source_marker = VisualizationMarkers(cfg.origin_marker)

        self._sample_goal()

    # -- Scene setup --------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._frame_transformer = FrameTransformer(self.cfg.frame_transformer)
        self.scene.sensors["frame_transformer"] = self._frame_transformer

        self.cfg.table.func(
            "/World/envs/env_0/Table",
            self.cfg.table,
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )

        self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraLeftPole",
            self.cfg.camera_pole_spawn_cfg,
            translation=(-TABLE_WIDTH + 0.07, 0.05, TABLE_HEIGHT + 0.37),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )
        self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraRightPole",
            self.cfg.camera_pole_spawn_cfg,
            translation=(-0.05, TABLE_DEPTH - 0.05, TABLE_HEIGHT + 0.37),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )

        spawn_ground_plane(self.cfg.terrain.prim_path, GroundPlaneCfg())
        setup_dome_light(intensity=2000.0)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    # -- Step logic ---------------------------------------------------------

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

    # -- Observations -------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        frame_data = self._frame_transformer.data
        ee_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        ee_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        obs = torch.cat(
            [
                ee_pos_source,       # 3
                ee_quat_source,      # 4
                joint_pos,           # 6
                joint_vel,           # 6
                self.goal_pos_source,   # 3
                self.goal_quat_source,  # 4
            ],
            dim=1,
        )
        return {"policy": obs}

    # -- Rewards ------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        frame_data = self._frame_transformer.data
        ee_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        ee_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        position_error = torch.norm(self.goal_pos_source - ee_pos_source, dim=1)
        quat_dot = torch.sum(self.goal_quat_source * ee_quat_source, dim=1)
        orientation_error = 1.0 - torch.abs(quat_dot)
        action_cost = torch.sum(self.actions ** 2, dim=1)
        action_rate_cost = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)

        # EE marker visualisation (debug only)
        if self.cfg.debug_visualization:
            self.ee_marker.visualize(
                frame_data.target_pos_w[:, self._ee_frame_idx, :],
                frame_data.target_quat_w[:, self._ee_frame_idx, :],
            )

        reward = (
            self.cfg.ee_position_penalty * position_error
            + self.cfg.ee_orientation_penalty * orientation_error
            + self.cfg.action_penalty * action_cost
            + self.cfg.action_rate_penalty * action_rate_cost
        )

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(
            {
                "action_penalty": (self.cfg.action_penalty * action_cost).mean(),
                "action_rate_penalty": (self.cfg.action_rate_penalty * action_rate_cost).mean(),
                "ori_error": orientation_error.mean(),
                "ori_reward": (self.cfg.ee_orientation_penalty * orientation_error).mean(),
                "pos_error": position_error.mean(),
                "pos_reward": (self.cfg.ee_position_penalty * position_error).mean(),
                "rewards_mean": reward.mean(),
                "rewards_std": reward.std(),
            }
        )
        return reward

    # -- Termination --------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # -- Reset --------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore
        env_ids = env_ids.to(self.device, dtype=torch.long)

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

        # debug markers
        if self.cfg.debug_visualization:
            marker_idx = torch.zeros(len(env_ids), dtype=torch.int64, device=self.device)
            self.goal_marker.visualize(
                self.goal_pos_source[env_ids] + self.env_origins[env_ids],
                self.goal_quat_source[env_ids],
                marker_indices=marker_idx,
            )
            self.origin_marker.visualize(self.env_origins, marker_indices=marker_idx)
            self._visualize_source_frame(env_ids, marker_idx)

    # -- Goal sampling ------------------------------------------------------

    def _sample_goal(self, env_ids: torch.Tensor | None = None):
        """Sample random goal poses within a cylindrical volume around the robot base (source frame)."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        angle = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        radius = torch.empty(len(env_ids), device=self.device).uniform_(0.3, 0.75)
        height = torch.empty(len(env_ids), device=self.device).uniform_(0.1, 0.6)

        self.goal_pos_source[env_ids, 0] = self.robot_base_local[0] + radius * torch.cos(angle)
        self.goal_pos_source[env_ids, 1] = self.robot_base_local[1] + radius * torch.sin(angle)
        self.goal_pos_source[env_ids, 2] = height

        delta_quat = torch.randn(len(env_ids), 4, device=self.device)
        delta_quat = delta_quat / torch.norm(delta_quat, dim=1, keepdim=True)
        self.goal_quat_source[env_ids] = delta_quat

    # -- Debug helpers ------------------------------------------------------

    def _visualize_source_frame(self, env_ids: torch.Tensor, marker_idx: torch.Tensor):
        """Visualise the FrameTransformer source-frame offset in world coordinates."""
        try:
            base_idx = self._robot.body_names.index("base_link")
            src_off = torch.tensor(
                self.cfg.frame_transformer.source_frame_offset.pos,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0).expand(len(env_ids), -1)
            src_rot = torch.tensor(
                self.cfg.frame_transformer.source_frame_offset.rot,
                device=self.device,
                dtype=torch.float32,
            )

            body_pos_w = self._robot.data.body_pos_w[env_ids, base_idx]
            body_quat_w = self._robot.data.body_quat_w[env_ids, base_idx]

            q_w = body_quat_w[:, 0:1]
            q_vec = body_quat_w[:, 1:4]
            rotated = src_off + 2.0 * torch.cross(
                q_vec, torch.cross(q_vec, src_off) + q_w * src_off, dim=1
            )
            src_world = body_pos_w + rotated
            self.source_marker.visualize(src_world, src_rot, marker_indices=marker_idx)
        except Exception:
            pass  # best-effort; don't break reset


# ---------------------------------------------------------------------------
# V1 configuration – adds contact sensor + domain randomisation
# ---------------------------------------------------------------------------


@configclass
class PoseOrientationSim2RealV1Cfg(PoseOrientationSim2RealCfg):
    """Extended config with contact sensor and domain-randomisation settings."""

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/ur5e/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )
    contact_debug_interval: int = 0

    episode_length_s = 5.0

    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()


# ---------------------------------------------------------------------------
# V1 – domain randomisation + contact penalty
# ---------------------------------------------------------------------------


class PoseOrientationSim2RealV1(PoseOrientationSim2RealV0):
    """V1 adds domain randomisation (action delay/noise, obs noise, actuator randomisation)
    and a contact-force penalty."""

    cfg: PoseOrientationSim2RealV1Cfg

    def __init__(self, cfg: PoseOrientationSim2RealV1Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._debug_step_count = 0

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

    # -- Scene (adds contact sensor) ----------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._frame_transformer = FrameTransformer(self.cfg.frame_transformer)
        self.scene.sensors["frame_transformer"] = self._frame_transformer

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.table.func(
            "/World/envs/env_0/Table",
            self.cfg.table,
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )
        self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraLeftPole",
            self.cfg.camera_pole_spawn_cfg,
            translation=(-TABLE_WIDTH + 0.07, 0.05, TABLE_HEIGHT + 0.37),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )
        self.cfg.camera_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraRightPole",
            self.cfg.camera_pole_spawn_cfg,
            translation=(-0.05, TABLE_DEPTH - 0.05, TABLE_HEIGHT + 0.37),
            orientation=(0.7071068, 0.0, 0.0, 0.7071068),
        )

        spawn_ground_plane(self.cfg.terrain.prim_path, GroundPlaneCfg())
        setup_dome_light(intensity=2000.0)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    # -- Domain-randomised overrides ----------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

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

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs_dict = super()._get_observations()
        obs_dict["policy"] = self._obs_buffer.append_and_get(obs_dict["policy"])
        return obs_dict

    def _reset_idx(self, env_ids: torch.Tensor):  # type: ignore
        super()._reset_idx(env_ids)
        env_ids = env_ids.to(self.device, dtype=torch.long)
        self._action_buffer.reset(env_ids)
        self._obs_buffer.reset(env_ids)
        self._actuator_randomizer.sample_and_apply(env_ids)

    # -- Contact metric -----------------------------------------------------

    def _compute_contact_metric(self) -> torch.Tensor:
        """Max normal contact force per environment (N)."""
        if self._contact_sensor is None:
            return torch.zeros(self._num_envs, device=self.device)

        data = self._contact_sensor.data
        if data is None or data.net_forces_w is None:
            return torch.zeros(self._num_envs, device=self.device)

        forces = data.net_forces_w  # (num_envs, num_bodies, 3)
        magnitudes = torch.norm(forces, dim=2)
        max_per_env, max_body_idx = magnitudes.max(dim=1)

        if self.cfg.contact_debug_interval > 0 and self._debug_step_count % self.cfg.contact_debug_interval == 0:
            body_names = self._contact_sensor.body_names
            env0_mags = magnitudes[0]
            parts = [f"{name}: {env0_mags[i].item():.2f}N" for i, name in enumerate(body_names)]
            top_body = body_names[max_body_idx[0].item()] if body_names else "?"
            print(
                f"[ContactDebug step={self._debug_step_count}] "
                f"env0 max={max_per_env[0].item():.2f}N ({top_body}) | "
                f"all_envs max={max_per_env.max().item():.2f}N mean={max_per_env.mean().item():.2f}N\n"
                f"  per-body: {', '.join(parts)}"
            )

        return max_per_env

    # -- Rewards (adds contact penalty) -------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        base = super()._get_rewards()
        self._debug_step_count += 1

        contact_forces = self._compute_contact_metric()
        excess = torch.relu(contact_forces - self.cfg.contact_force_threshold_penalty)
        contact_penalty = self.cfg.contact_penalty_scale * excess

        reward = base + contact_penalty

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(
            {
                "contact_force_max": contact_forces.max(),
                "contact_force_mean": contact_forces.mean(),
                "contact_penalty_mean": contact_penalty.mean(),
            }
        )
        return reward
