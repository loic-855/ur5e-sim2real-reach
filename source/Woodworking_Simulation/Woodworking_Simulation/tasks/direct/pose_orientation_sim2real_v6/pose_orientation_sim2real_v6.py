# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pose-and-orientation reaching task for a UR5e arm (sim-to-real variant V6).

Same environment as V3 but designed to be used with an **LSTM policy**.
The recurrent architecture learns temporal context implicitly, making manual
observation stacking unnecessary.

Observations (24-dim, same as V3):
    pos_error (3), ori_error (3), joint_pos (6), joint_vel (6),
    tcp_linear_vel (3), tcp_angular_vel (3).

Actions (12-dim, same as V3):
    position increments (6) + velocity feedforward (6).

The agent config uses RslRlPpoActorCriticRecurrentCfg with LSTM.
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
from isaaclab.utils.math import sample_uniform, quat_error_magnitude, quat_box_minus, quat_apply_yaw, quat_inv

# Shared project helpers
from Woodworking_Simulation.common.robot_configs import (
    ENV_ORIGIN_OFFSET,
    MOUNT_HEIGHT,
    TABLE_DEPTH,
    TABLE_HEIGHT,
    TABLE_WIDTH,
    JOINT_LIMITS,
    MAX_REACH,
    MAX_JOINT_VEL,
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
BASE_OFFSET_LOCAL = (-(TABLE_WIDTH / 2 - 0.08), TABLE_DEPTH / 2 - 0.08, -MOUNT_HEIGHT)
BASE_ROTATION_LOCAL = (0.0, 0.0, 0.0, 1.0)

TCP_OFFSET_LOCAL = (0.0, 0.0, 0.14)
TCP_ROTATION_LOCAL = (0.0, 0.0, 0.0, 1.0)

from Woodworking_Simulation.common.domain_randomization import (
    ActionBuffer,
    ActuatorRandomizer,
    DomainRandomizationV4Cfg,
    MassComRandomizer,
    ObservationBuffer,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class PoseOrientationSim2RealV6Cfg(DirectRLEnvCfg):
    """Configuration for V6 – same as V3 but with LSTM policy."""

    # env timing
    episode_length_s = 10.0
    decimation = 2

    # spaces – same as V3
    action_space = 12
    observation_space = 24
    state_space = 0

    # simulation
    try:
        sim: SimulationCfg = SimulationCfg(
            dt=1 / 120,
            render_interval=decimation,
            physx=PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True),
        )
    except:
        print("This version of Isaac Sim may not support the 'enable_external_forces_every_iteration' option.")
        sim: SimulationCfg = SimulationCfg(
            dt=1 / 120,
            render_interval=decimation,
            physx=PhysxCfg(solver_type=1),
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
            pos=BASE_OFFSET_LOCAL,
            rot=BASE_ROTATION_LOCAL,
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ur5e/wrist_3_link",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=TCP_OFFSET_LOCAL,
                    rot=TCP_ROTATION_LOCAL,
                ),
            )
        ],
    )
    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/ur5e/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )

    # camera poles
    camera_pole_spawn_cfg = get_camera_pole_cfg()

    # action and coef scaling
    action_scale = 3.0
    velocity_scale = 1.0
    env_reset = 1.0
    progressive_reset: bool = False
    progressive_reset_steps: int = 25000
    position_exp_scale = 0.2
    stability_reward_scale = 0.3
    orientation_exp_scale = 0.4
    curric = [5000, 10000, 15000]
    curric_active = False

    # reward weights
    ee_position_penalty = -0.30
    ee_position_reward = 1.2
    ee_orientation_penalty = -0.20
    ee_orientation_reward = 0.60

    # penalty weights
    action_penalty_scale = -0.001
    velocity_action_penalty_scale = -0.001
    velocity_penalty_scale = -0.001
    contact_penalty_scale = -0.01
    contact_force_threshold_penalty = 5.0
    joint_limit_penalty_scale = -0.01

    # TCP velocity normalization (m/s)
    tcp_max_speed = 2.0

    # Bonus reward
    pos_threshold = 0.02
    rot_threshold = 0.1
    required_frames = 60
    goal_success_bonus = 15.0

    # --- Debug ---
    debug = False
    contact_debug_interval: int = 0
    norm_obs: bool = True

    # --- Domain Randomization ---
    domain_rand: DomainRandomizationV4Cfg = DomainRandomizationV4Cfg()


# ---------------------------------------------------------------------------
# V6
# ---------------------------------------------------------------------------


class PoseOrientationSim2RealV6(DirectRLEnv):
    """Pose + orientation reaching with velocity feedforward (LSTM policy)."""

    cfg: PoseOrientationSim2RealV6Cfg

    def __init__(
        self,
        cfg: PoseOrientationSim2RealV6Cfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs
        self.reward_buf = torch.zeros(self._num_envs, device=self.device, dtype=torch.float32)
        self._debug_step_count = 0
        self.c_idx = 0

        # env origins with global offset
        self.env_origins = (
            self.scene.env_origins + ENV_ORIGIN_OFFSET.to(device=self.device)
        ).to(device=self.device, dtype=torch.float32)

        # frame transformer
        self._frame_transformer = self.scene.sensors["frame_transformer"]
        self._ee_frame_idx = self._frame_transformer.data.target_frame_names.index(
            "ee_tcp"
        )

        # joint limits & speed scales
        limits = list(JOINT_LIMITS.values())[:6]
        lower = torch.tensor(
            [v[0] for v in limits], device=self.device, dtype=torch.float32
        )
        upper = torch.tensor(
            [v[1] for v in limits], device=self.device, dtype=torch.float32
        )

        self.robot_dof_lower_limits = lower
        self.robot_dof_upper_limits = upper
        self.joint_pos_norm = torch.zeros((self._num_envs, 6), device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = self._robot.data.joint_pos.clone()

        # Velocity feedforward targets (rad/s)
        self.robot_speed_targets = torch.zeros(
            (self._num_envs, 6), device=self.device, dtype=torch.float32
        )

        # goal tensors (source frame = table-relative)
        self.robot_base_local = torch.tensor([-0.52, 0.32, 0.0], device=self.device)
        self.goal_pos_source = torch.zeros(
            (self._num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.goal_quat_source = torch.zeros(
            (self._num_envs, 4), device=self.device, dtype=torch.float32
        )

        # buffers – full 12-dim action vector
        self.actions = torch.zeros(
            (self._num_envs, self.cfg.action_space),
            device=self.device,
            dtype=torch.float32,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.success_frames_count = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.goal_steps_elapsed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.goal_max_steps = int(5.0 / self.dt)

        # visualisation markers
        if self.cfg.debug:
            self.goal_marker = VisualizationMarkers(cfg.goal_marker)
            self.origin_marker = VisualizationMarkers(cfg.origin_marker)
            self.ee_marker = VisualizationMarkers(cfg.ee_marker)
            self.source_marker = VisualizationMarkers(cfg.origin_marker)

        self._sample_goal()

        # --- Domain Randomization helpers ---
        self._action_buffer = ActionBuffer(
            num_envs=self._num_envs,
            action_dim=self.cfg.action_space,
            num_joints=6,
            cfg=self.cfg.domain_rand,
            device=self.device,
        )
        self._obs_buffer = ObservationBuffer(
            num_envs=self._num_envs,
            obs_dim=self.cfg.observation_space,
            num_joints=6,
            cfg=self.cfg.domain_rand,
            device=self.device,
        )
        self._actuator_randomizer: ActuatorRandomizer | None = None
        self._mass_com_randomizer: MassComRandomizer | None = None

    # -- Scene setup --------------------------------------------------------

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

    # -- Step logic ---------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Apply domain-randomised action delay/noise (toggles checked internally)
        effective_actions = self._action_buffer.push(self.actions)

        # Split the 12-dim action vector: first 6 = position, last 6 = velocity
        pos_actions = effective_actions[:, :6]
        vel_actions = effective_actions[:, 6:]

        # --- Position increments ---
        increments = (
            self.robot_dof_speed_scales.unsqueeze(0)
            * self.dt
            * pos_actions
            * self.cfg.action_scale
        )
        targets = self.robot_dof_targets + increments
        self.robot_dof_targets[:] = torch.clamp(
            targets,
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
        )

        # --- Velocity targets ---
        self.robot_speed_targets[:] = vel_actions * self.cfg.velocity_scale

        # Debug
        if self.cfg.debug and self.common_step_counter % 100 == 0:
            try:
                raw_sample = actions[0].cpu().numpy()
                clamped_sample = self.actions[0].cpu().numpy()
                inc_sample = increments[0].cpu().numpy()
                targ_sample = targets[0].cpu().numpy()
                speed_sample = self.robot_speed_targets[0].cpu().numpy()
                mins = self.actions.min(dim=0).values.cpu().numpy()
                maxs = self.actions.max(dim=0).values.cpu().numpy()
                print(
                    f"Action(raw) sample: {raw_sample}\n"
                    f"Action(clamped) sample: {clamped_sample}\n"
                    f"Pos increments sample: {inc_sample}\n"
                    f"Pos targets sample (pre-clamp): {targ_sample}\n"
                    f"Pos targets sample (post-clamp): {self.robot_dof_targets[0].cpu().numpy()}\n"
                    f"Vel targets sample: {speed_sample}\n"
                    f"Action min: {mins}\nAction max: {maxs}"
                )
            except Exception:
                pass

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)
        self._robot.set_joint_velocity_target(self.robot_speed_targets)

    # -- Observations -------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if not getattr(self.cfg, "norm_obs", True):
            return self._get_observations_raw()

        frame_data = self._frame_transformer.data
        tcp_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        tcp_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        # Normalized observations
        to_target_norm = (self.goal_pos_source - tcp_pos_source) / MAX_REACH
        orientation_error_norm = (
            quat_box_minus(self.goal_quat_source, tcp_quat_source) / torch.pi
        )
        self.joint_pos_norm = (
            2
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        joint_vel_norm = self._robot.data.joint_vel / MAX_JOINT_VEL
        tcp_angular_vel, tcp_linear_vel = self.compute_tcp_states()
        tcp_linear_vel_norm = tcp_linear_vel / self.cfg.tcp_max_speed
        tcp_angular_vel_norm = tcp_angular_vel / torch.pi

        obs = torch.cat(
            [
                to_target_norm,           # 3
                orientation_error_norm,   # 3
                self.joint_pos_norm,      # 6
                joint_vel_norm,           # 6
                tcp_linear_vel_norm,      # 3
                tcp_angular_vel_norm,     # 3
            ],
            dim=1,
        )
        # Domain randomisation on observations (toggles checked internally)
        obs = self._obs_buffer.append_and_get(obs)

        if self.cfg.debug and self.common_step_counter % 100 == 0:
            sample = obs[0].cpu().numpy()
            print(
                f"Observations: tcp_pos_source={tcp_pos_source[0].cpu().numpy()}, tcp_quat_source={tcp_quat_source[0].cpu().numpy()}, "
                f"Observation vector sample: {sample}"
            )
            mins = obs.min(dim=0).values.cpu().numpy()
            maxs = obs.max(dim=0).values.cpu().numpy()
            print(f"Obs min: {mins}\nObs max: {maxs}")
        return {"policy": obs}

    def _get_observations_raw(self) -> dict[str, torch.Tensor]:
        """Return raw (unnormalized) observations."""
        frame_data = self._frame_transformer.data
        tcp_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        tcp_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        to_target = self.goal_pos_source - tcp_pos_source
        orientation_error = quat_box_minus(self.goal_quat_source, tcp_quat_source)

        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        tcp_angular_vel, tcp_linear_vel = self.compute_tcp_states()

        obs = torch.cat(
            [
                to_target,            # 3
                orientation_error,    # 3
                joint_pos,            # 6
                joint_vel,            # 6
                tcp_linear_vel,       # 3
                tcp_angular_vel,      # 3
            ],
            dim=1,
        )

        if self.cfg.debug and self.common_step_counter % 100 == 0:
            sample = obs[0].cpu().numpy()
            print(
                f"[Raw Observations] tcp_pos_source={tcp_pos_source[0].cpu().numpy()}, tcp_quat_source={tcp_quat_source[0].cpu().numpy()}, "
                f"Observation vector sample: {sample}"
            )
            mins = obs.min(dim=0).values.cpu().numpy()
            maxs = obs.max(dim=0).values.cpu().numpy()
            print(f"[Raw Obs min]: {mins}\n[Raw Obs max]: {maxs}")

        return {"policy": obs}

    # -- Rewards ------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        self._debug_step_count += 1

        frame_data = self._frame_transformer.data
        tcp_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        tcp_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        # Position and orientation error/reward
        position_error = torch.norm(self.goal_pos_source - tcp_pos_source, dim=1)
        position_exp_error = torch.exp(-position_error / self.cfg.position_exp_scale)

        orientation_error = quat_error_magnitude(
            self.goal_quat_source, tcp_quat_source
        )
        orientation_exp_error = torch.exp(
            -orientation_error / self.cfg.orientation_exp_scale
        )
        # Penalties
        pos_action_cost = torch.sum(self.actions[:, :6]**2, dim=1)
        vel_action_cost = torch.sum(self.actions[:, 6:]**2, dim=1)
        velocity_cost = torch.sum(self._robot.data.joint_vel**2, dim=1)

        # Resample goals: timeout
        self.goal_steps_elapsed += 1
        timed_out = self.goal_steps_elapsed >= self.goal_max_steps
        if torch.any(timed_out):
            timeout_ids = timed_out.nonzero(as_tuple=False).flatten()
            self._sample_goal(timeout_ids)
            self.success_frames_count[timeout_ids] = 0

        # Contact penalty
        contact_forces = torch.clamp_max(self._compute_contact_metric(), 1000.0)
        excess = torch.relu(contact_forces - self.cfg.contact_force_threshold_penalty)
        contact_penalty = self.cfg.contact_penalty_scale * excess

        # Joint limit penalty
        threshold = 0.9
        out_of_bounds = torch.abs(self.joint_pos_norm) - threshold
        penalty_val = torch.where(
            out_of_bounds > 0, out_of_bounds**2, torch.zeros_like(out_of_bounds)
        )
        reward_joint_limits = penalty_val.sum(dim=-1)

        # Continuous stability reward
        closeness = torch.exp(-position_error / self.cfg.position_exp_scale)
        stillness = torch.exp(-velocity_cost / 10.0)
        stability_reward = self.cfg.stability_reward_scale * closeness * stillness

        # tcp marker visualisation (debug only)
        if self.cfg.debug:
            self.ee_marker.visualize(
                frame_data.target_pos_w[:, self._ee_frame_idx, :],
                frame_data.target_quat_w[:, self._ee_frame_idx, :],
            )

        reward = (
            self.cfg.ee_position_penalty * position_error
            + self.cfg.ee_position_reward * position_exp_error
            + self.cfg.ee_orientation_penalty * orientation_error
            + self.cfg.ee_orientation_reward * orientation_exp_error
            + self.cfg.action_penalty_scale * pos_action_cost
            + self.cfg.velocity_action_penalty_scale * vel_action_cost
            + self.cfg.velocity_penalty_scale * velocity_cost
            + self.cfg.joint_limit_penalty_scale * reward_joint_limits
            + contact_penalty
            + stability_reward
        )

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(
            {
                "pos_action_penalty": (self.cfg.action_penalty_scale * pos_action_cost).mean(),
                "vel_action_penalty": (self.cfg.velocity_action_penalty_scale * vel_action_cost).mean(),
                "stability_reward_mean": stability_reward.mean(),
                "contact_force_max": contact_forces.max(),
                "contact_force_mean": contact_forces.mean(),
                "contact_penalty_mean": contact_penalty.mean(),
                "joint_limit_penalty_mean": (
                    self.cfg.joint_limit_penalty_scale * reward_joint_limits
                ).mean(),
                "ori_error": orientation_error.mean(),
                "ori_reward": (
                    self.cfg.ee_orientation_reward * orientation_exp_error
                ).mean(),
                "pos_error": position_error.mean(),
                "pos_reward": (self.cfg.ee_position_penalty * position_error).mean(),
                "rewards_mean": reward.mean(),
                "rewards_std": reward.std(),
                "step_counter": self.common_step_counter,
                "velocity_penalty": (
                    self.cfg.velocity_penalty_scale * velocity_cost
                ).mean(),
            }
        )
        return reward

    # -- Termination --------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # -- Reset --------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # Lazily create the physical randomizers
        if self._actuator_randomizer is None:
            self._actuator_randomizer = ActuatorRandomizer(
                robot=self._robot,
                cfg=self.cfg.domain_rand,
                device=self.device,
            )
        if self._mass_com_randomizer is None:
            self._mass_com_randomizer = MassComRandomizer(
                robot=self._robot,
                cfg=self.cfg.domain_rand,
                device=self.device,
            )

        # Split envs: some reset to home, others fully random
        mask = torch.rand(len(env_ids), device=self.device) < self.cfg.env_reset
        home_ids = env_ids[mask]
        random_ids = env_ids[~mask]
        if len(home_ids) > 0:
            self._reset_to_home(home_ids)
        if len(random_ids) > 0:
            self._reset_random(random_ids)

        # Write reset joint state to simulation
        self._robot.write_joint_state_to_sim(
            self._robot.data.joint_pos[env_ids],
            self._robot.data.joint_vel[env_ids],
            env_ids=env_ids,
        )

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.robot_speed_targets[env_ids] = 0.0
        self.success_frames_count[env_ids] = 0

        # Reset DR buffers & apply physical randomisation (toggles checked internally)
        self._action_buffer.reset(env_ids)
        self._obs_buffer.reset(env_ids)
        self._actuator_randomizer.sample_and_apply(env_ids)
        self._mass_com_randomizer.sample_and_apply(env_ids)

        self._sample_goal(env_ids)

        # debug markers
        if self.cfg.debug:
            marker_idx = torch.zeros(
                len(env_ids), dtype=torch.int64, device=self.device
            )
            self.origin_marker.visualize(self.env_origins, marker_indices=marker_idx)
            self._visualize_source_frame(env_ids, marker_idx)

    # -- Goal sampling ------------------------------------------------------

    def _sample_goal(self, env_ids: torch.Tensor | None = None):
        """Sample random goal poses within a cylindrical volume around the robot base."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        angle = torch.empty(len(env_ids), device=self.device).uniform_(
            -torch.pi, torch.pi
        )
        radius = torch.empty(len(env_ids), device=self.device).uniform_(0.3, 0.75)
        height = torch.empty(len(env_ids), device=self.device).uniform_(0.1, 0.6)

        self.goal_pos_source[env_ids, 0] = self.robot_base_local[0] + radius * torch.cos(angle)
        self.goal_pos_source[env_ids, 1] = self.robot_base_local[1] + radius * torch.sin(angle)
        self.goal_pos_source[env_ids, 2] = height

        delta_quat = torch.randn(len(env_ids), 4, device=self.device)
        delta_quat = delta_quat / torch.norm(delta_quat, dim=1, keepdim=True)
        self.goal_quat_source[env_ids] = delta_quat

        self.goal_steps_elapsed[env_ids] = 0

        if self.cfg.debug:
            marker_idx = torch.zeros(
                len(env_ids), dtype=torch.int64, device=self.device
            )
            self.goal_marker.visualize(
                self.goal_pos_source[env_ids] + self.env_origins[env_ids],
                self.goal_quat_source[env_ids],
                marker_indices=marker_idx,
            )

    # -- Debug helpers ------------------------------------------------------

    def _visualize_source_frame(self, env_ids: torch.Tensor, marker_idx: torch.Tensor):
        """Visualise the FrameTransformer source-frame offset in world coordinates."""
        try:
            base_idx = self._robot.body_names.index("base_link")
            src_off = (
                torch.tensor(
                    self.cfg.frame_transformer.source_frame_offset.pos,
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .expand(len(env_ids), -1)
            )
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
            pass

    # -- Contact metric -----------------------------------------------------

    def _compute_contact_metric(self) -> torch.Tensor:
        """Max normal contact force per environment (N)."""
        if self._contact_sensor is None:
            return torch.zeros(self._num_envs, device=self.device)

        data = self._contact_sensor.data
        if data is None or data.net_forces_w is None:
            return torch.zeros(self._num_envs, device=self.device)

        forces = data.net_forces_w
        magnitudes = torch.norm(forces, dim=2)
        max_per_env, max_body_idx = magnitudes.max(dim=1)

        if (
            self.cfg.contact_debug_interval > 0
            and self._debug_step_count % self.cfg.contact_debug_interval == 0
        ):
            body_names = self._contact_sensor.body_names
            env0_mags = magnitudes[0]
            parts = [
                f"{name}: {env0_mags[i].item():.2f}N"
                for i, name in enumerate(body_names)
            ]
            top_body = body_names[max_body_idx[0].item()] if body_names else "?"
            print(
                f"[Contact Debug step={self._debug_step_count}] "
                f"env0 max={max_per_env[0].item():.2f}N ({top_body}) | "
                f"all_envs max={max_per_env.max().item():.2f}N mean={max_per_env.mean().item():.2f}N\n"
                f"  per-body: {', '.join(parts)}"
            )

        return max_per_env

    def _reset_to_home(self, env_ids: torch.Tensor):
        """Reset around the home pose."""
        home = self._robot.data.default_joint_pos[env_ids]

        if self.cfg.progressive_reset:
            progress = min(1.0, self.common_step_counter / self.cfg.progressive_reset_steps)

            initial_half = 0.125
            tight_low = (home - initial_half).clamp(min=self.robot_dof_lower_limits)
            tight_high = (home + initial_half).clamp(max=self.robot_dof_upper_limits)

            low = tight_low + progress * (self.robot_dof_lower_limits - tight_low)
            high = tight_high + progress * (self.robot_dof_upper_limits - tight_high)

            joint_pos = sample_uniform(
                low, high,
                (len(env_ids), self._robot.num_joints),
                self.device,
            )
        else:
            joint_pos = home + sample_uniform(
                -0.125, 0.125,
                (len(env_ids), self._robot.num_joints),
                self.device,
            )

        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.data.joint_pos[env_ids] = joint_pos
        self._robot.data.joint_vel[env_ids] = 0.0

    def _reset_random(self, env_ids: torch.Tensor):
        """Reset to a random pose within the joint limits."""
        joint_pos = sample_uniform(
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.data.joint_pos[env_ids] = joint_pos
        self._robot.data.joint_vel[env_ids] = 0.0

    def compute_tcp_states(self):
        wrist_quat_w = self._robot.data.body_link_quat_w[:, 6]
        wrist_vel_w = self._robot.data.body_link_vel_w[:, 6]

        lin_vel_wrist_w = wrist_vel_w[:, :3]
        ang_vel_wrist_w = wrist_vel_w[:, 3:]

        tcp_offset = (
            torch.tensor(TCP_OFFSET_LOCAL, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(wrist_quat_w.shape[0], -1)
        )
        r_offset_w = quat_apply_yaw(wrist_quat_w, tcp_offset)

        v_tcp_w = lin_vel_wrist_w + torch.cross(ang_vel_wrist_w, r_offset_w, dim=-1)

        base_rot = (
            torch.tensor(BASE_ROTATION_LOCAL, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(wrist_quat_w.shape[0], -1)
        )
        inv_base = quat_inv(base_rot)
        v_tcp_final = quat_apply_yaw(inv_base, v_tcp_w)
        ang_vel_final = quat_apply_yaw(inv_base, ang_vel_wrist_w)

        return v_tcp_final, ang_vel_final
