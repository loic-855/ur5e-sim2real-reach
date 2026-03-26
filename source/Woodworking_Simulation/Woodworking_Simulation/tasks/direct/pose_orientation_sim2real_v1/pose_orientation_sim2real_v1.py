# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Pose-and-orientation reaching task for a UR5e arm **with gripper** (sim-to-real variant V1).

Uses the GRIPPER_TCP robot model (8 joints: 6 arm + 2 finger).
The gripper is present in the simulation for visual/collision fidelity with
the real robot but is **not actuated by the policy**.  Only the 6 arm joints
are controlled.

Same architecture as V2 but with **position-only control**: the policy outputs
6 joint-position increments instead of 12 (no velocity feedforward head).

Observations (24-dim):
    pos_error (3), ori_error (3), joint_pos (6 arm), joint_vel (6 arm),
    tcp_linear_vel (3), tcp_angular_vel (3).

Actions (6-dim):
    position increments (6 arm joints only).

Gripper joints are held at their default position (closed).
"""

from __future__ import annotations

import torch
import math

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
from isaaclab.utils.math import sample_uniform, quat_error_magnitude, quat_box_minus, quat_apply_yaw, quat_inv, quat_apply, quat_mul

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

# TCP offset from wrist_3_link
TCP_OFFSET_LOCAL = (0.0, 0.0, 0.14)
TCP_ROTATION_LOCAL = (0.0, 0.0, 0.0, 1.0)

from Woodworking_Simulation.common.domain_randomization import (
    ActionBuffer,
    ActuatorRandomizer,
    DomainRandomizationV4Cfg,
    MassComRandomizer,
    ObservationBuffer,
)

# Number of UR5e arm joints (excluding gripper)
NUM_ARM_JOINTS = 6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class PoseOrientationSim2RealV1Cfg(DirectRLEnvCfg):
    """Configuration for V1 – UR5e + gripper, only arm controlled, position-only actions."""

    # env timing
    episode_length_s = 10.0
    decimation = 2

    # spaces – 6 actions: position increments only (arm)
    action_space = 6
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

    # GRIPPER_TCP robot model (8 joints: 6 arm + 2 finger)
    robot = get_robot_cfg(RobotType.GRIPPER_TCP, "/World/envs/env_.*/ur5e")

    # scene assets
    table = get_table_cfg()
    terrain = get_terrain_cfg()
    goal_marker = get_goal_marker_cfg()
    origin_marker = get_origin_marker_cfg()
    ee_marker = get_robot_grasp_marker_cfg()

    # frame transformer – TCP pose relative to table centre
    frame_transformer = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/ur5e/ur5e/base_link",
        source_frame_offset=OffsetCfg(
            pos=BASE_OFFSET_LOCAL,
            rot=BASE_ROTATION_LOCAL,
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ur5e/ur5e/wrist_3_link",
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
        prim_path="/World/envs/env_.*/ur5e/ur5e/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )

    # camera poles
    camera_pole_spawn_cfg = get_camera_pole_cfg()

    # action and coef scaling
    action_scale = 2.0
    reset_range = 0.125
    goal_timeout_s = 10.0

    # reward weights
    ee_position_penalty = -1.0
    ee_position_reward = 2.0
    ee_orientation_penalty = -0.8
    ee_orientation_reward = 0.0

    # --- Small curriculum for exponential scales ---
    enable_exp_curriculum: bool = True
    position_exp_scale_start: float = 0.2
    position_exp_scale_end: float = 0.05
    orientation_exp_scale_start: float = 0.2
    orientation_exp_scale_end: float = 0.05
    exp_curriculum_steps: int = 110000

    # penalty weights
    action_penalty_scale = -0.02
    velocity_penalty_scale = -0.02
    contact_penalty_scale = -0.01
    contact_force_threshold_penalty = 5.0
    joint_limit_penalty_scale = -0.02

    # TCP velocity normalization (m/s)
    tcp_max_speed = 2.0

    # Goal sampling: 0.0 = 100 % FK-based, 1.0 = 100 % random cylindrical
    goal_sampling_random_ratio: float = 0.3
    deterministic_goal_sampling: bool = False
    benchmark_goals: tuple[tuple[float, ...], ...] = ()
    goal_height = [0.1, 0.6]

    # --- Debug ---
    debug = False
    contact_debug_interval: int = 0

    # --- Domain Randomization ---
    domain_rand: DomainRandomizationV4Cfg = DomainRandomizationV4Cfg()


# ---------------------------------------------------------------------------
# V1
# ---------------------------------------------------------------------------


class PoseOrientationSim2RealV1(DirectRLEnv):
    """Pose + orientation reaching with position-only control – gripper present but passive."""

    cfg: PoseOrientationSim2RealV1Cfg

    def __init__(
        self,
        cfg: PoseOrientationSim2RealV1Cfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs
        self.reward_buf = torch.zeros(self._num_envs, device=self.device, dtype=torch.float32)
        self._debug_step_count = 0
        self.c_idx = 0

        # arm vs total joints
        self._num_arm_joints = NUM_ARM_JOINTS
        self._num_total_joints = self._robot.num_joints  # 8 for GRIPPER_TCP

        # env origins with global offset
        self.env_origins = (
            self.scene.env_origins + ENV_ORIGIN_OFFSET.to(device=self.device)
        ).to(device=self.device, dtype=torch.float32)

        # frame transformer
        self._frame_transformer = self.scene.sensors["frame_transformer"]
        self._ee_frame_idx = self._frame_transformer.data.target_frame_names.index("ee_tcp")

        # look up wrist body index dynamically
        self._wrist_body_idx = self._robot.body_names.index("wrist_3_link")

        # Pre-compute UR5e DH parameters
        import math
        self._dh_a = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
        self._dh_d = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
        self._dh_alpha = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
        self._dh_cos_alpha = [math.cos(a) for a in self._dh_alpha]
        self._dh_sin_alpha = [math.sin(a) for a in self._dh_alpha]

        # Joint limits for the 6 ARM joints only
        arm_limits = list(JOINT_LIMITS.values())[:NUM_ARM_JOINTS]
        lower = torch.tensor([v[0] for v in arm_limits], device=self.device, dtype=torch.float32)
        upper = torch.tensor([v[1] for v in arm_limits], device=self.device, dtype=torch.float32)
        self.robot_dof_lower_limits = lower  # (6,)
        self.robot_dof_upper_limits = upper  # (6,)
        self.joint_pos_norm = torch.zeros((self._num_envs, NUM_ARM_JOINTS), device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)  # (6,)

        # Full joint targets (8-dim) – arm + gripper
        self.robot_dof_targets = self._robot.data.joint_pos.clone()  # (N, 8)

        # Default gripper position
        self._gripper_default_pos = self._robot.data.default_joint_pos[0, NUM_ARM_JOINTS:].clone()  # (2,)

        # goal tensors (source frame = table-relative)
        self.robot_base_local = torch.tensor([-0.52, 0.32, 0.0], device=self.device)
        self._source_origin_in_base = torch.tensor(
            BASE_OFFSET_LOCAL, device=self.device, dtype=torch.float32
        )
        self._source_rot_in_base = torch.tensor(
            BASE_ROTATION_LOCAL, device=self.device, dtype=torch.float32
        )
        self.goal_pos_source = torch.zeros(
            (self._num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.goal_quat_source = torch.zeros(
            (self._num_envs, 4), device=self.device, dtype=torch.float32
        )
        self.benchmark_goal_pos_quat = torch.zeros(
            (1, 7), device=self.device, dtype=torch.float32
        )
        self._num_benchmark_goals = 0
        self._benchmark_goal_idx = 0

        if len(self.cfg.benchmark_goals) > 0:
            benchmark_goals = torch.tensor(
                self.cfg.benchmark_goals, device=self.device, dtype=torch.float32
            )
            if benchmark_goals.ndim != 2 or benchmark_goals.shape[1] != 7:
                raise ValueError(
                    "cfg.benchmark_goals must have shape (N, 7) with [x, y, z, qw, qx, qy, qz]."
                )
            benchmark_goals[:, 3:7] = benchmark_goals[:, 3:7] / torch.norm(
                benchmark_goals[:, 3:7], dim=1, keepdim=True
            ).clamp(min=1e-8)
            self.benchmark_goal_pos_quat = benchmark_goals
            self._num_benchmark_goals = int(benchmark_goals.shape[0])

        # Action and step buffers
        self.actions = torch.zeros(
            (self._num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.success_frames_count = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.goal_steps_elapsed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.goal_max_steps = int(self.cfg.goal_timeout_s / self.dt)

        # visualisation markers
        if self.cfg.debug:
            self.goal_marker = VisualizationMarkers(cfg.goal_marker)
            self.origin_marker = VisualizationMarkers(cfg.origin_marker)
            self.ee_marker = VisualizationMarkers(cfg.ee_marker)
            self.source_marker = VisualizationMarkers(cfg.origin_marker)

        self._sample_goal()

        # Domain Randomization helpers
        self._action_buffer = ActionBuffer(
            num_envs=self._num_envs,
            action_dim=self.cfg.action_space,
            num_joints=NUM_ARM_JOINTS,
            cfg=self.cfg.domain_rand,
            device=self.device,
        )
        self._obs_buffer = ObservationBuffer(
            num_envs=self._num_envs,
            obs_dim=self.cfg.observation_space,
            num_joints=NUM_ARM_JOINTS,
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

        effective_actions = self._action_buffer.push(self.actions)

        # Position increments only (6-dim)
        increments = (
            self.robot_dof_speed_scales.unsqueeze(0)
            * self.dt
            * effective_actions
            * self.cfg.action_scale
        )
        arm_targets = self.robot_dof_targets[:, :NUM_ARM_JOINTS] + increments
        self.robot_dof_targets[:, :NUM_ARM_JOINTS] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
        )
        self.robot_dof_targets[:, NUM_ARM_JOINTS:] = self._gripper_default_pos.unsqueeze(0)

        if self.cfg.debug and self.common_step_counter % 100 == 0:
            try:
                print(
                    f"Action(raw) sample: {actions[0].cpu().numpy()}\n"
                    f"Pos increments: {increments[0].cpu().numpy()}\n"
                    f"Arm targets (post-clamp): {self.robot_dof_targets[0, :NUM_ARM_JOINTS].cpu().numpy()}\n"
                    f"Gripper targets: {self.robot_dof_targets[0, NUM_ARM_JOINTS:].cpu().numpy()}"
                )
            except Exception as e:
                print(f"[DEBUG ERROR] {e}")

    def _apply_action(self):
        # Position-only control; no velocity targets sent
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # -- Observations -------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        frame_data = self._frame_transformer.data
        tcp_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        tcp_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        to_target_norm = (self.goal_pos_source - tcp_pos_source) / MAX_REACH
        orientation_error_norm = quat_box_minus(self.goal_quat_source, tcp_quat_source) / torch.pi

        arm_joint_pos = self._robot.data.joint_pos[:, :NUM_ARM_JOINTS]
        arm_joint_vel = self._robot.data.joint_vel[:, :NUM_ARM_JOINTS]

        self.joint_pos_norm = (
            2 * (arm_joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        joint_vel_norm = arm_joint_vel / MAX_JOINT_VEL
        tcp_angular_vel, tcp_linear_vel = self.compute_tcp_states()
        tcp_linear_vel_norm = tcp_linear_vel / self.cfg.tcp_max_speed
        tcp_angular_vel_norm = tcp_angular_vel / torch.pi

        obs = torch.cat(
            [
                to_target_norm,          # 3
                orientation_error_norm,  # 3
                self.joint_pos_norm,     # 6 (arm only)
                joint_vel_norm,          # 6 (arm only)
                tcp_linear_vel_norm,     # 3
                tcp_angular_vel_norm,    # 3
            ],
            dim=1,
        )
        obs = self._obs_buffer.append_and_get(obs)

        if self.cfg.debug and self.common_step_counter % 100 == 0:
            print(
                f"Observations: tcp_pos_source={tcp_pos_source[0].cpu().numpy()}, "
                f"Observation vector sample: {obs[0].cpu().numpy()}"
            )
        return {"policy": obs}

    # -- Rewards ------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        self._debug_step_count += 1

        frame_data = self._frame_transformer.data
        tcp_pos_source = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        tcp_quat_source = frame_data.target_quat_source[:, self._ee_frame_idx, :]

        if getattr(self.cfg, "enable_exp_curriculum", False):
            steps = max(1, int(self.cfg.exp_curriculum_steps))
            t = min(1.0, float(self.common_step_counter) / float(steps))
            pos_scale = (1.0 - t) * float(self.cfg.position_exp_scale_start) + t * float(self.cfg.position_exp_scale_end)
            ori_scale = (1.0 - t) * float(self.cfg.orientation_exp_scale_start) + t * float(self.cfg.orientation_exp_scale_end)
        else:
            pos_scale = float(self.cfg.position_exp_scale_end)
            ori_scale = float(self.cfg.orientation_exp_scale_end)

        position_error = torch.norm(self.goal_pos_source - tcp_pos_source, dim=1)
        position_exp_error = torch.exp(-position_error / pos_scale)

        orientation_error = quat_error_magnitude(self.goal_quat_source, tcp_quat_source)
        orientation_exp_error = torch.exp(-orientation_error / ori_scale)

        action_cost = torch.sum(self.actions ** 2, dim=1)
        velocity_cost = torch.sum(self._robot.data.joint_vel[:, :NUM_ARM_JOINTS] ** 2, dim=1)

        # Goal timeout resampling
        self.goal_steps_elapsed += 1
        timed_out = self.goal_steps_elapsed >= self.goal_max_steps
        if torch.any(timed_out):
            timeout_ids = timed_out.nonzero(as_tuple=False).flatten()
            self._sample_goal(timeout_ids)
            self.success_frames_count[timeout_ids] = 0

        contact_forces = torch.clamp_max(self._compute_contact_metric(), 1000.0)
        excess = torch.relu(contact_forces - self.cfg.contact_force_threshold_penalty)
        contact_penalty = self.cfg.contact_penalty_scale * excess

        threshold = 0.9
        out_of_bounds = torch.abs(self.joint_pos_norm) - threshold
        penalty_val = torch.where(out_of_bounds > 0, out_of_bounds ** 2, torch.zeros_like(out_of_bounds))
        reward_joint_limits = penalty_val.sum(dim=-1)

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
            + self.cfg.action_penalty_scale * action_cost
            + self.cfg.velocity_penalty_scale * velocity_cost
            + self.cfg.joint_limit_penalty_scale * reward_joint_limits
            + contact_penalty
        )

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(
            {
                "action_penalty": (self.cfg.action_penalty_scale * action_cost).mean(),
                "contact_force_max": contact_forces.max(),
                "contact_force_mean": contact_forces.mean(),
                "contact_penalty_mean": contact_penalty.mean(),
                "joint_limit_penalty_mean": (self.cfg.joint_limit_penalty_scale * reward_joint_limits).mean(),
                "ori_error": orientation_error.mean(),
                "ori_reward": (self.cfg.ee_orientation_reward * orientation_exp_error).mean(),
                "ori_penalty": (self.cfg.ee_orientation_penalty * orientation_error).mean(),
                "pos_error": position_error.mean(),
                "pos_reward": (self.cfg.ee_position_reward * position_exp_error).mean(),
                "pos_penalty": (self.cfg.ee_position_penalty * position_error).mean(),
                "rewards_mean": reward.mean(),
                "rewards_std": reward.std(),
                "step_counter": self.common_step_counter,
                "velocity_penalty": (self.cfg.velocity_penalty_scale * velocity_cost).mean(),
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

        if self._actuator_randomizer is None:
            self._actuator_randomizer = ActuatorRandomizer(
                robot=self._robot, cfg=self.cfg.domain_rand, device=self.device
            )
        if self._mass_com_randomizer is None:
            self._mass_com_randomizer = MassComRandomizer(
                robot=self._robot, cfg=self.cfg.domain_rand, device=self.device
            )

        self._reset_to_home(env_ids)

        self._robot.write_joint_state_to_sim(
            self._robot.data.joint_pos[env_ids],
            self._robot.data.joint_vel[env_ids],
            env_ids=env_ids,
        )

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.success_frames_count[env_ids] = 0
        self._benchmark_goal_idx = 0

        self._action_buffer.reset(env_ids)
        self._obs_buffer.reset(env_ids)
        self._actuator_randomizer.sample_and_apply(env_ids)
        self._mass_com_randomizer.sample_and_apply(env_ids)

        self._sample_goal(env_ids)

        if self.cfg.debug:
            marker_idx = torch.zeros(len(env_ids), dtype=torch.int64, device=self.device)
            self.origin_marker.visualize(self.env_origins, marker_indices=marker_idx)
            self._visualize_source_frame(env_ids, marker_idx)

    # -- Goal sampling ------------------------------------------------------

    def _ur5e_fk_batch(self, joint_angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analytical FK for UR5e using standard DH parameters (GPU-native)."""
        N = joint_angles.shape[0]
        dev = joint_angles.device
        dt = joint_angles.dtype

        a_list = self._dh_a
        d_list = self._dh_d
        cos_alpha = self._dh_cos_alpha
        sin_alpha = self._dh_sin_alpha

        T = torch.eye(4, device=dev, dtype=dt).unsqueeze(0).repeat(N, 1, 1)

        for i in range(6):
            theta = joint_angles[:, i]
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = cos_alpha[i]
            sa = sin_alpha[i]
            ai = a_list[i]
            di = d_list[i]

            Ti = torch.zeros(N, 4, 4, device=dev, dtype=dt)
            Ti[:, 0, 0] = ct
            Ti[:, 0, 1] = -st * ca
            Ti[:, 0, 2] =  st * sa
            Ti[:, 0, 3] =  ai * ct
            Ti[:, 1, 0] =  st
            Ti[:, 1, 1] =  ct * ca
            Ti[:, 1, 2] = -ct * sa
            Ti[:, 1, 3] =  ai * st
            Ti[:, 2, 1] =  sa
            Ti[:, 2, 2] =  ca
            Ti[:, 2, 3] =  di
            Ti[:, 3, 3] =  1.0

            T = torch.bmm(T, Ti)

        tcp_z = TCP_OFFSET_LOCAL[2]
        pos_base = torch.stack([
            T[:, 0, 3] + T[:, 0, 2] * tcp_z,
            T[:, 1, 3] + T[:, 1, 2] * tcp_z,
            T[:, 2, 3] + T[:, 2, 2] * tcp_z,
        ], dim=1)

        R = T[:, :3, :3]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        s0 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 2.0
        s1 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-8)) * 2.0
        s2 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-8)) * 2.0
        s3 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-8)) * 2.0

        q0 = torch.stack([0.25 * s0, (R[:, 2, 1] - R[:, 1, 2]) / s0, (R[:, 0, 2] - R[:, 2, 0]) / s0, (R[:, 1, 0] - R[:, 0, 1]) / s0], dim=1)
        q1 = torch.stack([(R[:, 2, 1] - R[:, 1, 2]) / s1, 0.25 * s1, (R[:, 0, 1] + R[:, 1, 0]) / s1, (R[:, 0, 2] + R[:, 2, 0]) / s1], dim=1)
        q2 = torch.stack([(R[:, 0, 2] - R[:, 2, 0]) / s2, (R[:, 0, 1] + R[:, 1, 0]) / s2, 0.25 * s2, (R[:, 1, 2] + R[:, 2, 1]) / s2], dim=1)
        q3 = torch.stack([(R[:, 1, 0] - R[:, 0, 1]) / s3, (R[:, 0, 2] + R[:, 2, 0]) / s3, (R[:, 1, 2] + R[:, 2, 1]) / s3, 0.25 * s3], dim=1)

        c0 = (trace > 0).unsqueeze(1)
        c1 = (~(trace > 0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])).unsqueeze(1)
        c2 = (~(trace > 0) & ~(R[:, 0, 0] > R[:, 1, 1]) & (R[:, 1, 1] > R[:, 2, 2])).unsqueeze(1)

        quat_base = torch.where(c0, q0, torch.where(c1, q1, torch.where(c2, q2, q3)))
        quat_base = quat_base / torch.norm(quat_base, dim=1, keepdim=True).clamp(min=1e-8)
        return pos_base, quat_base

    def _fk_to_source_frame(
        self, pos_base: torch.Tensor, quat_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert FK outputs from base_link frame to the FrameTransformer source frame."""
        N = pos_base.shape[0]
        inv_rot = quat_inv(self._source_rot_in_base.unsqueeze(0).expand(N, -1))
        delta = pos_base - self._source_origin_in_base.unsqueeze(0)
        pos_source = quat_apply(inv_rot, delta)
        quat_source = quat_mul(inv_rot, quat_base)
        return pos_source, quat_source

    def _sample_goal(self, env_ids: torch.Tensor | None = None):
        """Sample goal poses for the given environments."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        n = len(env_ids)
        if self.cfg.deterministic_goal_sampling:
            if self._num_benchmark_goals <= 0:
                raise RuntimeError(
                    "deterministic_goal_sampling=True requires cfg.benchmark_goals to contain at least one goal."
                )
            goal_idx = self._benchmark_goal_idx % self._num_benchmark_goals
            selected_goals = self.benchmark_goal_pos_quat[goal_idx].unsqueeze(0).expand(n, -1)
            self.goal_pos_source[env_ids] = selected_goals[:, :3]
            self.goal_quat_source[env_ids] = selected_goals[:, 3:7]
            self._benchmark_goal_idx += 1
            self.goal_steps_elapsed[env_ids] = 0
            if self.cfg.debug:
                marker_idx = torch.zeros(n, dtype=torch.int64, device=self.device)
                self.goal_marker.visualize(
                    self.goal_pos_source[env_ids] + self.env_origins[env_ids],
                    self.goal_quat_source[env_ids],
                    marker_indices=marker_idx,
                )
            return

        ratio = max(0.0, min(1.0, float(self.cfg.goal_sampling_random_ratio)))

        if ratio >= 1.0:
            random_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        elif ratio <= 0.0:
            random_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            random_mask = torch.rand(n, device=self.device) < ratio

        rand_ids = env_ids[random_mask]
        fk_ids   = env_ids[~random_mask]

        if len(rand_ids) > 0:
            m = len(rand_ids)
            angle  = torch.empty(m, device=self.device).uniform_(-torch.pi, torch.pi)
            radius = torch.empty(m, device=self.device).uniform_(0.3, 0.75)
            height = torch.empty(m, device=self.device).uniform_(self.cfg.goal_height[0], self.cfg.goal_height[1])
            self.goal_pos_source[rand_ids, 0] = self.robot_base_local[0] + radius * torch.cos(angle)
            self.goal_pos_source[rand_ids, 1] = self.robot_base_local[1] + radius * torch.sin(angle)
            self.goal_pos_source[rand_ids, 2] = height
            delta_quat = torch.randn(m, 4, device=self.device)
            delta_quat = delta_quat / torch.norm(delta_quat, dim=1, keepdim=True)
            self.goal_quat_source[rand_ids] = delta_quat

        if len(fk_ids) > 0:
            arm_joints = sample_uniform(
                self.robot_dof_lower_limits,
                self.robot_dof_upper_limits,
                (len(fk_ids), NUM_ARM_JOINTS),
                self.device,
            )
            ps, qs = self._fk_to_source_frame(*self._ur5e_fk_batch(arm_joints))

            below = ps[:, 2] < 0.0
            if torch.any(below):
                ps[below, 2] = -ps[below, 2]
                s = 2 ** -0.5
                q_mirror = torch.tensor(
                    [0.0, s, s, 0.0], device=self.device, dtype=qs.dtype
                ).unsqueeze(0).expand(int(below.sum()), -1)
                qs[below] = quat_mul(q_mirror, qs[below])

            self.goal_pos_source[fk_ids]  = ps
            self.goal_quat_source[fk_ids] = qs

        self.goal_steps_elapsed[env_ids] = 0

        if self.cfg.debug:
            marker_idx = torch.zeros(n, dtype=torch.int64, device=self.device)
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
                    device=self.device, dtype=torch.float32,
                )
                .unsqueeze(0).expand(len(env_ids), -1)
            )
            src_rot = torch.tensor(
                self.cfg.frame_transformer.source_frame_offset.rot,
                device=self.device, dtype=torch.float32,
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
            parts = [f"{name}: {env0_mags[i].item():.2f}N" for i, name in enumerate(body_names)]
            top_body = body_names[max_body_idx[0].item()] if body_names else "?"
            print(
                f"[Contact Debug step={self._debug_step_count}] "
                f"env0 max={max_per_env[0].item():.2f}N ({top_body}) | "
                f"all_envs max={max_per_env.max().item():.2f}N mean={max_per_env.mean().item():.2f}N\n"
                f"  per-body: {', '.join(parts)}"
            )
        return max_per_env

    # -- Reset helpers ------------------------------------------------------

    def _reset_to_home(self, env_ids: torch.Tensor):
        """Reset around the home pose (arm joints randomised, gripper at default)."""
        home = self._robot.data.default_joint_pos[env_ids]  # (N, 8)
        arm_pos = home[:, :NUM_ARM_JOINTS] + sample_uniform(
            -self.cfg.reset_range, self.cfg.reset_range,
            (len(env_ids), NUM_ARM_JOINTS),
            self.device,
        )
        joint_pos = torch.zeros(
            (len(env_ids), self._num_total_joints), device=self.device, dtype=torch.float32
        )
        joint_pos[:, :NUM_ARM_JOINTS] = arm_pos
        joint_pos[:, NUM_ARM_JOINTS:] = self._gripper_default_pos.unsqueeze(0)
        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.data.joint_pos[env_ids] = joint_pos
        self._robot.data.joint_vel[env_ids] = 0.0

    def _reset_random(self, env_ids: torch.Tensor):
        """Reset to a random arm pose within joint limits; gripper at default."""
        arm_pos = sample_uniform(
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
            (len(env_ids), NUM_ARM_JOINTS),
            self.device,
        )
        joint_pos = torch.zeros(
            (len(env_ids), self._num_total_joints), device=self.device, dtype=torch.float32
        )
        joint_pos[:, :NUM_ARM_JOINTS] = arm_pos
        joint_pos[:, NUM_ARM_JOINTS:] = self._gripper_default_pos.unsqueeze(0)
        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.data.joint_pos[env_ids] = joint_pos
        self._robot.data.joint_vel[env_ids] = 0.0

    # -- TCP velocity computation -------------------------------------------

    def compute_tcp_states(self):
        wrist_quat_w = self._robot.data.body_link_quat_w[:, self._wrist_body_idx]
        wrist_vel_w = self._robot.data.body_link_vel_w[:, self._wrist_body_idx]
        lin_vel_wrist_w = wrist_vel_w[:, :3]
        ang_vel_wrist_w = wrist_vel_w[:, 3:]

        tcp_offset = (
            torch.tensor(TCP_OFFSET_LOCAL, device=self.device, dtype=torch.float32)
            .unsqueeze(0).expand(wrist_quat_w.shape[0], -1)
        )
        r_offset_w = quat_apply(wrist_quat_w, tcp_offset)
        v_tcp_w = lin_vel_wrist_w + torch.cross(ang_vel_wrist_w, r_offset_w, dim=-1)

        base_rot = (
            torch.tensor(BASE_ROTATION_LOCAL, device=self.device, dtype=torch.float32)
            .unsqueeze(0).expand(wrist_quat_w.shape[0], -1)
        )
        inv_base = quat_inv(base_rot)
        v_tcp_final = quat_apply(inv_base, v_tcp_w)
        ang_vel_final = quat_apply(inv_base, ang_vel_wrist_w)
        return v_tcp_final, ang_vel_final
