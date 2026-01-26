# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Simple pose/orientation control for UR5e (6 joints, no gripper)."""

from __future__ import annotations
from pathlib import Path
import torch

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

REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"

# Table dimensions (origin at corner, we offset to center)
TABLE_DEPTH = 0.8   # x
TABLE_WIDTH = 1.2   # y
TABLE_HEIGHT = 0.842  # z
# Aluminium block on which the robot is mounted
MOUNT_HEIGHT = 0.02

# Normalization constants
MAX_REACH = 0.85  # UR5e reach ~850mm
MAX_JOINT_VEL = 3.14  # ~180°/s

# Joint limits for real robot (elbow has cable constraint) - as floats
JOINT_LIMITS = {
    "shoulder_pan_joint": (-6.283185307179586, 6.283185307179586),  # -2π, 2π
    "shoulder_lift_joint": (-6.283185307179586, 6.283185307179586),
    "elbow_joint": (-3.141592653589793, 3.141592653589793),  # -π, π (cable constraint)
    "wrist_1_joint": (-6.283185307179586, 6.283185307179586),
    "wrist_2_joint": (-6.283185307179586, 6.283185307179586),
    "wrist_3_joint": (-6.283185307179586, 6.283185307179586),
}


@configclass
class PoseOrientationNoGripper(DirectRLEnvCfg):
    episode_length_s = 8.0
    decimation = 2
    action_space = 6  # 6 joints UR5e
    observation_space = 19  # pos_error(3), quat_error(4), joint_pos(6), joint_vel(6)
    state_space = 0  # Not used in this task

    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0)

    # UR5e robot position in local frame (table center = origin)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",        
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"shoulder_lift_joint": -1.57, "wrist_1_joint": -1.57},
            pos=(0.08, 0.08, TABLE_HEIGHT + MOUNT_HEIGHT),  # Robot base in local frame
            rot=(0.7071, 0.0, 0.0, -0.7071),  # -90° around Z to match real setup
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                stiffness=800, damping=80
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                stiffness=500, damping=50
            ),
        },
    )

    table = sim_utils.UsdFileCfg(usd_path=str(USD_FILES_DIR / "woodworking_table.usd"))
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    goal_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/goal",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        )},
    )

    origin_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/origin",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        )},
    )

    camera_left_pole_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.07, 0.13, 0.7),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),                                     
    )

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
        
        # Cached constants on device
        self.env_origin_offset = torch.tensor(
            [TABLE_DEPTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT], device=self.device
        )
        self.env_origins = self.scene.env_origins + self.env_origin_offset
        
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
        self.dof_mid = (self.dof_lower + self.dof_upper) / 2
        self.dof_targets = self._robot.data.joint_pos.clone()
        
        # Cached tensors for _sample_goal
        self.robot_base_local = torch.tensor([-0.36, -0.54, 0.0], device=self.device)
        self.identity_quat = torch.tensor([1, 0, 0, 0], device=self.device)
        
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)
        self.origin_marker = VisualizationMarkers(cfg.origin_marker)
        self._sample_goal()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.table.func("/World/envs/env_0/Table", self.cfg.table)

        self.camera_left_pole = self.cfg.camera_left_pole_spawn_cfg.func(
            "/World/envs/env_0/CameraLeftPole", self.cfg.camera_left_pole_spawn_cfg,
            translation=(TABLE_DEPTH/2 + 0.365, TABLE_WIDTH/2 - 0.535, TABLE_HEIGHT + 0.37),
        )
        spawn_ground_plane(self.cfg.terrain.prim_path, GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

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


