# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Simple pose/orientation control for UR5e (6 joints, no gripper)."""


from __future__ import annotations
from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform, quat_mul, quat_conjugate, quat_error_magnitude


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


@configclass
class PoseOrientationNoGripper(DirectRLEnvCfg):
    episode_length_s = 3.0
    decimation = 2
    action_space = 6  # 6 joints UR5e
    observation_space = 19  # to_target(3), ori_error(4), joint_pos(6), joint_vel(6)
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
    ee_marker = get_robot_grasp_marker_cfg()

    # Frame transformer to compute TCP pose relative to table center
    frame_transformer = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/ur5e/base_link",
        #prim_path="/World/envs/env_.*/Table/woodworking_table",
        #the offset points to the table center.
        source_frame_offset=OffsetCfg(
            pos=(- (TABLE_WIDTH / 2 - 0.08), TABLE_DEPTH / 2 - 0.08, - MOUNT_HEIGHT),
            rot=(0.7071, 0.0, 0.0, 0.7071),
            #rot=(1.0, 0.0, 0.0, 0.0)     
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

    action_scale = 5.0
    reward_position = 0.5
    penalty_position = -0.2
    penalty_orientation = -0.13
    reset_frac = 1
    action_penalty_scale = -0.005
    elbow_penalty_scale = -0.5  # penalize elbow below table surface
    elbow_min_height = 0.035     # minimum height (m) above table surface
    std = 0.1 # scaling for tanh reward


class PoseOrientationNoGripperV0(DirectRLEnv):
    cfg: PoseOrientationNoGripper

    def __init__(self, cfg: PoseOrientationNoGripper, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation  # 1/60s = 60Hz policy
        
        # Cached constants on device - use shared origin offset
        self.env_origins = self.scene.env_origins + ENV_ORIGIN_OFFSET.to(device=self.device)      
        print(f"robot links availables: {self._robot.body_names}")

        # Frame transformer for TCP pose
        self._frame_transformer = self.scene.sensors["frame_transformer"]
        self._ee_frame_idx = self._frame_transformer.data.target_frame_names.index("ee_tcp")

        # Reset fraction for stochastic resets (not always the same environments)
        self.reset_frac = self.cfg.reset_frac
        self.num_envs_to_reset = int(self.num_envs * self.reset_frac)
        
        # Apply real robot joint limits (already as floats)
        self.q_min = torch.tensor(
            [JOINT_LIMITS[name][0] for name in self._robot.joint_names], device=self.device
        )
        self.q_max = torch.tensor(
            [JOINT_LIMITS[name][1] for name in self._robot.joint_names], device=self.device
        )
        self.q_des = self._robot.data.joint_pos.clone()

        # Elbow body index for table collision avoidance
        self._elbow_body_idx = self._robot.body_names.index("forearm_link")
        # Table surface z in world frame (env_origins already includes TABLE_HEIGHT)
        self._table_z = self.env_origins[:, 2]  # (num_envs,)
        
        #Add more speed to wrist 3 link
        self.robot_q_des_speed_scale = torch.ones_like(self.q_min)
        #self.robot_q_des_speed_scale[5] = 2  # last wrist move faster for orientation control
        
        # Cached tensors for _sample_goal
        self.robot_base_local = torch.tensor([-0.36, -0.54, 0.0], device=self.device)
        self.identity_quat = torch.tensor([1, 0, 0, 0], device=self.device)
        
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)
        self.origin_marker = VisualizationMarkers(cfg.origin_marker)
        self.ee_marker = VisualizationMarkers(cfg.ee_marker)
 
        self._sample_goal()
        print("Init of V0 complete.")
                


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._frame_transformer = FrameTransformer(self.cfg.frame_transformer)
        self.scene.sensors["frame_transformer"] = self._frame_transformer
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
        # store actions for logging / action penalty
        self.actions = actions.clamp(-1.0, 1.0)
        delta = self.dt * self.robot_q_des_speed_scale * self.cfg.action_scale * self.actions
        self.q_des = torch.clamp(self.q_des + delta, self.q_min, self.q_max)
        # Update marker (world frame)
        marker_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(self.goal_pos + self.env_origins, self.goal_quat, marker_indices=marker_idx)
        self.origin_marker.visualize(self.env_origins, self.identity_quat.unsqueeze(0).expand(self.num_envs, -1), marker_indices=marker_idx)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.q_des)

    def _get_observations(self):
        # EE TCP pose relative to table center (source frame) for position
        frame_data = self._frame_transformer.data
        ee_pos = frame_data.target_pos_source[:, self._ee_frame_idx, :] 
        to_target = self.goal_pos - ee_pos
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        
        # Orientation error: use WORLD frame for both ee and goal (goal_quat is world frame)
        ee_quat_w = frame_data.target_quat_w[:, self._ee_frame_idx, :]
        # quat_diff = goal * current^-1  (rotation FROM current TO goal)
        ori_error = quat_mul(self.goal_quat, quat_conjugate(ee_quat_w))
        # Ensure w >= 0 for shortest-path (double-cover) representation
        ori_error = torch.where(ori_error[:, 0:1] < 0, -ori_error, ori_error)
        
        # Normalized observations
        to_target_norm = torch.zeros_like(to_target)
        to_target_norm[:, 0:2] = to_target[:, 0:2] / MAX_REACH 
        to_target_norm[:, 2] = to_target[:, 2] / (MAX_REACH/2)
        joint_pos_norm = 2.0 * (joint_pos - self.q_min) / (self.q_max - self.q_min) - 1.0  # [-1, 1]
        joint_vel_norm = joint_vel / MAX_JOINT_VEL  # [-1, 1] approx
        
        # Update ee_marker
        ee_pos_w = frame_data.target_pos_w[:, self._ee_frame_idx, :]
        self.ee_marker.visualize(
            ee_pos_w,
            ee_quat_w,
            marker_indices=torch.zeros(self.num_envs, dtype=torch.int64, device=self.device),
        )
        elbow_pos = self._robot.data.body_pos_w[:, self._elbow_body_idx, :]
        elbow_quat = self._robot.data.body_quat_w[:, self._elbow_body_idx, :]


        obs = torch.cat(
            [
                to_target_norm,     # 3: position error (source frame)
                ori_error,          # 4: orientation error quaternion (world frame)
                joint_pos_norm,     # 6: normalized joint positions
                joint_vel_norm,     # 6: normalized joint velocities
            ], 
            dim=1
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}


    def _get_rewards(self) -> torch.Tensor:
        if "log" not in self.extras:
            self.extras["log"] = {}

        frame_data = self._frame_transformer.data
        ee_pos = frame_data.target_pos_source[:, self._ee_frame_idx, :]
        # Use WORLD frame orientation for both (goal_quat is in world frame)
        ee_quat_w = frame_data.target_quat_w[:, self._ee_frame_idx, :]
        # Penalize position using L2 norm
        pos_err = torch.norm(self.goal_pos - ee_pos, dim=1)
        # Reward tracking using the tanh kernel
        pos_rew = 1.0 - torch.tanh(pos_err / self.cfg.std)
        # Orientation error in world frame (both quaternions now in same frame)
        ori_err = quat_error_magnitude(ee_quat_w, self.goal_quat)
        action_penalty = torch.sum(self.actions **2, dim=-1)
        
        # Elbow above table: penalize when forearm_link z drops below table + margin
        elbow_z = self._robot.data.body_pos_w[:, self._elbow_body_idx, 2]
        elbow_below = torch.relu(self._table_z + self.cfg.elbow_min_height - elbow_z)

        rewards = (
            self.cfg.reward_position * pos_rew +
            self.cfg.penalty_position * pos_err + 
            self.cfg.penalty_orientation * ori_err + 
            self.cfg.action_penalty_scale * action_penalty +
            self.cfg.elbow_penalty_scale * elbow_below
        )

        self.extras["log"].update({
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "pos_error": pos_err.mean().item(),
            "pos_rew": pos_rew.mean().item(),
            "ori_error": ori_err.mean().item(),
            "action_penalty": action_penalty.mean().item(),
            "elbow_below_table": elbow_below.mean().item(),
        })
        return rewards

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
                    -0.4, 0.4, (len(envs_to_reset), self._robot.num_joints), self.device
                )
                self.q_des[envs_to_reset] = joint_pos
                self._robot.data.joint_pos[envs_to_reset] = joint_pos
                self._robot.data.joint_vel[envs_to_reset] = 0.0
                self._robot.set_joint_position_target(self.q_des[envs_to_reset], env_ids=envs_to_reset)
        
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
        
        # Orientation sampling per request:
        # Start from orientation with z pointing downward (180° about X),
        # then apply Rx in ±25°, then Rz in ±25°, then final Rz in ±180°.
        q_down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(n, -1)

        max_tilt = 25.0 * torch.pi / 180.0
        ax = torch.empty(n, device=self.device).uniform_(-max_tilt, max_tilt)
        az_small = torch.empty(n, device=self.device).uniform_(-max_tilt, max_tilt)
        az_big = torch.empty(n, device=self.device).uniform_(-torch.pi, torch.pi)

        qx = torch.stack([torch.cos(ax * 0.5), torch.sin(ax * 0.5), torch.zeros(n, device=self.device), torch.zeros(n, device=self.device)], dim=1)
        qz_small = torch.stack([torch.cos(az_small * 0.5), torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), torch.sin(az_small * 0.5)], dim=1)
        qz_big = torch.stack([torch.cos(az_big * 0.5), torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), torch.sin(az_big * 0.5)], dim=1)

        # Compose: q = Rz_big * Rz_small * Rx * q_down
        q = quat_mul(qz_big, quat_mul(qz_small, quat_mul(qx, q_down)))
        self.goal_quat[env_ids] = q / q.norm(dim=1, keepdim=True)

@configclass
class PoseOrientationNoGripperV1Cfg(PoseOrientationNoGripper):

    robot = get_robot_cfg(RobotType.GRIPPER_TCP_NO_ACTUATION, "/World/envs/env_.*/ur5e") # dans la config c'est le robot sans gripper
    
   # Frame transformer to compute TCP pose relative to table center
    frame_transformer = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Table/woodworking_table",
        source_frame_offset=OffsetCfg(
            pos=(TABLE_DEPTH / 2.0, TABLE_WIDTH / 2.0, TABLE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/ur5e/ur5e/wrist_3_link",

                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.15),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
        ],
    )

class PoseOrientationNoGripperV1(PoseOrientationNoGripperV0):
    cfg: PoseOrientationNoGripperV1Cfg

    def __init__(self, cfg: PoseOrientationNoGripperV1Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("Init of V1 complete.")



