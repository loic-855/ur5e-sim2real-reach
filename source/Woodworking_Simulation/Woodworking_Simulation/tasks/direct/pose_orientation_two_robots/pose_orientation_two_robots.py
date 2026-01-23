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
from isaaclab.utils.math import sample_uniform, quat_conjugate, quat_mul, quat_apply

#path constants
REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"


"""
The script implemented a pose and orientation control task with the gripper and screwdriver arms.
The architecture is a centralized policy controlling both arms simultaneously.
The controller uses the joint space to command the two arms.
"""


@configclass
class PoseOrientationTwoRobotsCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    decimation = 2
    # space definition:
    # 8 for gripper arm, 7 for screwdriver arm
    action_space = 15 
    """ obstervation space breakdown:
    obs = [
    # Robot 1 state
    ee1_pos (3)
    ee1_quat (4)
    ee1_lin_vel (3)
    ee1_ang_vel (3)
    joint_pos_1 (8)
    joint_vel_1 (8)

    # Robot 2 state
    ee2_pos (3)
    ee2_quat (4)
    ee2_lin_vel (3)
    ee2_ang_vel (3)
    joint_pos_2 (7)
    joint_vel_2 (7)

    # Relational information
    ee2_pos_relative_to_ee1 (3)
    ee2_quat_relative_to_ee1 (4)

    # Goals
    goal_gripper_pos (3)
    goal_gripper_quat (4)
    goal_screwdriver_pos (3)
    goal_screwdriver_quat (4)

    # Contacts  
    contact_robot1 (1)
    contact_robot2 (1)

    # Last action
    last_action_robot1 (8)
    last_action_robot2 (7) ]
    total 92 dim with contacts"""
    observation_space = 92
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True
    )
    # robots
    gripper_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e_gripper_tcp",
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(USD_FILES_DIR / "ur5e_gripper_tcp.usd"),
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

    screwdriver_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e_screwdriver_tcp",
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(USD_FILES_DIR / "ur5e_screwdriver_tcp.usd"),
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
                "wrist_2_joint": 1.57,
                "wrist_3_joint": 0.0,
                "joint0": 0.0,
            },
            pos=(0.72, 1.12, 0.842),
            rot=(0.0, 0.0, 0.0, 1.0),
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
            "screwdriver_action": ImplicitActuatorCfg(
                joint_names_expr = [
                    "joint0",
                    ],
                damping=20, stiffness=100),
        }
    )

    GRIPPER_TCP_OFFSET = (0.0, 0.0, 0.136)
    SCREWDRIVER_TCP_OFFSET = (0.0, -0.09685, -0.1665)
    ROBOT_BASE_OFFSET = (0.64, 1.04, 0.0)

    # Table asset placement: Width = 1.2m, Depth = 0.8m, Height = 0.842m
    table = sim_utils.UsdFileCfg(
        usd_path=str(REPO_ROOT / "USD_files" / "woodworking_table.usd")
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

    #reward scale
    ee_position_tracking = 1.0
    ee_orientation_tracking = 0.5
    action_penalty = -0.0001
    collision_penalty = -50.0

class PoseOrientationTwoRobotsV0(DirectRLEnv):
    cfg: PoseOrientationTwoRobotsCfg

    def __init__(self, cfg: PoseOrientationTwoRobotsCfg, render_mode: str | None = None, **kwargs):
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)

        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs
        self.env_origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32)
        #print("Available body names:", self._robot.body_names)

        stage = get_current_stage()
        #get robots base position
        robot_grip_base = self._get_env_local_pose(
            self.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp/ur5e/base_link")), # type: ignore
            self.device, # type: ignore
        )
        self.robot_grip_base = robot_grip_base[:3].to(self.device)
        self.robot_screw_base = robot_grip_base[:3].to(self.device) + torch.tensor(self.cfg.ROBOT_BASE_OFFSET, device=self.device)

        #get end effector indices
        print("Available body names:", self._robot_grip.body_names)
        self._ee_index_grip = self._robot_grip.body_names.index("base_link_0")
        self._ee_index_screw = self._robot_screw.body_names.index("link0")

        # Define TCP offsets
        self.gripper_tcp_offset = torch.tensor(self.cfg.GRIPPER_TCP_OFFSET, device=self.device)
        self.screwdriver_tcp_offset = torch.tensor(self.cfg.SCREWDRIVER_TCP_OFFSET, device=self.device)

        #get some infos about joints
        self.robot_grip_dof_lower_limits = self._robot_grip.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_grip_dof_upper_limits = self._robot_grip.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_grip_dof_speed_scales = torch.ones_like(self.robot_grip_dof_lower_limits)
        self.robot_screw_dof_lower_limits = self._robot_screw.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_screw_dof_upper_limits = self._robot_screw.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_screw_dof_speed_scales = torch.ones_like(self.robot_screw_dof_lower_limits)

        self.robot_grip_dof_targets = self._robot_grip.data.joint_pos.clone()
        self.robot_screw_dof_targets = self._robot_screw.data.joint_pos.clone()

        # define goal variables
        self.goal_gripper_pos = torch.zeros((self._num_envs, 3), device=self.device)
        self.goal_gripper_quat = torch.zeros((self._num_envs, 4), device=self.device)
        self.goal_screwdriver_pos = torch.zeros((self._num_envs, 3), device=self.device)
        self.goal_screwdriver_quat = torch.zeros((self._num_envs, 4), device=self.device)
        
        self.actions = torch.zeros((self._num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32) # type: ignore
        self._sample_goal()
        stage = get_current_stage




    def _setup_scene(self):
    
        self._robot_grip = Articulation(self.cfg.gripper_robot)
        self.scene.articulations["gripper_robot"] = self._robot_grip
        self._robot_screw = Articulation(self.cfg.screwdriver_robot)
        self.scene.articulations["screwdriver_robot"] = self._robot_screw

        #self._frame_transformer = FrameTransformer(self.cfg.frame_transformer)
        #self.scene.sensors["frame_transformer"] = self._frame_transformer

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

         # Initialize all markers
        num_envs = self.scene.cfg.num_envs
        n_markers = 2 * num_envs
        init_pos = torch.zeros((n_markers, 3), device=self.device)
        init_ori = torch.tensor([0, 0, 0, 1], device=self.device).repeat(n_markers, 1)
        marker_indices = torch.arange(n_markers, dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(init_pos, init_ori, marker_indices=marker_indices)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().to(self.device).clamp(-1.0, 1.0)

        actions_grip = self.actions[:, :8]
        actions_screw = self.actions[:, 8:]

        increments_grip  = (
            self.robot_grip_dof_speed_scales.unsqueeze(0)
            * self.dt
            * self.cfg.dof_velocity_scale
            * actions_grip
            * self.cfg.action_scale
        )
        increments_screw  = (
            self.robot_screw_dof_speed_scales.unsqueeze(0)
            * self.dt
            * self.cfg.dof_velocity_scale
            * actions_screw
            * self.cfg.action_scale
        )
        targets_grip  = self.robot_grip_dof_targets + increments_grip
        self.robot_grip_dof_targets[:] = torch.clamp(
            targets_grip,
            self.robot_grip_dof_lower_limits.unsqueeze(0),
            self.robot_grip_dof_upper_limits.unsqueeze(0),
        )
        targets_screw  = self.robot_screw_dof_targets + increments_screw
        self.robot_screw_dof_targets[:] = torch.clamp(
            targets_screw,
            self.robot_screw_dof_lower_limits.unsqueeze(0),
            self.robot_screw_dof_upper_limits.unsqueeze(0),
        )

        # Keep the marker always at the goal pose
        # Gripper markers: 0..num_envs-1
        # Screwdriver markers: num_envs..2*num_envs-1
        pos_grip = self.goal_gripper_pos + self.env_origins
        pos_screw = self.goal_screwdriver_pos + self.env_origins
        
        all_pos = torch.cat([pos_grip, pos_screw], dim=0)
        all_quat = torch.cat([self.goal_gripper_quat, self.goal_screwdriver_quat], dim=0)
        
        self.goal_marker.visualize(all_pos, all_quat)

    def _apply_action(self):
        self._robot_grip.set_joint_position_target(self.robot_grip_dof_targets)
        self._robot_screw.set_joint_position_target(self.robot_screw_dof_targets)

    def _get_observations(self): # type: ignore
        # Robot 1 (gripper) state
        ee_grip_lin_vel = self._robot_grip.data.body_lin_vel_w[:, self._ee_index_grip, :]
        ee_grip_ang_vel = self._robot_grip.data.body_ang_vel_w[:, self._ee_index_grip, :]
        joint_pos_grip = self._robot_grip.data.joint_pos
        joint_vel_grip = self._robot_grip.data.joint_vel
        
        # Get TCP pose from transformer
        ee_grip_link_pos_w = self._robot_grip.data.body_pos_w[:, self._ee_index_grip, :]
        ee_grip_tcp_quat_w = self._robot_grip.data.body_quat_w[:, self._ee_index_grip, :]
        
        # Apply TCP offset
        ee_grip_tcp_pos_w = ee_grip_link_pos_w + quat_apply(ee_grip_tcp_quat_w, self.gripper_tcp_offset.repeat(self._num_envs, 1))
        
        ee_grip_pos_local = ee_grip_tcp_pos_w - (self.robot_grip_base + self.env_origins)

        # Robot 2 (screwdriver) state
        ee_screw_lin_vel = self._robot_screw.data.body_lin_vel_w[:, self._ee_index_screw, :]
        ee_screw_ang_vel = self._robot_screw.data.body_ang_vel_w[:, self._ee_index_screw, :]
        joint_pos_screw = self._robot_screw.data.joint_pos
        joint_vel_screw = self._robot_screw.data.joint_vel
        
        # Get TCP pose from transformer
        ee_screw_link_pos_w = self._robot_screw.data.body_pos_w[:, self._ee_index_screw, :]
        ee_screw_tcp_quat_w = self._robot_screw.data.body_quat_w[:, self._ee_index_screw, :]
        
        # Apply TCP offset
        ee_screw_tcp_pos_w = ee_screw_link_pos_w + quat_apply(ee_screw_tcp_quat_w, self.screwdriver_tcp_offset.repeat(self._num_envs, 1))
        
        ee_screw_pos_local = ee_screw_tcp_pos_w - (self.robot_screw_base + self.env_origins)

        obs = torch.cat(
            [
            # Robot 1 state
            ee_grip_pos_local,
            ee_grip_tcp_quat_w,
            ee_grip_lin_vel,
            ee_grip_ang_vel,
            joint_pos_grip,
            joint_vel_grip,
            # Robot 2 state
            ee_screw_pos_local,
            ee_screw_tcp_quat_w,
            ee_screw_lin_vel,
            ee_screw_ang_vel,
            joint_pos_screw,
            joint_vel_screw,
            # Relational information
            ee_screw_pos_local - ee_grip_pos_local,
            quat_mul(quat_conjugate(ee_grip_tcp_quat_w), ee_screw_tcp_quat_w),
            # Goals
            self.goal_gripper_pos,
            self.goal_gripper_quat,
            self.goal_screwdriver_pos,
            self.goal_screwdriver_quat,
            # Contacts
            #is_colliding_grip.float(),
            #is_colliding_screw.float(),
            # Last action
            self.actions[:, :8],
            self.actions[:, 8:],
            ],
            dim = 1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # -- Get current state
        # Gripper
        ee_grip_link_pos_w = self._robot_grip.data.body_pos_w[:, self._ee_index_grip, :]
        ee_grip_tcp_quat_w = self._robot_grip.data.body_quat_w[:, self._ee_index_grip, :]
        
        # Apply TCP offset
        ee_grip_tcp_pos_w = ee_grip_link_pos_w + quat_apply(ee_grip_tcp_quat_w, self.gripper_tcp_offset.repeat(self._num_envs, 1))
        
        # Relative to env origin
        ee_grip_tcp_pos = ee_grip_tcp_pos_w - self.env_origins
        
        # Screwdriver
        ee_screw_link_pos_w = self._robot_screw.data.body_pos_w[:, self._ee_index_screw, :]
        ee_screw_tcp_quat_w = self._robot_screw.data.body_quat_w[:, self._ee_index_screw, :]
        
        # Apply TCP offset
        ee_screw_tcp_pos_w = ee_screw_link_pos_w + quat_apply(ee_screw_tcp_quat_w, self.screwdriver_tcp_offset.repeat(self._num_envs, 1))
        
        # Relative to env origin
        ee_screw_tcp_pos = ee_screw_tcp_pos_w - self.env_origins

        # -- Distance to goals
        d_grip = torch.norm(ee_grip_tcp_pos - self.goal_gripper_pos, dim=-1)
        d_screw = torch.norm(ee_screw_tcp_pos - self.goal_screwdriver_pos, dim=-1)

        # -- Orientation error
        # Gripper
        dot_grip = torch.sum(ee_grip_tcp_quat_w * self.goal_gripper_quat, dim=-1)
        angle_grip = 2.0 * torch.acos(torch.clamp(torch.abs(dot_grip), min=-1.0, max=1.0))
        # Screwdriver
        dot_screw = torch.sum(ee_screw_tcp_quat_w * self.goal_screwdriver_quat, dim=-1)
        angle_screw = 2.0 * torch.acos(torch.clamp(torch.abs(dot_screw), min=-1.0, max=1.0))

        # -- Rewards Calculation
        rewards = torch.zeros(self._num_envs, device=self.device)

        # 1. Position Tracking (L2 Norm Error)
        rew_pos = - (d_grip + d_screw)
        rewards += self.cfg.ee_position_tracking * rew_pos

        # 2. Orientation Tracking (L2-like Error)
        rew_rot = - (angle_grip + angle_screw)
        rewards += self.cfg.ee_orientation_tracking * rew_rot

        # 3. Action Penalty (Regularization)
        # Encourages smooth motion and prevents unnecessary jitter
        rewards += self.cfg.action_penalty * torch.sum(self.actions ** 2, dim=-1)

        # 4. Proximity / Collision with other robot
        # DISABLED: The previous implementation penalized proximity regardless of context,
        # causing robots to maximize distance (run away) instead of solving the task.
        # d_robots = torch.norm(ee_grip_tcp_pos_w - ee_screw_tcp_pos_w, dim=-1)
        # rewards -= 0.5 * torch.exp(-d_robots / 0.2)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # We do not terminate on collision, relying on the reward penalty instead.
        # This prevents the "constant resetting" loop and allows the agent to learn to move away.
        died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor): # type: ignore
        super() ._reset_idx(env_ids) # type: ignore
        env_ids = env_ids.to(self.device, dtype=torch.long)
        # robot state initialization with some noise for gripper
        joint_pos_grip = self._robot_grip.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot_grip.num_joints),
            self.device,
        )
        self.robot_grip_dof_targets[env_ids] = joint_pos_grip
        self._robot_grip.data.joint_pos[env_ids] = joint_pos_grip
        self._robot_grip.data.joint_vel[env_ids] = 0.0

        # robot state initialization with some noise for screwdriver
        joint_pos_screw = self._robot_screw.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot_screw.num_joints),
            self.device,
        )
        self.robot_screw_dof_targets[env_ids] = joint_pos_screw
        self._robot_screw.data.joint_pos[env_ids] = joint_pos_screw
        self._robot_screw.data.joint_vel[env_ids] = 0.0

        self.actions[env_ids] = 0.0

        self._robot_grip.set_joint_position_target(self.robot_grip_dof_targets)
        self._robot_screw.set_joint_position_target(self.robot_screw_dof_targets)
        self._sample_goal(env_ids)

        # Move marker to the goal for gripper
        self.goal_marker.visualize(
            self.goal_gripper_pos[env_ids] + self.env_origins[env_ids], 
            self.goal_gripper_quat[env_ids], 
            marker_indices=env_ids
        )

        # Move marker to the goal for screwdriver
        self.goal_marker.visualize(
            self.goal_screwdriver_pos[env_ids] + self.env_origins[env_ids], 
            self.goal_screwdriver_quat[env_ids], 
            marker_indices=env_ids + self._num_envs
        )

    def _sample_goal(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        num = env_ids.shape[0]

        # --- Sample Goal for Gripper Robot ---
        offsets_grip = torch.empty((num, 3), device=self.device)
        # Adjust ranges based on workspace limits relative to robot base
        offsets_grip[:, 0].uniform_(-0.08, 0.72/2)
        offsets_grip[:, 1].uniform_(-0.08, 1.12/2)
        offsets_grip[:, 2].uniform_(0.0, 1.0)

        self.goal_gripper_pos[env_ids] = self.robot_grip_base + offsets_grip

        delta_quat_grip = torch.randn(num, 4, device=self.device)
        delta_quat_grip = delta_quat_grip / torch.norm(delta_quat_grip, dim=1, keepdim=True)
        self.goal_gripper_quat[env_ids] = delta_quat_grip

        # --- Sample Goal for Screwdriver Robot ---
        offsets_screw = torch.empty((num, 3), device=self.device)
        # Adjust ranges based on workspace limits relative to robot base
        offsets_screw[:, 0].uniform_(-0.72/2, 0.08)
        offsets_screw[:, 1].uniform_(-1.12/2, 0.08)
        offsets_screw[:, 2].uniform_(0.0, 1.0)

        self.goal_screwdriver_pos[env_ids] = self.robot_screw_base + offsets_screw

        delta_quat_screw = torch.randn(num, 4, device=self.device)
        delta_quat_screw = delta_quat_screw / torch.norm(delta_quat_screw, dim=1, keepdim=True)
        self.goal_screwdriver_quat[env_ids] = delta_quat_screw

    @staticmethod
    def _get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device): # type: ignore
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