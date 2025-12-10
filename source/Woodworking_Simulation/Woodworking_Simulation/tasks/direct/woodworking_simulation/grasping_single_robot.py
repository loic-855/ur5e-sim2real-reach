# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from pathlib import Path
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, euler_xyz_from_quat
    
 #path constants
REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"
TABLE_ASSET_PATH = (REPO_ROOT/"USD_files"/"woodworking_table.usd")

# Physics constants
TABLE_HEIGHT = 0.842
BLOCK_THICKNESS = 0.015
BLOCK_MARGIN = 0.003
TABLE_WIDTH = 1.2 
TABLE_DEPTH = 0.8

"""
The script defines a grasping task for the gripper robot and a position holding task for the screwdriver robot.
The architecture is a centralized policiy controlling both arms simultaneously.
The controller uses the task space to command the two arms.
"""

@configclass
class GraspingSingleRobotCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.333  # 10 seconds
    decimation = 2

    # Action space is 7D: 6 joints for the arm + 1 prismatic actuator for the left finger (right finger is mimicked in USD)
    action_space = 7
    # Observation space is 27D: 7 scaled joint positions + 7 joint velocities + 3 to_target vector + 3 block position (xyz) + last actions.
    observation_space = 27
    state_space = 0  # Not used in this task

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )
    
    # robots
    gripper_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e_gripper_tcp_small",
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(USD_FILES_DIR / "ur5e_gripper_tcp_small.usd"),
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
                "left_finger_joint": 0.0,   #[0.0, 0.019] open-close
            },
            pos=(0.08, 0.08, 0.842),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "shoulder_action": ImplicitActuatorCfg(    
                joint_names_expr = [
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    ],
                damping=60, stiffness=800),  # UR5e shoulders: heavy joints
            "wrist_action": ImplicitActuatorCfg(    
                joint_names_expr = [
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                    ],
                    damping=35, stiffness=350),  # UR5e wrist: lighter joints
            "gripper_action": ImplicitActuatorCfg(
                joint_names_expr = [
                    "left_finger_joint",
                    ],
                damping=35, stiffness=200),  # Realistic for OnRobot 2FG7 (140N grip)
        }
    )
    
    # Table asset placement: Width = 1.2m, Depth = 0.8m, Height = 0.842m
    table = sim_utils.UsdFileCfg(
        usd_path=str(TABLE_ASSET_PATH),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    #block
    wooden_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wooden_block",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.06, 0.03),           
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                disable_gravity=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.27, 0.07)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )


    # Grasp pose markers for visualization
    robot_grasp_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/robot_grasp_markers",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.03, 0.03, 0.03),
            ),
        },
    )
    
    block_grasp_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/block_grasp_markers",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.03, 0.03, 0.03),
            ),
        },
    )

    left_finger_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/left_finger_markers",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.02, 0.02, 0.02),
            ),
        },
    )

    right_finger_marker = VisualizationMarkersCfg(
        prim_path="/Visuals/right_finger_markers",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.02, 0.02, 0.02),
            ),
        },
    )

    action_scale = 4.0
    dof_velocity_scale = 0.2

    # visualization
    enable_debug_markers = True
    num_debug_markers = 15  # Number of envs to show markers for (starting from env_0)

    # reward scales
    dist_reward_scale = 1.2    # Increased to encourage approaching the block
    dist_penalty_scale = 1  # New penalty for distance to encourage closeness
    orientation_reward_scale = 0.8       # Reward for matching y-axis and opposed z-axis alignement
    finger_distance_reward_scale = 2  # New reward for finger closing when near the block
    lift_reward_scale = 5  
    action_penalty_scale = [0.0001, 0.005] # Curriculum on
    speed_penalty_scale = [0.0001, 0.005]


class GraspingSingleRobotV0(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: GraspingSingleRobotCfg

    def __init__(self, cfg: GraspingSingleRobotCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize episode counter for curriculum learning
 

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
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

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.env_origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32)

        # create auxiliary variables for computing applied action, observations and rewards
        self.g_robot_dof_lower_limits = self._g_robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.g_robot_dof_upper_limits = self._g_robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # Cache joint indices and define which DOFs are directly controlled (exclude mimicked right finger)
        self._left_finger_joint_idx = self._g_robot.find_joints("left_finger_joint")[0][0]
        self._right_finger_joint_idx = self._g_robot.find_joints("right_finger_joint")[0][0]
        self.control_joint_indices = torch.tensor(
            [i for i in range(self._g_robot.num_joints) if i != self._right_finger_joint_idx],
            device=self.device,
            dtype=torch.long,
        )

        self.g_robot_dof_speed_scales = torch.ones_like(self.g_robot_dof_lower_limits)
        self.g_robot_dof_speed_scales[self._left_finger_joint_idx] = 0.5
        self.g_robot_dof_speed_scales[self._right_finger_joint_idx] = 0.5

        self.g_robot_dof_targets = torch.zeros((self.num_envs, self._g_robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/ur5e/wrist_3_link")),
            self.device, # type: ignore
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/left_finger_link")),
            self.device, # type: ignore
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/right_finger_link")),
            self.device, # type: ignore
        )
        self.robot_base_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/ur5e/base_link")),
            self.device, # type: ignore
        )


        # Compute the grasp pose for the fingers. Apply small offsets to get the grasp points at the finger tips
        # Use the same approach as for the gripper: inverse + combine to get local offset
        lfinger_inv_rot, lfinger_inv_pos = tf_inverse(lfinger_pose[3:7], lfinger_pose[0:3])
        lfinger_grasp_pose_rot, lfinger_grasp_pose_pos = tf_combine(
            lfinger_inv_rot, lfinger_inv_pos, lfinger_pose[3:7], lfinger_pose[0:3]
        )
        # Add offset if needed (default: [0, 0, 0])
        lfinger_grasp_pose_pos += torch.tensor([0.015, 0.03, 0.03], device=self.device)
        
        rfinger_inv_rot, rfinger_inv_pos = tf_inverse(rfinger_pose[3:7], rfinger_pose[0:3])
        rfinger_grasp_pose_rot, rfinger_grasp_pose_pos = tf_combine(
            rfinger_inv_rot, rfinger_inv_pos, rfinger_pose[3:7], rfinger_pose[0:3]
        )
        # Add offset if needed (default: [0, 0, 0])
        rfinger_grasp_pose_pos += torch.tensor([0.015, 0.03, 0.03], device=self.device)
        
        self.lfinger_local_grasp_rot = lfinger_grasp_pose_rot.repeat((self.num_envs, 1))
        self.lfinger_local_grasp_pos = lfinger_grasp_pose_pos.repeat((self.num_envs, 1))
        self.rfinger_local_grasp_rot = rfinger_grasp_pose_rot.repeat((self.num_envs, 1))
        self.rfinger_local_grasp_pos = rfinger_grasp_pose_pos.repeat((self.num_envs, 1))

        # Compute the grasp pose for the robot gripper
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        g_robot_local_grasp_pose_rot, g_robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )

        g_robot_local_pose_pos += torch.tensor([0.012, 0.03, 0.03], device=self.device)
        self.g_robot_local_grasp_pos = g_robot_local_pose_pos.repeat((self.num_envs, 1))
        self.g_robot_local_grasp_rot = g_robot_local_grasp_pose_rot.repeat((self.num_envs, 1))
        
        # Compute the grasp pose for the wooden block (same approach as robot)
        # Get the block's root pose from the USD stage
        block_root_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/wooden_block")),
            self.device, # type: ignore
        )
        
        # The grasp point is at the geometric center of the block
        # Start with the block's root pose (center)
        block_grasp_pose = block_root_pose.clone()
        
        # Compute the inverse transform of the block root
        block_root_inv_rot, block_root_inv_pos = tf_inverse(block_root_pose[3:7], block_root_pose[0:3])
        
        # Combine to get the local grasp pose (currently identity since grasp = center)
        w_block_local_grasp_rot, w_block_local_grasp_pos = tf_combine(
            block_root_inv_rot, block_root_inv_pos, block_grasp_pose[3:7], block_grasp_pose[0:3]
        )
        
        # Add offset from center if needed (default: no offset, grasp at geometric center)
        w_block_local_grasp_pos += torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.w_block_local_grasp_pos = w_block_local_grasp_pos.repeat((self.num_envs, 1))
        self.w_block_local_grasp_rot = w_block_local_grasp_rot.repeat((self.num_envs, 1))

        # Simplify axis initialization
        self.gripper_plane_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.w_block_plane_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.w_block_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        self.hand_link_idx = self._g_robot.find_bodies("wrist_3_link")[0][0]
        self.left_finger_link_idx = self._g_robot.find_bodies("left_finger_link")[0][0]
        self.right_finger_link_idx = self._g_robot.find_bodies("right_finger_link")[0][0]
        
        # Initialize grasp pose tensors
        self.w_robot_lfinger_grasp_rot, self.w_robot_lfinger_grasp_pos = torch.zeros((self.num_envs, 4), device=self.device), torch.zeros((self.num_envs, 3), device=self.device)
        self.w_robot_rfinger_grasp_rot, self.w_robot_rfinger_grasp_pos = torch.zeros((self.num_envs, 4), device=self.device), torch.zeros((self.num_envs, 3), device=self.device)
        self.w_robot_grasp_rot, self.w_robot_grasp_pos = torch.zeros((self.num_envs, 4), device=self.device), torch.zeros((self.num_envs, 3), device=self.device)
        self.w_block_grasp_rot, self.w_block_grasp_pos = torch.zeros((self.num_envs, 4), device=self.device), torch.zeros((self.num_envs, 3), device=self.device)
        
        # Initialize grasp pose visualization markers (if enabled)
        markers_cfg = {
            "_robot_grasp_markers": self.cfg.robot_grasp_marker,
            "_block_grasp_markers": self.cfg.block_grasp_marker,
            "_left_finger_markers": self.cfg.left_finger_marker,
            "_right_finger_markers": self.cfg.right_finger_marker,
        }
        for marker_attr, marker_cfg in markers_cfg.items():
            setattr(self, marker_attr, VisualizationMarkers(marker_cfg) if self.cfg.enable_debug_markers else None)

    def _setup_scene(self):
        self._g_robot = Articulation(self.cfg.gripper_robot)
        self.scene.articulations["gripper_robot"] = self._g_robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create wooden block (object to grasp) and table
        self._w_block = RigidObject(self.cfg.wooden_block)
        self.scene.rigid_objects["wooden_block"] = self._w_block
        self.cfg.table.func(
            "/World/envs/env_0/WoodworkingTable",
            self.cfg.table, 
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        # Only apply actions to controllable joints (exclude mimicked right finger)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        lower = self.g_robot_dof_lower_limits[self.control_joint_indices]
        upper = self.g_robot_dof_upper_limits[self.control_joint_indices]
        deltas = (
            self.g_robot_dof_speed_scales[self.control_joint_indices]
            * self.dt
            * self.actions
            * self.cfg.action_scale
        )

        new_targets = self.g_robot_dof_targets[:, self.control_joint_indices] + deltas
        self.g_robot_dof_targets[:, self.control_joint_indices] = torch.clamp(new_targets, lower, upper)

    def _apply_action(self):
        self._g_robot.set_joint_position_target(self.g_robot_dof_targets)
        #self.print_tensor_values(env_num=0, interval_s=1, **{f"action_{i}": self.actions[:, i] for i in range(7)})
     # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Success termination: block lifted above 0.25m + table height
        success = self._w_block.data.root_pos_w[:, 2] > 0.25 + TABLE_HEIGHT + BLOCK_THICKNESS
        
        # Failure termination: block below table with margin (z < 0.835)
        below_table = self._w_block.data.root_pos_w[:, 2] < TABLE_HEIGHT - 0.007

        #Failure termination: block outside XY bounds, compute local positions for the block in each env
        block_pos_local = self._w_block.data.root_pos_w[:, :3] - self.env_origins[:, :3]
        
        # Check if block is outside XY bounds in local env coordinates
        out_of_bounds = (
            (block_pos_local[:, 0] < 0) | (block_pos_local[:, 0] > TABLE_DEPTH) |
            (block_pos_local[:, 1] < 0) | (block_pos_local[:, 1] > TABLE_WIDTH)
        )

        roll, pitch, yaw = euler_xyz_from_quat(self.w_block_grasp_rot)
        tilted = (torch.abs(roll) > torch.pi / 4) | (torch.abs(pitch) > torch.pi / 4)
            
        terminated = success | below_table | tilted | out_of_bounds
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards for V1. Customize this method to modify rewards."""
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        
        if self.common_step_counter < 50000:
            action_penalty_scale = self.cfg.action_penalty_scale[0]
            speed_penalty_scale = self.cfg.speed_penalty_scale[0]
        else:
            action_penalty_scale = self.cfg.action_penalty_scale[1]
            speed_penalty_scale = self.cfg.speed_penalty_scale[1]

        return self._compute_rewards(
            self.actions,
            self._w_block.data.root_pos_w,
            self.w_robot_grasp_pos,
            self.w_block_grasp_pos,
            self.w_robot_grasp_rot,
            self.w_block_grasp_rot,
            self.w_robot_lfinger_grasp_pos,
            self.w_robot_rfinger_grasp_pos,
            self.gripper_plane_axis,
            self.w_block_plane_axis,
            self.gripper_up_axis,
            self.w_block_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.dist_penalty_scale,
            self.cfg.finger_distance_reward_scale,
            self.cfg.orientation_reward_scale,
            self.cfg.lift_reward_scale,
            action_penalty_scale,
            speed_penalty_scale,
            self._g_robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None): # type: ignore
        super()._reset_idx(env_ids) # type: ignore
        # robot state
        joint_pos = self._g_robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._g_robot.num_joints), # type: ignore
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.g_robot_dof_lower_limits, self.g_robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.g_robot_dof_targets[env_ids] = joint_pos
        self._g_robot.set_joint_position_target(joint_pos, env_ids=env_ids) # type: ignore
        self._g_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids) # type: ignore

        # wooden block state, reset wooden block position with randomness root base pose = tensor([0.0800, 0.0800, 0.8420, 1.0000, 0.0000, 0.0000, 0.0000],
        # Spawn wooden block in a geometric zone: 30-80cm from robot base, 90° sector
        random_bonus = torch.where(torch.rand(len(env_ids), device=self.device) < 0.15, torch.rand(len(env_ids), device=self.device) * 0.01 + 0.01, 0.0) # pyright: ignore[reportArgumentType]
        radius = torch.rand(len(env_ids), device=self.device) * 0.5 + 0.3  # type: ignore # 0.3-0.8m
        angle = torch.rand(len(env_ids), device=self.device) * (torch.pi / 2)  # pyright: ignore[reportArgumentType] # 0-90°
        pos = self.robot_base_pose[0:3].repeat(len(env_ids), 1) # type: ignore
        pos[:, :2] += torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
        pos[:, 2] += BLOCK_MARGIN + BLOCK_THICKNESS / 2.0 + random_bonus
        pos += self.env_origins[env_ids]

        random_yaw = (torch.rand(len(env_ids), device=self.device) * 20 - 10) * torch.pi / 180 # type: ignore
        euler_angles = torch.zeros((len(env_ids), 3), device=self.device) # type: ignore
        euler_angles[:, 2] = random_yaw  # Set yaw (Z-axis rotation)
        ori = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        vel = torch.zeros((len(env_ids), 6), device=self.device) # type: ignore
        self._w_block.write_root_velocity_to_sim(vel, env_ids=env_ids) # type: ignore
        self._w_block.write_root_pose_to_sim(torch.cat([pos, ori], dim=1), env_ids=env_ids) #type: ignore

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        """Get observations for V1. Customize this method to modify observations."""
        dof_pos_scaled = (
            2.0
            * (self._g_robot.data.joint_pos - self.g_robot_dof_lower_limits)
            / (self.g_robot_dof_upper_limits - self.g_robot_dof_lower_limits)
            - 1.0
        )
        # Vector from robot grasp point to block grasp point in
        to_target = self.w_block_grasp_pos - self.w_robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,                                                 # 7D: scaled joint positions [-1, 1]
                self._g_robot.data.joint_vel * self.cfg.dof_velocity_scale,     # 7D: joint velocities (scaled)
                to_target,                                                      # 3D: vector pointing from gripper to block
                self._w_block.data.root_pos_w[:, :3] - self.env_origins[:, :3], # 3D: block position (xyz) in local frame
                self.actions                                                    # 7D: last applied action
            ),
            dim=-1,
        )
        # Total: 7 + 7 + 3 + 3 = 20D observations
        
        # Update grasp pose markers visualization
        if self._robot_grasp_markers is not None:
            n = min(self.cfg.num_debug_markers, self.num_envs)
            self._update_marker_visualization(self._robot_grasp_markers, self.w_robot_grasp_pos, self.w_robot_grasp_rot, n)
        
        if self._block_grasp_markers is not None:
            n = min(self.cfg.num_debug_markers, self.num_envs)
            self._update_marker_visualization(self._block_grasp_markers, self.w_block_grasp_pos, self.w_block_grasp_rot, n)

        if self._left_finger_markers is not None:
            n = min(self.cfg.num_debug_markers, self.num_envs)
            self._update_marker_visualization(self._left_finger_markers, self.w_robot_lfinger_grasp_pos, self.w_robot_lfinger_grasp_rot, n)

        if self._right_finger_markers is not None:
            n = min(self.cfg.num_debug_markers, self.num_envs)
            self._update_marker_visualization(self._right_finger_markers, self.w_robot_rfinger_grasp_pos, self.w_robot_rfinger_grasp_rot, n)
        
        return {"policy": torch.clamp(obs, -5.0, 5.0)}    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._g_robot._ALL_INDICES
        lfinger_rot = self._g_robot.data.body_quat_w[env_ids, self.left_finger_link_idx]
        lfinger_pos = self._g_robot.data.body_pos_w[env_ids, self.left_finger_link_idx]
        rfinger_rot = self._g_robot.data.body_quat_w[env_ids, self.right_finger_link_idx]
        rfinger_pos = self._g_robot.data.body_pos_w[env_ids, self.right_finger_link_idx]
        hand_rot = self._g_robot.data.body_quat_w[env_ids, self.hand_link_idx]
        hand_pos = self._g_robot.data.body_pos_w[env_ids, self.hand_link_idx]
        w_block_rot = self._w_block.data.body_quat_w[env_ids].squeeze(1)
        w_block_pos = self._w_block.data.root_pos_w[env_ids]
    
        (
            self.w_robot_lfinger_grasp_rot[env_ids],
            self.w_robot_lfinger_grasp_pos[env_ids],
            self.w_robot_rfinger_grasp_rot[env_ids],
            self.w_robot_rfinger_grasp_pos[env_ids],
            self.w_robot_grasp_rot[env_ids],
            self.w_robot_grasp_pos[env_ids],
            self.w_block_grasp_rot[env_ids],
            self.w_block_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            lfinger_rot,
            lfinger_pos,
            rfinger_rot,
            rfinger_pos,
            self.lfinger_local_grasp_rot[env_ids],
            self.lfinger_local_grasp_pos[env_ids],
            self.rfinger_local_grasp_rot[env_ids],
            self.rfinger_local_grasp_pos[env_ids],
            hand_rot,
            hand_pos,
            self.g_robot_local_grasp_rot[env_ids],
            self.g_robot_local_grasp_pos[env_ids],
            w_block_rot,
            w_block_pos,
            self.w_block_local_grasp_rot[env_ids],
            self.w_block_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        w_block_pos,
        robot_grasp_pos,
        block_grasp_pos,
        robot_grasp_rot,
        block_grasp_rot,
        robot_lfinger_pos,
        robot_rfinger_pos,
        gripper_plane_axis,
        w_block_plane_axis,
        gripper_up_axis,
        w_block_up_axis,
        num_envs,
        dist_reward_scale,
        dist_penalty_scale,
        finger_distance_reward_scale,
        orientation_reward_scale,
        lift_reward_scale,
        action_penalty_scale,
        speed_penalty_scale,

        joint_positions,
    ):
        """Compute rewards for V1. Override this to modify reward calculation."""
        # Initialize extras log if not present
        if "log" not in self.extras:
            self.extras["log"] = {}

        # Reward the agent for reaching the object using tanh-kernel
        d = torch.norm(block_grasp_pos - robot_grasp_pos, p=2, dim=-1)
        std = 0.02
        dist_reward = dist_reward_scale * (1.0 - torch.tanh(d / std)) - dist_penalty_scale * d

        # Reward the agent for aligning the gripper with the object along z and y axes. Reward for matching the y and opposed z alignement.
        axis1 = tf_vector(robot_grasp_rot, gripper_plane_axis)
        axis2 = tf_vector(block_grasp_rot, w_block_plane_axis)
        axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(block_grasp_rot, w_block_up_axis)

        dot1 = (torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1))
        dot2 = (torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1))

        orientation_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions and speed(summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)
        speed_penalty = torch.sum((self._g_robot.data.joint_vel[:, self.control_joint_indices])**2, dim=-1)

        """ # Reward fingers closing QUAND ils sont proches du bloc
        finger_distance = torch.norm(robot_lfinger_pos[:, :3] - robot_rfinger_pos[:, :3], dim=-1)
        # Active la récompense si distance à bloc < 5 cm PAR ENV
        finger_close_reward = torch.where(
            d < 0.03,
            1.0 / (finger_distance + 1e-6),  # Récompense fermeture
            torch.zeros_like(d)
        ) """
        # Pénalité continue, pas par étapes
        block_on_table = w_block_pos[:, 2] < (TABLE_HEIGHT + BLOCK_THICKNESS + 0.05)
        on_table_penalty = torch.where(block_on_table, 1.0, 0.0)  # Forte

        # Reward the agent for lifting the block (positive when block goes UP)
        lift_height = torch.clamp(w_block_pos[:, 2] - (TABLE_HEIGHT + BLOCK_THICKNESS), min=0.0)
        lift_reward = torch.pow(lift_height, 2)
        
        sucess_bonus = torch.where(self._w_block.data.root_pos_w[:, 2] > 0.25 + TABLE_HEIGHT + BLOCK_THICKNESS, 50.0, 0.0)

        rewards = (
            dist_reward
            + orientation_reward_scale * orientation_reward
            - 0.4 * on_table_penalty
            + lift_reward_scale * lift_reward
            - action_penalty_scale * action_penalty
            - speed_penalty_scale * speed_penalty
            + sucess_bonus
        )

        # bonus for lifting the block properly
        """ rewards = torch.where(lift_height > 0.005, rewards + 1.0, rewards)
        rewards = torch.where(lift_height > 0.015, rewards + 2.0, rewards)
        rewards = torch.where(lift_height > 0.05, rewards + 5.0, rewards)
        rewards = torch.where(lift_height > 0.10, rewards + 10.0, rewards) """

        self.extras["log"].update({
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(),
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "distance to block": d.mean(),
            "orientation_reward": (orientation_reward_scale * orientation_reward).mean(),
            "lift_reward": (lift_reward_scale * lift_reward).mean(),
            "lift_height": lift_height.mean(),
            #"finger_close_reward": finger_close_reward.mean(),
            "on_table_penalty": (-0.4 * on_table_penalty).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "speed_penalty": (-speed_penalty_scale * speed_penalty).mean(),       
            "episode_length": self.episode_length_buf.float().mean(),
            "self.common_step_counter": self.common_step_counter,
            "std": std,

        })

        return rewards

    def _compute_grasp_transforms(
        self,
        lfinger_rot, #quat
        lfinger_pos, # xyz
        rfinger_rot, #quat
        rfinger_pos, # xyz
        lfinger_local_grasp_rot, #quat
        lfinger_local_grasp_pos, #xyz
        rfinger_local_grasp_rot, #quat
        rfinger_local_grasp_pos, #xyz
        hand_rot, #quat
        hand_pos, # xyz
        ur5e_local_grasp_rot, #quat
        ur5e_local_grasp_pos, #xyz
        block_rot, 
        block_pos,
        block_local_grasp_rot,
        block_local_grasp_pos,
    ):
        global_lfinger_rot, global_lfinger_pos = tf_combine(
            lfinger_rot, lfinger_pos, lfinger_local_grasp_rot, lfinger_local_grasp_pos, 
        )
        global_rfinger_rot, global_rfinger_pos = tf_combine(
            rfinger_rot, rfinger_pos, rfinger_local_grasp_rot, rfinger_local_grasp_pos, 
        )
        global_ur5e_rot, global_ur5e_pos = tf_combine(
            hand_rot, hand_pos, ur5e_local_grasp_rot, ur5e_local_grasp_pos
        )
        global_block_rot, global_block_pos = tf_combine(
            block_rot, block_pos, block_local_grasp_rot, block_local_grasp_pos
        )

        return global_lfinger_rot, global_lfinger_pos, global_rfinger_rot, global_rfinger_pos, global_ur5e_rot, global_ur5e_pos, global_block_rot, global_block_pos
    
    def _update_marker_visualization(self, marker, translations, orientations, n):
        """Helper to visualize marker poses."""
        marker_indices = torch.zeros(n, dtype=torch.int32, device=self.device)
        marker.visualize(
            translations=translations[0:n],
            orientations=orientations[0:n],
            marker_indices=marker_indices,
        )

    def debug_tensor_sizes(self, message=None, **kwargs):
        """Debug function to print the sizes of multiple tensors."""
        if message is not None:
            print(message)
        for var_name, tensor in kwargs.items():
            print(f"[DEBUG] {var_name} is of size {tensor.size()}")

    def print_tensor_values(self, env_num=0, interval_s: float = 2.0, force: bool = False, **kwargs):
        """Print tensor values on one line, extracting only numeric values."""
        
        dt = getattr(self, "dt", self.cfg.sim.dt * getattr(self.cfg, "decimation", 1))
        step_interval = max(1, int(round(interval_s / dt))) if interval_s is not None else 1

        if not hasattr(self, "_print_step_counter"):
            self._print_step_counter = 0
        self._print_step_counter += 1

        if not force and (self._print_step_counter % step_interval != 0):
            return

        values_str = ", ".join([f"{name}={tensor[env_num].item():.4f}" for name, tensor in kwargs.items()])
        print(f"[DEBUG] env ({env_num}): {values_str}")

