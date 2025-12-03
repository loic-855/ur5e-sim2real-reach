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
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz

 #path constants
REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"
TABLE_ASSET_PATH = (REPO_ROOT/"USD_files"/"woodworking_table.usd")

"""
The script defines a grasping task for the gripper robot and a position holding task for the screwdriver robot.
The architecture is a centralized policiy controlling both arms simultaneously.
The controller uses the task space to command the two arms.
"""
## le repère est au milieu de la table sur le vrai setup, changer ca encore.
@configclass
class GraspingSingleRobotCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    decimation = 2

    # Action space is now 13D: 6 arm R1 + 6 arm R2 + 1 gripper
    action_space = 8
    observation_space = 19 #not sure about 19
    # = 3+3+4+3+3 + 3+3+4 + 3+1+1 + 13 = 46D

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True
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
                "shoulder_pan_joint": 0.785,
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
                damping=50, stiffness=700),
            "wrist_action": ImplicitActuatorCfg(    
                joint_names_expr = [
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                    ],
                    damping=30, stiffness=300),
            "gripper_action": ImplicitActuatorCfg(
                joint_names_expr = [
                    "left_finger_joint", # Only active joint
                    ],
                damping=14, stiffness=80),
        }
    )
    
    """ gripper_tcp_path = "/World/envs/{ENV_REGEX_NS}/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/tcp"
    GRIPPER_TCP_OFFSET = (0.0, 0.0, 0.136)
    SCREWDRIVER_TCP_OFFSET = (0.0, -0.09685, -0.1665)
    ROBOT_BASE_OFFSET = (0.64, 1.04, 0.0)
 """
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

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    lift_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


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

        self.g_robot_dof_speed_scales = torch.ones_like(self.g_robot_dof_lower_limits)
        self.g_robot_dof_speed_scales[self._g_robot.find_joints("left_finger_joint")[0]] = 0.1
        self.g_robot_dof_speed_scales[self._g_robot.find_joints("right_finger_joint")[0]] = 0.1

        self.g_robot_dof_targets = torch.zeros((self.num_envs, self._g_robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/ur5e/wrist_3_link")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/left_finger_link")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/right_finger_link")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        g_robot_local_grasp_pose_rot, g_robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )

        g_robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.g_robot_local_grasp_pos = g_robot_local_pose_pos.repeat((self.num_envs, 1))
        self.g_robot_local_grasp_rot = g_robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        #the grasp pose is where we want to grip the block relative to the block frame
        w_block_local_grasp_pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.w_block_local_grasp_pos = w_block_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.w_block_local_grasp_rot = w_block_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_plane_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.w_block_plane_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.w_block_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._g_robot.find_bodies("wrist_3_link")[0][0]
        self.left_finger_link_idx = self._g_robot.find_bodies("left_finger_link")[0][0]
        self.right_finger_link_idx = self._g_robot.find_bodies("right_finger_link")[0][0]
        # self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0] no need to initialize for the block

        self.w_robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.w_robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.w_block_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.w_block_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self._g_robot = Articulation(self.cfg.gripper_robot)
        # self._cabinet = Articulation(self.cfg.cabinet) no need the block is not articulated
        self.scene.articulations["gripper_robot"] = self._g_robot
        # self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

         # Create wooden block (object to grasp)
        self._w_block = RigidObject(self.cfg.wooden_block)

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
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.g_robot_dof_targets + self.g_robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.g_robot_dof_targets[:] = torch.clamp(targets, self.g_robot_dof_lower_limits, self.g_robot_dof_upper_limits)

    def _apply_action(self):
        self._g_robot.set_joint_position_target(self.g_robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        #self.debug_tensor_sizes("_get_dones outputs:", _w_block_body_pos_cul_z=self._w_block.data.root_pos_w[:, 2])
        terminated = self._w_block.data.root_pos_w[:, 2] > 0.25 + 0.842 # table height
        
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._g_robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._g_robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self._w_block.data.root_pos_w,
            self.w_robot_grasp_pos,
            self.w_block_grasp_pos,
            self.w_robot_grasp_rot,
            self.w_block_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_plane_axis,
            self.w_block_plane_axis,
            self.gripper_up_axis,
            self.w_block_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._g_robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._g_robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._g_robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.g_robot_dof_lower_limits, self.g_robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._g_robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._g_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # wooden block state, reset wooden block position with randomness
        pos = torch.tensor([0.4, 0.6, 0.842 + 0.015 + 0.001], device=self.device).repeat(len(env_ids), 1) + self.env_origins[env_ids]
        pos += sample_uniform(
            torch.tensor([-0.05, -0.05, 0.0], device=self.device),
            torch.tensor([0.05, 0.05, 0.0], device=self.device),
            (len(env_ids), 3),
            self.device,
        )
        random_yaw = (torch.rand(len(env_ids), device=self.device) * 10 - 5) * torch.pi / 180
        euler_angles = torch.zeros((len(env_ids), 3), device=self.device)
        euler_angles[:, 2] = random_yaw  # Set yaw (Z-axis rotation)
        ori = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        w_block_local_grasp_pose = torch.cat([pos, ori], dim=1)
        self._w_block.write_root_pose_to_sim(w_block_local_grasp_pose)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._g_robot.data.joint_pos - self.g_robot_dof_lower_limits)
            / (self.g_robot_dof_upper_limits - self.g_robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.w_block_grasp_pos - self.w_robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled, #7
                self._g_robot.data.joint_vel * self.cfg.dof_velocity_scale, #7
                to_target, #3
                #self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),
                #self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),
                self._w_block.data.root_pos_w[:, 2].unsqueeze(-1), #1
                self._w_block.data.root_vel_w[:, 2].unsqueeze(-1), #1
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._g_robot._ALL_INDICES

        hand_rot = self._g_robot.data.body_quat_w[env_ids, self.hand_link_idx]
        hand_pos = self._g_robot.data.body_pos_w[env_ids, self.hand_link_idx]
        w_block_rot = self._w_block.data.body_quat_w[env_ids].squeeze(1)
        w_block_pos = self._w_block.data.root_pos_w[env_ids]
    
        (
            self.w_robot_grasp_rot[env_ids],
            self.w_robot_grasp_pos[env_ids],
            self.w_block_grasp_rot[env_ids],
            self.w_block_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
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
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_plane_axis,
        w_block_plane_axis,
        gripper_up_axis,
        w_block_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        lift_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_plane_axis)
        axis2 = tf_vector(drawer_grasp_rot, w_block_plane_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, w_block_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        lift_reward = w_block_pos[:, 2]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + lift_reward_scale * lift_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "lift_reward": (lift_reward_scale * lift_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for lifting the block properly
        rewards = torch.where(w_block_pos[:, 2] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(w_block_pos[:, 2] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(w_block_pos[:, 2] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot, #quat
        hand_pos, # xyz
        ur5e_local_grasp_rot, #quat
        ur5e_local_grasp_pos, #xyz
        block_rot, 
        block_pos,
        block_local_grasp_rot,
        block_local_grasp_pos,
    ):
        
        """ self.debug_tensor_sizes(
            "_compute_grasp_transforms inputs:",
            hand_rot=hand_rot,
            hand_pos=hand_pos,
            ur5e_local_grasp_rot=ur5e_local_grasp_rot,
            ur5e_local_grasp_pos=ur5e_local_grasp_pos,
            block_rot=block_rot,
            block_pos=block_pos,
            block_local_grasp_rot=block_local_grasp_rot,
            block_local_grasp_pos=block_local_grasp_pos,
        ) """

        global_ur5e_rot, global_ur5e_pos = tf_combine(
            hand_rot, hand_pos, ur5e_local_grasp_rot, ur5e_local_grasp_pos
        )
        global_block_rot, global_block_pos = tf_combine(
            block_rot, block_pos, block_local_grasp_rot, block_local_grasp_pos
        )

        return global_ur5e_rot, global_ur5e_pos, global_block_rot, global_block_pos

    def debug_tensor_sizes(self, message=None, **kwargs):
        """Debug function to print the sizes of multiple tensors."""
        if message is not None:
            print(message)
        for var_name, tensor in kwargs.items():
            print(f"[DEBUG] {var_name} is of size {tensor.size()}")