from __future__ import annotations
from pathlib import Path

import torch
import isaaclab.sim as sim_utils

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_conjugate, quat_mul, quat_from_euler_xyz, axis_angle_from_quat
from isaaclab.assets import RigidObject, RigidObjectCfg

#path constants
REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"
TABLE_ASSET_PATH = (REPO_ROOT/"USD_files"/"woodworking_table.usd")


def collapse_obs_dict(obs_dict: dict, obs_order: list) -> torch.Tensor:
    """Collapse observation dictionary into a single tensor based on obs_order.
    
    Args:
        obs_dict: Dictionary of observation tensors
        obs_order: List of observation keys in desired order
    
    Returns:
        Concatenated observation tensor of shape (num_envs, total_obs_dim)
    """
    obs_list = []
    for obs_name in obs_order:
        if obs_name in obs_dict:
            obs_tensor = obs_dict[obs_name]
            # Flatten if needed
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(-1)
            obs_list.append(obs_tensor)
    return torch.cat(obs_list, dim=-1)


"""
The script defines a grasping task for the gripper robot and a position holding task for the screwdriver robot.
The architecture is a centralized policiy controlling both arms simultaneously.
The controller uses the task space to command the two arms.
"""
OBS_DIM_CFG = {
    # Robot 1 (active grasping)
    "tcp1_pos": 3,
    "tcp1_pos_rel_block": 3,
    "tcp1_quat": 4,
    "tcp1_linvel": 3,
    "tcp1_angvel": 3,
    
    # Robot 2 (passive holding)
    "tcp2_pos": 3,
    "tcp2_pos_rel_hold_target": 3,  # Position error from its hold target
    "tcp2_quat": 4,
    
    # Shared observations
    "block_pos": 3,
    "gripper_width1": 1,
    "gripper_width2": 1,
}

@configclass
class GraspingDualRobot(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    decimation = 2

    obs_order = [
        # Robot 1 (control robot)
        "tcp1_pos_rel_block",
        "tcp1_quat",
        "tcp1_linvel",
        "tcp1_angvel",
        
        # Robot 2 (holding robot)
        "tcp2_pos_rel_hold_target",
        "tcp2_quat",
        
        # Shared
        "block_pos",
        "gripper_width1",
        "gripper_width2",
    ]

    # Action space is now 13D: 6 arm R1 + 6 arm R2 + 1 gripper
    action_space = 13
    observation_space = sum([OBS_DIM_CFG[obs] for obs in obs_order]) + action_space
    # = 3+3+4+3+3 + 3+3+4 + 3+1+1 + 13 = 46D

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=3.0, replicate_physics=True
    )
    
    # IK Controller Settings
    ik_controller = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )

    # robots
    gripper_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e_gripper_tcp",
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
    gripper_tcp_path = "/World/envs/{ENV_REGEX_NS}/ur5e_gripper_tcp_small/onrobot_2fg7_tcp_small/tcp"

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
        usd_path=str(TABLE_ASSET_PATH),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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

    #reward scale
    ee_position_tracking = 1.0
    ee_orientation_tracking = 0.5
    action_penalty = -0.0001
    collision_penalty = -50.0

class GraspingDualRobotV0(DirectRLEnv):
    cfg: GraspingDualRobot

    def __init__(self, cfg: GraspingDualRobot, render_mode: str | None = None, **kwargs):
        self.goal_marker = VisualizationMarkers(cfg.goal_marker)

        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._num_envs = self.scene.cfg.num_envs
        self.env_origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32)

        # Initialize Differential IK Controllers
        self.ik_controller_r1 = DifferentialIKController(self.cfg.ik_controller, num_envs=self.num_envs, device=self.device)
        self.ik_controller_r2 = DifferentialIKController(self.cfg.ik_controller, num_envs=self.num_envs, device=self.device)

        self._init_tensors()
   

    def _init_tensors(self):
        """Initialize tensors for dual robot setup."""
        # Robot 1 (Gripper)
        # Find TCP body index
        tcp1_names = [name for name in self._robot1.body_names if "tcp" in name]
        if tcp1_names:
            self.tcp1_body_idx = self._robot1.body_names.index(tcp1_names[0])
        else:
            # Fallback or error
            self.tcp1_body_idx = self._robot1.body_names.index("base_link_0") # Fallback
            
        self.gripper1_left_idx = self._robot1.body_names.index("left_finger_link")
        self.gripper1_right_idx = self._robot1.body_names.index("right_finger_link")

        # Robot 1 Joint Indices
        r1_arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        r1_gripper_joint_names = ["left_finger_joint"]
        self.r1_arm_indices, _ = self._robot1.find_joints(r1_arm_joint_names)
        self.r1_gripper_indices, _ = self._robot1.find_joints(r1_gripper_joint_names)
        self.r1_joint_indices = self.r1_arm_indices + self.r1_gripper_indices

        # Robot 2 (Screwdriver)
        tcp2_names = [name for name in self._robot2.body_names if "tcp" in name]
        if tcp2_names:
            self.tcp2_body_idx = self._robot2.body_names.index(tcp2_names[0])
        else:
             self.tcp2_body_idx = self._robot2.body_names.index("link0") # Fallback

        # Robot 2 Joint Indices
        r2_arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        r2_tool_joint_names = ["joint0"]
        self.r2_arm_indices, _ = self._robot2.find_joints(r2_arm_joint_names)
        self.r2_tool_indices, _ = self._robot2.find_joints(r2_tool_joint_names)
        self.r2_joint_indices = self.r2_arm_indices + self.r2_tool_indices

        # Robot 2 holding position (stays constant during episode)
        self.hold_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hold_target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Finite-differencing tensors for velocities
        self.prev_tcp1_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_tcp1_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        self.prev_tcp2_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_tcp2_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        # Action scaling factors
        self.pos_action_scale = 0.05  # Scale for position actions
        self.rot_action_scale = 0.5   # Scale for rotation actions
        
        # Observation noise
        self.init_block_pos_noise = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Gripper state tracking
        self.gripper_target = torch.zeros((self.num_envs, 1), device=self.device)  # Current gripper target position

    def _compute_intermediate_values(self, dt):
        """Compute state for both robots and block via forward kinematics and finite differencing."""
        # === ROBOT 1 (Active Gripper) ===
        self.tcp1_pos = (self._robot1.data.body_pos_w[:, self.tcp1_body_idx] - self.scene.env_origins)
        self.tcp1_quat = self._robot1.data.body_quat_w[:, self.tcp1_body_idx]

        # Finite-difference velocities (more stable than simulator velocities)
        self.tcp1_linvel_fd = (self.tcp1_pos - self.prev_tcp1_pos) / dt
        self.prev_tcp1_pos = self.tcp1_pos.clone()

        rot_diff_quat = quat_mul(
            self.tcp1_quat,
            quat_conjugate(self.prev_tcp1_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.tcp1_angvel_fd = rot_diff_aa / dt
        self.prev_tcp1_quat = self.tcp1_quat.clone()

        # === ROBOT 2 (Passive Holding) ===
        self.tcp2_pos = (
            self._robot2.data.body_pos_w[:, self.tcp2_body_idx] - self.scene.env_origins
        )
        self.tcp2_quat = self._robot2.data.body_quat_w[:, self.tcp2_body_idx]
        
        # === BLOCK ===
        self.block_pos = self._wooden_block.data.root_pos_w - self.scene.env_origins
        self.block_quat = self._wooden_block.data.root_quat_w

    def _setup_scene(self):
        """Setup the simulation scene with both robots, block, and table."""
        # Create Robot 1 (Gripper)
        self._robot1 = Articulation(self.cfg.gripper_robot)
        self.scene.articulations["robot1"] = self._robot1
        
        # Create Robot 2 (Screwdriver)
        self._robot2 = Articulation(self.cfg.screwdriver_robot)
        self.scene.articulations["robot2"] = self._robot2

        # Spawn table
        self.cfg.table.func(
            "/World/envs/env_0/woodworking_table",
            self.cfg.table,
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Create wooden block (object to grasp)
        self._wooden_block = RigidObject(self.cfg.wooden_block)

        # Spawn ground plane
        spawn_ground_plane(prim_path=self.cfg.terrain.prim_path, cfg=GroundPlaneCfg())

        # Clone environments for parallel simulation
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Initialize visualization markers
        num_envs = self.scene.cfg.num_envs
        n_markers = 2 * num_envs
        init_pos = torch.zeros((n_markers, 3), device=self.device)
        init_ori = torch.tensor([0, 0, 0, 1], device=self.device).repeat(n_markers, 1)
        marker_indices = torch.arange(n_markers, dtype=torch.int64, device=self.device)
        self.goal_marker.visualize(init_pos, init_ori, marker_indices=marker_indices)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply task-space control actions to both robots with learned gripper control.
        
        Actions are interpreted as:
        - [0:3] = Robot 1 arm position deltas
        - [3:6] = Robot 1 arm rotation deltas
        - [6:9] = Robot 2 arm position deltas
        - [9:12] = Robot 2 arm rotation deltas
        - [12] = Gripper closing command (-1 to 1, maps to 0-0.019)
        """
        self.actions = actions.clone().to(self.device).clamp(-1.0, 1.0)

        # === ROBOT 1: Apply learned task-space actions ===
        # Actions 0-6: Pose Delta
        actions_r1 = self.actions[:, 0:6]
        # Scale actions
        actions_r1[:, 0:3] *= self.pos_action_scale
        actions_r1[:, 3:6] *= self.rot_action_scale

        # Get current state for IK
        tcp1_pos_w = self._robot1.data.body_pos_w[:, self.tcp1_body_idx]
        tcp1_quat_w = self._robot1.data.body_quat_w[:, self.tcp1_body_idx]
        
        # Jacobian: (num_envs, num_bodies, 6, num_dofs) -> (num_envs, 6, num_dofs)
        # We need to select the arm joints using the stored indices
        jacobian_r1 = self._robot1.root_physx_view.get_jacobians()[:, self.tcp1_body_idx, :, :][:, :, :, self.r1_arm_indices]
        joint_pos_r1_arm = self._robot1.data.joint_pos[:, self.r1_arm_indices]

        # Set command and compute IK
        self.ik_controller_r1.set_command(actions_r1, tcp1_pos_w, tcp1_quat_w)
        joint_targets_r1_arm = self.ik_controller_r1.compute(
            tcp1_pos_w, tcp1_quat_w, jacobian_r1, joint_pos_r1_arm
        )

        # === GRIPPER: Extract learned closing command ===
        gripper_action = self.actions[:, 12:13]  # Shape: (num_envs, 1), range [-1, 1]
        # Map [-1, 1] to [0, 0.019] meters (full range of gripper opening)
        gripper_target = (gripper_action + 1.0) * 0.5 * 0.019  # Shape: (num_envs, 1)
        self.gripper_target = gripper_target.clone()
        
        # Combine arm and gripper targets
        joint_targets_r1 = torch.cat([joint_targets_r1_arm, gripper_target], dim=-1)
        self._robot1.set_joint_position_target(joint_targets_r1, joint_ids=self.r1_joint_indices)
        
        # === ROBOT 2: Apply learned task-space actions ===
        # Actions 6-12: Pose Delta
        actions_r2 = self.actions[:, 6:12]
        # Scale actions
        actions_r2[:, 0:3] *= self.pos_action_scale
        actions_r2[:, 3:6] *= self.rot_action_scale

        # Get current state for IK
        tcp2_pos_w = self._robot2.data.body_pos_w[:, self.tcp2_body_idx]
        tcp2_quat_w = self._robot2.data.body_quat_w[:, self.tcp2_body_idx]
        
        # Jacobian: (num_envs, num_bodies, 6, num_dofs) -> (num_envs, 6, num_dofs)
        # We need to select the arm joints using the stored indices
        jacobian_r2 = self._robot2.root_physx_view.get_jacobians()[:, self.tcp2_body_idx, :, :][:, :, :, self.r2_arm_indices]
        joint_pos_r2_arm = self._robot2.data.joint_pos[:, self.r2_arm_indices]

        # Set command and compute IK
        self.ik_controller_r2.set_command(actions_r2, tcp2_pos_w, tcp2_quat_w)
        joint_targets_r2_arm = self.ik_controller_r2.compute(
            tcp2_pos_w, tcp2_quat_w, jacobian_r2, joint_pos_r2_arm
        )

        # Screwdriver tool target (fixed at 0)
        screwdriver_target = torch.zeros((self.num_envs, 1), device=self.device)
        
        # Combine arm and tool targets
        joint_targets_r2 = torch.cat([joint_targets_r2_arm, screwdriver_target], dim=-1)
        self._robot2.set_joint_position_target(joint_targets_r2, joint_ids=self.r2_joint_indices)

    def _apply_action(self):
        """Apply actions to the simulator."""
        pass

    def _get_observations(self): # type: ignore
        """Get task-space observations for both robots and block.
        
        Policy sees symmetric observations (no privileged critic information).
        """
        # Update state and velocities
        self._compute_intermediate_values(dt=self.dt)

        # Add sensor noise to block position (simulates real sensor uncertainty)
        noisy_block_pos = self.block_pos + self.init_block_pos_noise
        
        # Compute gripper opening width (distance between fingers)
        gripper_width_r1 = torch.linalg.norm(
            self._robot1.data.body_pos_w[:, self.gripper1_right_idx] -
            self._robot1.data.body_pos_w[:, self.gripper1_left_idx],
            dim=-1,
            keepdim=True
        )
        
        # Robot 2 does not have a gripper, so width is 0
        gripper_width_r2 = torch.zeros((self.num_envs, 1), device=self.device)

        obs_dict = {
            # Robot 1 (active gripper)
            "tcp1_pos": self.tcp1_pos,
            "tcp1_pos_rel_block": self.tcp1_pos - noisy_block_pos,
            "tcp1_quat": self.tcp1_quat,
            "tcp1_linvel": self.tcp1_linvel_fd,
            "tcp1_angvel": self.tcp1_angvel_fd,
            
            # Robot 2 (passive screwdriver)
            "tcp2_pos": self.tcp2_pos,
            "tcp2_pos_rel_hold_target": self.tcp2_pos - self.hold_target_pos,
            "tcp2_quat": self.tcp2_quat,
            
            # Shared object state
            "block_pos": self.block_pos,
            "gripper_width1": gripper_width_r1,
            "gripper_width2": gripper_width_r2,
        }
        
        # Collapse to tensor using obs_order
        obs_tensors = collapse_obs_dict(obs_dict, self.cfg.obs_order)
        
        # Symmetric critic: both policy and value function see same observations
        return {"policy": obs_tensors, "critic": obs_tensors}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for block lifting task."""
        # Distance to block (how close is the gripper TCP to the block)
        dist_to_block = torch.norm(self.tcp1_pos - self.block_pos, dim=-1)
        
        # Current reward components
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Position tracking: Reward for bringing gripper close to block
        reward_proximity = -dist_to_block
        rewards += self.cfg.ee_position_tracking * reward_proximity
        
        # 2. Action regularization: Penalize large/jerky motions
        reward_action = -torch.sum(self.actions ** 2, dim=-1)
        rewards += self.cfg.action_penalty * reward_action
        
        # FIX: Optional - implement collision detection or holding error penalty for Robot 2
        
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # We do not terminate on collision, relying on the reward penalty instead.
        # This prevents the "constant resetting" loop and allows the agent to learn to move away.
        died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor): # type: ignore
        """Reset environments: robots, block position, and velocities."""
        super()._reset_idx(env_ids) # type: ignore
        env_ids = env_ids.to(self.device, dtype=torch.long)

        # Reset wooden block position with randomness
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
        pose = torch.cat([pos, ori], dim=1)
        self._wooden_block.write_root_pose_to_sim(pose)

        # Compute intermediate values to update joint positions
        self._compute_intermediate_values(dt=self.dt)

        # Reset Robot 2 (screwdriver) hold target
        block_pos = self._wooden_block.data.root_pos_w[env_ids]
        self.hold_target_pos[env_ids] = block_pos + torch.tensor(
            [0.0, 0.3, 0.15],  # Offset from block (x, y, z)
            device=self.device
        )
        self.hold_target_quat[env_ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],  # Neutral upright orientation
            device=self.device
        ).repeat(len(env_ids), 1)
        
        # Reset robots to initial state defined in config
        # This is handled by super()._reset_idx(env_ids) which calls _robot.reset(env_ids)
        # But we need to ensure the joints are set to the init_state
        
        # Reset finite-differencing buffers for velocity calculation
        self.prev_tcp1_pos[env_ids] = self.tcp1_pos[env_ids].clone()
        self.prev_tcp1_quat[env_ids] = self.tcp1_quat[env_ids].clone()

