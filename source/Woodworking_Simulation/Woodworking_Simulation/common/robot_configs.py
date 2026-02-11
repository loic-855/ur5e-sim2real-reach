# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Common robot configurations and scene assets for Woodworking Simulation environments.

This module provides factory functions and constants to eliminate configuration duplication
across different robot tasks. Based on pose_orientation_no_gripper.py as reference.
"""

from __future__ import annotations
from pathlib import Path
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# === Path Constants ===
REPO_ROOT = Path(__file__).resolve().parents[4]  # Go up to Woodworking_Simulation root
USD_FILES_DIR = REPO_ROOT / "USD_files"
TABLE_ASSET_PATH = USD_FILES_DIR / "woodworking_table.usd"

# === Physical Constants ===
# Table dimensions (origin at corner, we offset to center)
TABLE_DEPTH = 0.8   # x-axis
TABLE_WIDTH = 1.2   # y-axis  
TABLE_HEIGHT = 0.842  # z-axis
# Aluminium block on which the robot is mounted
MOUNT_HEIGHT = 0.02

# Environment origin offset to place origin at table center at plateau height
ENV_ORIGIN_OFFSET = torch.tensor([TABLE_DEPTH / 2.0, TABLE_WIDTH / 2.0, TABLE_HEIGHT])

# === UR5e Robot Constants ===
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
    # Gripper joints (added for completeness, even if not actuated in some versions)
    "left_finger_joint": (0.0,0.02),
    "right_finger_joint": (0.0,0.02),
}

# === Robot Type Constants ===
class RobotType:
    """Robot configuration identifiers."""
    NO_GRIPPER = "no_gripper"
    GRIPPER_TCP = "gripper_tcp"
    GRIPPER_TCP_NO_ACTUATION = "gripper_tcp_no_actuation"
    GRIPPER_TCP_WIDE = "gripper_tcp_wide"
    SCREWDRIVER_TCP = "screwdriver_tcp"


# === Standard Configuration Factory Functions ===

def get_table_cfg() -> sim_utils.UsdFileCfg:
    """Get woodworking table configuration."""
    return sim_utils.UsdFileCfg(usd_path=str(TABLE_ASSET_PATH))

def get_terrain_cfg() -> TerrainImporterCfg:
    """Get ground plane/terrain configuration."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane"
    )

def get_goal_marker_cfg(scale: tuple[float, float, float] = (0.05, 0.05, 0.05)) -> VisualizationMarkersCfg:
    """Get goal marker configuration for debugging."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/goal",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=scale,
        )},
    )

def get_origin_marker_cfg(scale: tuple[float, float, float] = (0.02, 0.02, 0.02)) -> VisualizationMarkersCfg:
    """Get origin marker configuration for debugging."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/origin",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=scale,
        )},
    )

def get_robot_grasp_marker_cfg(scale: tuple[float, float, float] = (0.05, 0.05, 0.05)) -> VisualizationMarkersCfg:
    """Get robot grasp marker configuration for debugging."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/robot_grasp_markers",
        markers={"frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=scale,
        )},
    )

def get_camera_pole_cfg(
    size: tuple[float, float, float] = (0.1, 0.18, 0.7),
    color: tuple[float, float, float] = (0.0, 1.0, 0.0)
) -> sim_utils.CuboidCfg:
    """Get camera pole configuration (size and color only)."""
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )

# === Robot Configuration Factory Functions ===

def get_robot_cfg(robot_type: str, prim_path: str) -> ArticulationCfg:
    """
    Factory function for robot configurations.
    
    Args:
        robot_type: One of RobotType constants
        prim_path: USD prim path for the robot
        
    Returns:
        ArticulationCfg configured for the specified robot type
        
    Raises:
        ValueError: If robot_type is not supported
    """
    
    # Robot position in local frame (table center = origin)
    robot_local_pos = (0.08, 0.08, TABLE_HEIGHT + MOUNT_HEIGHT)
    #robot_local_pos = (0.0, 0.0, 0.0)
    # Standard rotation: -90° around Z to match real setup (consistent across all robots)
    robot_local_rot = (0.7071, 0.0, 0.0, -0.7071)
    
    if robot_type == RobotType.NO_GRIPPER:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",        
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, 
                    solver_position_iteration_count=12, 
                    solver_velocity_iteration_count=1
                ),
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_lift_joint": -1.57, 
                    "wrist_1_joint": -1.57
                },
                pos=robot_local_pos,
                rot=robot_local_rot,
            ),
            # joint_pos={
            #         "shoulder_pan_joint": -0.38,
            #         "shoulder_lift_joint": -2.15,
            #         "elbow_joint": -1.737,
            #         "wrist_1_joint": -0.742,
            #         "wrist_2_joint": 1.659,
            #         "wrist_3_joint": -0.299,
            #     },
            actuators={
                "shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                    stiffness=200, damping=35,
                    effort_limit_sim=150.0, 
                    velocity_limit_sim=MAX_JOINT_VEL,
                ),
                "wrist": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                    stiffness=80, damping=15,
                    effort_limit_sim=28.0,
                    velocity_limit_sim=MAX_JOINT_VEL,
                ),
            },
        )
    
    elif robot_type == RobotType.GRIPPER_TCP:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(USD_FILES_DIR / "ur5e_gripper_tcp_small.usd"),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, 
                    solver_position_iteration_count=12, 
                    solver_velocity_iteration_count=1
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
                pos=robot_local_pos,
                rot=robot_local_rot,  # Same rotation as no_gripper
            ),
            actuators={
                "shoulder_action": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                    damping=60, stiffness=800
                ),
                "wrist_action": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                    damping=35, stiffness=350
                ),
                "gripper_action": ImplicitActuatorCfg(
                    joint_names_expr=["left_finger_joint", "right_finger_joint"],
                    damping=5, stiffness=1200
                ),
            },
        )
    
    elif robot_type == RobotType.GRIPPER_TCP_WIDE:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(USD_FILES_DIR / "ur5e_gripper_tcp.usd"),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, 
                    solver_position_iteration_count=12, 
                    solver_velocity_iteration_count=1
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
                pos=robot_local_pos,
                rot=robot_local_rot,  # Same rotation as no_gripper
            ),
            actuators={
                "shoulder_action": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                    damping=50, stiffness=700
                ),
                "wrist_action": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                    damping=30, stiffness=300
                ),
                "gripper_action": ImplicitActuatorCfg(
                    joint_names_expr=["left_finger_joint", "right_finger_joint"],
                    damping=14, stiffness=80
                ),
            },
        )
    
    elif robot_type == RobotType.SCREWDRIVER_TCP:
        # Position for screwdriver robot (placeholder, adjust based on your tests)
        screwdriver_pos = (0.72, 1.12, TABLE_HEIGHT + MOUNT_HEIGHT)
        # Rotation placeholder (will be tested by user)
        screwdriver_rot = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion, adjust later
        
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(USD_FILES_DIR / "ur5e_screwdriver_tcp.usd"),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, 
                    solver_position_iteration_count=12, 
                    solver_velocity_iteration_count=1
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
                    "joint0": 0.0,  # Screwdriver joint
                },
                pos=screwdriver_pos,
                rot=screwdriver_rot,  # To be tested and updated
            ),
            actuators={
                "shoulder_action": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                    damping=50, stiffness=700
                ),
                "wrist_action": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
                    damping=30, stiffness=300
                ),
                "screwdriver_action": ImplicitActuatorCfg(
                    joint_names_expr=["joint0"],
                    damping=20, stiffness=100
                ),
            },
        )
    elif robot_type == RobotType.GRIPPER_TCP_NO_ACTUATION:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(USD_FILES_DIR / "ur5e_gripper_tcp_unactuated.usd"),
                #usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=1
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_lift_joint": -1.57, 
                    "wrist_1_joint": -1.57
                },
                pos=robot_local_pos,
                rot=robot_local_rot,
            ),
            actuators={
                "shoulder_action": DelayedPDActuatorCfg(
                    joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
                    stiffness=160.0,
                    damping=28.0,
                    effort_limit=120.0,
                    velocity_limit=MAX_JOINT_VEL,
                    min_delay=0,
                    max_delay=0,
                ),
                "wrist_action_1": DelayedPDActuatorCfg(
                    joint_names_expr=["wrist_1_joint"],
                    stiffness=125.0,
                    damping=24.0,
                    effort_limit=60.0,
                    velocity_limit=MAX_JOINT_VEL,
                    min_delay=0,
                    max_delay=0,
                ),
                "wrist_action_2": DelayedPDActuatorCfg(
                    joint_names_expr=["wrist_2_joint"],
                    stiffness=100.0,
                    damping=22.0,
                    effort_limit=60.0,
                    velocity_limit=MAX_JOINT_VEL,
                    min_delay=0,
                    max_delay=0,
                ),
                "wrist_action_3": DelayedPDActuatorCfg(
                    joint_names_expr=["wrist_3_joint"],
                    stiffness=80.0,
                    damping=20.0,
                    effort_limit=60.0,
                    velocity_limit=MAX_JOINT_VEL,
                    min_delay=0,
                    max_delay=0,
                ),
            },
        )

    else:
        raise ValueError(f"Unsupported robot type: {robot_type}. "
                         f"Use one of: {[getattr(RobotType, attr) for attr in dir(RobotType) if not attr.startswith('_')]}")
    


# === Lighting Configuration ===

def setup_dome_light(intensity: float = 2000.0, color: tuple[float, float, float] = (0.75, 0.75, 0.75)) -> None:
    """Setup standard dome lighting for the scene."""
    light_cfg = sim_utils.DomeLightCfg(intensity=intensity, color=color)
    light_cfg.func("/World/Light", light_cfg)