# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


REPO_ROOT = Path(__file__).resolve().parents[6]
USD_FILES_DIR = REPO_ROOT / "USD_files"
TABLE_ASSET_PATH = (
    REPO_ROOT
    / "source"
    / "Woodworking_Simulation"
    / "Woodworking_Simulation"
    / "tasks"
    / "manager_based"
    / "woodworking_simulation"
    / "asset"
    / "assambly_table"
    / "Assambly_Table_Physics.usd"
)


def _make_robot_cfg(usd_path: Path, prim_name: str, init_pos: tuple[float, float, float], init_rot: tuple[float, float, float, float]) -> ArticulationCfg:
    cfg = ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        actuators={
            "arm_action": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=50, stiffness=5000),
        },
    )
    cfg.init_state.pos = init_pos
    cfg.init_state.rot = init_rot
    #is the parked position
    cfg.init_state.joint_pos = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.57,
        "elbow_joint": 1.57,
        "wrist_1_joint": -1.57,
        "wrist_2_joint": 1.57,
        "wrist_3_joint": 0.0,
    }
    return cfg


@configclass
class WoodworkingSimulationEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    gripper_robot_cfg: ArticulationCfg = _make_robot_cfg(USD_FILES_DIR / "ur5e_gripper.usd", "UR5eGripper", (0.08, 0.08, 0.842), (1.0, 0.0, 0.0, 0.0))
    screwdriver_robot_cfg: ArticulationCfg = _make_robot_cfg(USD_FILES_DIR / "ur5e_screwdriver.usd", "UR5eScrewdriver", (0.72, 1.12, 0.842), (0.0, 0.0, 0.0, 1.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.5, replicate_physics=True)

    # Table asset placement: Width = 1.2m, Depth = 0.8m, Height = 0.842m
    table_usd_path: str = str(TABLE_ASSET_PATH)
    table_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # custom parameters/scales
    action_scale = 1.0  # effort scaling for joint torques