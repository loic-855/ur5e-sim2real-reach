# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Pose-Orientation-Gripper-Robot-Direct-v0",
    entry_point=f"{__name__}.pose_orientation_gripper_robot:PoseOrientationGripperRobotV0",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_orientation_gripper_robot:PoseOrientationGripperRobot",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_small_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Pose-Orientation-Gripper-Robot-Direct-v1",
    entry_point=f"{__name__}.pose_orientation_gripper_robot_v1:PoseOrientationGripperRobotV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_orientation_gripper_robot_v1:PoseOrientationGripperRobotV1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_small_cfg:PPORunnerCfg",
    },
)
