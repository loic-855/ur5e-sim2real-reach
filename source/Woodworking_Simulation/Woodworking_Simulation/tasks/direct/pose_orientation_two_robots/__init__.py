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
    id="WWSim-Pose-Orientation-Two-Robots-v0",
    entry_point=f"{__name__}.pose_orientation_two_robots:PoseOrientationTwoRobotsV0",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_orientation_two_robots:PoseOrientationTwoRobotsCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
