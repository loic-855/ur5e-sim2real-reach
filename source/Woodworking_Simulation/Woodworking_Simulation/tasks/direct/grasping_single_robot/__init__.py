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
    id="WWSim-Grasping-Single-Robot-Direct-v0",
    entry_point=f"{__name__}.grasping_single_robot:GraspingSingleRobotV0",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasping_single_robot:GraspingSingleRobotCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_single_grasp:PPORunnerCfg",
    },
)

gym.register(
    id="WWSim-Grasping-Single-Robot-Direct-v1",
    entry_point=f"{__name__}.grasping_single_robot:GraspingSingleRobotV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasping_single_robot:GraspingSingleRobotV1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_single_grasp:PPORunnerCfg",
    },
)
