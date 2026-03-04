# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# V4 – UR5e with gripper (gripper not controlled)
gym.register(
    id="WWSim-Pose-Orientation-Sim2Real-Direct-v4",
    entry_point=f"{__name__}.pose_orientation_sim2real_v4:PoseOrientationSim2RealV4",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_orientation_sim2real_v4:PoseOrientationSim2RealV4Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
