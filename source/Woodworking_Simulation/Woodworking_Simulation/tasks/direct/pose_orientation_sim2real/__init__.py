# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# V0 – base (no domain randomisation, no contact sensor)
gym.register(
    id="WWSim-Pose-Orientation-Sim2Real-Direct-v0",
    entry_point=f"{__name__}.pose_orientation_sim2real:PoseOrientationSim2RealV0",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_orientation_sim2real:PoseOrientationSim2RealCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# V1 – domain randomisation + contact penalty (small PPO network)
gym.register(
    id="WWSim-Pose-Orientation-Sim2Real-Direct-v1",
    entry_point=f"{__name__}.pose_orientation_sim2real:PoseOrientationSim2RealV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_orientation_sim2real:PoseOrientationSim2RealV1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# V1-ext – same env as V1 but with extended (larger) PPO network
gym.register(
    id="WWSim-Pose-Orientation-Sim2Real-Direct-v1-ext",
    entry_point=f"{__name__}.pose_orientation_sim2real:PoseOrientationSim2RealV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pose_orientation_sim2real:PoseOrientationSim2RealV1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_extended:PPORunnerCfg",
    },
)
