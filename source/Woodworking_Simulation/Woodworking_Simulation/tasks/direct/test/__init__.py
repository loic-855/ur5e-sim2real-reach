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
    id="Template-Test-Direct-v0",
    entry_point=f"{__name__}.test:TestV0",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test:TestCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Test-Direct-v1",
    entry_point=f"{__name__}.test:TestV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test:TestV1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        # Use the extended PPO runner config for v1 by default
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_extended:PPORunnerCfg",
    },
)

# Duplicate of v1 but using the extended PPO config
gym.register(
    id="Template-Test-Direct-v1-ext",
    entry_point=f"{__name__}.test:TestV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test:TestV1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_extended:PPORunnerCfg",
    },
)
