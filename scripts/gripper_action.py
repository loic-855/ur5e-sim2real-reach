# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Woodworking_Simulation.tasks  # noqa: F401


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    # initialize variables for the ramp on the 7th joint
    val = torch.tensor(-1.0, device=env.unwrapped.device)
    direction = torch.tensor(1.0, device=env.unwrapped.device)
    step_size = 0.01  # adjust this value to control the speed of the ramp (smaller = slower)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # create actions tensor with zeros
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # update the ramp value for the 7th joint
            val = val + direction * step_size*10  # increased multiplier from 3 to 6 to make it faster
            # clamp and reverse direction at limits
            if val >= 1.0:
                val = torch.tensor(1.0, device=env.unwrapped.device)
                direction = -1.0
            elif val <= -1.0:
                val = torch.tensor(-1.0, device=env.unwrapped.device)
                direction = 1.0
            # set the 7th joint action to the ramp value
            actions[..., 6] = val
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
