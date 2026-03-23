# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight test runner for RSL-RL checkpoints in Isaac Sim.

This is a small variant of ``play.py`` intended for sim2real validation runs.
It adds CLI overrides that are useful for deterministic testing / benchmarking:

- disable domain randomization at runtime
- control the reset split between home and random resets
- control the noise applied around the home joint configuration
- optionally override the goal sampling ratio

Example:
    ./isaaclab.sh -p scripts/rsl_rl/test.py \
        --task Woodworking_Simulation-Pose-Orientation-Sim2Real-V4-Direct-v0 \
        --checkpoint logs/rsl_rl/<run>/model_2499.pt \
        --num_envs 1 \
        --seed 1234 \
        --disable-domain-rand \
        --home-joint-noise 0.0 \
        --env-reset 1.0
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Test an RSL-RL checkpoint in Isaac Sim.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disable-domain-rand",
    action="store_true",
    default=False,
    help="Disable noise, delay, and physical domain randomization at runtime.",
)
parser.add_argument(
    "--home-joint-noise",
    type=float,
    default=None,
    help="Uniform half-range around the home arm joint pose during reset. Use 0.0 for exact deterministic home resets.",
)
parser.add_argument(
    "--env-reset",
    type=float,
    default=None,
    help="Probability of using the home reset path instead of the fully random reset path.",
)
parser.add_argument(
    "--goal-sampling-random-ratio",
    type=float,
    default=None,
    help="Override goal sampling ratio: 0.0=FK only, 1.0=random cylindrical only.",
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=0,
    help="Optional maximum number of policy steps. Use 0 to run until the simulator is closed.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    try:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Could not import 'get_published_pretrained_checkpoint' from either 'isaaclab_rl' or 'isaaclab'. "
            "Please install the appropriate package or add it to PYTHONPATH."
        ) from e

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Woodworking_Simulation.tasks  # noqa: F401


def _apply_runtime_env_overrides(env_cfg):
    if args_cli.env_reset is not None and hasattr(env_cfg, "env_reset"):
        env_cfg.env_reset = float(args_cli.env_reset)

    if args_cli.home_joint_noise is not None and hasattr(env_cfg, "home_joint_pos_noise"):
        env_cfg.home_joint_pos_noise = float(args_cli.home_joint_noise)

    if args_cli.goal_sampling_random_ratio is not None and hasattr(env_cfg, "goal_sampling_random_ratio"):
        env_cfg.goal_sampling_random_ratio = float(args_cli.goal_sampling_random_ratio)

    if args_cli.disable_domain_rand and hasattr(env_cfg, "domain_rand"):
        for attr in ("enable_physical_rand", "enable_noise", "enable_delay"):
            if hasattr(env_cfg.domain_rand, attr):
                setattr(env_cfg.domain_rand, attr, False)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_runtime_env_overrides(env_cfg)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        if os.path.isdir(resume_path):
            raise ValueError(
                f"Provided --checkpoint is a directory ({resume_path}).\n"
                "Please provide the explicit checkpoint file path, e.g. "
                "logs/rsl_rl/<run>/model_2499.pt"
            )
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "test"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during execution.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(
        "[INFO] Runtime overrides: "
        f"num_envs={env_cfg.scene.num_envs}, seed={env_cfg.seed}, "
        f"env_reset={getattr(env_cfg, 'env_reset', 'n/a')}, "
        f"home_joint_pos_noise={getattr(env_cfg, 'home_joint_pos_noise', 'n/a')}, "
        f"goal_sampling_random_ratio={getattr(env_cfg, 'goal_sampling_random_ratio', 'n/a')}"
    )
    if hasattr(env_cfg, "domain_rand"):
        print(
            "[INFO] Domain randomization: "
            f"physical={getattr(env_cfg.domain_rand, 'enable_physical_rand', 'n/a')}, "
            f"noise={getattr(env_cfg.domain_rand, 'enable_noise', 'n/a')}, "
            f"delay={getattr(env_cfg.domain_rand, 'enable_delay', 'n/a')}"
        )

    obs = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
        if args_cli.num_steps > 0 and timestep >= args_cli.num_steps:
            break

        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()