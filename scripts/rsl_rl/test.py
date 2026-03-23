# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight test runner for RSL-RL checkpoints in Isaac Sim.

This is a small variant of ``play.py`` intended for deterministic
benchmark preparation in simulation.

Example:
    ./isaaclab.sh -p scripts/rsl_rl/test.py \
        --task Woodworking_Simulation-Pose-Orientation-Sim2Real-V4-Direct-v0 \
        --checkpoint logs/rsl_rl/<run>/model_2499.pt \
        --num_envs 1 \
        --seed 1234 \
        --goals-file scripts/rsl_rl/default_benchmark_goals.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOALS_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "default_benchmark_goals.json"


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
    "--goals-file",
    type=str,
    default=str(DEFAULT_GOALS_FILE),
    help="Path to a JSON file containing [[x, y, z, qw, qx, qy, qz], ...].",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required.")

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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import Woodworking_Simulation.tasks  # noqa: F401


def _load_benchmark_goals(goals_file: str) -> tuple[tuple[float, ...], ...]:
    path = Path(goals_file)
    with open(path, "r", encoding="utf-8") as f:
        goals = json.load(f)

    if not isinstance(goals, list) or len(goals) == 0:
        raise ValueError("Goals file must contain a non-empty list of goals.")

    parsed_goals = []
    for goal in goals:
        if not isinstance(goal, list) or len(goal) != 7:
            raise ValueError("Each goal must be [x, y, z, qw, qx, qy, qz].")
        parsed_goals.append(tuple(float(v) for v in goal))
    return tuple(parsed_goals)


def _apply_runtime_env_overrides(env_cfg):
    env_cfg.reset_range = 0.0
    env_cfg.deterministic_goal_sampling = True
    env_cfg.benchmark_goals = _load_benchmark_goals(args_cli.goals_file)
    env_cfg.domain_rand.enable_physical_rand = False
    env_cfg.domain_rand.enable_noise = False
    env_cfg.domain_rand.enable_delay = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_runtime_env_overrides(env_cfg)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = retrieve_file_path(args_cli.checkpoint)
    if os.path.isdir(resume_path):
        raise ValueError(
            f"Provided --checkpoint is a directory ({resume_path}).\n"
            "Please provide the explicit checkpoint file path, e.g. "
            "logs/rsl_rl/<run>/model_2499.pt"
        )

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
        f"reset_range={env_cfg.reset_range}, "
        f"deterministic_goal_sampling={env_cfg.deterministic_goal_sampling}, "
        f"num_benchmark_goals={len(env_cfg.benchmark_goals)}"
    )
    print(
        "[INFO] Domain randomization: "
        f"physical={env_cfg.domain_rand.enable_physical_rand}, "
        f"noise={env_cfg.domain_rand.enable_noise}, "
        f"delay={env_cfg.domain_rand.enable_delay}"
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

        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()