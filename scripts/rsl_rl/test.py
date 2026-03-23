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
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOALS_FILE = REPO_ROOT / "scripts" / "rsl_rl" / "default_benchmark_goals.json"

GOAL_TIMEOUT_S = 10.0
IN_AREA_POS_M = 0.05
IN_AREA_ROT_RAD = 15.0 * math.pi / 180.0
ON_GOAL_POS_M = 0.02
ON_GOAL_ROT_RAD = 10.0 * math.pi / 180.0


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
from isaaclab.utils.math import quat_error_magnitude
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
    env_cfg.debug = True
    env_cfg.episode_length_s = 10.0 * len(env_cfg.benchmark_goals)


def _make_goal_result(goal_index: int) -> dict:
    return {
        "goal_index": goal_index,
        "time_to_in_area_s": None,
        "time_to_on_goal_s": None,
        "samples_in_area": 0,
        "samples_on_goal": 0,
        "sum_pos_err_in_area": 0.0,
        "sum_rot_err_in_area": 0.0,
        "reached_in_area": False,
        "reached_on_goal": False,
    }


def _finalize_goal_result(goal_result: dict) -> dict:
    samples_in_area = int(goal_result["samples_in_area"])
    if samples_in_area > 0:
        goal_result["mean_pos_err_in_area_m"] = goal_result["sum_pos_err_in_area"] / samples_in_area
        goal_result["mean_rot_err_in_area_rad"] = goal_result["sum_rot_err_in_area"] / samples_in_area
    else:
        goal_result["mean_pos_err_in_area_m"] = None
        goal_result["mean_rot_err_in_area_rad"] = None
    del goal_result["sum_pos_err_in_area"]
    del goal_result["sum_rot_err_in_area"]
    return goal_result


def _read_pose_errors(task) -> tuple[float, float]:
    frame_data = task._frame_transformer.data
    tcp_pos = frame_data.target_pos_source[0, task._ee_frame_idx, :]
    tcp_quat = frame_data.target_quat_source[0, task._ee_frame_idx, :]
    goal_pos = task.goal_pos_source[0]
    goal_quat = task.goal_quat_source[0]
    pos_err = torch.norm(goal_pos - tcp_pos).item()
    rot_err = quat_error_magnitude(goal_quat.unsqueeze(0), tcp_quat.unsqueeze(0))[0].item()
    return float(pos_err), float(rot_err)


def _update_goal_result(goal_result: dict, goal_time_s: float, pos_err: float, rot_err: float):
    in_area = pos_err <= IN_AREA_POS_M and rot_err <= float(IN_AREA_ROT_RAD)
    on_goal = pos_err <= ON_GOAL_POS_M and rot_err <= float(ON_GOAL_ROT_RAD)

    if in_area:
        goal_result["reached_in_area"] = True
        if goal_result["time_to_in_area_s"] is None:
            goal_result["time_to_in_area_s"] = goal_time_s
        goal_result["samples_in_area"] += 1
        goal_result["sum_pos_err_in_area"] += pos_err
        goal_result["sum_rot_err_in_area"] += rot_err

    if on_goal:
        goal_result["reached_on_goal"] = True
        if goal_result["time_to_on_goal_s"] is None:
            goal_result["time_to_on_goal_s"] = goal_time_s
        goal_result["samples_on_goal"] += 1


def _build_episode_summary(episode_index: int, goal_results: list[dict]) -> dict:
    in_area_times = [g["time_to_in_area_s"] for g in goal_results if g["time_to_in_area_s"] is not None]
    on_goal_times = [g["time_to_on_goal_s"] for g in goal_results if g["time_to_on_goal_s"] is not None]
    mean_pos_in_area = [g["mean_pos_err_in_area_m"] for g in goal_results if g["mean_pos_err_in_area_m"] is not None]
    mean_rot_in_area = [g["mean_rot_err_in_area_rad"] for g in goal_results if g["mean_rot_err_in_area_rad"] is not None]
    goal_count = len(goal_results)

    return {
        "episode_index": episode_index,
        "goal_count": goal_count,
        "goals": goal_results,
        "goals_reached_in_area": sum(1 for g in goal_results if g["reached_in_area"]),
        "goals_reached_on_goal": sum(1 for g in goal_results if g["reached_on_goal"]),
        "mean_time_to_in_area_s": sum(in_area_times) / len(in_area_times) if in_area_times else None,
        "mean_time_to_on_goal_s": sum(on_goal_times) / len(on_goal_times) if on_goal_times else None,
        "mean_pos_err_in_area_m": sum(mean_pos_in_area) / len(mean_pos_in_area) if mean_pos_in_area else None,
        "mean_rot_err_in_area_rad": sum(mean_rot_in_area) / len(mean_rot_in_area) if mean_rot_in_area else None,
    }


def _save_benchmark_results(log_dir: str, resume_path: str, goals: tuple[tuple[float, ...], ...], results: list[dict]):
    benchmark_dir = REPO_ROOT / "logs" / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_path = benchmark_dir / f"_{timestamp}_sim_pose_benchmark.json"

    all_in_area_times = [g["time_to_in_area_s"] for ep in results for g in ep["goals"] if g["time_to_in_area_s"] is not None]
    all_on_goal_times = [g["time_to_on_goal_s"] for ep in results for g in ep["goals"] if g["time_to_on_goal_s"] is not None]
    all_pos_in_area = [g["mean_pos_err_in_area_m"] for ep in results for g in ep["goals"] if g["mean_pos_err_in_area_m"] is not None]
    all_rot_in_area = [g["mean_rot_err_in_area_rad"] for ep in results for g in ep["goals"] if g["mean_rot_err_in_area_rad"] is not None]

    payload = {
        "metadata": {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "log_dir": log_dir,
            "seed": args_cli.seed,
            "goal_count": len(goals),
            "goal_timeout_s": GOAL_TIMEOUT_S,
            "thresholds": {
                "in_area": {"pos_m": IN_AREA_POS_M, "rot_rad": float(IN_AREA_ROT_RAD)},
                "on_goal": {"pos_m": ON_GOAL_POS_M, "rot_rad": float(ON_GOAL_ROT_RAD)},
            },
            "goals": [list(goal) for goal in goals],
        },
        "summary": {
            "episodes_completed": len(results),
            "goal_count": len(goals),
            "total_goals_executed": sum(len(ep["goals"]) for ep in results),
            "mean_time_to_in_area_s": sum(all_in_area_times) / len(all_in_area_times) if all_in_area_times else None,
            "mean_time_to_on_goal_s": sum(all_on_goal_times) / len(all_on_goal_times) if all_on_goal_times else None,
            "mean_pos_err_in_area_m": sum(all_pos_in_area) / len(all_pos_in_area) if all_pos_in_area else None,
            "mean_rot_err_in_area_rad": sum(all_rot_in_area) / len(all_rot_in_area) if all_rot_in_area else None,
            "goals_reached_in_area": sum(ep["goals_reached_in_area"] for ep in results),
            "goals_reached_on_goal": sum(ep["goals_reached_on_goal"] for ep in results),
        },
        "episodes": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Benchmark results saved to: {out_path}")



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
    env.unwrapped.goal_max_steps = int(GOAL_TIMEOUT_S / env.unwrapped.step_dt)

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
    episode_idx = 0
    episode_step = 0
    goal_timeout_steps = int(GOAL_TIMEOUT_S / env.unwrapped.step_dt)
    goal_count = len(env_cfg.benchmark_goals)
    steps_per_episode = goal_timeout_steps * goal_count
    episode_goal_results: list[dict] = []
    benchmark_results: list[dict] = []
    current_goal_result = _make_goal_result(0)

    while simulation_app.is_running() and episode_idx < len(env_cfg.benchmark_goals):
        start_time = time.time()
        goal_time_s = (episode_step % goal_timeout_steps) * env.unwrapped.step_dt
        pos_err, rot_err = _read_pose_errors(env.unwrapped)
        _update_goal_result(current_goal_result, goal_time_s, pos_err, rot_err)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        timestep += 1
        episode_step += 1

        if episode_step % goal_timeout_steps == 0:
            episode_goal_results.append(_finalize_goal_result(current_goal_result))
            next_goal_idx = episode_step // goal_timeout_steps
            if next_goal_idx < goal_count:
                current_goal_result = _make_goal_result(next_goal_idx)

        if episode_step >= steps_per_episode:
            benchmark_results.append(_build_episode_summary(episode_idx, episode_goal_results))
            print(f"[INFO] Completed benchmark episode {episode_idx + 1}/{len(env_cfg.benchmark_goals)}")
            episode_idx += 1
            episode_step = 0
            episode_goal_results = []
            current_goal_result = _make_goal_result(0)

        if args_cli.video and timestep == args_cli.video_length:
            break

        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    _save_benchmark_results(log_dir, resume_path, env_cfg.benchmark_goals, benchmark_results)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()