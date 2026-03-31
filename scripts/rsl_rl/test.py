# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight test runner for RSL-RL checkpoints in Isaac Sim.

This is a small variant of ``play.py`` intended for deterministic
benchmark preparation in simulation.

Example:
    python scripts/rsl_rl/test.py \
        --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
        --checkpoint logs/rsl_rl/sim2real_v1_ablation/2026-03-25_00-53-16_rand-False/model_1499.pt\
        --goals-file scripts/benchmark_settings/default_benchmark_goals.json

    python scripts/rsl_rl/test.py \
        --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
        --checkpoint scripts/benchmark_settings/default_benchmark_checkpoints.json \
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOALS_FILE = REPO_ROOT / "scripts" / "benchmark_settings" / "default_benchmark_goals.json"
DEFAULT_CHECKPOINTS_FILE = REPO_ROOT / "scripts" / "benchmark_settings" / "default_benchmark_checkpoints.json"

GOAL_TIMEOUT_S = 10.0
IN_AREA_POS_M = 0.08


parser = argparse.ArgumentParser(description="Test an RSL-RL checkpoint in Isaac Sim.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

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


def _load_checkpoints(checkpoint_arg: str) -> list[str]:
    checkpoint_path = Path(checkpoint_arg)
    if checkpoint_path.suffix == ".pt":
        return [checkpoint_arg]

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = payload.get("checkpoints")

    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(
            "Checkpoint file must be a non-empty JSON list of checkpoint paths or an object with a 'checkpoints' list."
        )

    checkpoints: list[str] = []
    for entry in payload:
        if isinstance(entry, str):
            entry_path = Path(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
            entry_path = Path(entry["path"])
        else:
            raise ValueError("Each checkpoint entry must be a string or an object containing a 'path' field.")
        if not entry_path.is_absolute():
            entry_path = REPO_ROOT / entry_path
        checkpoints.append(str(entry_path))
    return checkpoints


def _yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _yaml_dump(value, indent: int = 0) -> str:
    indent_str = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{indent_str}{key}:")
                lines.append(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{indent_str}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{indent_str}-")
                lines.append(_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{indent_str}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{indent_str}{_yaml_scalar(value)}"


def _format_goal_line(goal: tuple[float, ...]) -> str:
    return "[" + ",".join(f"{value:.3f}" for value in goal) + "]"


def _extract_run_name_from_checkpoint(resume_path: str) -> str:
    agent_yaml_path = Path(resume_path).resolve().parent / "params" / "agent.yaml"
    if not agent_yaml_path.is_file():
        raise FileNotFoundError(f"Could not find agent config file: {agent_yaml_path}")

    with open(agent_yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("run_name:"):
                run_name = stripped.split(":", 1)[1].strip()
                if run_name:
                    return run_name

    raise ValueError(f"Could not find 'run_name' in {agent_yaml_path}")


def _apply_runtime_env_overrides(env_cfg):
    env_cfg.reset_range = 0.0
    env_cfg.deterministic_goal_sampling = True
    env_cfg.benchmark_goals = _load_benchmark_goals(args_cli.goals_file)
    env_cfg.goal_timeout_s = GOAL_TIMEOUT_S
    env_cfg.domain_rand.enable_actuator_rand = False
    env_cfg.domain_rand.enable_noise = False
    env_cfg.domain_rand.enable_delay = False
    env_cfg.domain_rand.enable_mass_com_rand = False
    env_cfg.debug = True
    env_cfg.episode_length_s = GOAL_TIMEOUT_S * len(env_cfg.benchmark_goals)


def _make_goal_result(goal_index: int) -> dict:
    return {
        "goal_index": goal_index,
        "time_to_area_s": None,
        "samples_total": 0,
        "samples_area": 0,
        "sum_pos_err": 0.0,
        "sum_rot_err": 0.0,
        "sum_pos_err_area": 0.0,
        "sum_rot_err_area": 0.0,
        "reached_area": False,
    }


def _finalize_goal_result(goal_result: dict) -> dict:
    samples_total = int(goal_result["samples_total"])
    samples_area = int(goal_result["samples_area"])
    if samples_total > 0:
        goal_result["mean_pos_err_area_m"] = goal_result["sum_pos_err_area"] / samples_area if samples_area > 0 else None
        goal_result["mean_rot_err_area_rad"] = goal_result["sum_rot_err_area"] / samples_area if samples_area > 0 else None
    else:
        goal_result["mean_pos_err_area_m"] = None
        goal_result["mean_rot_err_area_rad"] = None
    del goal_result["sum_pos_err"]
    del goal_result["sum_rot_err"]
    del goal_result["sum_pos_err_area"]
    del goal_result["sum_rot_err_area"]
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
    goal_result["samples_total"] += 1
    goal_result["sum_pos_err"] += pos_err
    goal_result["sum_rot_err"] += rot_err

    in_area = pos_err <= IN_AREA_POS_M

    if in_area:
        goal_result["reached_area"] = True
        if goal_result["time_to_area_s"] is None:
            goal_result["time_to_area_s"] = goal_time_s
        goal_result["samples_area"] += 1
        goal_result["sum_pos_err_area"] += pos_err
        goal_result["sum_rot_err_area"] += rot_err


def _build_episode_summary(episode_index: int, goal_results: list[dict]) -> dict:
    area_times = [g["time_to_area_s"] for g in goal_results if g["time_to_area_s"] is not None]
    area_pos = [g["mean_pos_err_area_m"] for g in goal_results if g["mean_pos_err_area_m"] is not None]
    area_rot = [g["mean_rot_err_area_rad"] for g in goal_results if g["mean_rot_err_area_rad"] is not None]
    goal_count = len(goal_results)

    return {
        "episode_index": episode_index,
        "goal_count": goal_count,
        "goals": goal_results,
        "goals_reached_area": sum(1 for g in goal_results if g["reached_area"]),
        "mean_time_to_area_s": sum(area_times) / len(area_times) if area_times else None,
        "mean_pos_err_area_m": sum(area_pos) / len(area_pos) if area_pos else None,
        "mean_rot_err_area_rad": sum(area_rot) / len(area_rot) if area_rot else None,
    }


def _save_benchmark_results(log_dir: str, resume_path: str, goals: tuple[tuple[float, ...], ...], results: list[dict]):
    benchmark_dir = REPO_ROOT / "logs" / "benchmarks" / "sim_pose"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_name = _extract_run_name_from_checkpoint(resume_path)
    out_path = benchmark_dir / f"{timestamp}_{run_name}.yaml"

    all_area_times = [g["time_to_area_s"] for ep in results for g in ep["goals"] if g["time_to_area_s"] is not None]
    all_area_pos = [g["mean_pos_err_area_m"] for ep in results for g in ep["goals"] if g["mean_pos_err_area_m"] is not None]
    all_area_rot = [g["mean_rot_err_area_rad"] for ep in results for g in ep["goals"] if g["mean_rot_err_area_rad"] is not None]

    payload = {
        "metadata": {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "run_name": run_name,
            "log_dir": log_dir,
            "seed": args_cli.seed,
            "goal_count": len(goals),
            "goal_timeout_s": GOAL_TIMEOUT_S,
            "goal_convention": "x y z qw qx qy qz",
            "thresholds": {
                "area": {"pos_m": IN_AREA_POS_M},
            },
        },
        "summary": {
            "episodes_completed": len(results),
            "goal_count": len(goals),
            "total_goals_executed": sum(len(ep["goals"]) for ep in results),
            "mean_time_to_area_s": sum(all_area_times) / len(all_area_times) if all_area_times else None,
            "mean_pos_err_area_m": sum(all_area_pos) / len(all_area_pos) if all_area_pos else None,
            "mean_rot_err_area_rad": sum(all_area_rot) / len(all_area_rot) if all_area_rot else None,
            "goals_reached_area": sum(ep["goals_reached_area"] for ep in results),
        },
        "episodes": results,
        "goals": [_format_goal_line(goal) for goal in goals],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_yaml_dump(payload))
        f.write("\n")

    print(f"[INFO] Benchmark results saved to: {out_path}")


def _create_benchmark_env(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    log_dir: str,
):
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

    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def _run_single_checkpoint_benchmark(
    env,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
):
    log_dir = os.path.dirname(resume_path)


    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    print(
        "[INFO] Runtime overrides: "
        f"num_envs={env_cfg.scene.num_envs}, seed={env_cfg.seed}, "
        f"reset_range={env_cfg.reset_range}, "
        f"deterministic_goal_sampling={env_cfg.deterministic_goal_sampling}, "
        f"num_benchmark_goals={len(env_cfg.benchmark_goals)}"
    )
    print(
        "[INFO] Domain randomization: "
        f"actuator={env_cfg.domain_rand.enable_actuator_rand}, "
        f"mass_com={env_cfg.domain_rand.enable_mass_com_rand}, "
        f"noise={env_cfg.domain_rand.enable_noise}, "
        f"delay={env_cfg.domain_rand.enable_delay}"
    )

    obs, _ = env.reset()
    timestep = 0
    episode_idx = 0
    episode_step = 0
    goal_timeout_steps = int(GOAL_TIMEOUT_S / env.unwrapped.step_dt)
    goal_count = len(env_cfg.benchmark_goals)
    steps_per_episode = goal_timeout_steps * goal_count
    episode_goal_results: list[dict] = []
    benchmark_results: list[dict] = []
    current_goal_result = _make_goal_result(0)

    while simulation_app.is_running() and episode_idx < 1:
        start_time = time.time()
        goal_time_s = (episode_step % goal_timeout_steps) * env.unwrapped.step_dt
        pos_err, rot_err = _read_pose_errors(env.unwrapped)
        _update_goal_result(current_goal_result, goal_time_s, pos_err, rot_err)

        with torch.no_grad():
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
            print(f"[INFO] Completed benchmark episode {episode_idx + 1}/1")
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

    checkpoint_paths = _load_checkpoints(args_cli.checkpoint)
    print(f"[INFO] Running benchmark for {len(checkpoint_paths)} checkpoint(s)")

    first_resume_path = retrieve_file_path(checkpoint_paths[0])
    if os.path.isdir(first_resume_path):
        raise ValueError(
            f"Provided checkpoint is a directory ({first_resume_path}).\n"
            "Please provide the explicit checkpoint file path, e.g. logs/rsl_rl/<run>/model_2499.pt"
        )

    env = _create_benchmark_env(copy.deepcopy(env_cfg), copy.deepcopy(agent_cfg), os.path.dirname(first_resume_path))

    try:
        for checkpoint_idx, checkpoint_arg in enumerate(checkpoint_paths, start=1):
            resume_path = retrieve_file_path(checkpoint_arg)
            if os.path.isdir(resume_path):
                raise ValueError(
                    f"Provided checkpoint is a directory ({resume_path}).\n"
                    "Please provide the explicit checkpoint file path, e.g. logs/rsl_rl/<run>/model_2499.pt"
                )

            print(f"[INFO] Benchmark {checkpoint_idx}/{len(checkpoint_paths)}")
            _run_single_checkpoint_benchmark(env, copy.deepcopy(env_cfg), copy.deepcopy(agent_cfg), resume_path)

            if not simulation_app.is_running():
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()