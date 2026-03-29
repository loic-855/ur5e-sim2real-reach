#!/usr/bin/env python3
"""
Sim2Real Node for UR5e pose control via RTDE – **V1** (normalised observations).

This node:
1. Connects to the UR5e robot via RTDE and runs a **dedicated reader
   thread at 125 Hz** that continuously caches the latest robot state
   (actual_q, actual_qd, actual_TCP_pose, actual_TCP_speed)
2. Subscribes to /goal_pose (ROS2) for target end-effector pose
3. The **control loop runs at 60 Hz**, reads the most recent cached
   state, builds 24-dim normalised observations matching IsaacSim v2,
   and runs policy inference
4. Sends q_des via RTDE input registers (continuous stream)
5. URScript on the robot runs an impedance controller with first-order
   interpolation filter at 500 Hz for smooth motion

Decoupling the RTDE reader (125 Hz) from the policy loop (60 Hz)
ensures observations are at most ~8 ms stale instead of ~16.7 ms.

Frame convention:
  - RTDE TCP pose / velocity are in **robot-base** frame.
  - The policy expects everything in the **table-centre** frame.
  - Position conversion: simple offset ``ee_pos_table = ee_pos_base + ROBOT_BASE_LOCAL``
  - Quaternion and velocities are used as-is (no rotation between frames).
  - This matches the v1 sim2real_node convention (validated on real robot).

Usage:
    python3 sim2real_node.py 
    python3 sim2real_node.py --rate 60 --action-scale 0.5 --model path/to/policy.pt
    python3 sim2real_node.py --robot-gain tuned
For benchmarking:
    python scripts/sim2real/v1/sim2real_node.py \
    --benchmark \
    --goals-file scripts/benchmark_settings/goals_handmade.json \
    --goal-timeout-s 10 \
    --num-goals 7 \
    --num-takes 3 \
    --robot-gain naive \
    --action-scale 0.3 \
    --model logs/rsl_rl/sim2real_v1_ablation_10s/2026-03-25_10-20-56__rand-False_10s-Timeout/exported/policy.pt \

Then publish goals using goal_publisher.py in benchmark mode:
    python scripts/sim2real/goal_publisher.py --goals-file scripts/benchmark_settings/goals_handmade.json --update 10 --benchmark
"""

import math
import torch
import numpy as np
import argparse
import logging
import time
import socket
import sys
import json
import os
from datetime import datetime, timezone

from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# Local imports
from observation_builder import (
    RobotState, GoalState,
    build_observation,
    compute_dof_targets,
    quat_box_minus,
    JOINT_NAMES_SIM,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
)
from policy_inference import load_policy

# ============================================================================
# RTDE constants
# ============================================================================
ROBOT_HOST = "192.168.1.101"
ROBOT_PORT = 30004
ROBOT_PRIMARY_PORT = 30001

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
RTDE_CONFIG_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "rtde_input_v1.xml")
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control.script")
URSCRIPT_FILE_TUNED = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_tuned.script")
URSCRIPT_FILE_NAIVE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_naive.script")

# Home position
HOME_Q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

# ============================================================================
# Robot base position relative to TABLE CENTRE (table frame origin).
# actual_TCP_pose from RTDE is in the robot-base frame; we add this offset
# to convert to the table-centre frame used by the policy / goal publisher.
# Matches v1 sim2real_node (validated on real robot).
# ============================================================================
ROBOT_BASE_LOCAL = np.array([-0.52, 0.32, 0.02], dtype=np.float32)
IN_AREA_POS_M = 0.08


# ============================================================================
# Rotation-vector ↔ quaternion conversion
# ============================================================================

def rotvec_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert a rotation vector (axis-angle) to a unit quaternion (w, x, y, z)."""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    half = angle / 2.0
    s = math.sin(half) / angle
    return np.array([math.cos(half), rx * s, ry * s, rz * s], dtype=np.float32)


def load_benchmark_goals(goals_file: str) -> tuple[tuple[float, ...], ...]:
    path = Path(goals_file)
    with open(path, "r", encoding="utf-8") as f:
        goals = json.load(f)

    if not isinstance(goals, list) or len(goals) == 0:
        raise ValueError("Goals file must contain a non-empty list of goals.")

    parsed_goals = []
    for goal in goals:
        if not isinstance(goal, list) or len(goal) != 7:
            raise ValueError("Each goal must be [x, y, z, qw, qx, qy, qz].")
        parsed_goals.append(tuple(float(value) for value in goal))
    return tuple(parsed_goals)


def yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def yaml_dump(value, indent: int = 0) -> str:
    indent_str = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{indent_str}{key}:")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{indent_str}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{indent_str}-")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{indent_str}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{indent_str}{yaml_scalar(value)}"


def format_goal_line(goal: tuple[float, ...]) -> str:
    return "[" + ",".join(f"{value:.3f}" for value in goal) + "]"


def extract_run_name_from_model_path(model_path: Optional[str]) -> str:
    if model_path is None:
        return "default_policy"

    model = Path(model_path).resolve()
    candidate_paths = [
        model.parent / "params" / "agent.yaml",
        model.parent.parent / "params" / "agent.yaml",
    ]
    for agent_yaml_path in candidate_paths:
        if not agent_yaml_path.is_file():
            continue
        with open(agent_yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("run_name:"):
                    run_name = stripped.split(":", 1)[1].strip()
                    if run_name:
                        return run_name
    return model.stem


# ============================================================================
# RTDE communication helper
# ============================================================================

class RTDEController:
    """Manages RTDE connection, URScript upload, q_des streaming,
    and a background reader thread that caches robot state at 125 Hz."""

    def __init__(
        self,
        robot_host: str = ROBOT_HOST,
        robot_port: int = ROBOT_PORT,
        primary_port: int = ROBOT_PRIMARY_PORT,
        config_file: str = RTDE_CONFIG_FILE,
        urscript_file: str = URSCRIPT_FILE,
        rtde_frequency: float = 125.0,
        control_rate: float = 60.0,
    ):
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.primary_port = primary_port
        self.config_file = config_file
        self.urscript_file = urscript_file
        self.rtde_frequency = rtde_frequency
        self.control_rate = control_rate
        self.con: Optional[rtde.RTDE] = None
        self.setp = None
        self.stop_reg = None
        self.control_rate_reg = None
        self.connected = False

        # --- Cached robot state (updated by reader thread) -----------------
        self._state_lock = Lock()
        self._cached_q: Optional[np.ndarray] = None
        self._cached_qd: Optional[np.ndarray] = None
        self._cached_tcp_pose: Optional[list] = None   # 6 floats
        self._cached_tcp_speed: Optional[list] = None   # 6 floats
        self._state_seq: int = 0          # monotonic counter
        self._reader_running = False
        self._reader_thread: Optional[Thread] = None

    # -- connection ---------------------------------------------------------

    def connect(self):
        """Open RTDE connection, configure recipes, start synchronisation."""
        conf = rtde_config.ConfigFile(self.config_file)
        output_names, output_types = conf.get_recipe("out")
        q_des_names, q_des_types = conf.get_recipe("q_des")
        stop_name, stop_type = conf.get_recipe("stop")
        control_rate_info, control_rate_type = conf.get_recipe("control_rate_info")

        self.con = rtde.RTDE(self.robot_host, self.robot_port)
        self.con.connect()

        if not self.con.send_output_setup(output_names, output_types,
                                           frequency=self.rtde_frequency):
            raise RuntimeError("RTDE: unable to configure output recipe")

        self.setp = self.con.send_input_setup(q_des_names, q_des_types)
        if self.setp is None:
            raise RuntimeError("RTDE: unable to configure q_des input")

        self.stop_reg = self.con.send_input_setup(stop_name, stop_type)
        if self.stop_reg is None:
            raise RuntimeError("RTDE: unable to configure stop signal")
        
        self.control_rate_reg = self.con.send_input_setup(control_rate_info, control_rate_type)
        if self.control_rate_reg is None:
            raise RuntimeError("RTDE: unable to configure control rate info input")

        if not self.con.send_start():
            raise RuntimeError("RTDE: unable to start synchronisation")

        self.connected = True
        print("[RTDE] Connected and synchronised")

    # -- URScript -----------------------------------------------------------

    def send_home_movement(self, timeout: float = 5.0, wait_time: float = 5.2):
        """Send a simple movej command to home position before impedance control."""
        print("[RTDE] Sending home movement command...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.robot_host, self.primary_port))
            s.sendall(f"movej({HOME_Q}, a=0.7, v=0.6, t={timeout}, r=0)\n".encode("utf-8"))
            s.close()
            print("[RTDE] Home movement command sent")
        except Exception as e:
            raise RuntimeError(f"[RTDE] Failed to send home movement: {e}")

        time.sleep(wait_time)
        print(f"[RTDE] Home movement completed")
    def send_urscript(self):
        """Upload and start the URScript impedance controller on the robot."""
        self.stop_reg.input_bit_register_64 = False
        self.con.send(self.stop_reg)
        self.control_rate_reg.input_int_register_36 = int(self.control_rate)
        self.con.send(self.control_rate_reg)

        self._write_q_des(HOME_Q)

        print(f"[RTDE] Sending URScript: {self.urscript_file}")
        try:
            with open(self.urscript_file, "r") as f:
                program = f.read()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.robot_host, self.primary_port))
            s.sendall((program + "\n").encode("utf-8"))
            s.close()
            print("[RTDE] URScript sent successfully")
        except Exception as e:
            raise RuntimeError(f"[RTDE] Failed to send URScript: {e}")

        time.sleep(3.0)
        print("[RTDE] URScript started on robot")

    # -- q_des streaming ----------------------------------------------------

    def _write_q_des(self, q_des):
        """Write 6 joint values into RTDE input registers 24-29."""
        for i in range(6):
            self.setp.__dict__[f"input_double_register_{24 + i}"] = float(q_des[i])
        self.con.send(self.setp)

    def send_q_des(self, q_des: np.ndarray):
        self._write_q_des(q_des.tolist())

    def receive_state(self):
        """Receive the latest robot state from RTDE (synchronous).

        Returns:
            RTDE state object, or None on failure.
        """
        if self.con is None:
            return None
        return self.con.receive()

    # -- Background reader thread -------------------------------------------

    def start_reader(self):
        """Start the background thread that polls RTDE at ``rtde_frequency``
        and caches the latest robot state."""
        if self._reader_thread is not None:
            return
        self._reader_running = True
        self._reader_thread = Thread(
            target=self._reader_loop, daemon=True, name="rtde_reader"
        )
        self._reader_thread.start()
        print(f"[RTDE] Reader thread started at {self.rtde_frequency} Hz")

    def stop_reader(self):
        """Signal the reader thread to stop."""
        self._reader_running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

    def _reader_loop(self):
        """Continuously read RTDE state and cache it under a lock."""
        period = 1.0 / self.rtde_frequency
        while self._reader_running and self.connected:
            t0 = time.monotonic()
            try:
                state = self.con.receive()
                if state is not None:
                    with self._state_lock:
                        self._cached_q = np.array(state.actual_q, dtype=np.float32)
                        self._cached_qd = np.array(state.actual_qd, dtype=np.float32)
                        self._cached_tcp_pose = list(state.actual_TCP_pose)
                        self._cached_tcp_speed = list(state.actual_TCP_speed)
                        self._state_seq += 1
            except Exception:
                pass  # transient RTDE error; next iteration will retry
            # Sleep for the remainder of the period
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_cached_state(self):
        """Return the latest cached state as a tuple, or None if not yet available.

        Returns:
            (q, qd, tcp_pose, tcp_speed, seq) or None.
        """
        with self._state_lock:
            if self._cached_q is None:
                return None
            return (
                self._cached_q.copy(),
                self._cached_qd.copy(),
                list(self._cached_tcp_pose),
                list(self._cached_tcp_speed),
                self._state_seq,
            )

    # -- shutdown -----------------------------------------------------------

    def stop_robot(self):
        """Send stop signal so the URScript returns to home and exits."""
        if self.con is None or not self.connected:
            return
        try:
            self.stop_reg.input_bit_register_64 = True
            self.con.send(self.stop_reg)
            time.sleep(2.0)
            print("[RTDE] Stop signal sent - robot returning to home")
        except Exception as e:
            print(f"[RTDE] Error sending stop: {e}")

    def disconnect(self):
        self.stop_reader()
        if self.con is None:
            return
        try:
            self.con.send_pause()
            self.con.disconnect()
        except Exception:
            pass
        self.connected = False
        print("[RTDE] Disconnected")


# ============================================================================
# ROS2 + RTDE sim2real node
# ============================================================================

class Sim2RealNode(Node):
    """ROS2 node for sim2real policy deployment using RTDE for joint control."""

    def __init__(
        self,
        robot_prefix: str = "gripper",
        model_path: Optional[str] = None,
        control_rate: float = 60.0,
        rtde_rate: float = 125.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        action_scale: float = 7.0,
        robot_host: str = ROBOT_HOST,
        robot_gain: str = "tuned",
        num_takes: int = 1,
        benchmark: bool = False,
        goal_timeout_s: float = 10.0,
        num_goals: int = 0,
        goals_file: Optional[str] = None,
    ):
        super().__init__("sim2real_policy_node_v1")

        self.robot_prefix = robot_prefix
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        self.action_scale = action_scale

        # State variables (protected by lock)
        self.lock = Lock()
        self.joint_positions: Optional[np.ndarray] = None
        self.joint_velocities: Optional[np.ndarray] = None
        self.ee_position: Optional[np.ndarray] = None       # source frame
        self.ee_quaternion: Optional[np.ndarray] = None      # source frame
        self.tcp_linear_vel: Optional[np.ndarray] = None     # source frame
        self.tcp_angular_vel: Optional[np.ndarray] = None    # source frame
        self.goal_position: Optional[np.ndarray] = None
        self.goal_quaternion: Optional[np.ndarray] = None
        self.old_goal: Optional[np.ndarray] = None

        # DOF targets (tracked across control steps)
        self.dof_targets: Optional[np.ndarray] = None

        # Control state
        self.is_running = False

        # Benchmarking
        self._benchmark = benchmark
        self._goal_timeout_s = goal_timeout_s
        self._benchmark_num_goals = num_goals
        self._benchmark_goals_file = goals_file
        self._benchmark_goals = load_benchmark_goals(goals_file) if goals_file is not None else ()
        if self._benchmark_num_goals == 0 and self._benchmark_goals:
            self._benchmark_num_goals = len(self._benchmark_goals)
        self._benchmark_started_at: Optional[float] = None
        self._benchmark_current_goal_index = -1
        self._benchmark_results: list[dict] = []
        self._current_goal_result: Optional[dict] = None
        self._benchmark_completed = False
        self._benchmark_output_dir = REPO_ROOT / "logs" / "benchmarks" / "sim_pose_real"
        self._model_path = model_path
        self._robot_gain = robot_gain
        self._num_takes = num_takes
        self._current_take = 0
        self._all_episode_results: list[list[dict]] = []
        if robot_gain == "naive":
            urscript_file = URSCRIPT_FILE_NAIVE
        else:
            urscript_file = URSCRIPT_FILE_TUNED

        # ==================================================================
        # Load policy
        # ==================================================================
        self.get_logger().info(f"Loading policy from: {model_path or 'default'}")
        self.policy = load_policy(model_path=model_path, device=device)
        self.get_logger().info("Policy loaded successfully!")

        # ==================================================================
        # RTDE controller  (reader runs at rtde_rate, decoupled from policy)
        # ==================================================================
        self.rtde = RTDEController(
            robot_host=robot_host,
            rtde_frequency=rtde_rate,
            control_rate=control_rate,
            urscript_file=urscript_file,
        )
        self.get_logger().info("Connecting to robot via RTDE...")
        self.rtde.connect()
        self.get_logger().info("Sending home movement...")
        self.rtde.send_home_movement()
        self.get_logger().info("Uploading URScript (impedance)...")
        self.rtde.send_urscript()
        self.get_logger().info("Starting RTDE reader thread...")
        self.rtde.start_reader()
        self.get_logger().info(
            f"RTDE ready!  reader={rtde_rate} Hz  policy={control_rate} Hz"
        )
        self._last_state_seq = 0

        # ==================================================================
        # ROS2 Subscribers (goal pose)
        # ==================================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_pose_callback,
            qos,
        )

        # Benchmark control publisher (for coordinating with goal_publisher)
        if self._benchmark:
            self._benchmark_control_pub = self.create_publisher(String, "/benchmark_control", 10)

        # ==================================================================
        # Control timer
        # ==================================================================
        self.callback_group = ReentrantCallbackGroup()
        self.control_timer = self.create_timer(
            self.dt,
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info(f"Sim2Real V1 node initialised at {control_rate} Hz")
        self.get_logger().info(
            f"Action scale: {action_scale}"
        )

        # Delayed initial "start" signal for benchmark goal publisher
        if self._benchmark and self._num_takes > 0:
            self._initial_start_timer = self.create_timer(
                2.0, self._publish_initial_start, callback_group=self.callback_group
            )

    # -- Benchmark control signals -----------------------------------------

    def _publish_initial_start(self):
        """One-shot: publish initial 'start' to goal publisher, then cancel."""
        self._publish_start_signal()
        self._initial_start_timer.cancel()

    def _publish_start_signal(self):
        msg = String()
        msg.data = "start"
        self._benchmark_control_pub.publish(msg)
        self.get_logger().info("Published benchmark 'start' signal to goal publisher")

    # -- Benchmarking helpers ----------------------------------------------

    def _make_goal_result(self, goal_index: int) -> dict:
        return {
            "goal_index": goal_index,
            "time_to_area_s": None,
            "samples_total": 0,
            "samples_area": 0,
            "sum_pos_err_area": 0.0,
            "sum_rot_err_area": 0.0,
            "reached_area": False,
        }

    def _update_goal_result(self, goal_result: dict, goal_time_s: float, pos_err: float, rot_err: float):
        goal_result["samples_total"] += 1
        in_area = pos_err <= IN_AREA_POS_M
        if in_area:
            goal_result["reached_area"] = True
            if goal_result["time_to_area_s"] is None:
                goal_result["time_to_area_s"] = goal_time_s
            goal_result["samples_area"] += 1
            goal_result["sum_pos_err_area"] += pos_err
            goal_result["sum_rot_err_area"] += rot_err

    def _finalize_goal_result(self, goal_result: dict) -> dict:
        samples_area = int(goal_result["samples_area"])
        goal_result["mean_pos_err_area_m"] = (
            goal_result["sum_pos_err_area"] / samples_area if samples_area > 0 else None
        )
        goal_result["mean_rot_err_area_rad"] = (
            goal_result["sum_rot_err_area"] / samples_area if samples_area > 0 else None
        )
        del goal_result["sum_pos_err_area"]
        del goal_result["sum_rot_err_area"]
        return goal_result

    def _build_episode_summary(self, episode_index: int = 0, results: Optional[list[dict]] = None) -> dict:
        goal_results = (results if results is not None else self._benchmark_results)[: self._benchmark_num_goals]
        area_times = [g["time_to_area_s"] for g in goal_results if g["time_to_area_s"] is not None]
        area_pos = [g["mean_pos_err_area_m"] for g in goal_results if g["mean_pos_err_area_m"] is not None]
        area_rot = [g["mean_rot_err_area_rad"] for g in goal_results if g["mean_rot_err_area_rad"] is not None]
        return {
            "episode_index": episode_index,
            "goal_count": len(goal_results),
            "goals": goal_results,
            "goals_reached_area": sum(1 for g in goal_results if g["reached_area"]),
            "mean_time_to_area_s": sum(area_times) / len(area_times) if area_times else None,
            "mean_pos_err_area_m": sum(area_pos) / len(area_pos) if area_pos else None,
            "mean_rot_err_area_rad": sum(area_rot) / len(area_rot) if area_rot else None,
        }

    def _save_benchmark_results(self):
        if not self._all_episode_results:
            self.get_logger().warn("No benchmark results collected; nothing to save")
            return None

        self._benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        run_name = extract_run_name_from_model_path(self._model_path)
        out_path = self._benchmark_output_dir / f"{timestamp}_{run_name}.yaml"

        episodes = []
        for take_idx, take_results in enumerate(self._all_episode_results):
            episodes.append(self._build_episode_summary(take_idx, take_results))

        # Aggregate across all episodes
        all_area_times = [ep["mean_time_to_area_s"] for ep in episodes if ep["mean_time_to_area_s"] is not None]
        all_area_pos = [ep["mean_pos_err_area_m"] for ep in episodes if ep["mean_pos_err_area_m"] is not None]
        all_area_rot = [ep["mean_rot_err_area_rad"] for ep in episodes if ep["mean_rot_err_area_rad"] is not None]
        total_reached = sum(ep["goals_reached_area"] for ep in episodes)
        total_goals = sum(ep["goal_count"] for ep in episodes)

        payload = {
            "metadata": {
                "model_path": self._model_path,
                "run_name": run_name,
                "robot": self.robot_prefix,
                "robot_gain": self._robot_gain,
                "rate_hz": self.control_rate,
                "action_scale": self.action_scale,
                "goals_file": self._benchmark_goals_file,
                "goal_count": self._benchmark_num_goals,
                "goal_timeout_s": self._goal_timeout_s,
                "num_takes": self._num_takes,
                "goal_convention": "x y z qw qx qy qz",
                "thresholds": {
                    "area": {"pos_m": IN_AREA_POS_M},
                },
            },
            "summary": {
                "episodes_completed": len(episodes),
                "goal_count": total_goals,
                "total_goals_executed": total_goals,
                "mean_time_to_area_s": sum(all_area_times) / len(all_area_times) if all_area_times else None,
                "mean_pos_err_area_m": sum(all_area_pos) / len(all_area_pos) if all_area_pos else None,
                "mean_rot_err_area_rad": sum(all_area_rot) / len(all_area_rot) if all_area_rot else None,
                "goals_reached_area": total_reached,
            },
            "episodes": episodes,
            "goals": [format_goal_line(goal) for goal in self._benchmark_goals[: self._benchmark_num_goals]],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(yaml_dump(payload))
            f.write("\n")
        self.get_logger().info(f"Benchmark results saved: {out_path}")
        return out_path

    def _start_benchmark_if_ready(self):
        if not self._benchmark or self._benchmark_started_at is not None:
            return
        if self.goal_position is None or self.goal_quaternion is None:
            return
        self._benchmark_started_at = time.time()
        self._benchmark_current_goal_index = 0
        self._current_goal_result = self._make_goal_result(0)
        self.get_logger().info("Benchmark started on first received goal")

    def _complete_benchmark(self):
        if self._benchmark_completed:
            return
        # Finalize current take's results
        if self._current_goal_result is not None and len(self._benchmark_results) < self._benchmark_num_goals:
            self._benchmark_results.append(self._finalize_goal_result(self._current_goal_result))
            self._current_goal_result = None
        self._benchmark_results = self._benchmark_results[: self._benchmark_num_goals]
        self._all_episode_results.append(self._benchmark_results)
        self._current_take += 1
        self.get_logger().info(f"Take {self._current_take}/{self._num_takes} completed")

        if self._current_take >= self._num_takes:
            self._benchmark_completed = True
            self._save_benchmark_results()
            self.get_logger().info("All takes completed; shutting down.")
            self.shutdown()
            try:
                rclpy.shutdown()
            except Exception:
                pass
        else:
            # Start next take in background thread
            Thread(target=self._reset_for_next_take, daemon=True, name="take_reset").start()

    def _reset_for_next_take(self):
        """Home robot, re-upload URScript, reset state, signal goal publisher."""
        self.is_running = False
        self.get_logger().info("Homing robot between takes...")

        # Stop URScript (robot goes home)
        self.rtde.stop_robot()
        time.sleep(8.0)

        # Re-upload URScript
        self.rtde.send_urscript()

        # Reset benchmark state
        with self.lock:
            self.goal_position = None
            self.goal_quaternion = None
            self.old_goal = None
            self.dof_targets = None

        self._benchmark_started_at = None
        self._benchmark_current_goal_index = -1
        self._benchmark_results = []
        self._current_goal_result = None

        self.is_running = True

        # Signal goal publisher to start next cycle
        self._publish_start_signal()
        self.get_logger().info(f"Take {self._current_take + 1}/{self._num_takes} armed. Waiting for goals...")

    def _update_benchmark(self, robot_state: RobotState, goal_state: GoalState):
        if not self._benchmark or self._benchmark_completed:
            return
        self._start_benchmark_if_ready()
        if self._benchmark_started_at is None or self._current_goal_result is None:
            return
        if self._benchmark_num_goals <= 0:
            return

        elapsed = time.time() - self._benchmark_started_at
        total_duration = self._benchmark_num_goals * self._goal_timeout_s
        target_goal_index = min(int(elapsed / self._goal_timeout_s), self._benchmark_num_goals - 1)

        while self._benchmark_current_goal_index < target_goal_index and len(self._benchmark_results) < self._benchmark_num_goals - 1:
            self._benchmark_results.append(self._finalize_goal_result(self._current_goal_result))
            self._benchmark_current_goal_index += 1
            self._current_goal_result = self._make_goal_result(self._benchmark_current_goal_index)
            self.get_logger().info(f"Benchmark goal window {self._benchmark_current_goal_index + 1}/{self._benchmark_num_goals}")

        pos_err = float(np.linalg.norm(robot_state.ee_position - goal_state.position))
        ori_err = float(np.linalg.norm(quat_box_minus(goal_state.quaternion, robot_state.ee_quaternion)))
        goal_time_s = elapsed - self._benchmark_current_goal_index * self._goal_timeout_s
        self._update_goal_result(self._current_goal_result, goal_time_s, pos_err, ori_err)

        if elapsed >= total_duration:
            self._complete_benchmark()

    # -- RTDE state reading (from cached reader thread) ---------------------

    def update_robot_state_from_cache(self) -> bool:
        """Fetch the latest cached RTDE state (written by the 125 Hz reader
        thread) and convert to the table-centre frame.

        Position: simple offset ``ee_pos = ee_pos_base + ROBOT_BASE_LOCAL``
        Quaternion / velocities: used as-is (no rotation between frames).
        This matches the v1 sim2real_node (validated on real robot).

        Returns:
            True if state was updated successfully.
        """
        cached = self.rtde.get_cached_state()
        if cached is None:
            self.get_logger().warn(
                "RTDE cache empty (reader not producing data yet)",
                throttle_duration_sec=1.0,
            )
            return False

        positions, velocities, tcp, tcp_speed, seq = cached

        # Skip if we already consumed this exact sample
        if seq == self._last_state_seq:
            return True  # state unchanged but still valid
        self._last_state_seq = seq

        # TCP pose: (x, y, z, rx, ry, rz) in robot-base frame
        ee_pos_base = np.array([tcp[0], tcp[1], tcp[2]], dtype=np.float32)
        ee_quat = rotvec_to_quat(tcp[3], tcp[4], tcp[5])

        # Shift position from robot-base frame to table-centre frame
        # (same as v1 – no rotation, just translation)
        ee_pos_table = ee_pos_base + ROBOT_BASE_LOCAL

        # TCP velocity: (vx, vy, vz, wx, wy, wz) – used as-is (no rotation)
        tcp_lin_vel = np.array([tcp_speed[0], tcp_speed[1], tcp_speed[2]], dtype=np.float32)
        tcp_ang_vel = np.array([tcp_speed[3], tcp_speed[4], tcp_speed[5]], dtype=np.float32)

        with self.lock:
            self.joint_positions = positions
            self.joint_velocities = velocities
            self.ee_position = ee_pos_table
            self.ee_quaternion = ee_quat
            self.tcp_linear_vel = tcp_lin_vel
            self.tcp_angular_vel = tcp_ang_vel

            if self.dof_targets is None:
                self.dof_targets = positions.copy()
                self.get_logger().info(f"Initialised DOF targets from RTDE: {self.dof_targets}")

        return True

    # -- ROS2 callbacks (goal) ----------------------------------------------

    def goal_pose_callback(self, msg: PoseStamped):
        """Handle incoming goal pose (already in source frame from goal_publisher)."""
        with self.lock:
            self.goal_position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ], dtype=np.float32)

            self.goal_quaternion = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ], dtype=np.float32)

            if self.old_goal is None or not np.allclose(self.old_goal, self.goal_position, atol=1e-6):
                pos_s = f"pos=({self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}, {self.goal_position[2]:.3f})"
                quat_s = f"quat=({self.goal_quaternion[0]:.3f}, {self.goal_quaternion[1]:.3f}, {self.goal_quaternion[2]:.3f}, {self.goal_quaternion[3]:.3f})"
                self.get_logger().info(f"New goal: {pos_s}, {quat_s}")
            self.old_goal = self.goal_position.copy()

        self._start_benchmark_if_ready()

    # -- Main control loop --------------------------------------------------

    def control_loop(self):
        """Main control loop running at 60 Hz."""
        if not self.is_running:
            return

        # 1. Read latest cached state (updated by 125 Hz reader thread)
        if not self.update_robot_state_from_cache():
            return

        # 2. Check that all data is available
        with self.lock:
            if any(x is None for x in [
                self.joint_positions, self.joint_velocities,
                self.ee_position, self.ee_quaternion,
                self.tcp_linear_vel, self.tcp_angular_vel,
                self.goal_position, self.goal_quaternion,
                self.dof_targets,
            ]):
                self.get_logger().info(
                    "Waiting for data...",
                    throttle_duration_sec=2.0,
                )
                return

            robot_state = RobotState(
                joint_positions=self.joint_positions.copy(),
                joint_velocities=self.joint_velocities.copy(),
                ee_position=self.ee_position.copy(),
                ee_quaternion=self.ee_quaternion.copy(),
                tcp_linear_vel=self.tcp_linear_vel.copy(),
                tcp_angular_vel=self.tcp_angular_vel.copy(),
            )
            goal_state = GoalState(
                position=self.goal_position.copy(),
                quaternion=self.goal_quaternion.copy(),
            )
            current_targets = self.dof_targets.copy()

        # 3. Build observation (24-dim, normalised)
        observation = build_observation(robot_state, goal_state)

        self._update_benchmark(robot_state, goal_state)
        if self._benchmark_completed:
            return

        # 4. Policy inference
        actions = self.policy.get_action(observation)

        # 5. Compute new DOF targets
        new_targets = compute_dof_targets(current_targets, actions, self.dt, self.action_scale)

        with self.lock:
            self.dof_targets = new_targets

        # 6. Send q_des via RTDE
        self.rtde.send_q_des(new_targets)

        # DEBUG log at ~2 Hz
        pos_err = np.linalg.norm(goal_state.position - robot_state.ee_position)
        ori_err = np.linalg.norm(quat_box_minus(goal_state.quaternion, robot_state.ee_quaternion))
        self.get_logger().info(
            f"pos_err={pos_err:.4f}m  ori_err={ori_err:.3f}rad | "
            f"EE=({robot_state.ee_position[0]:.3f},{robot_state.ee_position[1]:.3f},{robot_state.ee_position[2]:.3f},"
            f"{robot_state.ee_quaternion[0]:.3f},{robot_state.ee_quaternion[1]:.3f},{robot_state.ee_quaternion[2]:.3f},{robot_state.ee_quaternion[3]:.3f}) \n"
            f"Act=({actions[0]:.2f},{actions[1]:.2f},{actions[2]:.2f},{actions[3]:.2f},{actions[4]:.2f},{actions[5]:.2f})",
            throttle_duration_sec=0.5,
        )

    # -- Lifecycle ----------------------------------------------------------

    def start(self):
        self.is_running = True
        self.get_logger().info("Control loop started!")

    def stop(self):
        self.is_running = False
        self.get_logger().info("Control loop stopped!")

    def shutdown(self):
        self.stop()
        self.rtde.stop_robot()
        self.rtde.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Sim2Real V1 Policy Deployment (RTDE)")
    parser.add_argument(
        "--robot", type=str, default="gripper",
        choices=["gripper"],
        help="Robot prefix",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to TorchScript policy (.pt file)",
    )
    parser.add_argument(
        "--rate", type=float, default=60.0,
        help="Control loop rate in Hz",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for policy inference",
    )
    parser.add_argument(
        "--action-scale", type=float, default=0.5,
        help="Action scaling factor (sim v1 uses 2.0)",
    )
    parser.add_argument(
        "--rtde-rate", type=float, default=125.0,
        help="RTDE reader thread rate in Hz (default 125, UR native)",
    )
    parser.add_argument(
        "--robot-ip", type=str, default=ROBOT_HOST,
        help=f"Robot IP address (default {ROBOT_HOST})",
    )
    parser.add_argument(
        "--robot-gain", type=str, default="tuned",
        choices=["tuned", "naive"],
        help="Select the URScript gain profile to upload",
    )
    parser.add_argument(
        "--benchmark", action="store_true", default=False,
        help="Run the real benchmark with timed goal windows",
    )
    parser.add_argument("--goal-timeout-s", type=float, default=10.0, help="Benchmark timeout per goal in seconds")
    parser.add_argument("--num-goals", type=int, default=None, help="Number of benchmark goals to evaluate")
    parser.add_argument("--num-takes", type=int, default=1, help="Number of benchmark takes (episodes) to run")
    parser.add_argument("--goals-file", type=str, default=None, help="Optional JSON goals file for benchmark metadata")
    args = parser.parse_args()

    rclpy.init()

    node = None
    try:
        node = Sim2RealNode(
            robot_prefix=args.robot,
            model_path=args.model,
            control_rate=args.rate,
            rtde_rate=args.rtde_rate,
            device=args.device,
            action_scale=args.action_scale,
            robot_host=args.robot_ip,
            robot_gain=args.robot_gain,
            num_takes=args.num_takes,
            benchmark=args.benchmark,
            goal_timeout_s=args.goal_timeout_s,
            num_goals=args.num_goals,
            goals_file=args.goals_file,
        )

        node.start()

        if args.benchmark:
            if args.num_goals <= 0 and args.goals_file is None:
                raise ValueError("Benchmark mode requires --num-goals or --goals-file.")
            node.get_logger().info(
                f"Benchmark armed: goal_timeout_s={args.goal_timeout_s}, num_goals={node._benchmark_num_goals}. Waiting for first goal..."
            )

            rclpy.spin(node)
        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        if node is not None:
            node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
