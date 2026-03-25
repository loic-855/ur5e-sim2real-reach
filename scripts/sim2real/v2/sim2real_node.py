#!/usr/bin/env python3
"""
Sim2Real Node for UR5e pose control via RTDE – **V2** (velocity feedforward).

Extension of V2: the policy outputs 12 actions instead of 6.
  - actions[0:6]  → position increments (sent as q_des via registers 24-29)
  - actions[6:12] → velocity feedforward targets (sent as qdot_des via registers 30-35)

This node:
1. Connects to the UR5e robot via RTDE and runs a **dedicated reader
   thread at 125 Hz** that continuously caches the latest robot state
2. Subscribes to /goal_pose (ROS2) for target end-effector pose
3. The **control loop runs at 60 Hz**, builds 24-dim normalised
   observations, runs 12-dim policy inference
4. Sends q_des AND qdot_des via RTDE input registers (continuous stream)
5. URScript on the robot runs an impedance controller V3 with velocity
   feedforward at 500 Hz

Frame convention (same as V2):
  - RTDE TCP pose / velocity are in **robot-base** frame.
  - The policy expects everything in the **table-centre** frame.
  - Position conversion: simple offset ``ee_pos_table = ee_pos_base + ROBOT_BASE_LOCAL``

Usage:
    source ~/wwro_ws/install/local_setup.bash
    python3 sim2real_node.py --robot gripper --model path/to/policy.pt
    python3 scripts/sim2real/v2/sim2real_node.py --robot gripper --rate 60 --action-scale 0.3 --velocity-scale 0.15 --model path/to/policy.pt
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

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# Local imports
from observation_builder import (
    RobotState, GoalState,
    build_observation,
    compute_dof_targets_v2,
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
RTDE_CONFIG_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "rtde_input_v3.xml")
#URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_v3.script")
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_test.script")

# Home position
HOME_Q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

# ============================================================================
# Robot base position relative to TABLE CENTRE (table frame origin).
# ============================================================================
ROBOT_BASE_LOCAL = np.array([-0.52, 0.32, 0.02], dtype=np.float32)


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


# ============================================================================
# RTDE communication helper (q_des + qdot_des)
# ============================================================================

class RTDEController:
    """Manages RTDE connection, URScript upload, q_des + qdot_des streaming,
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
        self.setp = None          # q_des registers 24-29
        self.setp_vel = None      # qdot_des registers 30-35
        self.stop_reg = None
        self.control_rate_reg = None
        self.connected = False

        # --- Cached robot state (updated by reader thread) -----------------
        self._state_lock = Lock()
        self._cached_q: Optional[np.ndarray] = None
        self._cached_qd: Optional[np.ndarray] = None
        self._cached_tcp_pose: Optional[list] = None
        self._cached_tcp_speed: Optional[list] = None
        self._state_seq: int = 0
        self._reader_running = False
        self._reader_thread: Optional[Thread] = None

    # -- connection ---------------------------------------------------------

    def connect(self):
        """Open RTDE connection, configure recipes, start synchronisation."""
        conf = rtde_config.ConfigFile(self.config_file)
        output_names, output_types = conf.get_recipe("out")
        q_des_names, q_des_types = conf.get_recipe("q_des")
        qdot_des_names, qdot_des_types = conf.get_recipe("qdot_des")
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

        self.setp_vel = self.con.send_input_setup(qdot_des_names, qdot_des_types)
        if self.setp_vel is None:
            raise RuntimeError("RTDE: unable to configure qdot_des input")

        self.stop_reg = self.con.send_input_setup(stop_name, stop_type)
        if self.stop_reg is None:
            raise RuntimeError("RTDE: unable to configure stop signal")

        self.control_rate_reg = self.con.send_input_setup(control_rate_info, control_rate_type)
        if self.control_rate_reg is None:
            raise RuntimeError("RTDE: unable to configure control rate info input")

        if not self.con.send_start():
            raise RuntimeError("RTDE: unable to start synchronisation")

        self.connected = True
        print("[RTDE] Connected and synchronised (V2 – with qdot_des)")

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
        self._write_qdot_des([0.0] * 6)

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

    # -- q_des + qdot_des streaming -----------------------------------------

    def _write_q_des(self, q_des):
        """Write 6 joint values into RTDE input registers 24-29."""
        for i in range(6):
            self.setp.__dict__[f"input_double_register_{24 + i}"] = float(q_des[i])
        self.con.send(self.setp)

    def _write_qdot_des(self, qdot_des):
        """Write 6 velocity values into RTDE input registers 30-35."""
        for i in range(6):
            self.setp_vel.__dict__[f"input_double_register_{30 + i}"] = float(qdot_des[i])
        self.con.send(self.setp_vel)

    def send_targets(self, q_des: np.ndarray, qdot_des: np.ndarray):
        """Send both position and velocity targets in a single call."""
        self._write_q_des(q_des.tolist())
        self._write_qdot_des(qdot_des.tolist())

    def send_q_des(self, q_des: np.ndarray):
        """Send position targets only (backward compatibility)."""
        self._write_q_des(q_des.tolist())

    def receive_state(self):
        """Receive the latest robot state from RTDE (synchronous)."""
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
                pass
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_cached_state(self):
        """Return the latest cached state as a tuple, or None if not yet available."""
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
            # Zero velocity before stopping
            self._write_qdot_des([0.0] * 6)
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
# ROS2 + RTDE sim2real node (V2)
# ============================================================================

class Sim2RealNode(Node):
    """ROS2 node for sim2real policy deployment with velocity feedforward."""

    def __init__(
        self,
        robot_prefix: str = "gripper",
        model_path: Optional[str] = None,
        control_rate: float = 60.0,
        rtde_rate: float = 125.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        action_scale: float = 0.3,
        velocity_scale: float = 0.7,
        robot_host: str = ROBOT_HOST,
    ):
        super().__init__("sim2real_policy_node_v2")

        self.robot_prefix = robot_prefix
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        self.action_scale = action_scale
        self.velocity_scale = velocity_scale

        # State variables (protected by lock)
        self.lock = Lock()
        self.joint_positions: Optional[np.ndarray] = None
        self.joint_velocities: Optional[np.ndarray] = None
        self.ee_position: Optional[np.ndarray] = None
        self.ee_quaternion: Optional[np.ndarray] = None
        self.tcp_linear_vel: Optional[np.ndarray] = None
        self.tcp_angular_vel: Optional[np.ndarray] = None
        self.goal_position: Optional[np.ndarray] = None
        self.goal_quaternion: Optional[np.ndarray] = None
        self.old_goal: Optional[np.ndarray] = None

        # DOF targets (tracked across control steps)
        self.dof_targets: Optional[np.ndarray] = None

        # Control state
        self.is_running = False

        # Benchmarking
        self._benchmark = False
        self._benchmark_samples = []
        self._benchmark_lock = Lock()

        # ==================================================================
        # Load policy (24-dim obs → 12-dim actions)
        # ==================================================================
        self.get_logger().info(f"Loading policy from: {model_path or 'default'}")
        self.policy = load_policy(model_path=model_path, device=device)
        self.get_logger().info("Policy loaded successfully!")

        # ==================================================================
        # RTDE controller (reader runs at rtde_rate, decoupled from policy)
        # ==================================================================
        self.rtde = RTDEController(
            robot_host=robot_host,
            rtde_frequency=rtde_rate,
            control_rate=control_rate,
        )
        self.get_logger().info("Connecting to robot via RTDE (V2)...")
        self.rtde.connect()
        self.get_logger().info("Sending home movement...")
        self.rtde.send_home_movement()
        self.get_logger().info("Uploading URScript (impedance + velocity feedforward)...")
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

        # ==================================================================
        # Control timer
        # ==================================================================
        self.callback_group = ReentrantCallbackGroup()
        self.control_timer = self.create_timer(
            self.dt,
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info(f"Sim2Real V2 node initialised at {control_rate} Hz")
        self.get_logger().info(
            f"Action scale: {action_scale}  |  Velocity scale: {velocity_scale}"
        )

    # -- Benchmarking helpers ----------------------------------------------

    def _record_benchmark_sample(self, robot_state: RobotState, goal_state: GoalState):
        pos_err = float(np.linalg.norm(robot_state.ee_position - goal_state.position))
        ori_err_vec = quat_box_minus(goal_state.quaternion, robot_state.ee_quaternion)
        ori_err = float(np.linalg.norm(ori_err_vec))

        with self._benchmark_lock:
            self._benchmark_samples.append((time.time(), pos_err, ori_err))

    def _save_benchmark_results(self, out_path: str, metadata: dict):
        with self._benchmark_lock:
            samples = list(self._benchmark_samples)

        if len(samples) == 0:
            self.get_logger().warn("No benchmark samples collected; nothing to save")
            return

        pos_errs = [s[1] for s in samples]
        ori_errs = [s[2] for s in samples]

        results = {
            "metadata": metadata,
            "samples_count": len(samples),
            "mean_position_error_m": float(np.mean(pos_errs)),
            "median_position_error_m": float(np.median(pos_errs)),
            "mean_orientation_error_rad": float(np.mean(ori_errs)),
            "median_orientation_error_rad": float(np.median(ori_errs)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            mp = results["mean_position_error_m"]
            mo = results["mean_orientation_error_rad"]
            results["global_error"] = float((mp + mo) / 2.0)
        except Exception:
            results["global_error"] = None

        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            self.get_logger().info(f"Benchmark results saved: {out_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save benchmark results: {e}")

        return results

    # -- RTDE state reading (from cached reader thread) ---------------------

    def update_robot_state_from_cache(self) -> bool:
        """Fetch the latest cached RTDE state and convert to table-centre frame."""
        cached = self.rtde.get_cached_state()
        if cached is None:
            self.get_logger().warn(
                "RTDE cache empty (reader not producing data yet)",
                throttle_duration_sec=1.0,
            )
            return False

        positions, velocities, tcp, tcp_speed, seq = cached

        if seq == self._last_state_seq:
            return True
        self._last_state_seq = seq

        ee_pos_base = np.array([tcp[0], tcp[1], tcp[2]], dtype=np.float32)
        ee_quat = rotvec_to_quat(tcp[3], tcp[4], tcp[5])
        ee_pos_table = ee_pos_base + ROBOT_BASE_LOCAL

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

    # -- Main control loop --------------------------------------------------

    def control_loop(self):
        """Main control loop running at 60 Hz."""
        if not self.is_running:
            return

        # 1. Read latest cached state
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

        # 3. Build observation (24-dim, normalised – same as V2)
        observation = build_observation(robot_state, goal_state)

        # Benchmark recording
        if self._benchmark:
            try:
                self._record_benchmark_sample(robot_state, goal_state)
            except Exception:
                pass

        # 4. Policy inference (12-dim output)
        actions = self.policy.get_action(observation)

        # 5. Compute new position targets + velocity targets
        new_targets, velocity_targets = compute_dof_targets_v2(
            current_targets, actions, self.dt,
            self.action_scale, self.velocity_scale,
        )

        with self.lock:
            self.dof_targets = new_targets

        # 6. Send q_des + qdot_des via RTDE
        self.rtde.send_targets(new_targets, velocity_targets)

        # DEBUG log at ~2 Hz
        pos_err = np.linalg.norm(goal_state.position - robot_state.ee_position)
        ori_err = np.linalg.norm(quat_box_minus(goal_state.quaternion, robot_state.ee_quaternion))
        self.get_logger().info(
            f"pos_err={pos_err:.4f}m  ori_err={ori_err:.3f}rad | "
            f"EE=({robot_state.ee_position[0]:.3f},{robot_state.ee_position[1]:.3f},{robot_state.ee_position[2]:.3f}) "
            f"Goal=({goal_state.position[0]:.3f},{goal_state.position[1]:.3f},{goal_state.position[2]:.3f}) | "
            f"PosAct=({actions[0]:.2f},{actions[1]:.2f},{actions[2]:.2f},{actions[3]:.2f},{actions[4]:.2f},{actions[5]:.2f}) "
            f"VelAct=({actions[6]:.2f},{actions[7]:.2f},{actions[8]:.2f},{actions[9]:.2f},{actions[10]:.2f},{actions[11]:.2f})",
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
    parser = argparse.ArgumentParser(description="Sim2Real V2 Policy Deployment (RTDE + velocity feedforward)")
    parser.add_argument(
        "--robot", type=str, default="gripper",
        choices=["gripper", "screwdriver"],
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
        "--action-scale", type=float, default=0.3,
        help="Action scaling factor for position increments",
    )
    parser.add_argument(
        "--velocity-scale", type=float, default=0.7,
        help="Velocity scaling factor for feedforward (rad/s)",
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
        "--benchmark", action="store_true", default=False,
        help="Run policy for a fixed duration and record errors",
    )
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
            velocity_scale=args.velocity_scale,
            robot_host=args.robot_ip,
        )

        node.start()

        if args.benchmark:
            BENCHMARK_DURATION = 20.0
            node.get_logger().info(f"Starting benchmark for {BENCHMARK_DURATION}s...")
            node._benchmark = True

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            out_path = str(REPO_ROOT / "logs" / "benchmarks" / f"policy_bench_v2_{ts}.json")

            def _bench_thread():
                try:
                    time.sleep(BENCHMARK_DURATION)
                except Exception:
                    pass

                node._benchmark = False
                metadata = {
                    "model": Path(args.model).name if args.model else "default_policy",
                    "model_path": args.model,
                    "robot": args.robot,
                    "rate_hz": args.rate,
                    "action_scale": args.action_scale,
                    "velocity_scale": args.velocity_scale,
                    "duration_s": BENCHMARK_DURATION,
                    "version": "v2",
                }
                try:
                    node._save_benchmark_results(out_path, metadata)
                except Exception as e:
                    node.get_logger().error(f"Benchmark thread error: {e}")
                finally:
                    try:
                        node.get_logger().info("Benchmark finished; shutting down.")
                    except Exception:
                        pass
                    try:
                        node.shutdown()
                    except Exception:
                        pass
                    try:
                        rclpy.shutdown()
                    except Exception:
                        pass
                    try:
                        os._exit(0)
                    except Exception:
                        pass

            t = Thread(target=_bench_thread, daemon=True)
            t.start()

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
