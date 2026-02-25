#!/usr/bin/env python3
"""
Sim2Real Node for UR5e pose control via RTDE – **V2** (normalised observations).

This node:
1. Connects to the UR5e robot via RTDE to read joint states (actual_q, actual_qd),
   TCP pose (actual_TCP_pose) and TCP velocity (actual_TCP_speed)
2. Subscribes to /goal_pose (ROS2) for target end-effector pose
3. Builds 24-dim normalised observations matching IsaacSim v2
4. Runs policy inference at a configurable rate (default 60 Hz)
5. Sends q_des via RTDE input registers (continuous stream)
6. URScript on the robot runs an impedance controller with first-order
   interpolation filter at 500 Hz for smooth motion

Frame convention:
  - RTDE TCP pose / velocity are in **robot-base** frame.
  - The policy expects everything in the **source** (table-centre) frame.
  - Conversion uses ``subtract_frame_transforms`` logic identical to
    IsaacLab's FrameTransformer with ``source_frame_offset``.

Usage:
    source ~/wwro_ws/install/local_setup.bash
    python3 sim2real_node.py --robot gripper
    python3 sim2real_node.py --robot gripper --rate 60 --action-scale 7.0
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
    compute_dof_targets,
    quat_box_minus,
    quat_conjugate,
    quat_multiply,
    quat_rotate_inverse,
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
REPO_ROOT = Path(__file__).resolve().parents[2]
RTDE_CONFIG_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "rtde_input_v2.xml")
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control.script")

# Home position
HOME_Q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

# ============================================================================
# Source frame offset (table-centre frame relative to robot base)
# Must match the FrameTransformerCfg.source_frame_offset in the sim:
#   pos  = (-(TABLE_WIDTH/2 - 0.08), TABLE_DEPTH/2 - 0.08, -MOUNT_HEIGHT)
#        = (-0.52, 0.32, -0.02)
#   rot  = (0, 0, 0, 1)  → 180° around Z
# ============================================================================
SOURCE_POS = np.array([-0.52, 0.32, -0.02], dtype=np.float32)
SOURCE_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # (w,x,y,z) = 180° Z


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
# Frame transform helpers  (numpy, single-sample)
# ============================================================================

def subtract_frame_transforms(
    t01: np.ndarray,
    q01: np.ndarray,
    t02: np.ndarray,
    q02: np.ndarray,
) -> tuple:
    """T_{12} = T_{01}^{-1} · T_{02}.

    Converts a pose expressed in frame 0 into frame 1.
    Matches ``isaaclab.utils.math.subtract_frame_transforms`` exactly.

    Args:
        t01: Position of frame 1 w.r.t. frame 0 (3,).
        q01: Orientation of frame 1 w.r.t. frame 0 (w,x,y,z) (4,).
        t02: Position of frame 2 w.r.t. frame 0 (3,).
        q02: Orientation of frame 2 w.r.t. frame 0 (w,x,y,z) (4,).

    Returns:
        (t12, q12): Pose of frame 2 in frame 1.
    """
    q10 = quat_conjugate(q01)                        # q01^{-1}
    q12 = quat_multiply(q10, q02)                    # rotation
    t12 = quat_rotate_inverse(q01, t02 - t01)        # quat_apply(q10, t02 - t01)
    # Normalise quaternion
    q12 = q12 / np.linalg.norm(q12)
    return t12.astype(np.float32), q12.astype(np.float32)


def rotate_velocity_to_source(vel: np.ndarray, source_quat: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector from robot-base frame to source frame.

    Equivalent to ``quat_apply(quat_inv(source_quat), vel)``.
    Since source_quat represents source→base, we need base→source = inv(source_quat).
    """
    return quat_rotate_inverse(source_quat, vel)


# ============================================================================
# RTDE communication helper
# ============================================================================

class RTDEController:
    """Manages RTDE connection, URScript upload, and q_des streaming."""

    def __init__(
        self,
        robot_host: str = ROBOT_HOST,
        robot_port: int = ROBOT_PORT,
        primary_port: int = ROBOT_PRIMARY_PORT,
        config_file: str = RTDE_CONFIG_FILE,
        urscript_file: str = URSCRIPT_FILE,
        rtde_frequency: float = 125.0,
    ):
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.primary_port = primary_port
        self.config_file = config_file
        self.urscript_file = urscript_file
        self.rtde_frequency = rtde_frequency

        self.con: Optional[rtde.RTDE] = None
        self.setp = None
        self.stop_reg = None
        self.connected = False

    # -- connection ---------------------------------------------------------

    def connect(self):
        """Open RTDE connection, configure recipes, start synchronisation."""
        conf = rtde_config.ConfigFile(self.config_file)
        output_names, output_types = conf.get_recipe("out")
        q_des_names, q_des_types = conf.get_recipe("q_des")
        stop_name, stop_type = conf.get_recipe("stop")

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

        if not self.con.send_start():
            raise RuntimeError("RTDE: unable to start synchronisation")

        self.connected = True
        print("[RTDE] Connected and synchronised")

    # -- URScript -----------------------------------------------------------

    def send_urscript(self):
        """Upload and start the URScript impedance controller on the robot."""
        self.stop_reg.input_bit_register_64 = False
        self.con.send(self.stop_reg)

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
        """Receive the latest robot state from RTDE.

        Returns:
            RTDE state object with ``actual_q``, ``actual_qd``,
            ``actual_TCP_pose`` and ``actual_TCP_speed`` fields,
            or None on failure.
        """
        if self.con is None:
            return None
        return self.con.receive()

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        action_scale: float = 7.0,
        robot_host: str = ROBOT_HOST,
    ):
        super().__init__("sim2real_policy_node_v2")

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
        self._benchmark = False
        self._benchmark_samples = []
        self._benchmark_lock = Lock()

        # ==================================================================
        # Load policy
        # ==================================================================
        self.get_logger().info(f"Loading policy from: {model_path or 'default'}")
        self.policy = load_policy(model_path=model_path, device=device)
        self.get_logger().info("Policy loaded successfully!")

        # ==================================================================
        # RTDE controller
        # ==================================================================
        self.rtde = RTDEController(
            robot_host=robot_host,
            rtde_frequency=control_rate,
        )
        self.get_logger().info("Connecting to robot via RTDE...")
        self.rtde.connect()
        self.get_logger().info("Uploading URScript (impedance + first-order filter)...")
        self.rtde.send_urscript()
        self.get_logger().info("RTDE ready!")

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
        self.get_logger().info(f"Action scale: {action_scale} (sim uses 7.0)")

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

    # -- RTDE state reading -------------------------------------------------

    def update_robot_state_from_rtde(self) -> bool:
        """Read actual_q, actual_qd, actual_TCP_pose and actual_TCP_speed
        from RTDE and convert to the source (table-centre) frame.

        The FrameTransformer in IsaacSim computes:
          T_{source→tcp} = T_{source→base}^{-1} · T_{base→tcp}

        Here ``source`` is the table-centre frame whose offset from
        the robot base is given by SOURCE_POS / SOURCE_QUAT.
        We replicate this with ``subtract_frame_transforms``.

        For velocities, the sim does:
          v_source = quat_apply(quat_inv(SOURCE_QUAT), v_base)
        """
        state = self.rtde.receive_state()
        if state is None:
            self.get_logger().warn("RTDE receive failed", throttle_duration_sec=1.0)
            return False

        positions = np.array(state.actual_q, dtype=np.float32)
        velocities = np.array(state.actual_qd, dtype=np.float32)

        # TCP pose: (x, y, z, rx, ry, rz) in robot-base frame
        tcp = state.actual_TCP_pose
        ee_pos_base = np.array([tcp[0], tcp[1], tcp[2]], dtype=np.float32)
        ee_quat_base = rotvec_to_quat(tcp[3], tcp[4], tcp[5])

        # TCP velocity: (vx, vy, vz, wx, wy, wz) in robot-base frame
        tcp_speed = state.actual_TCP_speed
        tcp_lin_vel_base = np.array([tcp_speed[0], tcp_speed[1], tcp_speed[2]], dtype=np.float32)
        tcp_ang_vel_base = np.array([tcp_speed[3], tcp_speed[4], tcp_speed[5]], dtype=np.float32)

        # Transform TCP pose from base frame to source frame
        # T_source_tcp = T_source_base^{-1} · T_base_tcp
        # Where T_source_base is (SOURCE_POS, SOURCE_QUAT) in base frame
        ee_pos_source, ee_quat_source = subtract_frame_transforms(
            SOURCE_POS, SOURCE_QUAT,  # frame 1 (source) w.r.t. frame 0 (base)
            ee_pos_base, ee_quat_base,  # frame 2 (tcp) w.r.t. frame 0 (base)
        )

        # Transform velocities from base frame to source frame
        # The sim applies quat_apply(quat_inv(BASE_ROTATION_LOCAL), vel)
        # which rotates from base to source frame
        tcp_lin_vel_source = rotate_velocity_to_source(tcp_lin_vel_base, SOURCE_QUAT)
        tcp_ang_vel_source = rotate_velocity_to_source(tcp_ang_vel_base, SOURCE_QUAT)

        with self.lock:
            self.joint_positions = positions
            self.joint_velocities = velocities
            self.ee_position = ee_pos_source
            self.ee_quaternion = ee_quat_source
            self.tcp_linear_vel = tcp_lin_vel_source
            self.tcp_angular_vel = tcp_ang_vel_source

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

        # 1. Read joint state + TCP pose + TCP velocity from RTDE
        if not self.update_robot_state_from_rtde():
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

        # If benchmarking: record sample
        if self._benchmark:
            try:
                self._record_benchmark_sample(robot_state, goal_state)
            except Exception:
                pass

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
            f"EE=({robot_state.ee_position[0]:.3f},{robot_state.ee_position[1]:.3f},{robot_state.ee_position[2]:.3f}) "
            f"Goal=({goal_state.position[0]:.3f},{goal_state.position[1]:.3f},{goal_state.position[2]:.3f}) | "
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
    parser = argparse.ArgumentParser(description="Sim2Real V2 Policy Deployment (RTDE)")
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
        "--action-scale", type=float, default=7.0,
        help="Action scaling factor (sim v2 uses 7.0)",
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
            device=args.device,
            action_scale=args.action_scale,
            robot_host=args.robot_ip,
        )

        node.start()

        if args.benchmark:
            BENCHMARK_DURATION = 90.0
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
