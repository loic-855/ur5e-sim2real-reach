#!/usr/bin/env python3
"""
Sim2Real Node for UR5e pose control via RTDE.

This node:
1. Connects to the UR5e robot via RTDE to read joint states (actual_q, actual_qd)
   and TCP pose (actual_TCP_pose)
2. Subscribes to /goal_pose (ROS2) for target end-effector pose
3. Builds observations matching IsaacSim normalization
4. Runs policy inference at a configurable rate (default 60 Hz)
5. Sends q_des via RTDE input registers (continuous stream)
6. URScript on the robot runs an impedance controller with first-order
   interpolation filter at 500 Hz for smooth motion

The TCP pose from RTDE is in the robot base frame.  A constant offset
(ROBOT_BASE_LOCAL) is applied to express it in the *table-centre* frame
used by the policy.

Usage:
    # Source ROS2 workspace first (needed for goal_pose subscriber)
    source ~/wwro_ws/install/local_setup.bash

    # Run the node (60 Hz default)
    python3 sim2real_node.py --robot gripper

    # Run at 30 Hz with custom action scale
    python3 sim2real_node.py --robot gripper --rate 30 --action-scale 5.0
"""

import math
import torch
import numpy as np
import argparse
import logging
import time
import socket
import sys
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
RTDE_CONFIG_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "rtde_input.xml")
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control.script")

# Home position
HOME_Q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

# Robot base position relative to TABLE CENTRE (table frame origin).
# actual_TCP_pose from RTDE is in the robot-base frame; we add this offset
# to convert to the table-centre frame used by the policy / goal publisher.
ROBOT_BASE_LOCAL = np.array([-0.52, -0.32, 0.0], dtype=np.float32)


# ============================================================================
# Rotation-vector  →  quaternion  conversion
# ============================================================================

def rotvec_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert a rotation vector (axis-angle) to a unit quaternion (w, x, y, z).

    The UR controller reports tool orientation as a rotation vector whose
    direction is the rotation axis and whose magnitude is the angle in
    radians.

    Args:
        rx, ry, rz: Rotation vector components.

    Returns:
        Quaternion as np.float32 array [w, x, y, z].
    """
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    half = angle / 2.0
    s = math.sin(half) / angle
    return np.array([
        math.cos(half),
        rx * s,
        ry * s,
        rz * s,
    ], dtype=np.float32)


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
        rtde_frequency: float = 60.0,
    ):
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.primary_port = primary_port
        self.config_file = config_file
        self.urscript_file = urscript_file
        self.rtde_frequency = rtde_frequency

        self.con: Optional[rtde.RTDE] = None
        self.setp = None  # q_des input registers
        self.stop_reg = None  # stop boolean register
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
        # First clear the stop flag
        self.stop_reg.input_bit_register_64 = False
        self.con.send(self.stop_reg)

        # Send initial home position so the controller has valid q_des
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

        # Wait for the script to start on the robot
        time.sleep(3.0)
        print("[RTDE] URScript started on robot")

    # -- q_des streaming ----------------------------------------------------

    def _write_q_des(self, q_des):
        """Write 6 joint values into RTDE input registers 24-29."""
        for i in range(6):
            self.setp.__dict__[f"input_double_register_{24 + i}"] = float(q_des[i])
        self.con.send(self.setp)

    def send_q_des(self, q_des: np.ndarray):
        """Send desired joint positions to the robot via RTDE.

        Args:
            q_des: Joint position targets [6] in simulation order
                   (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
        """
        self._write_q_des(q_des.tolist())

    def receive_state(self):
        """Receive the latest robot state from RTDE.

        Returns:
            RTDE state object with ``actual_q``, ``actual_qd`` and
            ``actual_TCP_pose`` fields, or None on failure.
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
            print("[RTDE] Stop signal sent – robot returning to home")
        except Exception as e:
            print(f"[RTDE] Error sending stop: {e}")

    def disconnect(self):
        """Clean disconnect."""
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
        super().__init__("sim2real_policy_node")

        self.robot_prefix = robot_prefix
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        self.action_scale = action_scale

        # State variables (protected by lock)
        self.lock = Lock()
        self.joint_positions: Optional[np.ndarray] = None
        self.joint_velocities: Optional[np.ndarray] = None
        self.ee_position: Optional[np.ndarray] = None
        self.ee_quaternion: Optional[np.ndarray] = None
        self.goal_position: Optional[np.ndarray] = None
        self.goal_quaternion: Optional[np.ndarray] = None
        self.old_goal: Optional[np.ndarray] = None

        # DOF targets (tracked across control steps)
        self.dof_targets: Optional[np.ndarray] = None

        # Control state
        self.is_running = False

        # ==================================================================
        # Load policy
        # ==================================================================
        self.get_logger().info(f"Loading policy from: {model_path or 'default'}")
        self.policy = load_policy(model_path=model_path, device=device)
        self.get_logger().info("Policy loaded successfully!")

        # ==================================================================
        # RTDE controller (replaces ROS2 trajectory action client)
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
        # Control timer (60 Hz)
        # ==================================================================
        self.callback_group = ReentrantCallbackGroup()
        self.control_timer = self.create_timer(
            self.dt,
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info(f"Sim2Real node initialised at {control_rate} Hz")
        self.get_logger().info(f"Robot prefix: {robot_prefix}")
        self.get_logger().info(f"Action scale: {action_scale} (IsaacSim uses 7.5)")

    # -- RTDE state reading -------------------------------------------------

    def update_robot_state_from_rtde(self) -> bool:
        """Read actual_q, actual_qd and actual_TCP_pose from RTDE.

        The TCP pose (x, y, z, rx, ry, rz) is in the robot-base frame.
        We convert the rotation vector to a quaternion and shift the
        position by ROBOT_BASE_LOCAL so that it is expressed in the
        table-centre frame expected by the policy.

        Returns:
            True if state was updated successfully.
        """
        state = self.rtde.receive_state()
        if state is None:
            self.get_logger().warn("RTDE receive failed", throttle_duration_sec=1.0)
            return False

        positions = np.array(state.actual_q, dtype=np.float32)
        velocities = np.array(state.actual_qd, dtype=np.float32)

        # TCP pose: (x, y, z, rx, ry, rz) in robot-base frame
        tcp = state.actual_TCP_pose  # list of 6 floats
        ee_pos_base = np.array([tcp[0], tcp[1], tcp[2]], dtype=np.float32)
        ee_quat = rotvec_to_quat(tcp[3], tcp[4], tcp[5])

        # Shift position from robot-base frame to table-centre frame
        ee_pos_table = ee_pos_base + ROBOT_BASE_LOCAL

        with self.lock:
            self.joint_positions = positions
            self.joint_velocities = velocities
            self.ee_position = ee_pos_table
            self.ee_quaternion = ee_quat

            # Initialise dof_targets on first read
            if self.dof_targets is None:
                self.dof_targets = positions.copy()
                self.get_logger().info(f"Initialised DOF targets from RTDE: {self.dof_targets}")

        return True

    # -- ROS2 callbacks (goal) ----------------------------------------------

    def goal_pose_callback(self, msg: PoseStamped):
        """Handle incoming goal pose."""
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

        # 1. Read joint state + TCP pose from RTDE
        if not self.update_robot_state_from_rtde():
            return

        # 2. Check that all data is available
        with self.lock:
            if any(x is None for x in [
                self.joint_positions, self.joint_velocities,
                self.ee_position, self.ee_quaternion,
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
            )
            goal_state = GoalState(
                position=self.goal_position.copy(),
                quaternion=self.goal_quaternion.copy(),
            )
            current_targets = self.dof_targets.copy()

        # 3. Build observation
        observation = build_observation(robot_state, goal_state)

        # 4. Policy inference
        actions = self.policy.get_action(observation)

        # 5. Compute new DOF targets
        new_targets = compute_dof_targets(current_targets, actions, self.dt, self.action_scale)

        with self.lock:
            self.dof_targets = new_targets

        # 6. Send q_des via RTDE
        self.rtde.send_q_des(new_targets)

        # DEBUG log at ~2 Hz
        self.get_logger().info(
            f"EE: ({robot_state.ee_position[0]:.3f}, {robot_state.ee_position[1]:.3f}, {robot_state.ee_position[2]:.3f}) | "
            f"Goal: ({goal_state.position[0]:.3f}, {goal_state.position[1]:.3f}, {goal_state.position[2]:.3f}) | "
            f"Act: ({actions[0]:.2f}, {actions[1]:.2f}, {actions[2]:.2f}, {actions[3]:.2f}, {actions[4]:.2f}, {actions[5]:.2f})",
            throttle_duration_sec=0.5,
        )

    # -- Lifecycle ----------------------------------------------------------

    def start(self):
        """Start the control loop."""
        self.is_running = True
        self.get_logger().info("Control loop started!")

    def stop(self):
        """Stop the control loop and send robot home."""
        self.is_running = False
        self.get_logger().info("Control loop stopped!")

    def shutdown(self):
        """Full shutdown: stop robot, disconnect RTDE."""
        self.stop()
        self.rtde.stop_robot()
        self.rtde.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Sim2Real Policy Deployment (RTDE)")
    parser.add_argument(
        "--robot",
        type=str,
        default="gripper",
        choices=["gripper", "screwdriver"],
        help="Robot prefix (gripper or screwdriver)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to TorchScript policy (.pt file)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=60.0,
        help="Control loop rate in Hz",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for policy inference",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=3.0,
        help="Action scaling factor (default 3.0, IsaacSim uses 7.5)",
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=ROBOT_HOST,
        help=f"Robot IP address (default {ROBOT_HOST})",
    )
    args = parser.parse_args()

    # Initialise ROS2 (needed for goal_pose subscriber)
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
