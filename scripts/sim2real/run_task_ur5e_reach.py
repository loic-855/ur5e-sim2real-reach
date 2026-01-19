#!/usr/bin/env python3
"""
run_task_ur5e_reach.py
----------------------

ROS 2 node to run the UR5e reach policy (27D obs / 7D actions) trained in
Isaac Lab. It mirrors the sim-side observation/action preprocessing and sends
position targets to the scaled_joint_trajectory_controller.

- Subscribes: /joint_states (sensor_msgs/JointState)
- TF: base_link -> wrist_3_link (grasp pose proxy)
- Publishes: /scaled_joint_trajectory_controller/joint_trajectory (JointTrajectory)
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

from robots.ur5e_gripper_reach import UR5eGripperReachPolicy


class UR5eReachNode(Node):
    JOINT_STATE_TOPIC = "/joint_states"
    CMD_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
    BASE_FRAME = "base_link"
    GRASP_FRAME = "wrist_3_link"

    def __init__(self, policy_path: Optional[Path], env_path: Optional[Path]) -> None:
        super().__init__("ur5e_reach_policy")
        self.policy = UR5eGripperReachPolicy(policy_path=policy_path, env_path=env_path)

        # Base pose used in sim to place the robot/table
        self.base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_pos = UR5eGripperReachPolicy.sample_goal(self.base_pos)
        self.goal_interval_s = 5.0
        self.last_goal_time = self.get_clock().now()

        # Joint order mapping: robot real sends joints in different order than policy expects
        # Robot real order: [elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
        # Policy order:     [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
        self.robot_joint_order = [
            "elbow_joint",
            "shoulder_lift_joint",
            "shoulder_pan_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.policy_joint_order = self.policy.dof_names[:6]  # Exclude fingers for now
        # Create mapping: index in robot order -> index in policy order
        self.joint_reorder_indices = [
            self.policy_joint_order.index(joint) for joint in self.robot_joint_order
        ]
        self.get_logger().info(
            f"Joint order mapping: robot {self.robot_joint_order} -> policy {self.policy_joint_order}"
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(JointState, self.JOINT_STATE_TOPIC, self._state_cb, 10)
        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)

        # Policy runs at sim control rate (60 Hz = dt * decimation)
        self.timer = self.create_timer(self.policy.policy_dt, self._step_cb)
        self.get_logger().info("UR5e reach policy node started (60 Hz).")
        
        # Flag to track if we've received joint state data
        self.has_received_joint_state = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _state_cb(self, msg: JointState) -> None:
        """Callback for joint state updates from real robot.
        
        Reorders joints to match policy training order since robot real
        publishes in a different order than the controller expects.
        """
        positions = []
        velocities = []
        # Map incoming joint names to the robot order
        name_to_value = {n: v for n, v in zip(msg.name, msg.position)}
        name_to_vel = {n: v for n, v in zip(msg.name, msg.velocity)}
        
        # Get values in robot real order
        for name in self.robot_joint_order:
            positions.append(name_to_value.get(name, 0.0))
            velocities.append(name_to_vel.get(name, 0.0))
        
        # Reorder to policy order
        positions_reordered = [positions[i] for i in self.joint_reorder_indices]
        velocities_reordered = [velocities[i] for i in self.joint_reorder_indices]
        
        # Add finger joints (not in real robot feedback, will be set to 0)
        positions_reordered.extend([0.0, 0.0])  # left_finger, right_finger
        velocities_reordered.extend([0.0, 0.0])
        
        self.policy.update_joint_state(positions_reordered, velocities_reordered)
        self.has_received_joint_state = True

    def _step_cb(self) -> None:
        if not self.has_received_joint_state:
            return

        # Resample goal periodically
        now = self.get_clock().now()
        if (now - self.last_goal_time).nanoseconds * 1e-9 >= self.goal_interval_s:
            self.goal_pos = UR5eGripperReachPolicy.sample_goal(self.base_pos)
            self.last_goal_time = now

        grasp_pos = self._lookup_grasp_position()
        if grasp_pos is None:
            return

        targets = self.policy.forward(self.policy.policy_dt, self.goal_pos, grasp_pos)
        if targets is None:
            return

        # Reorder targets from policy order back to robot real order (first 6 joints only)
        # targets are in policy order [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        # robot real expects order [elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3]
        targets_policy_order = targets[:6]  # Only first 6 (exclude fingers)
        targets_robot_order = [0.0] * 6
        for policy_idx, robot_idx in enumerate(self.joint_reorder_indices):
            targets_robot_order[robot_idx] = targets_policy_order[policy_idx]

        traj = JointTrajectory()
        traj.joint_names = self.robot_joint_order
        point = JointTrajectoryPoint()
        point.positions = targets_robot_order
        point.time_from_start = Duration(sec=0, nanosec=int(self.policy.policy_dt * 1e9))
        traj.points.append(point)
        self.pub.publish(traj)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _lookup_grasp_position(self) -> Optional[np.ndarray]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.BASE_FRAME,
                self.GRASP_FRAME,
                Time(),
                timeout=RclpyDuration(seconds=0.01),
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"TF lookup failed: {exc}", throttle_duration_sec=2.0)
            return None


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Run UR5e reach policy on ROS2.")
    parser.add_argument("--policy", type=Path, default=None, help="Path to TorchScript policy (.pt)")
    parser.add_argument("--env", type=Path, default=None, help="Path to env.yaml")
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = UR5eReachNode(parsed.policy, parsed.env)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
