#!/usr/bin/env python3
"""
run_task_ur5e_reach.py
----------------------

ROS 2 node to run the UR5e reach policy trained in Isaac Lab on a dual-arm
robot setup. Controls only the gripper robot (6 arm joints), gripper stays open.
Screwdriver robot remains at default position.

- Subscribes: /joint_states (sensor_msgs/JointState)
- TF: base_link -> gripper_wrist_3_link (grasp pose proxy)
- Publishes: /gripper_scaled_joint_trajectory_controller/joint_trajectory
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
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
import tf2_ros

from robots.ur5e_gripper_reach import UR5eGripperReachPolicy


class UR5eReachNode(Node):
    JOINT_STATE_TOPIC = "/joint_states"
    CMD_TOPIC = "/gripper_scaled_joint_trajectory_controller/joint_trajectory"
    GOAL_MARKER_TOPIC = "/goal_marker"
    BASE_FRAME = "table"
    GRASP_FRAME = "gripper_tcp"  # Use gripper robot's end effector frame

    def __init__(self, policy_path: Optional[Path], env_path: Optional[Path]) -> None:
        super().__init__("ur5e_reach_policy")
        self.policy = UR5eGripperReachPolicy(policy_path=policy_path, env_path=env_path)

        # Base pose used in sim to place the robot/table
        self.base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_pos = UR5eGripperReachPolicy.sample_goal(self.base_pos)
        self.goal_interval_s = 5.0
        self.last_goal_time = self.get_clock().now()

        # Joint names in ROS2 order (how robot publishes them)
        self.robot_joint_names_ros_order = [
            "gripper_elbow_joint",
            "gripper_shoulder_lift_joint",
            "gripper_shoulder_pan_joint",
            "gripper_wrist_1_joint",
            "gripper_wrist_2_joint",
            "gripper_wrist_3_joint",
        ]
        
        # Joint names in policy order (how Isaac Lab trained)
        self.robot_joint_names_policy_order = [
            "gripper_shoulder_pan_joint",
            "gripper_shoulder_lift_joint",
            "gripper_elbow_joint",
            "gripper_wrist_1_joint",
            "gripper_wrist_2_joint",
            "gripper_wrist_3_joint",
        ]
        
        # Create mapping: ROS2 index -> Policy index
        self.ros_to_policy_indices = [
            self.robot_joint_names_policy_order.index(name)
            for name in self.robot_joint_names_ros_order
        ]
        # Create mapping: Policy index -> ROS2 index
        self.policy_to_ros_indices = [
            self.robot_joint_names_ros_order.index(name)
            for name in self.robot_joint_names_policy_order
        ]

        self.tf_buffer = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(JointState, self.JOINT_STATE_TOPIC, self._state_cb, 10)
        self.pub = self.create_publisher(JointTrajectory, self.CMD_TOPIC, 10)
        self.marker_pub = self.create_publisher(Marker, self.GOAL_MARKER_TOPIC, 10)

        # Policy runs at sim control rate
        self.timer = self.create_timer(self.policy.policy_dt, self._step_cb)
        self.get_logger().info(
            f"UR5e reach policy node started (control rate: {1.0/self.policy.policy_dt:.1f} Hz)."
        )
        
        # Flag to track if we've received joint state data
        self.has_received_joint_state = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _state_cb(self, msg: JointState) -> None:
        """Callback for joint state updates from real robot.
        
        Extracts gripper robot joint states and reorders them from ROS2 order to policy order.
        """
        # Create mapping from joint name to position/velocity
        name_to_pos = {n: v for n, v in zip(msg.name, msg.position)}
        name_to_vel = {n: v for n, v in zip(msg.name, msg.velocity)}
        
        # Extract gripper robot joints in ROS2 order
        positions_ros_order = []
        velocities_ros_order = []
        for name in self.robot_joint_names_ros_order:
            if name in name_to_pos:
                positions_ros_order.append(name_to_pos[name])
                velocities_ros_order.append(name_to_vel.get(name, 0.0))
            else:
                self.get_logger().warn(f"Joint {name} not found in joint_states", throttle_duration_sec=5.0)
                return
        
        # Reorder from ROS2 order to policy order
        positions_policy_order = [positions_ros_order[i] for i in self.ros_to_policy_indices]
        velocities_policy_order = [velocities_ros_order[i] for i in self.ros_to_policy_indices]
        
        # Update policy with joint states in policy order
        self.policy.update_joint_state(positions_policy_order, velocities_policy_order)
        
        if not self.has_received_joint_state:
            self.get_logger().info(f"First joint state received:")
            self.get_logger().info(f"ROS2 order -> Policy order mapping:")
            for i, ros_name in enumerate(self.robot_joint_names_ros_order):
                policy_name = self.robot_joint_names_policy_order[self.ros_to_policy_indices[i]]
                self.get_logger().info(f"  {ros_name} (pos={positions_ros_order[i]:.3f}) -> {policy_name} (pos={positions_policy_order[self.ros_to_policy_indices[i]]:.3f})")
        
        self.has_received_joint_state = True

    def _step_cb(self) -> None:
        """Policy control loop - runs at policy rate."""
        if not self.has_received_joint_state:
            return

        # Resample goal periodically
        now = self.get_clock().now()
        if (now - self.last_goal_time).nanoseconds * 1e-9 >= self.goal_interval_s:
            self.goal_pos = UR5eGripperReachPolicy.sample_goal(self.base_pos)
            self.last_goal_time = now
            self.get_logger().info(f"New goal: {self.goal_pos}")
            self._publish_goal_marker()

        grasp_pos = self._lookup_grasp_position()
        if grasp_pos is None:
            return

        # Get 6 joint targets from policy (arm joints only, in policy order)
        targets_policy_order = self.policy.forward(self.policy.policy_dt, self.goal_pos, grasp_pos)
        if targets_policy_order is None:
            return
        
        # Reorder targets from policy order to ROS2 order for publishing
        targets_ros_order = [targets_policy_order[i] for i in self.policy_to_ros_indices]
        
        # Debug: Log targets periodically (every 2 seconds)
        if int(now.nanoseconds * 1e-9) % 2 == 0:
            self.get_logger().info(f"Grasp pos: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
            self.get_logger().info(f"Targets (policy order): {[f'{t:.3f}' for t in targets_policy_order]}")
            self.get_logger().info(f"Targets (ROS2 order): {[f'{t:.3f}' for t in targets_ros_order]}")

        # Publish trajectory command
        traj = JointTrajectory()
        traj.joint_names = self.robot_joint_names_ros_order
        point = JointTrajectoryPoint()
        point.positions = targets_ros_order
        point.time_from_start = Duration(sec=0, nanosec=int(self.policy.policy_dt * 1e9))
        traj.points.append(point)
        self.pub.publish(traj)
        
        # Publish goal marker for visualization
        self._publish_goal_marker()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _publish_goal_marker(self) -> None:
        """Publish a sphere marker at the goal position for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = self.BASE_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(self.goal_pos[0])
        marker.pose.position.y = float(self.goal_pos[1])
        marker.pose.position.z = float(self.goal_pos[2])
        marker.pose.orientation.w = 1.0
        
        # Scale (5cm diameter sphere)
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Color (bright green, semi-transparent)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.lifetime = Duration(sec=0, nanosec=0)  # Persist until replaced
        
        self.marker_pub.publish(marker)
    
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
        # Ensure any policy loggers are closed and flushed
        try:
            if getattr(node, 'policy', None) is not None:
                node.policy.close_logger()
        except Exception as e:
            node.get_logger().warn(f"Failed to close policy logger: {e}", throttle_duration_sec=5.0)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
