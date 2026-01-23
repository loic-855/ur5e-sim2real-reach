#!/usr/bin/env python3
"""
run_task_ur5e_reach.py
----------------------
ROS2 node for UR5e reach policy with gripper control and periodic resets.

Subscribes: /joint_states
Publishes:  /gripper_scaled_joint_trajectory_controller/joint_trajectory
Action:     /gripper_scaled_joint_trajectory_controller/follow_joint_trajectory (homing)
Services:   /on_twofg7_grip_external, /on_twofg7_release_external (gripper control)
TF:         table -> gripper_tcp (grasp position)

Reset behavior:
    - Every 5 seconds: send action to return to home
    - Wait for action success callback
    - Reset policy state and sample new goal
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from rclpy.action import ActionClient
from rclpy.action.client import GoalStatus

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from control_msgs.action import FollowJointTrajectory
import tf2_ros

from robots.ur5e_gripper_reach import UR5eGripperReachPolicy

# Import gripper service (optional - will work without if not available)
HAS_GRIPPER_SERVICE = False
OnTwofg7 = None
try:
    from wwro_msgs.srv import OnTwofg7
    HAS_GRIPPER_SERVICE = True
except ImportError:
    print("Warning: wwro_msgs not found - gripper control disabled")


class ControlState(Enum):
    """Control state machine."""
    RUNNING = 0           # Policy is running
    WAITING_FOR_HOME = 1  # Waiting for home action to complete


class UR5eReachNode(Node):
    """ROS2 node running UR5e reach policy with gripper control and resets."""
    
    # Control loop frequency (Hz) - easy to adjust
    CONTROL_LOOP_HZ = 5.0
    
    # Home position (rad) - in POLICY order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    HOME_POSITION_POLICY_ORDER = [0.0, -1.57, -1.57, -1.57, 0.0, 0.0]
    RESET_INTERVAL_S = 15.0
    HOME_DURATION_S = 3.0  # Time to reach home position
    
    # Joint names in ROS2 alphabetical order (how robot publishes)
    ROS2_JOINT_ORDER = [
        "gripper_elbow_joint",
        "gripper_shoulder_lift_joint", 
        "gripper_shoulder_pan_joint",
        "gripper_wrist_1_joint",
        "gripper_wrist_2_joint",
        "gripper_wrist_3_joint",
    ]
    
    # Joint names in policy order (how Isaac Lab trained)
    POLICY_JOINT_ORDER = [
        "gripper_shoulder_pan_joint",
        "gripper_shoulder_lift_joint",
        "gripper_elbow_joint", 
        "gripper_wrist_1_joint",
        "gripper_wrist_2_joint",
        "gripper_wrist_3_joint",
    ]

    def __init__(self, policy_path: Optional[Path], env_path: Optional[Path]) -> None:
        super().__init__("ur5e_reach_policy")
        
        # Load policy
        self.policy = UR5eGripperReachPolicy(policy_path=policy_path, env_path=env_path)
        
        # Goal management
        self.base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal_pos = UR5eGripperReachPolicy.sample_goal(self.base_pos)
        
        # Reset timer
        self.reset_interval_s = self.RESET_INTERVAL_S
        self.last_reset_time = self.get_clock().now()
        
        # State machine
        self.control_state = ControlState.RUNNING
        
        # Joint reordering indices
        self._ros_to_policy = [self.POLICY_JOINT_ORDER.index(n) for n in self.ROS2_JOINT_ORDER]
        self._policy_to_ros = [self.ROS2_JOINT_ORDER.index(n) for n in self.POLICY_JOINT_ORDER]
        
        # Compute home position in ROS2 order
        self._home_position_ros_order = [self.HOME_POSITION_POLICY_ORDER[i] for i in self._policy_to_ros]
        
        # TF for grasp position
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribers and publishers
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/gripper_scaled_joint_trajectory_controller/joint_trajectory", 10
        )
        self.marker_pub = self.create_publisher(Marker, "/goal_marker", 10)
        
        # Action client for homing (FollowJointTrajectory)
        self._home_action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            "/gripper_scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self._home_goal_handle = None
        
        # Gripper service client
        self._gripper_client = None
        self._last_gripper_cmd_mm = None
        if HAS_GRIPPER_SERVICE:
            self._gripper_client = self.create_client(OnTwofg7, "/on_twofg7_grip_external")
        
        # Gripper state tracking (single joint command)
        self._gripper_opening_mm = 40.0  # Assume open at start
        self._gripper_opening_mm_prev = 40.0  # Previous position for velocity computation
        self._control_loop_dt = 1.0 / self.CONTROL_LOOP_HZ
        
        # State flags
        self._has_joint_state = False
        
        # Control loop timer - run at CONTROL_LOOP_HZ for better responsiveness
        self.timer = self.create_timer(self._control_loop_dt, self._control_loop)
        self.get_logger().info(f"Node started (rate: {self.CONTROL_LOOP_HZ:.1f}Hz, reset every {self.reset_interval_s}s)")

    def _update_gripper_state(self, gripper_opening_mm: float) -> Tuple[float, float]:
        """Process gripper opening and compute velocity.
        
        Args:
            gripper_opening_mm: Current gripper opening (0-40mm, total width)
            
        Returns:
            (finger_pos_m, finger_vel_m): Single finger position (0-0.02m) and velocity
        """
        # Convert total gripper opening (0-40mm) to single finger position (0-0.02m)
        finger_pos_m = np.clip(gripper_opening_mm / 2.0 / 1000.0, 0.0, 0.02)
        
        # Compute finger velocity from position delta
        finger_pos_m_prev = np.clip(self._gripper_opening_mm_prev / 2.0 / 1000.0, 0.0, 0.02)
        finger_vel_m = (finger_pos_m - finger_pos_m_prev) / self._control_loop_dt if self._control_loop_dt > 0 else 0.0
        
        # Update tracking
        self._gripper_opening_mm_prev = gripper_opening_mm
        self._gripper_opening_mm = gripper_opening_mm
        
        return finger_pos_m, finger_vel_m

    def _joint_state_cb(self, msg: JointState) -> None:
        """Process joint state and reorder to policy order."""
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))
        
        # Extract arm joints in ROS2 order
        pos_ros = [name_to_pos.get(n, 0.0) for n in self.ROS2_JOINT_ORDER]
        vel_ros = [name_to_vel.get(n, 0.0) for n in self.ROS2_JOINT_ORDER]
        
        # Reorder to policy order
        pos_policy = np.array([pos_ros[i] for i in self._ros_to_policy])
        vel_policy = np.array([vel_ros[i] for i in self._ros_to_policy])
        
        # Process gripper state separately (single joint)
        finger_pos_m, finger_vel_m = self._update_gripper_state(self._gripper_opening_mm)
        
        # Update policy with 7 joint states (6 arm + 1 finger)
        self.policy.update_joint_state(pos_policy, vel_policy, finger_pos_m, finger_vel_m)
        self._has_joint_state = True

    def _send_home_action(self) -> None:
        """Send action to go to home position."""
        if not self._home_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Home action server not available!")
            self.control_state = ControlState.RUNNING
            return
        
        # Build trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.ROS2_JOINT_ORDER
        
        point = JointTrajectoryPoint()
        point.positions = self._home_position_ros_order
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=int(self.HOME_DURATION_S), nanosec=0)
        goal_msg.trajectory.points.append(point)
        
        self.get_logger().info(f"Sending home action: {self._home_position_ros_order}")
        
        # Send goal asynchronously
        send_goal_future = self._home_action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._home_goal_response_callback)

    def _home_goal_response_callback(self, future) -> None:
        """Called when home action goal is accepted/rejected."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Home action rejected!")
            self.control_state = ControlState.RUNNING
            return
        
        self.get_logger().info("Home action accepted, waiting for result...")
        self._home_goal_handle = goal_handle
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._home_result_callback)

    def _home_result_callback(self, future) -> None:
        """Called when home action completes."""
        result = future.result()
        status = result.status
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Home action SUCCEEDED!")
        else:
            self.get_logger().warn(f"Home action ended with status: {status}")
        
        # Reset and continue regardless of status
        self._complete_reset()

    def _complete_reset(self) -> None:
        """Complete the reset after home action finishes."""
        # Reset policy with home position (in policy order!)
        self.policy.reset(home_position=self.HOME_POSITION_POLICY_ORDER)
        self.goal_pos = UR5eGripperReachPolicy.sample_goal(self.base_pos)
        self.last_reset_time = self.get_clock().now()
        self.control_state = ControlState.RUNNING
        self._home_goal_handle = None
        self.get_logger().info(f"Reset complete! New goal: {self.goal_pos}")

    def _control_loop(self) -> None:
        """Main control loop - runs at policy rate."""
        if not self._has_joint_state:
            return
        
        now = self.get_clock().now()
        
        # --- STATE: WAITING FOR HOME ACTION ---
        if self.control_state == ControlState.WAITING_FOR_HOME:
            return  # Just wait for action callback
        
        # --- STATE: RUNNING ---
        # Check if reset interval elapsed
        elapsed = (now - self.last_reset_time).nanoseconds * 1e-9
        if elapsed >= self.reset_interval_s:
            self.get_logger().info(f"Reset triggered after {elapsed:.1f}s")
            self.control_state = ControlState.WAITING_FOR_HOME
            self._send_home_action()
            return
        
        # Get grasp position from TF
        grasp_pos = self._lookup_grasp_position()
        if grasp_pos is None:
            return
        
        # Run policy
        arm_targets, gripper_mm = self.policy.forward(self.policy.policy_dt, self.goal_pos, grasp_pos)
        if arm_targets is None:
            return
        
        # Publish arm trajectory (reorder from policy to ROS2 order)
        targets_ros = [arm_targets[i] for i in self._policy_to_ros]
        self._publish_trajectory(targets_ros)
        
        # Command gripper (if available)
        # gripper_mm is now the raw action [-1, 1]
        if gripper_mm is not None:
            self._command_gripper(gripper_mm)
        
        # Publish goal marker for visualization
        self._publish_goal_marker()

    def _publish_trajectory(self, targets: list) -> None:
        """Publish joint trajectory command."""
        traj = JointTrajectory()
        traj.joint_names = self.ROS2_JOINT_ORDER
        point = JointTrajectoryPoint()
        point.positions = targets
        point.time_from_start = Duration(sec=0, nanosec=int(self.policy.policy_dt * 1e9))
        traj.points.append(point)
        self.traj_pub.publish(traj)

    def _command_gripper(self, finger_action: float) -> None:
        """Send gripper command via ROS2 service.
        
        Args:
            finger_action: Raw action in [-1, 1]
                          -1 → 0mm opening (closed)
                          +1 → 40mm opening (open)
        """
        if self._gripper_client is None:
            return
        
        # Scale action [-1, 1] to gripper opening [0, 40]mm
        opening_mm = (finger_action + 1.0) / 2.0 * 40.0
        opening_mm = np.clip(opening_mm, 0.0, 40.0)
        
        # Avoid commanding same position repeatedly (within 1mm)
        if self._last_gripper_cmd_mm is not None and abs(opening_mm - self._last_gripper_cmd_mm) < 1.0:
            return
        
        if not self._gripper_client.service_is_ready():
            return
        
        # Create request (OnTwofg7 is imported when HAS_GRIPPER_SERVICE is True)
        req = OnTwofg7.Request()  # type: ignore[union-attr]
        req.gripper_operation.tool_index = 0
        req.gripper_operation.width_mm = float(opening_mm)
        req.gripper_operation.force_n = 20  # Constant grip force
        req.gripper_operation.speed = 50
        
        # Async call (non-blocking)
        self._gripper_client.call_async(req)
        self._last_gripper_cmd_mm = opening_mm
        self._gripper_opening_mm = opening_mm  # Update estimate

    def _lookup_grasp_position(self) -> Optional[np.ndarray]:
        """Get TCP position from TF."""
        try:
            tf = self.tf_buffer.lookup_transform("table", "gripper_tcp", Time(), 
                                                  timeout=RclpyDuration(seconds=0.01))
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z], dtype=np.float32)
        except Exception:
            return None

    def _publish_goal_marker(self) -> None:
        """Publish visualization marker at goal."""
        marker = Marker()
        marker.header.frame_id = "table"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.goal_pos[0])
        marker.pose.position.y = float(self.goal_pos[1])
        marker.pose.position.z = float(self.goal_pos[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.g = 1.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=None)
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    node = UR5eReachNode(parsed.policy, parsed.env)
    try:
        rclpy.spin(node)
    finally:
        if hasattr(node, 'policy'):
            node.policy.close_logger()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
