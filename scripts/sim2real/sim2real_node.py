#!/usr/bin/env python3
"""
Sim2Real ROS2 Node for UR5e pose control.

This node:
1. Subscribes to /joint_states for joint positions and velocities
2. Subscribes to /goal_pose for target end-effector pose
3. Uses TF to get current end-effector pose (wrist_3_link)
4. Builds observations matching IsaacSim normalization
5. Runs policy inference at 60Hz
6. Sends joint trajectory actions via scaled_joint_trajectory_controller

Usage:
    # Source ROS2 workspace first
    source ~/wwro_ws/install/local_setup.bash
    
    # Run the node
    python3 sim2real_node.py --robot gripper
"""

import numpy as np
import argparse
from pathlib import Path
from threading import Lock
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tf2_ros
from tf2_ros import TransformException

# Local imports
from observation_builder import (
    RobotState, GoalState, 
    build_observation, 
    compute_dof_targets,
    reorder_joints_from_ros,
    JOINT_NAMES_SIM,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
)
from policy_inference import load_policy


class Sim2RealNode(Node):
    """ROS2 node for sim2real policy deployment."""
    
    def __init__(
        self,
        robot_prefix: str = "gripper",
        model_path: Optional[str] = None,
        control_rate: float = 60.0,
        device: str = "cpu",
        action_scale: float = 3.0,
    ):
        super().__init__("sim2real_policy_node")
        
        self.robot_prefix = robot_prefix
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        self.action_scale = action_scale
        
        # Joint names with robot prefix for SENDING commands (must match ROS2 controller)
        # Order must match simulation order for the trajectory command
        self.joint_names_for_command = [f"{robot_prefix}_{name}" for name in JOINT_NAMES_SIM]
        
        # State variables (protected by lock)
        self.lock = Lock()
        self.joint_positions: Optional[np.ndarray] = None
        self.joint_velocities: Optional[np.ndarray] = None
        self.ee_position: Optional[np.ndarray] = None
        self.ee_quaternion: Optional[np.ndarray] = None
        self.goal_position: Optional[np.ndarray] = None
        self.goal_quaternion: Optional[np.ndarray] = None
        # Last received goal position (used to avoid repeated logs)
        self.old_goal: Optional[np.ndarray] = None
        
        # DOF targets (tracked across control steps)
        self.dof_targets: Optional[np.ndarray] = None
        
        # Control state
        self.is_running = False
        self.action_in_progress = False
        
        # ====================================================================
        # Load policy
        # ====================================================================
        self.get_logger().info(f"Loading policy from: {model_path or 'default'}")
        self.policy = load_policy(model_path=model_path, device=device)
        self.get_logger().info("Policy loaded successfully!")
        
        # ====================================================================
        # TF2 for end-effector pose
        # ====================================================================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Frame names
        #self.base_frame = f"{robot_prefix}_base_link"
        self.base_frame = f"table"
        self.ee_frame = f"{robot_prefix}_wrist_3_link"
        
        # ====================================================================
        # Subscribers
        # ====================================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            qos
        )
        
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_pose_callback,
            qos
        )
        
        # ====================================================================
        # Action client for trajectory controller
        # ====================================================================
        self.callback_group = ReentrantCallbackGroup()
        
        action_name = f"/{robot_prefix}_scaled_joint_trajectory_controller/follow_joint_trajectory"
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            action_name,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f"Waiting for action server: {action_name}")
        if not self.trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"Action server not available: {action_name}")
            raise RuntimeError("Action server not available")
        self.get_logger().info("Action server connected!")
        
        # ====================================================================
        # Control timer (60Hz)
        # ====================================================================
        self.control_timer = self.create_timer(
            self.dt,
            self.control_loop,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f"Sim2Real node initialized at {control_rate}Hz")
        self.get_logger().info(f"Robot prefix: {robot_prefix}")
        self.get_logger().info(f"Action scale: {action_scale} (IsaacSim uses 7.5)")
        self.get_logger().info(f"Joint names for commands (sim order): {self.joint_names_for_command}")
        
    def joint_state_callback(self, msg: JointState):
        """Handle incoming joint states."""
        try:
            positions, velocities = reorder_joints_from_ros(
                list(msg.name),
                list(msg.position),
                list(msg.velocity) if msg.velocity else [0.0] * len(msg.position),
                robot_prefix=self.robot_prefix
            )
            
            with self.lock:
                self.joint_positions = positions
                self.joint_velocities = velocities
                
                # Initialize dof_targets on first message
                if self.dof_targets is None:
                    self.dof_targets = positions.copy()
                    self.get_logger().info(f"Initialized DOF targets: {self.dof_targets}")
                    
        except ValueError as e:
            self.get_logger().warn(f"Joint state parsing error: {e}")
    
    def goal_pose_callback(self, msg: PoseStamped):
        """Handle incoming goal pose."""
        with self.lock:
            self.goal_position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ], dtype=np.float32)
            
            # Quaternion in (w, x, y, z) order
            self.goal_quaternion = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ], dtype=np.float32)
            
            # Log only when goal position changes (avoid repeated messages)
            if self.old_goal is None or not np.allclose(self.old_goal, self.goal_position, atol=1e-6):
                pos_s = f"pos=({self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}, {self.goal_position[2]:.3f})"
                quat_s = f"quat=({self.goal_quaternion[0]:.3f}, {self.goal_quaternion[1]:.3f}, {self.goal_quaternion[2]:.3f}, {self.goal_quaternion[3]:.3f})"
                self.get_logger().info(f"New goal: {pos_s}, {quat_s}")
            self.old_goal = self.goal_position.copy()
    
    
    def update_ee_pose(self) -> bool:
        """Update end-effector pose from TF.
        
        Returns:
            True if pose was updated successfully
        """
        try:
            # Get transform from base to EE
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            t = transform.transform.translation
            r = transform.transform.rotation
            
            with self.lock:
                self.ee_position = np.array([t.x, t.y, t.z], dtype=np.float32)
                self.ee_quaternion = np.array([r.w, r.x, r.y, r.z], dtype=np.float32)
            
            return True
            
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed: {e}",
                throttle_duration_sec=1.0
            )
            return False
    
    def control_loop(self):
        """Main control loop running at 60Hz."""
        # Skip if action still in progress
        if self.action_in_progress:
            return
        
        # Update EE pose from TF
        if not self.update_ee_pose():
            return
        
        # Check if we have all required data
        with self.lock:
            if any(x is None for x in [
                self.joint_positions, self.joint_velocities,
                self.ee_position, self.ee_quaternion,
                self.goal_position, self.goal_quaternion,
                self.dof_targets
            ]):
                self.get_logger().info(
                    "Waiting for data...",
                    throttle_duration_sec=2.0
                )
                return
            
            # Copy state (release lock quickly)
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
        
        # Build observation
        observation = build_observation(robot_state, goal_state)
        
        # Run policy inference
        actions = self.policy.get_action(observation)
        
        # Compute new DOF targets (with configurable action scale)
        new_targets = compute_dof_targets(current_targets, actions, self.dt, self.action_scale)
        
        # Update stored targets
        with self.lock:
            self.dof_targets = new_targets
        
        # Send trajectory command
        self.send_trajectory(new_targets)
        
        # DEBUG: Log observation details periodically
        pos_error = observation[0:3]
        quat_error = observation[3:7]
        self.get_logger().info(
            f"EE: ({robot_state.ee_position[0]:.3f}, {robot_state.ee_position[1]:.3f}, {robot_state.ee_position[2]:.3f}) | "
            f"Goal: ({goal_state.position[0]:.3f}, {goal_state.position[1]:.3f}, {goal_state.position[2]:.3f}) | "
            f"PosErr: ({pos_error[0]:.3f}, {pos_error[1]:.3f}, {pos_error[2]:.3f}) | "
            f"Act: ({actions[0]:.2f}, {actions[1]:.2f}, {actions[2]:.2f}, {actions[3]:.2f}, {actions[4]:.2f}, {actions[5]:.2f})",
            throttle_duration_sec=0.5
        )
    
    def send_trajectory(self, targets: np.ndarray):
        """Send joint trajectory command.
        
        Args:
            targets: Joint position targets [6] in SIMULATION order
                     (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
        """
        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names_for_command  # Sim order with prefix
        
        # Single point with short duration for immediate execution
        point = JointTrajectoryPoint()
        point.positions = targets.tolist()
        point.velocities = [0.0] * 6  # Zero velocity at target
        point.time_from_start = Duration(sec=0, nanosec=int(self.dt * 1e9))  # ~16ms
        
        trajectory.points = [point]
        
        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        
        # Send asynchronously
        self.action_in_progress = True
        future = self.trajectory_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Handle trajectory goal response."""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warn("Trajectory goal rejected")
            self.action_in_progress = False
            return
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Handle trajectory result."""
        self.action_in_progress = False
        result = future.result().result
        
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().warn(
                f"Trajectory failed with error code: {result.error_code}",
                throttle_duration_sec=1.0
            )
    
    def start(self):
        """Start the control loop."""
        self.is_running = True
        self.get_logger().info("Control loop started!")
    
    def stop(self):
        """Stop the control loop."""
        self.is_running = False
        self.get_logger().info("Control loop stopped!")


def main():
    parser = argparse.ArgumentParser(description="Sim2Real Policy Deployment")
    parser.add_argument(
        "--robot", 
        type=str, 
        default="gripper",
        choices=["gripper", "screwdriver"],
        help="Robot prefix (gripper or screwdriver)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to TorchScript policy (.pt file)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=60.0,
        help="Control loop rate in Hz"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for policy inference"
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=3.0,
        help="Action scaling factor (default 3.0, IsaacSim uses 7.5)"
    )
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create node
        node = Sim2RealNode(
            robot_prefix=args.robot,
            model_path=args.model,
            control_rate=args.rate,
            device=args.device,
            action_scale=args.action_scale,
        )
        
        # Start control
        node.start()
        
        # Spin
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
