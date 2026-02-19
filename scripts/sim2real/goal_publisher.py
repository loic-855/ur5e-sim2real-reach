#!/usr/bin/env python3
"""
Goal Pose Publisher for Sim2Real Testing with RViz Visualization.

Publishes a goal pose to /goal_pose for the sim2real node to track.
Also publishes visualization markers to /visualization_marker for RViz display.

Usage:
    # Static goal
    source ~/wwro_ws/install/local_setup.bash
    python3 goal_publisher.py --x 0.3 --y 0.0 --z 0.4
    
    # Interactive mode (change goal from terminal)
    python3 goal_publisher.py --interactive

    #Random goal every 10 seconds
    python3 goal_publisher.py --random --update 10
    
RViz Setup:
    1. Start RViz: ros2 run rviz2 rviz2
    2. In RViz:
       - Set Fixed Frame to "table"
       - Add Marker display with topic "/visualization_marker"
       - You'll see a coordinate frame (X=red, Y=green, Z=blue) at the goal position and orientation
    
    # Interactive mode
    python3 goal_publisher.py --interactive
"""

import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Vector3
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

# Table dimensions - NOTE: In ROS, "table" frame origin is at TABLE CENTER SURFACE (0,0,0)
# So all coordinates here are relative to the center of the table surface!
TABLE_DEPTH = 0.8   # x
TABLE_WIDTH = 1.2   # y
TABLE_HEIGHT = 0.842  # z (only used if converting from corner-origin coords)

# Robot base position relative to TABLE CENTER (not corner!)
# In IsaacSim: robot at (0.08, 0.08) from corner = (0.08 - 0.4, 0.08 - 0.6) = (-0.32, -0.52) from center
ROBOT_BASE_LOCAL = np.array([-0.52, -0.32, 0.0])


class GoalPublisher(Node):
    """Simple goal pose publisher for testing with RViz visualization."""
    
    def __init__(self, rate: float = 10.0, update_interval: float = None, cycling_goals: list = None):
        super().__init__("goal_publisher")
        
        self.publisher = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            10
        )
        
        # Publisher for RViz visualization
        self.marker_publisher = self.create_publisher(
            Marker,
            "/visualization_marker",
            10
        )
        
        # Default goal (in front of robot, slightly elevated)
        # NOTE: Coordinates relative to table CENTER (frame "table" at 0,0,0)
        self.goal_position = np.array([0.0, 0.0, 0.3])  # 30cm above table center
        # Internal quaternion convention: (w, x, y, z)
        # Use 180deg rotation around X (0,1,0,0) so +Z points DOWN by default
        self.goal_quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # (w, x, y, z)
        
        # Cycling goals feature
        self.cycling_goals = cycling_goals if cycling_goals is not None else []
        self.current_goal_index = 0
        self.goal_cycle_timer = None
        
        # Timer to publish goal
        self.timer = self.create_timer(1.0 / rate, self.publish_goal)
        
        # Timer for random goal updates if enabled
        self.update_timer = None
        if update_interval is not None:
            self.update_timer = self.create_timer(update_interval, self.update_random_goal)
            self.update_random_goal()  # Set initial random goal
        
        # Timer for cycling through goals (8 seconds per goal)
        if self.cycling_goals:
            self.goal_cycle_timer = self.create_timer(8.0, self.cycle_to_next_goal)
            self.cycle_to_next_goal()  # Set initial goal
        
        self.get_logger().info("Goal publisher started")
        self.get_logger().info("RViz: Add a Marker display with topic '/visualization_marker' to see the goal coordinate frame")
    
    def update_random_goal(self):
        """Update to a new random goal pose."""
        # Random quaternion (uniform sampling on unit sphere)
        rand_u, rand_v, rand_w = np.random.uniform(0.0, 1.0, size=3)
        xq = np.sqrt(1 - rand_u) * np.sin(2 * np.pi * rand_v)
        yq = np.sqrt(1 - rand_u) * np.cos(2 * np.pi * rand_v)
        zq = np.sqrt(rand_u) * np.sin(2 * np.pi * rand_w)
        wq = np.sqrt(rand_u) * np.cos(2 * np.pi * rand_w)
        # rand quaternion components computed as (x, y, z, w) -- convert to (w, x, y, z)
        # 360° around robot base (cylindrical sampling)
        # Random angle and radius for full 360° coverage
        angle = np.random.uniform(0, np.pi/2)
        radius = np.random.uniform(0.2, 0.65)  # within reach
        height = np.random.uniform(0.1, 0.6)   # above table       
        # Cylindrical to Cartesian (relative to robot base)
        rand_x = ROBOT_BASE_LOCAL[0] + radius * np.cos(angle)
        rand_y = ROBOT_BASE_LOCAL[1] + radius * np.sin(angle)
        rand_z = height
        self.set_goal(rand_x, rand_y, rand_z, wq, xq, yq, zq)
    
    def publish_coordinate_frame(self):
        """Publish RViz coordinate frame markers at goal position and orientation."""
        # Quaternions for rotating to each axis
        # X-axis: no rotation
        quat_x = self.goal_quaternion
        # Y-axis: rotate 90 deg around Z
        quat_y_rot = np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865475])
        quat_y = self.quaternion_multiply(self.goal_quaternion, quat_y_rot)
        # Z-axis: rotate -90 deg around Y to map +X -> +Z (right-hand rule)
        quat_z_rot = np.array([0.7071067811865476, 0.0, -0.7071067811865475, 0.0])
        quat_z = self.quaternion_multiply(self.goal_quaternion, quat_z_rot)
        
        axes = [
            ("x", quat_x, [1.0, 0.0, 0.0, 0.8]),  # Red
            ("y", quat_y, [0.0, 1.0, 0.0, 0.8]),  # Green
            ("z", quat_z, [0.0, 0.0, 1.0, 0.8]),  # Blue
        ]
        
        for i, (axis, quat, color) in enumerate(axes):
            marker = Marker()
            marker.header.frame_id = "table"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.ns = "goal_axes"
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(self.goal_position[0])
            marker.pose.position.y = float(self.goal_position[1])
            marker.pose.position.z = float(self.goal_position[2])
            
            marker.pose.orientation.w = float(quat[0])
            marker.pose.orientation.x = float(quat[1])
            marker.pose.orientation.y = float(quat[2])
            marker.pose.orientation.z = float(quat[3])
            
            marker.scale.x = 0.05
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 0
            
            self.marker_publisher.publish(marker)
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])
    
    def cycle_to_next_goal(self):
        """Switch to next goal in the cycling goals list."""
        if not self.cycling_goals:
            return
        
        goal = self.cycling_goals[self.current_goal_index]
        self.set_goal(goal[0], goal[1], goal[2])
        self.current_goal_index = (self.current_goal_index + 1) % len(self.cycling_goals)
    
    
    def set_goal(
        self,
        x: float, y: float, z: float,
        qw: float = 0.0, qx: float = 1.0, qy: float = 0.0, qz: float = 0.0
    ):
        """Set new goal pose."""
        self.goal_position = np.array([x, y, z])
        # Store internally as (w, x, y, z)
        self.goal_quaternion = np.array([qw, qx, qy, qz])
        
        # Normalize quaternion
        norm = np.linalg.norm(self.goal_quaternion)
        if norm > 0:
            self.goal_quaternion /= norm
        
        self.get_logger().info(
            f"Goal set: pos=[{x:.3f}, {y:.3f}, {z:.3f}], "
            f"quat=[{self.goal_quaternion[0]:.3f}, {self.goal_quaternion[1]:.3f}, {self.goal_quaternion[2]:.3f}, {self.goal_quaternion[3]:.3f}] (w,x,y,z)"
        )
    
    def publish_goal(self):
        """Publish current goal pose."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "table"  # Reference frame
        
        msg.pose.position.x = float(self.goal_position[0])
        msg.pose.position.y = float(self.goal_position[1])
        msg.pose.position.z = float(self.goal_position[2])
        
        msg.pose.orientation.w = float(self.goal_quaternion[0])
        msg.pose.orientation.x = float(self.goal_quaternion[1])
        msg.pose.orientation.y = float(self.goal_quaternion[2])
        msg.pose.orientation.z = float(self.goal_quaternion[3])
        
        self.publisher.publish(msg)
        
        # Also publish visualization markers
        self.publish_coordinate_frame()


def main():
    parser = argparse.ArgumentParser(description="Goal pose publisher")
    parser.add_argument("--x", type=float, default=None, help="Goal X position")
    parser.add_argument("--y", type=float, default=None, help="Goal Y position")
    parser.add_argument("--z", type=float, default=None, help="Goal Z position")
    parser.add_argument("--qw", type=float, default=0.0, help="Quaternion W")
    parser.add_argument("--qx", type=float, default=0.0, help="Quaternion X")
    parser.add_argument("--qy", type=float, default=1.0, help="Quaternion Y")
    parser.add_argument("--qz", type=float, default=0.0, help="Quaternion Z")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate Hz")
    parser.add_argument("--random", action="store_true", help="Use random goal position")
    parser.add_argument("--update", type=float, default=10.0, help="Update goal every N seconds in random mode")
    args = parser.parse_args()
    
    rclpy.init()
    
    # Default cycling goals (three positions)
    default_cycling_goals = [
        (-0.15, 0.15, 0.6),
        (0.0, -0.2, 0.5),
        (-0.3, -0.05, 0.3),
    ]
    
    # Determine behavior based on arguments
    cycling_goals = None
    update_interval = None
    
    if args.x is not None and args.y is not None and args.z is not None:
        # Static goal specified
        cycling_goals = None
        update_interval = None
    elif args.random:
        # Random goal mode
        cycling_goals = None
        update_interval = args.update
    else:
        # Default: cycle through three positions
        cycling_goals = default_cycling_goals
        update_interval = None
    
    node = GoalPublisher(rate=args.rate, update_interval=update_interval, cycling_goals=cycling_goals)
    
    # Set static goal if specified
    if args.x is not None and args.y is not None and args.z is not None:
        node.set_goal(args.x, args.y, args.z, args.qw, args.qx, args.qy, args.qz)
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive goal publisher")
        print("Commands:")
        print("  x y z       - Set position")
        print("  x y z qw qx qy qz - Set pose")
        print("  q           - Quit")
        print("")
        
        import threading
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        
        try:
            while True:
                cmd = input("Goal> ").strip()
                if cmd.lower() == 'q':
                    break
                
                parts = cmd.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    if len(parts) >= 7:
                        qw, qx, qy, qz = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                        node.set_goal(x, y, z, qw, qx, qy, qz)
                    else:
                        node.set_goal(x, y, z)
                else:
                    print("Invalid input. Use: x y z [qw qx qy qz]")
                    
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        # Static or cycling mode
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
