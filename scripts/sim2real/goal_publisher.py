#!/usr/bin/env python3
"""
Goal Pose Publisher for Sim2Real Testing with RViz Visualization.

Publishes a goal pose to /goal_pose for the sim2real node to track.
Also publishes visualization markers to /visualization_marker for RViz display.
In overview mode, publishes a MarkerArray to /visualization_marker_array.

Usage:
    # Static goal
    source ~/wwro_ws/install/local_setup.bash
    python3 goal_publisher.py --x 0.3 --y 0.0 --z 0.4
    
    # Interactive mode (change goal from terminal)
    python3 goal_publisher.py --interactive

    #Random goal every 10 seconds
    python3 goal_publisher.py --random --update 10

    # Persistent RViz overview of all goals in a file
    python3 goal_publisher.py --goals-file scripts/benchmark_settings/goals_handmade.json --goal-overview
    
RViz Setup:
    1. Start RViz: ros2 run rviz2 rviz2
    2. In RViz:
       - Set Fixed Frame to "table"
       - Add Marker display with topic "/visualization_marker"
         - For --goal-overview, add a MarkerArray display with topic "/visualization_marker_array"
         - You'll see a coordinate frame (X=red, Y=green, Z=blue) at the goal position and orientation
         - In --goal-overview mode, you'll see every goal in the file at once, with labels
    
    # Interactive mode
    python3 goal_publisher.py --interactive
"""

import argparse
import json
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String

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
    
    def __init__(
        self,
        rate: float = 10.0,
        random_update_interval: float | None = None,
        cycling_goals: list[tuple[float, ...]] | None = None,
        overview_goals: list[tuple[float, ...]] | None = None,
        cycling_goal_interval: float | None = None,
        stop_after_single_cycle: bool = False,
        benchmark_mode: bool = False,
    ):
        super().__init__("goal_publisher")

        overview_goal_count = len(overview_goals) if overview_goals is not None else 0
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        self.publisher = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            10
        )
        
        # Publisher for RViz visualization
        self.marker_publisher = self.create_publisher(
            Marker,
            "/visualization_marker",
            marker_qos
        )
        self.marker_array_publisher = self.create_publisher(
            MarkerArray,
            "/visualization_marker_array",
            marker_qos,
        )
        
        # Default goal (in front of robot, slightly elevated)
        # NOTE: Coordinates relative to table CENTER (frame "table" at 0,0,0)
        self.goal_position = np.array([0.0, 0.0, 0.3])  # 30cm above table center
        # Internal quaternion convention: (w, x, y, z)
        # Use 180deg rotation around X (0,1,0,0) so +Z points DOWN by default
        self.goal_quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # (w, x, y, z)
        
        # Cycling goals feature
        self.cycling_goals = cycling_goals if cycling_goals is not None else []
        self.overview_goals = overview_goals if overview_goals is not None else []
        self.current_goal_index = 0
        self.goal_cycle_timer = None
        self.stop_after_single_cycle = stop_after_single_cycle
        self.finished_cycling = False
        self.shutdown_after_publish = False
        self._benchmark_mode = benchmark_mode
        self._overview_mode = bool(self.overview_goals)
        self._cycling_goal_interval = cycling_goal_interval if cycling_goal_interval is not None else 8.0
        self._publishing_active = not benchmark_mode and not self._overview_mode

        self.clear_markers()

        # Benchmark control subscription
        if benchmark_mode:
            self._benchmark_control_sub = self.create_subscription(
                String, "/benchmark_control", self._benchmark_control_callback, 10
            )

        self.get_logger().info("Goal publisher started")
        if self._overview_mode:
            self.get_logger().info(
                "RViz: Add a MarkerArray display with topic '/visualization_marker_array' to see all goal frames from the file"
            )
        else:
            self.get_logger().info(
                "RViz: Add a Marker display with topic '/visualization_marker' to see the goal coordinate frame"
            )
        
        # Timer to publish goal / markers. Overview mode republishes slowly because
        # the full marker set is persistent and frequent bursts can overwhelm RViz.
        publish_period = 1.0 if self._overview_mode else 1.0 / rate
        self.timer = self.create_timer(publish_period, self.publish_goal)

        if self._overview_mode:
            self.publish_goal_overview()
        
        # Timer for random goal updates if enabled
        self.update_timer = None
        self._random_update_interval = random_update_interval
        if random_update_interval is not None and not benchmark_mode and not self._overview_mode:
            self.update_timer = self.create_timer(random_update_interval, self.update_random_goal)
            self.update_random_goal()  # Set initial random goal
        
        # Timer for cycling through goals (skip in benchmark mode; "start" signal will trigger)
        if self.cycling_goals and not benchmark_mode and not self._overview_mode:
            goal_cycle_interval = cycling_goal_interval if cycling_goal_interval is not None else 8.0
            self.goal_cycle_timer = self.create_timer(goal_cycle_interval, self.cycle_to_next_goal)
            self.cycle_to_next_goal()  # Set initial goal
    
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
    
    def _normalized_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Return a normalized quaternion in (w, x, y, z) format."""
        normalized = quaternion.astype(float).copy()
        norm = np.linalg.norm(normalized)
        if norm > 0:
            normalized /= norm
        return normalized

    def clear_markers(self):
        """Clear stale RViz markers on startup."""
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.marker_publisher.publish(marker)
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_array_publisher.publish(marker_array)

    def build_coordinate_frame_markers(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        marker_id_base: int = 0,
        namespace: str = "goal_axes",
        label: str | None = None,
    ) -> list[Marker]:
        """Build RViz coordinate frame markers at a given pose."""
        normalized_quaternion = self._normalized_quaternion(quaternion)
        quat_y_rot = np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865475])
        quat_z_rot = np.array([0.7071067811865476, 0.0, -0.7071067811865475, 0.0])
        axes = [
            (normalized_quaternion, [1.0, 0.0, 0.0, 0.8]),
            (self.quaternion_multiply(normalized_quaternion, quat_y_rot), [0.0, 1.0, 0.0, 0.8]),
            (self.quaternion_multiply(normalized_quaternion, quat_z_rot), [0.0, 0.0, 1.0, 0.8]),
        ]
        markers: list[Marker] = []

        for i, (quat, color) in enumerate(axes):
            marker = Marker()
            marker.header.frame_id = "table"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id_base + i
            marker.ns = namespace
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
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
            markers.append(marker)

        if label is not None:
            text_marker = Marker()
            text_marker.header.frame_id = "table"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.id = marker_id_base + len(axes)
            text_marker.ns = f"{namespace}_labels"
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(position[0])
            text_marker.pose.position.y = float(position[1])
            text_marker.pose.position.z = float(position[2] + 0.05)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.04
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.95
            text_marker.text = label
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 0
            markers.append(text_marker)

        return markers

    def publish_coordinate_frame(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        marker_id_base: int = 0,
        namespace: str = "goal_axes",
        label: str | None = None,
    ):
        """Publish RViz coordinate frame markers at a given pose."""
        for marker in self.build_coordinate_frame_markers(
            position=position,
            quaternion=quaternion,
            marker_id_base=marker_id_base,
            namespace=namespace,
            label=label,
        ):
            self.marker_publisher.publish(marker)

    def publish_goal_overview(self):
        """Publish all goals from a file as persistent RViz markers."""
        marker_array = MarkerArray()
        for goal_index, goal in enumerate(self.overview_goals):
            position = np.array(goal[:3], dtype=float)
            quaternion = np.array(goal[3:], dtype=float)
            marker_array.markers.extend(self.build_coordinate_frame_markers(
                position=position,
                quaternion=quaternion,
                marker_id_base=goal_index * 4,
                namespace="goal_overview",
                label=f"goal_{goal_index + 1}",
            ))
        self.marker_array_publisher.publish(marker_array)
    
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
        if not self.cycling_goals or self.finished_cycling:
            return
        
        goal = self.cycling_goals[self.current_goal_index]
        if len(goal) == 7:
            self.set_goal(goal[0], goal[1], goal[2], goal[3], goal[4], goal[5], goal[6])
        else:
            self.set_goal(goal[0], goal[1], goal[2])

        is_last_goal = self.current_goal_index == len(self.cycling_goals) - 1
        if self.stop_after_single_cycle and is_last_goal:
            if self.goal_cycle_timer is not None:
                self.goal_cycle_timer.cancel()
            if self._benchmark_mode:
                self.get_logger().info("Goal cycle completed; waiting for next 'start' signal")
            else:
                self.shutdown_after_publish = True
                self.get_logger().info("Completed one goal cycle from file; exiting.")
            return

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
            f"Goal set: pos=[{x:.3f}, {y:.3f}, {z:.3f}, {self.goal_quaternion[0]:.3f}, {self.goal_quaternion[1]:.3f}, {self.goal_quaternion[2]:.3f}, {self.goal_quaternion[3]:.3f}] (w,x,y,z)], "
        )
        
    def _benchmark_control_callback(self, msg: String):
        """Handle benchmark control signals from sim2real node."""
        if msg.data != "start":
            return

        self._publishing_active = True

        # Cycling goals mode
        if self.cycling_goals:
            self.current_goal_index = 0
            self.finished_cycling = False
            self.shutdown_after_publish = False
            if self.goal_cycle_timer is not None:
                self.goal_cycle_timer.cancel()
            self.goal_cycle_timer = self.create_timer(
                self._cycling_goal_interval, self.cycle_to_next_goal
            )
            self.cycle_to_next_goal()  # publish first goal immediately
            self.get_logger().info("Benchmark control: started goal cycle")

        # Random goals mode
        elif self._random_update_interval is not None:
            if self.update_timer is not None:
                self.update_timer.cancel()
            self.update_timer = self.create_timer(
                self._random_update_interval, self.update_random_goal
            )
            self.update_random_goal()  # publish first random goal immediately
            self.get_logger().info("Benchmark control: started random goals")

    def publish_goal(self):
        """Publish current goal pose."""
        if self._overview_mode:
            self.publish_goal_overview()
            return
        if not self._publishing_active:
            return
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

        if self.shutdown_after_publish:
            self.shutdown_after_publish = False
            self.finished_cycling = True
        
        # Also publish visualization markers
        self.publish_coordinate_frame(self.goal_position, self.goal_quaternion)


def main():
    def load_goals_file(goals_file: str) -> list[tuple[float, ...]]:
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
        return parsed_goals

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
    parser.add_argument("--update", type=float, default=None, help="Update goal every N seconds in random or goals-file cycling mode")
    parser.add_argument("--goals-file", type=str, default=None, help="JSON file containing [x, y, z, qw, qx, qy, qz] goals")
    parser.add_argument("--goal-overview", action="store_true", default=False, help="Publish all goals from --goals-file as persistent RViz markers without cycling or publishing /goal_pose")
    parser.add_argument("--benchmark", action="store_true", default=False, help="Benchmark mode: wait for start signals from sim2real node")
    args = parser.parse_args()

    static_goal_requested = all(value is not None for value in (args.x, args.y, args.z))
    partial_static_goal = any(value is not None for value in (args.x, args.y, args.z)) and not static_goal_requested
    update_interval = args.update if args.update is not None else 10.0

    if partial_static_goal:
        parser.error("Provide --x, --y, and --z together for a static goal.")

    if args.interactive and (static_goal_requested or args.random or args.goals_file is not None):
        parser.error("--interactive cannot be combined with static, random, or goals-file modes.")

    if args.goal_overview and args.goals_file is None:
        parser.error("--goal-overview requires --goals-file.")

    if static_goal_requested and args.update is not None:
        parser.error("--update cannot be used with static goal (--x, --y, --z) because there is no cycling.")

    if args.interactive and args.update is not None:
        parser.error("--update cannot be used with interactive mode.")

    if args.goal_overview and args.update is not None:
        parser.error("--goal-overview cannot be combined with --update because overview mode does not cycle goals.")

    if args.goal_overview and (args.interactive or args.random or args.benchmark or static_goal_requested):
        parser.error("--goal-overview cannot be combined with interactive, static, random, or benchmark modes.")
    
    rclpy.init()
    
    # Default cycling goals (three positions)
    default_cycling_goals = [
        (-0.15, 0.15, 0.6),
        (0.0, -0.2, 0.5),
        (-0.3, -0.05, 0.3),
    ]
    
    # Determine behavior based on arguments
    cycling_goals = None
    overview_goals = None
    random_update_interval = None
    cycling_goal_interval = None
    
    if static_goal_requested:
        # Static goal specified
        cycling_goals = None
        overview_goals = None
        random_update_interval = None
        cycling_goal_interval = None
    elif args.goal_overview:
        cycling_goals = None
        overview_goals = load_goals_file(args.goals_file)
        random_update_interval = None
        cycling_goal_interval = None
    elif args.goals_file is not None:
        cycling_goals = load_goals_file(args.goals_file)
        overview_goals = None
        cycling_goal_interval = update_interval
    elif args.random:
        # Random goal mode
        cycling_goals = None
        overview_goals = None
        random_update_interval = update_interval
    elif args.interactive:
        # Interactive mode
        cycling_goals = None
        overview_goals = None
        random_update_interval = None
        cycling_goal_interval = None
    else:
        # Default: cycle through three positions
        cycling_goals = default_cycling_goals
        overview_goals = None
        cycling_goal_interval = update_interval if args.update is not None else None
    
    node = GoalPublisher(
        rate=args.rate,
        random_update_interval=random_update_interval,
        cycling_goals=cycling_goals,
        overview_goals=overview_goals,
        cycling_goal_interval=cycling_goal_interval,
        stop_after_single_cycle=args.benchmark,
        benchmark_mode=args.benchmark,
    )
    
    # Set static goal if specified
    if static_goal_requested:
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
                cmd = cmd.replace(",", " ").replace("["," ").replace("]"," ")  # Allow comma-separated input

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
            if args.benchmark or args.goal_overview:
                rclpy.spin(node)  # Stay alive; controlled by /benchmark_control
            else:
                while rclpy.ok() and not node.finished_cycling:
                    rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
