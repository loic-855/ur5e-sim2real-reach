#!/usr/bin/env python3
"""Publish a cumulative nav_msgs/Path from an input PoseStamped topic.

This helper is intended for RViz qualitative trajectory visualization.
It subscribes to an existing TCP pose topic and appends each accepted pose
into a Path message.
"""

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_srvs.srv import Empty


class EePathFromPose(Node):
    """Build and publish a path from incoming PoseStamped messages."""

    def __init__(
        self,
        input_topic: str,
        output_topic: str,
        max_points: int,
        min_dt: float,
        enable_clear_service: bool,
    ):
        super().__init__("ee_path_from_pose")

        self.input_topic = input_topic
        self.output_topic = output_topic
        self.max_points = max(1, int(max_points))
        self.min_dt = max(0.0, float(min_dt))

        self.path_pub = self.create_publisher(Path, self.output_topic, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, self.input_topic, self._pose_callback, 50
        )

        self.path_msg = Path()
        self._last_stamp_sec: float | None = None

        self._clear_service = None
        if enable_clear_service:
            self._clear_service = self.create_service(
                Empty, f"{self.output_topic}/clear", self._clear_callback
            )

        self.get_logger().info(
            f"Listening on {self.input_topic} and publishing path to {self.output_topic}"
        )
        self.get_logger().info(
            f"Path settings: max_points={self.max_points}, min_dt={self.min_dt:.3f}s"
        )
        if self._clear_service is not None:
            self.get_logger().info(
                f"Manual clear service available at {self.output_topic}/clear"
            )

    @staticmethod
    def _stamp_to_seconds(msg: PoseStamped) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _pose_callback(self, msg: PoseStamped) -> None:
        stamp_sec = self._stamp_to_seconds(msg)

        # If source has zero timestamps, fall back to node clock.
        if stamp_sec <= 0.0:
            now = self.get_clock().now().to_msg()
            msg.header.stamp = now
            stamp_sec = self._stamp_to_seconds(msg)

        if self._last_stamp_sec is not None and self.min_dt > 0.0:
            if (stamp_sec - self._last_stamp_sec) < self.min_dt:
                return

        self._last_stamp_sec = stamp_sec

        self.path_msg.header = msg.header
        self.path_msg.poses.append(msg)

        if len(self.path_msg.poses) > self.max_points:
            # Keep only the newest points to avoid RViz slowdowns over long runs.
            overflow = len(self.path_msg.poses) - self.max_points
            self.path_msg.poses = list(self.path_msg.poses)[overflow:]

        self.path_pub.publish(self.path_msg)

    def _clear_callback(self, request: Empty.Request, response: Empty.Response) -> Empty.Response:
        del request
        self.path_msg = Path()
        self._last_stamp_sec = None
        self.get_logger().info("Path cleared")
        return response


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a PoseStamped stream into a cumulative nav_msgs/Path for RViz"
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default="/gripper_tcp_pose_broadcaster/pose",
        help="Input PoseStamped topic",
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default="/ee_path",
        help="Output Path topic",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum number of poses to keep in path",
    )
    parser.add_argument(
        "--min-dt",
        type=float,
        default=0.03,
        help="Minimum accepted time gap between points in seconds",
    )
    parser.add_argument(
        "--no-clear-service",
        action="store_true",
        help="Disable manual clear service on <output-topic>/clear",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rclpy.init()
    node = EePathFromPose(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        max_points=args.max_points,
        min_dt=args.min_dt,
        enable_clear_service=not args.no_clear_service,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
