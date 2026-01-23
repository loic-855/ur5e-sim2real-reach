# Controlling the OnRobot 2FG7 Gripper in ROS2

This guide explains how to control the OnRobot 2FG7 gripper fingers using ROS2 services, based on the setup in the Woodworking Simulation project.

## Overview
- **Gripper Model**: OnRobot 2FG7 (2-finger parallel gripper).
- **Control Method**: ROS2 services via the `/on_twofg7_controller` node.
- **No Joint Control**: The gripper fingers are not controlled as ROS2 joints; instead, use services for positioning and force.
- **Workspaces Required**: Source both ROS2 Jazzy and the custom `wwro_ws` workspace.

## Prerequisites
1. ROS2 Jazzy installed.
2. Custom workspace `~/wwro_ws` built and sourced (contains `wwro_msgs` package).
3. Gripper controller node running (launched via your setup scripts).

Source the environments in every terminal:
```bash
source /opt/ros/jazzy/setup.bash
source ~/wwro_ws/install/local_setup.bash
```

## Available Services
The `/on_twofg7_controller` provides 4 services:
- `/on_twofg7_grip_external`: Grip externally (close fingers inward).
- `/on_twofg7_grip_internal`: Grip internally (expand fingers outward).
- `/on_twofg7_release_external`: Release externally (open fingers).
- `/on_twofg7_release_internal`: Release internally (contract fingers).

All use the message type: `wwro_msgs/srv/OnTwofg7`.

## Service Message Structure
**Request**:
- `gripper_operation` (nested object):
  - `tool_index` (int64): Tool index (usually `0`).
  - `width_mm` (float64): Desired gripper width in millimeters (0.0 = fully closed, 40.0 = fully open).
  - `force_n` (int64): Gripping force in Newtons (0-50+; 0 for release).
  - `speed` (int64): Movement speed (1-100; higher = faster).

**Response**:
- `success` (bool): `true` if successful, `false` otherwise.

## Example Commands
### Close Gripper (Grip to 10mm with 20N force)
```bash
ros2 service call /on_twofg7_grip_external wwro_msgs/srv/OnTwofg7 "{gripper_operation: {tool_index: 0, width_mm: 10.0, force_n: 20, speed: 50}}"
```

### Open Gripper (Release to 40mm)
```bash
ros2 service call /on_twofg7_release_external wwro_msgs/srv/OnTwofg7 "{gripper_operation: {tool_index: 0, width_mm: 40.0, force_n: 0, speed: 50}}"
```

### With 30-Second Delay
```bash
sleep 30 && ros2 service call /on_twofg7_grip_external wwro_msgs/srv/OnTwofg7 "{gripper_operation: {tool_index: 0, width_mm: 10.0, force_n: 20, speed: 50}}"
```

## Notes
- **Width Limits**: Based on code, 0.0–0.04 meters (0–40mm). Do not exceed hardware limits.
- **Force/Speed**: Start low (e.g., force=20, speed=50) and adjust. High force can damage objects or the gripper.
- **External vs. Internal**: Use "external" for standard operations; "internal" for specialized grips.
- **Safety**: Never place fingers or body parts near the gripper. Use test objects. Industrial grippers can cause injury.
- **Integration**: For code, use `rclpy` service clients to call these services programmatically.
- **Troubleshooting**: If services fail, ensure nodes are running (`ros2 node list`) and workspaces are sourced.

## Related Controllers
- Arm joints: Controlled via `/gripper_scaled_joint_trajectory_controller` (action for trajectory).
- Other: IO/status via `/gripper_io_and_status_controller`.

## Get the current gripper opening (example command, not optimal):
    python3 -c "
    import xmlrpc.client
    server = xmlrpc.client.ServerProxy('http://192.168.1.111:41414/', allow_none=True)
    try:
        width = server.twofg_get_external_width(0)  # tool_index 0
        print(f'Current external width: {width} mm')
    except Exception as e:
        print(f'Error: {e}') "


## Gripper loop to query its position via xmlrp (network xml instruction, not ros2)
python3 -c "
import time
import xmlrpc.client
server = xmlrpc.client.ServerProxy('http://192.168.1.111:41414/', allow_none=True)
while True:
    try:
        width = server.twofg_get_external_width(0)
        print(f'Current external width: {width} mm')
    except Exception as e:
        print(f'Error: {e}')
    time.sleep(1)
"



For more details, check ROS2 docs or the `wwro_msgs` package.

