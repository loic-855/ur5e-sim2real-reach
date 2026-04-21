# Sim-to-Real Deployment

> **Environment:** All scripts in this folder require `conda activate sim2real` and a sourced ROS 2 workspace (`source ~/wwro_ws/install/local_setup.bash`).
>
> **Driver:** The custom UR5e ROS 2 driver is available at <https://github.com/IDEALLab/Woodworking_Robots>. It is needed to visualize the robot state and goals in RViz, but the core sim2real nodes can run without it.

## Folder Layout

| Path | Description |
|---|---|
| `v1/` | Position-only deployment (6-dim actions) |
| `v2/` | Velocity-feedforward deployment (12-dim actions) |
| `URscript/` | Robot-side impedance controllers and RTDE recipe files |
| `goal_publisher.py` | Publish goal poses for manual testing and benchmark playback |
| `send_urscript.py` | Legacy TCP socket helper (reference only) |

## End-to-End Flow

1. Train in simulation.
2. Validate the checkpoint with `scripts/rsl_rl/play.py` or `scripts/rsl_rl/benchmark.py`.
3. Export `policy.pt` through `play.py` or `benchmark.py`.
4. Start the robot driver and external control.
5. Run the matching sim2real node.

## v1 vs v2

### v1

- Observation size: 24.
- Action size: 6.
- Meaning: position increments only.
- Node: `scripts/sim2real/v1/sim2real_node.py`.
- Policy loader: `scripts/sim2real/v1/policy_inference.py`.

### v2

- Observation size: 24.
- Action size: 12.
- Meaning: first 6 outputs are position increments, last 6 outputs are velocity feedforward targets.
- Node: `scripts/sim2real/v2/sim2real_node.py`.
- Policy loader: `scripts/sim2real/v2/policy_inference.py`.

Do not mix v1 and v2 artifacts. The action interface is part of the policy contract.

## Real-Robot Control Rates

The current deployment code uses:

- RTDE reader thread at 125 Hz
- policy loop at 60 Hz
- URScript impedance controller at 500 Hz

## Driver Bring-Up

Current documented network setup:

- ROS2 workstation: `192.168.1.105`
- gripper robot: `192.168.1.101`
- screwdriver robot: `192.168.1.103`
- OnRobot gripper: `192.168.1.111`

Open the driver ports:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 50001:50008 proto tcp
```

Start the ROS2 workspace:

```bash
cd /home/robots/wwro_ws
source install/local_setup.bash
```

Launch the robot control stack:

```bash
ros2 launch wwro_startup wwro_control.launch.py \
	gripper_robot_ip:=192.168.1.101 \
	screwdriver_robot_ip:=192.168.1.103 \
```


On each teach pendant:

1. Switch to remote mode.
2. Load the External Control program.
3. Press play.

Verify the connections:

```bash
ss -tnp | grep "5000[1-8]"
ros2 service call /gripper_dashboard_client/program_state ur_dashboard_msgs/srv/GetProgramState
ros2 service call /screwdriver_dashboard_client/program_state ur_dashboard_msgs/srv/GetProgramState
ros2 topic echo /gripper_io_and_status_controller/robot_program_running --qos-reliability reliable --qos-durability transient_local --once
```

## Running the Deployment Nodes

Position-only policy:

```bash
python scripts/sim2real/v1/sim2real_node.py --model /path/to/exported/policy.pt
```

Velocity-feedforward policy:

```bash
python scripts/sim2real/v2/sim2real_node.py --model /path/to/exported/policy.pt
```

## Goal and RViz Utilities

Publish a goal:

```bash
source ~/wwro_ws/install/local_setup.bash
python scripts/sim2real/goal_publisher.py --x 0.3 --y 0.0 --z 0.4
```

Show a benchmark goal set in RViz:

```bash
source ~/wwro_ws/install/local_setup.bash
python scripts/sim2real/goal_publisher.py \
	--goals-file scripts/benchmark_settings/goals_handmade.json \
	--goal-overview
```


## Impedance Tuning

The tuning scripts and the simulation gain logger have been consolidated under `scripts/tuning/`.
See `scripts/tuning/README.md` for the full 3-step tuning workflow.

The URScript controllers loaded by those tools are still stored here under `URscript/`:

**Deployment controllers**
- `URscript/impedance_control_naive.script` — v1 (position-only), fixed gains
- `URscript/impedance_control_tuned.script` — v1 (position-only), experimentally tuned gains
- `URscript/impedance_control_ff.script` — v2 (velocity feedforward), naive gains

**Tuning controllers** (loaded by `scripts/tuning/`)
- `URscript/impedance_control_tuning.script` — step-response sweeps (`step_tuner.py`)
- `URscript/impedance_control_tuning_zeta.script` — Kp stiffness sweeps (`auto_tuner.py`)

**RTDE recipes**
- `URscript/rtde_input.xml` — used by all deployment nodes and `impedance_tuner.py`
- `URscript/rtde_input_tuning.xml` — extends `rtde_input.xml` with kp, ζ and go_home registers; used by `auto_tuner.py` and `step_tuner.py`

## OnRobot 2FG7 Gripper Services

The gripper is controlled through ROS2 services rather than as ordinary ROS2 joints.

Main services:

- `/on_twofg7_grip_external`
- `/on_twofg7_grip_internal`
- `/on_twofg7_release_external`
- `/on_twofg7_release_internal`

Service type:

- `wwro_msgs/srv/OnTwofg7`

Close the gripper:

```bash
ros2 service call /on_twofg7_grip_external wwro_msgs/srv/OnTwofg7 "{gripper_operation: {tool_index: 0, width_mm: 10.0, force_n: 20, speed: 50}}"
```

Open the gripper:

```bash
ros2 service call /on_twofg7_release_external wwro_msgs/srv/OnTwofg7 "{gripper_operation: {tool_index: 0, width_mm: 40.0, force_n: 0, speed: 50}}"
```

## Troubleshooting

If the robot does not move:

1. re-check the firewall, might need to open ports
2. check the ethernet  connection
3. check the installation of the ROS2 driver and the workspace setup
4. check the robot is in remote mode
