# Testing Checklist

Use this checklist before release or after major changes to verify the full pipeline.

---

## 1. Simulation (conda: `env_isaaclab`)

```bash
conda activate env_isaaclab
cd <repo-root>
```

### 1.1 Extension installed correctly

```bash
python scripts/list_envs.py
```

- [ ] Script runs without import errors
- [ ] All expected `WWSim-*` tasks are listed

### 1.2 Environments load (dummy agents)

```bash
python scripts/zero_agent.py  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 --num_envs 2
python scripts/zero_agent.py  --task WWSim-Pose-Orientation-Sim2Real-Direct-v2 --num_envs 2
python scripts/random_agent.py --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 --num_envs 2
```

- [ ] Each command starts without errors
- [ ] Sim window appears (or headless runs without crash)
- [ ] Agent steps for at least 100 iterations without exception

### 1.3 Training starts

```bash
python scripts/rsl_rl/train.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --headless --num_envs 64 --max_iterations 10
```

- [ ] Training loop begins (iteration 1/10 printed)
- [ ] Checkpoint `model_9.pt` is written under `logs/rsl_rl/`

### 1.4 Playback & export

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint <checkpoint-from-1.3>
```

- [ ] Policy loads and sim runs
- [ ] `exported/policy.pt` and `exported/policy.onnx` are created next to the checkpoint

### 1.5 Simulation benchmark

```bash
python scripts/rsl_rl/benchmark.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint <checkpoint-from-1.3> \
  --goals-file scripts/benchmark_settings/goals_handmade.json
```

- [ ] Benchmark completes for all goals
- [ ] YAML result file is written under `logs/benchmarks/`

### 1.6 Pre-trained policies (smoke-test)

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint policies/2026-03-25_10-20-56__rand-False_10s-Timeout/model_1499.pt \
  --num_envs 4
```

- [ ] Pre-trained checkpoint loads successfully
- [ ] Robot reaches target poses in simulation

---

## 2. Sim-to-Real — offline checks (conda: `sim2real`)

These tests do not need the real robot or ROS 2.

```bash
conda activate sim2real
```

### 2.1 Policy inference module

```python
# Quick import test
python -c "
from scripts.sim2real.v1.policy_inference import PolicyInference
print('v1 PolicyInference OK')
from scripts.sim2real.v2.policy_inference import PolicyInference
print('v2 PolicyInference OK')
"
```

- [ ] Both imports succeed
- [ ] (Optional) Load an exported `policy.pt` and run `get_action()` with a dummy 24-dim tensor

### 2.2 Observation builder

```python
python -c "
from scripts.sim2real.v1.observation_builder import build_observation
import numpy as np
obs = build_observation(
    joint_pos=np.zeros(6), joint_vel=np.zeros(6),
    tcp_pos=np.zeros(3), tcp_orient=np.zeros(4),
    tcp_linear_vel=np.zeros(3), tcp_angular_vel=np.zeros(3),
    target_pos=np.array([0.3, 0.0, 0.4]), target_orient=np.array([1,0,0,0])
)
assert obs.shape == (24,), f'Expected 24, got {obs.shape}'
print('observation_builder OK')
"
```

- [ ] Observation vector has 24 dimensions
- [ ] No NaN values with zero inputs

### 2.3 Plotting scripts (no ROS needed)

```bash
# Plot DR study (needs result YAMLs)
python scripts/utils/plot_DR_study.py results/results_sim/ --no-recursive

# Plot tuning CSVs (if available)
# python scripts/tuning/plot_sim_gain_tuner_csv.py --file <path-to-csv>
```

- [ ] Scripts run or fail gracefully if input data is missing

---

## 3. Sim-to-Real — with real robot (conda: `sim2real` + ROS 2)

> **Safety:** Ensure the workspace is clear and the e-stop is accessible.

```bash
conda activate sim2real
source /opt/ros/jazzy/setup.bash
source ~/wwro_ws/install/local_setup.bash
```

### 3.1 Driver bring-up

```bash
ros2 launch wwro_startup wwro_control.launch.py \
  gripper_robot_ip:=192.168.1.101 \
  screwdriver_robot_ip:=192.168.1.103
```

- [ ] No crash on launch
- [ ] Teach pendants: Remote mode → External Control → Play
- [ ] `ss -tnp | grep "5000[1-8]"` shows ESTAB connections
- [ ] `ros2 service call /gripper_dashboard_client/program_state ur_dashboard_msgs/srv/GetProgramState` → `PLAYING`

### 3.2 Goal publisher

```bash
python scripts/sim2real/goal_publisher.py --x 0.3 --y 0.0 --z 0.4
```

- [ ] `/goal_pose` topic is published (`ros2 topic echo /goal_pose --once`)
- [ ] RViz marker is visible (if RViz is running)

### 3.3 Deploy a v1 policy

```bash
python scripts/sim2real/v1/sim2real_node.py \
  --model <path-to-exported/policy.pt>
```

- [ ] RTDE connection established (no timeout)
- [ ] Robot moves toward the goal pose
- [ ] Ctrl+C stops the node cleanly (robot holds position)

### 3.4 Deploy a v2 policy (if applicable)

```bash
python scripts/sim2real/v2/sim2real_node.py \
  --model <path-to-exported/policy.pt>
```

- [ ] Same checks as v1
- [ ] Velocity feedforward active (smoother motion compared to v1)

### 3.5 Benchmark run on real robot

```bash
python scripts/sim2real/goal_publisher.py \
  --goals-file scripts/benchmark_settings/goals_handmade.json \
  --benchmark
```

- [ ] Goals cycle through all entries
- [ ] Robot reaches each goal within the timeout

### 3.6 Gripper control

```bash
# Close
ros2 service call /on_twofg7_grip_external wwro_msgs/srv/OnTwofg7 \
  "{gripper_operation: {tool_index: 0, width_mm: 10.0, force_n: 20, speed: 50}}"

# Open
ros2 service call /on_twofg7_release_external wwro_msgs/srv/OnTwofg7 \
  "{gripper_operation: {tool_index: 0, width_mm: 40.0, force_n: 0, speed: 50}}"
```

- [ ] Gripper closes to ~10 mm
- [ ] Gripper opens to ~40 mm

---

## 4. Impedance Tuning (conda: `sim2real` + ROS 2)

See [scripts/tuning/README.md](scripts/tuning/README.md) for background.

### 4.1 Manual sinusoidal excitation

```bash
python scripts/tuning/impedance_tuner.py --joints 0 1 2 3 4 5 --duration 10
```

- [ ] Robot oscillates on selected joints
- [ ] CSV log file is created

### 4.2 Automated Kp sweep

```bash
python scripts/tuning/auto_tuner.py --joints 0 --dry-run
```

- [ ] Dry run prints the Kp grid without moving the robot

### 4.3 Damping ratio sweep

```bash
python scripts/tuning/step_tuner.py --joints 0 --dry-run
```

- [ ] Dry run prints the ζ grid without moving the robot

---

## Summary

| # | Test | Env | Robot needed |
|---|---|---|---|
| 1.1–1.6 | Simulation pipeline | `env_isaaclab` | No |
| 2.1–2.3 | Sim2real offline checks | `sim2real` | No |
| 3.1–3.6 | Real-robot deployment | `sim2real` + ROS 2 | Yes |
| 4.1–4.3 | Impedance tuning | `sim2real` + ROS 2 | Yes |
