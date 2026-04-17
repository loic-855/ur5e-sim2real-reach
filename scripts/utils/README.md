# Analysis Utilities

This directory contains analysis and helper scripts used across simulation and real-robot workflows.

## Script Inventory

- `plot_DR_study.py`: compare benchmark YAML results across model checkpoints and impedance gain configurations.
- `gripper_action.py`: drives the gripper joint in simulation with a bi-directional ramp for articulation testing.
- `ee_path_from_pose.py`: publish a TCP path for RViz, used for plotting the end-effector path.

## `plot_DR_study.py`

Scans a directory of benchmark YAML files (produced by `scripts/rsl_rl/benchmark.py` or the sim2real nodes),
groups runs by model checkpoint path, and compares `simulation` vs `tuned` impedance-gain configurations across three metrics:

- `mean_time_to_area_s` — average time to enter the goal tolerance
- `mean_pos_err_area_m` — average position error while inside the tolerance
- `mean_rot_err_area_rad` — average orientation error while inside the tolerance

Outputs one PDF bar-chart per metric and a YAML summary report.

### Usage

```bash
python scripts/utils/plot_DR_study.py <input_dir> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input_dir` | — | Folder containing benchmark YAML files |
| `--output-dir PATH` | `<input_dir>/impedance_gain_comparison` | Where to write the report and PDF plots |
| `--recursive` / `--no-recursive` | recursive | Whether to search sub-directories for YAML files |
| `--simulation` | off | Treat runs with a missing `robot_gain` field as `simulation` (equivalent to `naive`) |
| `--untuned-only` | off | Skip files with `robot_gain: tuned`; useful to inspect only baseline runs |

### YAML File Requirements

Each YAML file must contain:

```yaml
metadata:
  model_path: logs/rsl_rl/<experiment>/<run>/model_1499.pt   # or checkpoint
  robot_gain: simulation   # or: naive, tuned
  action_scale: 0.5        # optional
summary:
  mean_time_to_area_s: 3.2
  mean_pos_err_area_m: 0.012
  mean_rot_err_area_rad: 0.08
```

`model_path` and `checkpoint` are both accepted. `naive` is treated as equivalent to `simulation`.
Runs with an unrecognised `robot_gain` value are skipped with a warning.

### Model grouping

Runs are grouped by the **training-run directory**, not the exact checkpoint filename.
This means `<run>/model_1499.pt` and `<run>/exported/policy.pt` map to the same model key,
making it safe to mix simulation checkpoints and exported real-robot artifacts in the same comparison.

### Interactive prompts

The script prompts for a custom display name per model and an optional report title when running interactively.
These are skipped automatically when stdin is not a terminal (e.g. in CI).

### Examples

```bash
# Compare simulation vs tuned gains across all ablation variants
python scripts/utils/plot_DR_study.py \
    logs/benchmarks/sim_pose \
    --output-dir logs/benchmarks/sim_pose/gain_comparison

# Non-recursive: only YAML files at the top level
python scripts/utils/plot_DR_study.py \
    logs/benchmarks/sim_pose --no-recursive

# Only look at baseline (simulation/naive) runs
python scripts/utils/plot_DR_study.py \
    logs/benchmarks/sim_pose --untuned-only

# Treat runs with missing robot_gain as simulation baseline
python scripts/utils/plot_DR_study.py \
    logs/benchmarks/real_results --simulation
```

---

## `gripper_action.py`

Runs the gripper joint (joint index 6) through a continuous bi-directional ramp in simulation.
All other joints are held at zero. Useful for visually verifying gripper articulation and limits
without needing a full policy.

### Usage

Launch Isaac Sim with the target task and this script as the agent:

```bash
python scripts/utils/gripper_action.py \
    --task WWSim-Grasping-Single-Robot-Direct-v0 \
    --num_envs 1
```

The gripper ramps from −1 to +1 and back continuously until the simulation window is closed.
Adjust `step_size` inside the script to change the ramp speed.

## `ee_path_from_pose.py`
Subscribes to a TCP pose topic, constructs a path trace of the end-effector, and republishes it as a `nav_msgs/Path` for RViz visualization.

### Usage

Publish a TCP path trace:

```bash
python scripts/sim2real/ee_path_from_pose.py \
	--input-topic /gripper_tcp_pose_broadcaster/pose \
	--output-topic /ee_path \
	--max-points 5000 \
	--min-dt 0.03
```
In RViz, add a `Path` display subscribed to `/ee_path` and set the fixed frame to `world` to see the end-effector path trace.