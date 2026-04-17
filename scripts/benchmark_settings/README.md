# Benchmark Settings

This folder stores the fixed inputs used by the simulation and real-robot benchmark workflows.

## Files

- `goals_handmade.json`: manually selected pose goals.
- `goals_safe.json`: smaller, safer goal subset.
- `sim_benchmark_path.json`: list of simulation checkpoints to benchmark, used for automated evaluation with `scripts/rsl_rl/benchmark.py`.
- `real_benchmark_path.json`: list of exported real-robot policy artifacts, used for automated evaluation with `scripts/sim2real/v1/sim2real_node.py`.

## Goal Format

Each goal is stored as:

```json
[x, y, z, qw, qx, qy, qz]
```

The current benchmark runners expect a JSON list of those goals.

## Simulation Benchmark

The main simulation benchmark runner is `scripts/rsl_rl/benchmark.py`.

Typical usage:

```bash
python scripts/rsl_rl/benchmark.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_1499.pt \
  --num_envs 1 \
  --seed 12 \
  --goals-file scripts/benchmark_settings/goals_handmade.json
```

The current benchmark logic is built around deterministic goal playback, explicit goal files, and disabled domain randomization during evaluation.

## Path List Files

### `sim_benchmark_path.json`

Contains checkpoint paths, usually under a `checkpoints` list.

### `real_benchmark_path.json`

Contains exported `policy.pt` model paths, usually under a `models` list.

Keep these files together with the related evaluation campaign so the artifact set remains reproducible.

## Current Metrics and Status

The benchmark workflow tracks pose-reaching performance such as time to enter a tolerance area and error statistics while the robot remains near the goal.

The simulation runner exists today. The real-robot benchmark workflow is partially supported through the sim2real deployment code and artifact lists, but the evaluation path is still less unified than the simulation side.