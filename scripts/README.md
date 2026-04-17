# Scripts

This folder contains all the entry points for training, simulation playback, benchmarking, and real-robot deployment.

## Folder Map

| Folder / File | Description |
|---|---|
| `rsl_rl/` | RSL-RL training, playback, and simulation benchmark — **the RL library used in this project** |
| `sim2real/` | Real-robot deployment nodes, ROS2 utilities, URScript controllers |
| `tuning/` | Impedance controller gain-tuning tools (real robot + simulation logger) |
| `utils/` | Analysis utilities used across sim and real workflows |
| `benchmark_settings/` | Shared goal files and checkpoint lists for reproducible evaluation campaigns |
| `random_agent.py` | IsaacLab helper: runs a random-action agent in the sim environment |
| `zero_agent.py` | IsaacLab helper: runs a zero-action agent (useful for environment smoke-tests) |
| `list_envs.py` | IsaacLab helper: lists all registered task environments |
| `prim_path_helper.txt` | Notes on USD prim paths used in the simulation scene |

## Quick Reference

### Train (RSL-RL)

```bash
python scripts/rsl_rl/train.py --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 --headless
```

### Play / Export policy

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_2499.pt
```

Exports `exported/policy.pt` + `exported/policy.onnx`.

### Simulation benchmark

```bash
python scripts/rsl_rl/benchmark.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_1499.pt \
  --goals-file scripts/benchmark_settings/goals_handmade.json
```

### Real-robot deployment

```bash
# Position-only policy (v1)
python scripts/sim2real/v1/sim2real_node.py --model /path/to/exported/policy.pt

# Velocity-feedforward policy (v2)
python scripts/sim2real/v2/sim2real_node.py --model /path/to/exported/policy.pt
```

See `scripts/sim2real/README.md` for the full ROS2 bring-up sequence.

### Impedance controller tuning

```bash
# Step 1: sinusoidal excitation
python scripts/tuning/impedance_tuner.py

# Step 2: automated Kp sweep
python scripts/tuning/auto_tuner.py

# Step 3: damping sweep
python scripts/tuning/step_tuner.py
```

See `scripts/tuning/README.md` for the full 3-step workflow.
