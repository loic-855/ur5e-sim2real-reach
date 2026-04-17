# RSL-RL Workflows

This folder contains the main training, playback, and simulation benchmark entry points used in the repository.

## Files

- `train.py`: train a policy with RSL-RL.
- `play.py`: load a checkpoint, run inference in simulation, and export deployment artifacts.
- `benchmark.py`: deterministic simulation benchmark runner.
- `cli_args.py`: shared CLI argument handling.

## Training

Typical training command:

```bash
python scripts/rsl_rl/train.py --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 --headless
```

Outputs are stored under:

- `logs/rsl_rl/<experiment>/<run>/`

Important run artifacts include:

- `model_*.pt`
- `params/env.yaml`
- `params/agent.yaml`

## Playback and Export

Typical playback command:

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_2499.pt
```

`play.py` exports:

- `exported/policy.pt`
- `exported/policy.onnx`

If a run came from a sweep, use `--sweep_file` so run-specific Hydra overrides are recovered.

## Deterministic Benchmark

Typical benchmark command:

```bash
python scripts/rsl_rl/benchmark.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_1499.pt \
  --goals-file scripts/benchmark_settings/goals_handmade.json
```

The benchmark runner currently:

- loads explicit goal files
- forces deterministic goal playback
- disables domain randomization during evaluation
- writes YAML results under `logs/benchmarks/sim_pose/`

## Task IDs You Should Use

- `WWSim-Pose-Orientation-Sim2Real-Direct-v1`
- `WWSim-Pose-Orientation-Sim2Real-Direct-v2`
- `WWSim-Pose-Orientation-Two-Robots-v0`
- `WWSim-Grasping-Single-Robot-Direct-v0`
- `WWSim-Grasping-Single-Robot-Direct-v1`

Use the real task IDs exactly as registered in the environments, not older thesis naming.