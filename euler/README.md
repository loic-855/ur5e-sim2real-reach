# Isaac Lab Training on Euler (ETH Zurich)

This guide covers running **Isaac Lab** training jobs on the ETH Euler cluster using an Apptainer (`.sif`) container, and replaying checkpoints locally.

---

## Quick Commands

```bash
# Load Python in your session (required before running generate_sweep.py)
module load stack/2024-06 python/3.12.8
```

```bash
# Single training run (edit TASK_NAME and overrides in train_euler.sh first)
sbatch euler/train_euler.sh
```

```bash
# Sweep: generate sweep file + submit array job to SLURM
python euler/generate_sweep.py --config euler/sweep_sim2real_v1_ablation_10s_timeout.yaml --submit

# Preview sweep without writing files or submitting
python euler/generate_sweep.py --config euler/sweep_sim2real_v1_ablation_10s_timeout.yaml --dry-run
```

```bash
# Monitor and cancel
squeue -u $USER
scancel JOB_ID
```

```bash
# Play a trained model locally
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --num_envs 5 \
  --checkpoint "logs/rsl_rl/<run_folder>/model_1499.pt" \
  env.debug=True
```

---

## Prerequisites

Before starting, ensure you have:

1. **The SIF container** — download below (Polybox).
2. **WandB account** — for logging training metrics.
3. **ETH VPN** — Cisco AnyConnect (if outside the university network).
4. **Euler SSH access** — test with `ssh username@euler.ethz.ch`.

---

## Initial Setup

### Download the Required Assets

Two assets are needed and are hosted on Polybox:

| Asset | Link | Where to put it |
|-------|------|-----------------|
| **USD scene files** | [Polybox — USD files](https://polybox.ethz.ch/index.php/s/tdgY7imXcJH9qJn) | `source/Woodworking_Simulation/Woodworking_Simulation/assets/` (or wherever the config points) |
| **Isaac Sim `.sif` container** | [Polybox — SIF container](https://polybox.ethz.ch/index.php/s/5kYQr3z762brqJC) | `/cluster/scratch/$USER/` on Euler |

> **File transfer tip:** In Ubuntu's file explorer, go to *Other Locations* and enter `sftp://username@euler.ethz.ch/` to mount Euler's filesystem directly for large file transfers. Use Git for project code.

### Upload the SIF to Euler

```bash
scp /path/to/local/isaac_euler_salziegl.sif username@euler.ethz.ch:/cluster/scratch/username/
```

Then verify `SIF_PATH` in both `train_euler.sh` and `sweep_euler.sh` matches where you put it:

```bash
SIF_PATH="/cluster/scratch/$USER/isaac_euler_salziegl.sif"
```

### Configure WandB on Euler

1. Log in to [wandb.ai](https://wandb.ai) and copy your API key from settings.
2. SSH into Euler and save the key:

```bash
echo YOUR_LONG_API_KEY_HERE > ~/.wandb_key
chmod 600 ~/.wandb_key
```

The training scripts read `~/.wandb_key` automatically — they will exit with an error if it is missing.

---

## Single Training Run (`train_euler.sh`)

Use this for a one-off run with fully customized parameters.

**Before running:** open `euler/train_euler.sh` and edit:
- `TASK_NAME` — the registered Isaac Lab task to train (must match a task registered in the Python package).
- The Hydra overrides at the bottom of the `apptainer exec` block (domain randomization flags, reward weights, timeouts, etc.).

```bash
# SSH into Euler, pull latest code, then submit
ssh username@euler.ethz.ch
cd Woodworking_Simulation
git pull
sbatch euler/train_euler.sh
```

The script will:
1. Create a per-job scratch cache under `/cluster/scratch/$USER/isaac_cache/$SLURM_JOB_ID/`.
2. Load the ETH proxy module (required for internet/WandB access from compute nodes).
3. Install the project package in editable mode inside the container.
4. Run `scripts/rsl_rl/train.py` with the task and overrides you specified.
5. Clean up the scratch cache on exit.

Logs are written to `logs/train_<JOB_ID>.out` and `logs/train_<JOB_ID>.err`.

---

## Hyperparameter Sweep (`generate_sweep.py` + `sweep_euler.sh`)

The sweep system runs multiple training configurations in parallel across cluster nodes, with each node executing its assigned runs sequentially.

### Files

| File | Role |
|------|------|
| `euler/sweep_*.yaml` | Sweep config: task name, SLURM settings, global overrides, sweep dimensions |
| `euler/generate_sweep.py` | Reads YAML, generates `sweep_runs_<config_stem>.txt`, optionally submits |
| `euler/sweep_euler.sh` | SLURM array job script — reads config from the generated `.txt` file |
| `sweep_runs_<config_stem>.txt` | Generated output: metadata header + one run per line (not committed) |

---

### How Runs Are Distributed Across Nodes

The `slurm.nodes` field controls how many SLURM array tasks (cluster nodes) are used. The generator distributes all runs across them as evenly as possible:

```
total_runs = 9,  nodes = 3  →  each node runs 3 jobs sequentially
total_runs = 9,  nodes = 4  →  distribution: [2, 2, 2, 3]  (remainder to last)
total_runs = 9,  nodes = 9  →  each node runs 1 job
```

**Trade-off:**
- **More nodes** → more parallel execution, shorter wall-clock time, but more cluster resources consumed and more SLURM queue slots used.
- **Fewer nodes** → fewer resources, but each node runs more jobs sequentially, increasing wall-clock time.

The walltime per job is computed automatically:
```
job_walltime = (time_per_task + 15 min safety) × sequential_runs_on_that_node
```

This is passed directly to `sbatch --time=...`, so you never need to compute it manually.

---

### Sweep Configuration (YAML)

Create or edit a YAML file in `euler/` to define your sweep. The full structure:

```yaml
task_name: "WWSim-Pose-Orientation-Sim2Real-Direct-v1"

slurm:
  time_per_task: "07:00:00"   # Estimated wall time PER single run (HH:MM:SS). 
                               # 15 min safety is added automatically per run.
  nodes: 3                     # Number of SLURM array tasks (cluster nodes).
  gpus: "rtx_4090:1"           # GPU type and count (optional, defaults in sweep_euler.sh).
  cpus_per_task: 4             # CPUs per node.
  mem_per_cpu: 6000            # Memory per CPU in MB.

# Hydra overrides applied to EVERY run (global settings)
base_overrides:
  - "agent.max_iterations=1500"
  - "agent.wandb_project=my_project"
  - "env.goal_timeout_s=10.0"

# Each dimension defines a set of named presets.
# The sweep is the Cartesian product of all dimensions.
dimensions:
  rand:
    no_dr:
      - "env.domain_rand.enable_actuator_rand=False"
      - "env.domain_rand.enable_delay=False"
    all_enabled:
      - "env.domain_rand.enable_actuator_rand=True"
      - "env.domain_rand.enable_delay=True"
  reward:
    default: []          # Empty list = use code defaults, still creates a named preset
    strong:
      - "env.ee_position_reward=0.30"
```

Key fields:
- **`task_name`**: Registered Isaac Lab task identifier.
- **`slurm.time_per_task`**: Time budget for a *single* run. The generator multiplies this (plus 15 min safety) by the sequential count to get the per-job walltime.
- **`slurm.nodes`**: Number of parallel cluster nodes. Controls parallelism vs. resource usage (see above).
- **`base_overrides`**: Hydra-style overrides common to every run (e.g. iteration count, WandB project, timeouts).
- **`dimensions`**: Each key is a dimension, each sub-key is a named preset. Use `[]` for "use code defaults" — it still gets a unique name and participates in the Cartesian product.

**Example:** 2 dimensions with 2 and 3 presets → 6 total runs.

---

### Generated Sweep File Format

`generate_sweep.py` writes a `sweep_runs_<config_stem>.txt` file:

```
# META task_name=WWSim-Pose-Orientation-Sim2Real-Direct-v1
# META sequential_per_job=3
# META sequential_per_job_list=3,3,3
# META slurm_time=07:45:00
# META total_runs=9
# META config=sweep_sim2real_v1_ablation_10s_timeout.yaml
# META generated=2026-04-16T10:30:00
rand-no_dr|agent.max_iterations=1500 env.domain_rand.enable_actuator_rand=False ...
rand-all_enabled|agent.max_iterations=1500 env.domain_rand.enable_actuator_rand=True ...
...
```

- `sequential_per_job_list` is a comma-separated list with one entry per array task.
- Each run line is `run_name|hydra_overrides` — the run name is auto-generated from preset names.
- `sweep_euler.sh` reads the metadata header at runtime to determine how many runs to execute per node.

---

### Workflow

```bash
# On Euler:
ssh username@euler.ethz.ch
cd Woodworking_Simulation
module load stack/2024-06 python/3.12.8
git pull

# 1. Preview (no files written, no jobs submitted)
python euler/generate_sweep.py --config euler/sweep_sim2real_v1_ablation_10s_timeout.yaml --dry-run

# 2. Generate sweep file only (inspect before submitting)
python euler/generate_sweep.py --config euler/sweep_sim2real_v1_ablation_10s_timeout.yaml

# 3. Generate + submit
python euler/generate_sweep.py --config euler/sweep_sim2real_v1_ablation_10s_timeout.yaml --submit
```

Failed individual runs are logged but do not abort the remaining runs on that node.

---

## Playing Checkpoints Locally

Use `scripts/rsl_rl/play.py`. For a single checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --num_envs 5 \
  --checkpoint "logs/rsl_rl/<run_folder>/model_1499.pt" \
  env.debug=True
```

To replay all runs from a sweep (iterates over every entry in the sweep file):

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --num_envs 5 \
  --checkpoint "logs/rsl_rl/<any_run_in_sweep>/model_1499.pt" \
  --sweep_file "euler/sweep_runs_<config_stem>.txt" \
  env.debug=True
```

`--sweep_file` tells the script to iterate over every `run_name|overrides` entry in the file, applying each variant's overrides automatically. Hydra overrides without `--` (like `env.debug=True`) must be placed at the end of the command.
