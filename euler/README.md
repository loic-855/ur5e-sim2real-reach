# Isaac Lab Training on Euler

This README documents trainging on the Euler cluster, with a focus on the v1 ablation sweep defined in `euler/sweep_sim2real_v1_ablation_10s_timeout.yaml`.

Use it in two ways:

1. Edit `euler/train_euler.sh` for a single manual run.
2. Use `euler/generate_sweep.py` with `euler/sweep_euler.sh` to automate the full ablation sweep.

The same pattern can be reused for a future sweep file, but this README stays focused on the current v1 ablation setup.

## Files In This Folder

| File | Role |
|------|------|
| `euler/train_euler.sh` | Single manual training run with Hydra overrides written directly in the shell script |
| `euler/sweep_sim2real_v1_ablation_10s_timeout.yaml` | Sweep definition for the thesis ablation study |
| `euler/generate_sweep.py` | Expands the YAML into a concrete run list and can submit the sweep |
| `euler/sweep_euler.sh` | SLURM array script that reads the generated run list and executes each run |



## Before You Start

You need:

1. A clone of the repository on Euler, under `/cluster/scratch/$USER/ur5e-sim2real-reach`.
2. The Isaac Lab `.sif` container uploaded to Euler.
3. The USD assets copied to the repository root.
4. A WandB API key saved in `~/.wandb_key`.
5. Access to Euler over SSH.

Recommended setup:

```bash
ssh username@euler.ethz.ch
cd /cluster/scratch/$USER
git clone git@github.com:loic-855/ur5e-sim2real-reach.git
cd ur5e-sim2real-reach
```

Configure WandB once:

```bash
echo YOUR_WANDB_API_KEY > ~/.wandb_key
chmod 600 ~/.wandb_key
```

Upload the container once:

```bash
scp /path/to/isaac_euler_salziegl.sif username@euler.ethz.ch:/cluster/scratch/username/
```

If you use `generate_sweep.py`, load Python first:

```bash
module load stack/2024-06 python/3.12.8
```

## Quick Start

Single run with manual overrides:

```bash
cd /cluster/scratch/$USER/ur5e-sim2real-reach
git pull
sbatch euler/train_euler.sh
```

Preview the v1 ablation sweep:

```bash
cd /cluster/scratch/$USER/ur5e-sim2real-reach
git pull
module load stack/2024-06 python/3.12.8
python euler/generate_sweep.py --config euler/sweep_sim2real_v1_ablation_10s_timeout.yaml --dry-run
```

Generate and submit the full sweep:

```bash
cd /cluster/scratch/$USER/ur5e-sim2real-reach
module load stack/2024-06 python/3.12.8
python euler/generate_sweep.py --config euler/sweep_sim2real_v1_ablation_10s_timeout.yaml --submit
```

## Manual Overrides With `train_euler.sh`

`train_euler.sh` is the fastest way to launch one experiment variant by hand.

Inside the script, everything after `.../train.py` line  is passed to `scripts/rsl_rl/train.py` as Hydra overrides. That means you can directly edit the shell script to change the experiment without touching Python code.

The current `train_euler` script already contains a concrete example. 

## Automated Sweep With `generate_sweep.py` And `sweep_euler.sh`

The automated path is split into three parts:

1. `sweep_sim2real_v1_ablation_10s_timeout.yaml` defines the experiment.
2. `generate_sweep.py` expands the YAML into a concrete run file named `euler/sweep_runs_sweep_sim2real_v1_ablation_10s_timeout.txt`.
3. `sweep_euler.sh` reads that generated file and launches the runs as a SLURM array.

### Global Overrides

`base_overrides` in the YAML are applied to every generated run:

```yaml
base_overrides:
  - "agent.max_iterations=1500"
  - "agent.wandb_project=sim2real_v1_ablation"
  - "agent.experiment_name=sim2real_v1_ablation"
  - "env.goal_timeout_s=10.0"
```

This is the right place for settings that should stay identical across the full study.

### Per-Run Overrides

The `dimensions` section defines the actual experiment conditions. In the current file there is one dimension, `rand`, so each preset directly becomes one generated run.

Current presets:

- `no_dr`
- `all_enabled_10s-Timeout`
- `actuator_only_10s-Timeout`
- `masscom_only_10s-Timeout`
- `noise_only_10s-Timeout`
- `delay_only_action_1_2_10s-Timeout`
- `delay_only_action_0_1_10s-Timeout`
- `all_enabled_no_COM_delay_1_2_10s-Timeout`
- `actuator_plus_delay_1_2_10s-Timeout`

Each preset overrides a small set of toggles:

- `env.domain_rand.enable_actuator_rand`
- `env.domain_rand.enable_mass_com_rand`
- `env.domain_rand.enable_noise`
- `env.domain_rand.enable_delay`
- `env.domain_rand.action_delay_range`
- `env.domain_rand.obs_delay_range`

To adapt the sweep quickly for another study, duplicate one preset, rename it, and edit only those values.

### SLURM Resource Overrides

The YAML also controls the main SLURM settings used when `generate_sweep.py --submit` calls `sbatch`:

```yaml
slurm:
  time_per_task: "07:00:00"
  cpus_per_task: 4
  mem_per_cpu: 6000
  nodes: 3
```

What these fields do:

- `time_per_task`: expected runtime for one run. The generator adds a safety margin before submitting.
- `nodes`: number of SLURM array jobs.
- `cpus_per_task`: passed to `sbatch --cpus-per-task`.
- `mem_per_cpu`: passed to `sbatch --mem-per-cpu`.

`sweep_euler.sh` still contains default `#SBATCH` values, but the submitted values from the YAML override them when you use `generate_sweep.py --submit`.

### What `sweep_euler.sh` Actually Executes

For each generated run, `sweep_euler.sh` does this:

```bash
/isaac-sim/python.sh /workspace/isaaclab/$PROJECT_NAME/scripts/rsl_rl/train.py \
    --task=$TASK_NAME \
    --headless \
    --run_name=$RUN_NAME \
    $HYDRA_OVERRIDES
```

That means:

- `TASK_NAME` comes from the generated metadata.
- `RUN_NAME` comes from the preset name.
- `$HYDRA_OVERRIDES` is the concatenation of `base_overrides` and the preset-specific overrides.

`sweep_euler.sh` itself is mostly infrastructure: cache setup, container launch, editable install, WandB environment, and sequential execution inside each array job.

## Editing The Current v1 Ablation Sweep

The fastest edits are:

1. Change `base_overrides` for settings shared by every run.
2. Change a preset under `dimensions.rand` for one specific condition.
3. Change `slurm` if the sweep needs different runtime or resource allocation.
4. Change `SIF_PATH` in both `euler/train_euler.sh` and `euler/sweep_euler.sh` if your container path changes.

If you later create another sweep file, keep the same structure:

1. `task_name`
2. `slurm`
3. `base_overrides`
4. `dimensions`

## Monitoring

Check queue status:

```bash
squeue -u $USER
```

Check single-run logs:

```bash
cat logs/train_<JOB_ID>.out
```

Check sweep logs:

```bash
cat logs/sweep_<JOB_ID>_<ARRAY_ID>.out
```

Cancel a job:

```bash
scancel JOB_ID
```

## Replay A Sweep Checkpoint Locally

If you replay a checkpoint produced by the sweep, pass the generated sweep file so `scripts/rsl_rl/play.py` can re-apply the matching overrides.

Example:

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --num_envs 5 \
  --checkpoint "logs/rsl_rl/<run_folder>/model_1499.pt" \
  --sweep_file "euler/sweep_runs_sweep_sim2real_v1_ablation_10s_timeout.txt" \
  env.debug=True
```

If you copy results back from Euler, copy the checkpoint folder and the generated sweep file together.

