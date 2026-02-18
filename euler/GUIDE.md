
# Isaac Lab Training on Euler (ETH Zurich)

This guide details how to run **Isaac Lab** training jobs on the ETH Euler cluster using a custom Apptainer (`.sif`) container and play them back on the workstation.

## Quick Commands

```bash
# Load python in your session, git lfs
module load git-lfs/3.5.1 && module load stack/2024-06 python/3.12.8
```

```bash
# Single training run
sbatch train_euler.sh

# Sweep: generate + submit
python euler/generate_sweep.py --config euler/sweep_position_orientation.yaml --submit
```
```bash
#Play the trained sweep
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --num_envs 5 \
  --checkpoint "logs/rsl_rl/2026-02-18_00-32-14_reward_position-default_reward_orientation-default/model_2499.pt" \
  --sweep_file "euler/sweep_runs.txt" \
  env.debug=True
```


Before starting, ensure you have:

1. **The SIF Container:** You need the `isaac_euler_salziegl.sif` file (or your own build).
2. **WandB Account:** For logging training metrics (Weights & Biases).
3. **ETH VPN:** Connected via Cisco AnyConnect (if outside the university network).
4. **ETH Euler setup**: The connection between your computer and ETH Euler cluster is setup. Quick test via `ssh username@euler.ethz.ch`.

**(Optional) File transfer**:  Direct connect to Euler file system on Ubuntu for file transfer: 

Start the file explorer, go to `Other Locations`, entre the server adress `sftp://username@euler.ethz.ch/` and connect. Use it for transferring run or the container, but not for you project file, use git for that.


---

## 2. Initial Setup

### Transfer the SIF to Euler

Upload your `.sif` file from your local machine:

```bash
scp /path/to/local/isaac_euler_salziegl.sif username@euler.ethz.ch:/cluster/scratch/username/
```

### Configure WandB on Euler

1. Log in to [wandb.ai](https://wandb.ai) and copy your API Key from settings.
2. SSH into Euler and save the key:

```bash
echo YOUR_LONG_API_KEY_HERE > ~/.wandb_key
chmod 600 ~/.wandb_key
```

---

## 3. Single Training Run (`train_euler.sh`)

For running a single training job with fixed parameters.

**Important:** Update `TASK_NAME` and `SIF_PATH` in the script before running.

```bash
# 1. SSH into Euler and pull latest code
ssh username@euler.ethz.ch
cd Woodworking_Simulation
module load stack/2025-06 gcc/12.2.0 && module load git-lfs/3.5.1 && git pull

# 2. Submit
sbatch train_euler.sh
```

### Monitoring

* **Check status:** `squeue -u $USER`
* **Cancel job:** `scancel JOB_ID`
* **View Logs:** `cat logs/train_<JOB_ID>.out`
* **View Progress:** Go to your WandB Dashboard

---

## 4. Hyperparameter Sweep

The sweep system runs many training configurations in parallel on the cluster.

### Files

| File | Role |
|------|------|
| `euler/sweep_*.yaml` | Sweep config: task name, SLURM settings, dimensions, presets |
| `euler/generate_sweep.py` | Generates `sweep_runs.txt` from YAML, optionally submits to SLURM |
| `euler/sweep_euler.sh` | SLURM job script (reads config from `sweep_runs.txt` metadata) |
| `euler/sweep_runs.txt` | Generated output: metadata header + one line per run |

### 4.1 Sweep Config (YAML)

Create a YAML file in `euler/` defining your sweep. Example:

```yaml
task_name: "Template-Pose-Orientation-Sim2Real-Direct-v1-ext"

slurm:
  time: "06:30:00"
  gpus: "rtx_4090:1"
  cpus_per_task: 3
  mem_per_cpu: 4000

sequential_per_job: 6    # runs chained per SLURM job

base_overrides:
  - "agent.max_iterations=2500"

dimensions:
  reward_position:
    default: []           # use code defaults
    moderate:
      - "env.ee_position_penalty=-0.18"
      - "env.ee_position_reward=0.30"
  reward_orientation:
    default: []
    strong:
      - "env.ee_orientation_penalty=-0.18"
      - "env.ee_orientation_reward=0.30"
```

Key fields:
- **`task_name`**: Registered Isaac Lab task to train
- **`slurm`**: Resource settings (override `#SBATCH` defaults in `sweep_euler.sh`)
- **`sequential_per_job`**: How many runs to chain in one SLURM job
- **`base_overrides`**: Hydra overrides applied to every run
- **`dimensions`**: Each dimension has named presets. The sweep is the **Cartesian product** of all dimensions. Use `[]` for "use defaults".

Existing configs:
- `sweep_config.yaml` — actuators × DR × network × action_rate
- `sweep_reward_weight.yaml` — orientation reward only (5 presets)
- `sweep_position_orientation.yaml` — position × orientation reward grid

### 4.2 Workflow

```bash
# 1. SSH into Euler and pull latest code
ssh username@euler.ethz.ch
cd Woodworking_Simulation
module load stack/2025-06 gcc/12.2.0 && module load git-lfs/3.5.1 && git pull

# 2. Preview the sweep (no files written, no jobs submitted)
python euler/generate_sweep.py --config euler/sweep_position_orientation.yaml --dry-run

# 3. Generate sweep_runs.txt AND submit to SLURM
python euler/generate_sweep.py --config euler/sweep_position_orientation.yaml --submit

# 4. Monitor
squeue -u $USER
cat logs/sweep_<JOB_ID>_<ARRAY_ID>.out
```

### 4.3 How It Works

1. `generate_sweep.py` computes the Cartesian product of all dimension presets
2. Writes `sweep_runs.txt` with a metadata header and one line per run:
   ```
   # META task_name=Template-Pose-Orientation-Sim2Real-Direct-v1-ext
   # META sequential_per_job=6
   # META total_runs=36
   run_name_1|hydra_override_1 hydra_override_2
   run_name_2|hydra_override_1 hydra_override_2
   ```
3. Submits `sweep_euler.sh` as a SLURM array job (`sbatch --array=0-N --time=... --gpus=...`)
4. Each array task reads its chunk of runs from `sweep_runs.txt` and executes them sequentially
5. SLURM settings from the YAML override the `#SBATCH` defaults via command-line flags

### 4.4 Tips

- Use `--dry-run` first to verify the sweep grid
- Failed runs are logged but don't stop the remaining runs in the job
- Run names are auto-generated from preset names (e.g. `reward_position-moderate_reward_orientation-strong`)
- `SIF_PATH` in `sweep_euler.sh` must point to your `.sif` container
- Generate without submitting (omit `--submit`) to inspect `sweep_runs.txt` first

---

## 5. Moving Checkpoints (Euler -> Local)

When training is finished, your checkpoints are in `logs/rsl_rl/` on Euler.

```bash
# Download a specific run
scp -r username@euler.ethz.ch:~/Woodworking_Simulation/logs/rsl_rl/<experiment>/<timestamp> .

# Download all logs
scp -r username@euler.ethz.ch:~/Woodworking_Simulation/logs/rsl_rl .
```

You can view logs with TensorBoard locally if WandB sync didn't work.
Delete old checkpoints on Euler regularly to avoid running out of space.

## 6. Running the checkpoints

Once the runs are transferred on the local computer, the can be run with the modified play.py script in the repo. The modified variables of the sweep can be directly passed in command line form, it avoids modifiying the base script.

Special arguements need to be used: 
`--sweep_file <path_to_sweep_file>` to pass the modified variable. The sweep file is a `.txt` file with the structure `run_name | where.variable_name.modified_value `:

 ```
 reward_position-default_reward_orientation-default|agent.max_iterations=2500
reward_position-default_reward_orientation-conservative|agent.max_iterations=2500 env.ee_orientation_penalty=-0.10 env.ee_orientation_reward=0.0
...
```
`--checkpoint <path_to_checkpoint>` to give one of the checkpoint of the sweep.

`env.debug=True`to have visualisation if the script supports it. Argument without `--` must be at the end of the command

### Example

```bash
python scripts/rsl_rl/play.py \
  --task WWSim-Pose-Orientation-Sim2Real-Direct-v1 \
  --num_envs 5 \
  --checkpoint "logs/rsl_rl/2026-02-18_00-32-14_reward_position-default_reward_orientation-default/model_2499.pt" \
  --sweep_file "euler/sweep_runs.txt" \
  env.debug=True
```

