
---

# Isaac Lab Training on Euler (ETH Zurich)

This guide details how to run **Isaac Lab** training jobs on the ETH Euler cluster using a custom Apptainer (`.sif`) container.

## quick comands:

```bash
module load stack/2025-06 gcc/12.2.0 && module load git-lfs/3.5.1 && git pull

```

```bash
sbatch train_euler.sh

```



## 1. Prerequisites

Before starting, ensure you have:

1. **The SIF Container:** You need the `isaac_euler_salziegl.sif` file (or your own build).
2. **WandB Account:** For logging training metrics (Weights & Biases).
3. **ETH VPN:** Connected via Cisco AnyConnect (if outside the university network).

---

## 2. Initial Setup

### Transfer the SIF to Euler

If your `.sif` file is on your local machine, use `scp` (Secure Copy) to upload it to your home directory on Euler. Run this **from your local terminal**:

```bash
# Replace 'local/path/to/file' and 'username'
scp /path/to/local/isaac_euler_salziegl.sif username@euler.ethz.ch:/cluster/home/username/

```

### Configure WandB on Euler

1. Log in to [wandb.ai](https://www.google.com/search?q=https://wandb.ai) and copy your API Key from the settings.
2. SSH into Euler.
3. Save the key to a hidden file (replace `YOUR_LONG_API_KEY_HERE` with your actual key):

```bash
echo YOUR_LONG_API_KEY_HERE > ~/.wandb_key

```

4. Secure the file so only you can read it:

```bash
chmod 600 ~/.wandb_key

```

### Prepare Your Code (SKRL Example)

If you are using **skrl**, you must enable WandB in your agent configuration file (e.g., `skrl_ppo_cfg.yaml`). Add this to the `experiment:` section:

```yaml
experiment:
  # ... existing settings ...
  wandb: True
  wandb_kwargs:
    project: "My_Project_Name"  # Your Project Name on WandB
    entity: "my_team_name"         # Your WandB Username/Team
    tags: ["euler", "ur5e"]     # (Optional) Tags for filtering
    monitor_gym: True           # Log video/metrics from the environment

```

> **Note:** Ensure your local code works and connects to WandB before pushing to Euler.

---

## 3. The Euler Job Script (`train_euler.sh`)

Add this script to your repository and name it train_euler.sh
**Important:** make sure the `TASK_NAME` and `SIF_PATH` variables are correct and you use the appropriate amount of compute resource.

```bash
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --job-name="ur5e_train"
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# --- CONFIGURATION ---
#Your task name to run
TASK_NAME="Ur5e-Torque-Reach-v0"
# UPDATE THIS PATH to where you uploaded your .sif file
SIF_PATH="/cluster/home/$USER/isaac_euler_salziegl.sif"

PROJECT_PATH=$(pwd)
PROJECT_NAME=$(basename "$PROJECT_PATH")

# WandB API Key
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/.wandb_key)
else
    echo "Error: ~/.wandb_key not found!"
    exit 1
fi

# --- CACHE SETUP ---
# Creates writable scratch folders for Isaac Sim caches
JOB_CACHE="/cluster/scratch/$USER/isaac_cache/$SLURM_JOB_ID"

mkdir -p $JOB_CACHE/kit_cache $JOB_CACHE/kit_data $JOB_CACHE/ov $JOB_CACHE/pip \
         $JOB_CACHE/glcache $JOB_CACHE/computecache $JOB_CACHE/logs $JOB_CACHE/data \
         $JOB_CACHE/documents $JOB_CACHE/warp $JOB_CACHE/local_lib \
         $JOB_CACHE/wandb_cache $JOB_CACHE/wandb_config $JOB_CACHE/wandb_data

# Load Proxy (Required for internet access on compute nodes)
module load eth_proxy

echo "----------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Project: $PROJECT_NAME"
echo "Container: $SIF_PATH"
echo "----------------------------------------------------------------"

# --- EXECUTION ---
apptainer exec --nv \
    -B $JOB_CACHE/kit_cache:/isaac-sim/kit/cache:rw \
    -B $JOB_CACHE/kit_data:/isaac-sim/kit/data:rw \
    -B $JOB_CACHE/ov:$HOME/.cache/ov:rw \
    -B $JOB_CACHE/pip:$HOME/.cache/pip:rw \
    -B $JOB_CACHE/warp:$HOME/.cache/warp:rw \
    -B $JOB_CACHE/local_lib:$HOME/.local:rw \
    -B $JOB_CACHE/glcache:$HOME/.cache/nvidia/GLCache:rw \
    -B $JOB_CACHE/computecache:$HOME/.nv/ComputeCache:rw \
    -B $JOB_CACHE/logs:$HOME/.nvidia-omniverse/logs:rw \
    -B $JOB_CACHE/data:$HOME/.local/share/ov/data:rw \
    -B $JOB_CACHE/documents:$HOME/Documents:rw \
    -B $JOB_CACHE/wandb_cache:$HOME/.cache/wandb:rw \
    -B $JOB_CACHE/wandb_config:$HOME/.config/wandb:rw \
    -B $JOB_CACHE/wandb_data:$PROJECT_PATH/wandb:rw \
    -B $PROJECT_PATH:/workspace/isaaclab/$PROJECT_NAME:rw \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=$PROJECT_PATH \
    --env WANDB_CACHE_DIR=$HOME/.cache/wandb \
    --env WANDB_CONFIG_DIR=$HOME/.config/wandb \
    $SIF_PATH \
    bash -c "
        # 1. Install Project in Editable Mode
        echo 'Installing Project...'
        /isaac-sim/python.sh -m pip install --user -e /workspace/isaaclab/$PROJECT_NAME/source/$PROJECT_NAME

        # 2. Run Training
        echo 'Starting Training...'
        /isaac-sim/python.sh /workspace/isaaclab/$PROJECT_NAME/scripts/skrl/train.py \
            --task=$TASK_NAME \
            --headless
    "

# Cleanup Cache
rm -rf $JOB_CACHE

```

---

## 4. Workflow: Running a Job

1. **Push your code:** Make sure your local changes are pushed to GitHub.
2. **SSH into Euler:**
```bash
ssh username@euler.ethz.ch

```


3. **Navigate to your repo and update:**
```bash
cd your_project_name

```

```bash
module load stack/2025-06 gcc/12.2.0 && module load git-lfs/3.5.1 && git pull

```


*(Note: Ensure your GitHub repository name matches the folder name in `/source/Project_Name`, or update the paths in the script).*
4. **Start the training:**
```bash
sbatch train_euler.sh

```



### Monitoring

* **Check status:** `squeue` (Shows running/pending jobs)
* **Cancel job:** `scancel JOB_ID`
* **View Logs:** Check the `logs/` folder inside your repo (e.g., `cat logs/train_12345.out`).
* **View Progress:** Go to your WandB Dashboard

---

## 5. Moving Checkpoints (Euler -> Local)

When training is finished, your checkpoints (and logs) will be in your repository folder on Euler in `logs/skrl`.



1. **Download:**
```bash
# Downloads the whole 'runs' folder to your current local directory
scp -r username@euler.ethz.ch:~/your_project_name/logs/skrl .

```

you can still look at the logs with tensorboard locally, if the WandB synchronisation didn't work.
and make sure to delete the checkpoints on euler from time to time to avoid running out of space!
