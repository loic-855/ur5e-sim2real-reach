# Template for Isaac Lab Project Woodworking

## Overview

This is an external Isaac Lab project for the IDEALB Woodworking setup at ETHZ.![alt text](Main_setup.png)

## Installation
- Installation has been tested with Isaac Sim 5.0.0 and Isaac Lab 2.2.1

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/Woodworking_Simulation

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

## Import the Assets

Download files USD_files folder from polybox link https://polybox.ethz.ch/index.php/s/tdgY7imXcJH9qJn place them at the root folder of the repo.

## Running pre-trained policies

Pre-trained policies are available for the first two environments. These are working examples that demonstrate the capabilities but are **not optimized for performance**.

```bash
# Single gripper robot (10 environments with visualization)
python scripts/rsl_rl/play.py --task=Template-Pose-Orientation-Gripper-Robot-Direct-v0 --num_envs=10 --checkpoint=pretrained_models/model_pose_orientation_gripper_robot.pt

# Dual robot setup (10 environments with visualization)
python scripts/rsl_rl/play.py --task=Template-Pose-Orientation-Two-Robots-Direct-v0 --num_envs=10 --checkpoint=pretrained_models/model_pose_orientation_two_robots.pt
```

**⚠️ Attention:** These pre-trained policies are functional examples but have not been tuned for optimal performance. They serve as starting points for further training and experimentation. Below is an example.

![Pretrained network Animation](pose_orientation_tworobot.gif)

## Training the policies

This project includes several reinforcement learning environments for robotic manipulation tasks in the woodworking setup:

- **Template-Pose-Orientation-Gripper-Robot-Direct-v0**: Single UR5e robot with gripper for pose and orientation control tasks (tested with RSL-RL).
- **Template-Pose-Orientation-Two-Robots-Direct-v0**: Dual robot setup (UR5e with gripper and UR5e with screwdriver) for coordinated manipulation (tested with RSL-RL).
- **Template-Grasping-Single-Robot-Direct-v0**: Single robot grasping tasks (currently only supports random_agent.py and zero_agent.py, training not yet implemented).
- **Template-Grasping-Dual-Robot-Direct-v0**: Dual robot grasping tasks (currently only supports random_agent.py and zero_agent.py, training not yet implemented).

Each environment supports multiple RL algorithms (RSL-RL, SKRL, Stable Baselines 3) with configurable network architectures stored in the `agents/` folder.

### Environment Listing

Use `list_envs.py` to see all available environments. This script has been modified to also create a `list_envs.txt` file with the environment details for reference.

```bash
python scripts/list_envs.py
```

### Training Commands

**For the first two environments (pose/orientation tasks), use RSL-RL:**

```bash
# Single gripper robot (2048 environments, headless)
python scripts/rsl_rl/train.py --task=Template-Pose-Orientation-Gripper-Robot-Direct-v0 --headless --num_envs=2048

# Dual robot setup (2048 environments, headless) 
python scripts/rsl_rl/train.py --task=Template-Pose-Orientation-Two-Robots-Direct-v0 --headless --num_envs=2048
```

**Note:** Training for the grasping environments (3rd and 4th) is not yet implemented.

### Playing Trained Policies

To evaluate a trained policy (first two environments only):

```bash
# Single gripper robot
python scripts/rsl_rl/play.py --task=Template-Pose-Orientation-Gripper-Robot-Direct-v0 --num_envs=30

# Dual robot setup
python scripts/rsl_rl/play.py --task=Template-Pose-Orientation-Two-Robots-Direct-v0 --num_envs=30
```

### Testing with Dummy Agents

All environments support dummy agents for testing:

```bash
# Zero-action agent (all environments)
python scripts/zero_agent.py --task=<TASK_NAME>

# Random-action agent (all environments)
python scripts/random_agent.py --task=<TASK_NAME>
```

Training results and checkpoints are saved in the `outputs/` folder.

## Improving Policy Performance

For better performance than the provided pre-trained models, you can:

1. **Modify network architectures**: Adjust the neural network parameters in the configuration files located in `source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/woodworking_simulation/agents/`:
   - `rsl_rl_ppo_cfg.py` - For dual robot tasks
   - `rsl_rl_ppo_small_cfg.py` - For single robot tasks

2. **Experiment with hyperparameters**: Change learning rates, batch sizes, entropy coefficients, and other RL parameters in the same configuration files.

3. **Adjust training settings**: Modify the number of environments, training iterations, and other training parameters in the scripts.

4. **Fine-tune reward functions**: The reward parameters can be adjusted in the environment configuration files (`pose_orientation_gripper_robot.py` and `pose_orientation_two_robots.py`).

