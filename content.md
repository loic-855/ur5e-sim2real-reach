# Comprehensive Exploration Report: Sim2Real Transfer for UR5e Robot Control

## Executive Summary
This master thesis project implements **sim-to-real transfer learning for pose and orientation reaching tasks** using UR5e robots in Isaac Lab. The work evolves through 6 task versions (v1-v6) with progressively sophisticated domain randomization and control strategies, culminating in LSTM-based policies for handling real-world latencies and dynamics.

---

## 1. Task Environment Definitions (v1-v6)

### [V0/V1 - Base Task](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real/)
**Purpose**: Foundation pose + orientation reaching without gripper.
- **Observation** (26D): ee_pos (3), ee_quat (4), joint_pos (6), joint_vel (6), goal_pos (3), goal_quat (4)
- **Actions** (6D): Joint position increments
- **Key Features**: Contact sensing, basic domain randomization (V1), tanh-scaled rewards
- **Rewards**: Position/orientation penalties with precision bonuses, action/velocity penalties, alignment bonus when both pos & ori close

### [V2 - Simplified Observation & Better Rewards](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v2/)
**Purpose**: Improved observation normalization, contact penalties, goal resampling logic.
- **Observation** (24D): pos_error/0.85m (3), ori_error/π (3), joint_pos_norm (6), joint_vel_norm (6), tcp_lin_vel/2m/s (3), tcp_ang_vel/π (3)
- **Actions** (6D): Same as V1
- **Key Features**: 
  - Per-component normalization for all observation elements
  - Exponential reward functions: `exp(-error/scale)` for smooth shaping
  - **Goal Logic** (novel): Success when pos < 2cm AND ori < 0.1rad for 60 consecutive frames; timeout resample after 5s
  - Contact threshold: 5N penalty, 10N termination threshold
  - Joint limit penalties for approaching bounds (±90% of limits)
- **Domain Randomization**: ActionBuffer (0-5 step delay, packet loss 3%, noise 0.025), ObservationBuffer (0-3 step delay, structured noise per component)

### [V3 - Velocity Feedforward Head](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v3/)
**Purpose**: Dual-head control for both position tracking and velocity feed-forward.
- **Observation** (24D): Identical to V2
- **Actions** (12D): [0:6] = position increments, [6:12] = velocity targets (rad/s)
- **Key Features**:
  - Separate penalty scales for position actions vs velocity actions
  - Velocity feedforward enables implicit dynamics compensation
  - Separate velocity scale (1.0 rad/s) for feedforward targets
  - **Domain Randomization V3**: ActionBufferV3 applies separate noise stds to position (0.025) vs velocity (0.01) dims
  - Stability reward scales adjusted: position_exp_scale=0.2, stability bonus=0.3
- **Notable**: Enables policy to learn momentum compensation and reduces need for explicit state prediction

### [V4 - Gripper Addition + Physics Randomization](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v4/)
**Purpose**: Realistic gripper inclusion without actuation; mass/CoM randomization for physical robustness.
- **Observation** (24D): Same as V3 (only 6 arm joints observed, gripper hidden)
- **Actions** (12D): Same as V3 (arm only, gripper held at default position)
- **Robot Model**: UR5e with gripper TCP (8 total joints, 6 controlled)
- **Key Features**:
  - Gripper present in simulation for collision/visual fidelity with real robot
  - Gripper joints frozen at default position (no policy control)
  - **Domain Randomization V4**: Adds MassComRandomizer
    - Mass scaling: [0.85, 1.15] (±15%)
    - CoM offset: [-0.01, 0.01]m per axis
    - Recomputes inertia tensors (uniform density assumption)
  - Stricter action scales: action_scale=2.0 (vs 3.0 in V3), velocity_action_penalty=-0.05
- **Significance**: Bridges sim and real gripper model; physical randomization adds robustness without action-space changes

### [V5 - Observation Stacking](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v5/)
**Purpose**: Implicit temporal context via frame stacking instead of recurrence.
- **Observation** (72D): 3× stacked 24D frames [oldest, middle, newest]
- **Actions** (12D): Same as V3
- **Key Features**:
  - `obs_stack_frames=3` (configurable)
  - Enables MLP to implicitly learn temporal patterns without LSTM
  - Observation history provides latency/dynamics information
  - Useful when recurrent layers aren't available or desired
- **Trade-off**: Larger observation space vs simpler policy architecture

### [V6 - LSTM Policy](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v6/)
**Purpose**: Recurrent policy for temporal context handling without manual stacking.
- **Observation** (24D): Single frame (same as V3), recurrence handled by LSTM
- **Actions** (12D): Same as V3
- **Key Features**:
  - Policy uses `RslRlPpoActorCriticRecurrentCfg` with LSTM
  - Hidden state managed internally by JIT module
  - No domain randomization enabled (recurrence handles dynamics implicitly)
  - **Sim2real deployment**: PolicyInferenceLSTM with `.reset()` method for episode boundaries
- **Advantage**: Compact state-space, learns temporal dependencies naturally, no manual stacking needed

---

## 2. Robot Configurations

### UR5e Base Configuration ([robot_configs.py](source/Woodworking_Simulation/Woodworking_Simulation/common/robot_configs.py#L1))

**Physical Constants**:
- **Reach**: 0.85m (MAX_REACH)
- **Max Joint Velocity**: 3.14 rad/s (~180°/s)
- **Table Dimensions**: 1.2m (W) × 0.8m (D) × 0.842m (H)
- **Environment Origin Offset**: (-0.6, 0.4, 0.842) m (table center at plateau height)

**Joint Limits** (Real robot with cable constraints):
```
shoulder_pan:   ±2π rad
shoulder_lift:  ±2π rad
elbow:          ±π rad (cable constraint - narrower than others!)
wrist_1:        ±2π rad
wrist_2:        ±2π rad
wrist_3:        ±2π rad
gripper fingers: [0, 0.02] m (when present)
```

**Actuator Configurations** (socket groups):
- **Shoulder joints** (pan, lift, elbow): High stiffness for heavy lifting
  - Current: K=200, D=35
  - Mid: K=500, D=48
  - High: K=800, D=60
- **Wrist joints** (1, 2, 3): Lower stiffness for dexterity
  - Current: K=80, D=15
  - Mid: K=215, D=25
  - High: K=350, D=35
- **Gripper** (when present): K=1200, D=5 (fast closing response)

**TCP Offset** from wrist_3_link: (0, 0, 0.14-0.15m)

**Robot Types** (via RobotType enum):
- `NO_GRIPPER`: Bare UR5e (6 joints) - used in V0-V3, V5-V6
- `GRIPPER_TCP`: UR5e + OnRobot 2FG7 gripper (8 joints) - used in V4
- `GRIPPER_TCP_WIDE`, `SCREWDRIVER_TCP`: Other tool options

---

## 3. Domain Randomization Evolution

### [V1 - Action & Observation Buffers](source/Woodworking_Simulation/Woodworking_Simulation/common/domain_randomization.py#L1)

**ActionBuffer**:
- Ring-buffer FIFO queue (max_delay+1 entries)
- Per-environment random delay: [0, 5] decimated steps
- Additive Gaussian noise: std=0.025
- Packet loss: 3% probability (holds last action on drop)

**ObservationBuffer**:
- Per-component Gaussian noise (structured by observation layout)
- Per-environment random delay: [0, 3] decimated steps
- Noise standard deviations:
  - pos: 0.005m | quat: 0.025 | joint_pos: 0.012 rad | joint_vel: 0.025 rad/s

**ActuatorRandomizer**:
- Per-joint stiffness scale: [0.8, 1.2]
- Per-joint damping scale: [0.8, 1.2]
- Per-joint friction variation: ±35% around nominal values (8.24, 10.51, 7.9, 1.43, 1.05, 1.63)

### [V2 - Same Buffers + Observation Layout](source/Woodworking_Simulation/Woodworking_Simulation/common/domain_randomization_v2.py#L1)

**Key Addition**: Structured observation noise per component type:
```
parts: [pos_error(3), ori_error(3), joint_pos(6), joint_vel(6), tcp_lin_vel(3), tcp_ang_vel(3)]
noise_stds: [0.005, 0.025, 0.012, 0.025, 0.02, 0.02]  (additions in V2)
```

### [V3 - Split Action Noise](source/Woodworking_Simulation/Woodworking_Simulation/common/domain_randomization_v3.py#L1)

**ActionBufferV3**:
- Separate noise for position actions [0:6]: std=0.025
- Separate noise for velocity actions [6:12]: std=0.01
- Enables tuned, asymmetric randomization for 12D action space
- Re-exports ObservationBuffer, ActuatorRandomizer unchanged

### [V4 - Physical Randomization](source/Woodworking_Simulation/Woodworking_Simulation/common/domain_randomization_v4.py#L1)

**MassComRandomizer** (new - PhysX-level):
- **Mass scaling**: Per-link multiplicative factors [0.85, 1.15]
  - Equation: `mass_new = mass_default × scale`
  - Recomputes inertia (uniform density: `I ∝ m`)
- **CoM offset**: Per-body additive perturbations [-0.01, 0.01]m
- **Implementation**: Uses PhysX tensor API on CPU (not GPU)
- **Purpose**: Adds real-world uncertainty in mass distribution, improves transfer

**Config Dataclass** (`DomainRandomizationV4Cfg`):
```python
mass_scale_range: (0.85, 1.15)
recompute_inertia: True
com_offset_range: (-0.01, 0.01)  # meters
```

**Activation**: Called in `_reset_idx()` on environment resets; applied per sampled environment batch.

---

## 4. Sim2Real Deployment Scripts

### Directory Structure: `scripts/sim2real/`

#### [V1 - Basic](scripts/sim2real/v1/)
- `observation_builder.py`: Load observations from `/joint_states` topic
- `policy_inference.py`: Basic TorchScript loader
- `sim2real_node.py`: ROS2 node orchestrating observation→policy→action

#### [V2 - 24D Normalized](scripts/sim2real/v2/)
- `observation_builder.py`: Convert joint states to 24D normalized vector (v2 layout)
- `policy_inference.py`: Inference wrapper for v2 models
- **V2 observation normalization**:
  ```
  to_target / 0.85m, ori_error / π, joint_pos_norm, joint_vel_norm,
  tcp_linear_vel / 2m/s, tcp_angular_vel / π
  ```

#### [V3 - 12D Actions](scripts/sim2real/v3/)
- `policy_inference_v3.py`: Handles 12D action output (position + velocity)
- Action parsing: `[0:6]` → position increments, `[6:12]` → velocity feedforward

#### [V6 - LSTM Support](scripts/sim2real/v6/)
- `policy_inference_v6.py`: LSTM-aware wrapper
- Key methods:
  - `reset()`: Clear hidden/cell state (must call on episode start, goal achieved, timeout)
  - `get_action()`: Forward pass (updates internal state)
- **Critical**: Reset timing directly affects policy behavior

### URScript Control ([sim2real/URscript/](scripts/sim2real/URscript/))

**Files**:
- `impedance_control_v3.script`: Active impedance/force control
- `torque_control.script`: Direct torque commands
- `rtde_input_v*.xml`: Real-time data exchange (RTDE) protocol definitions

**Purpose**: Low-level joint control on real UR5e; receives desired joint velocities from ROS2.

### ROS2 Topics

**Subscriptions**:
- `/joint_states` (sensor_msgs/JointState): Real joint positions/velocities
- `/goal_pose` (geometry_msgs/PoseStamped): Target end-effector pose

**Published**:
- Joint velocity commands to UR controller (topic depends on controller type)

### Example Deployment Flow
1. **ROS2 Controller** launches (ur_robot_driver)
2. **sim2real_node** subscribes to `/joint_states`, `/goal_pose`
3. **Loop iteration**:
   - Receive joint states → normalize → policy forward pass
   - Get 12D action [pos_inc, vel_target]
   - Apply position increments + velocity feedforward (v3+)
   - Send to URscript controller
4. For **V6 (LSTM)**: Call `policy.reset()` on goal success/timeout

---

## 5. Reward Structure Across Versions

### [V2 Reference Implementation](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_sim2real_v2/pose_orientation_sim2real_v2.py#L550)

**Reward Equation**:
```python
reward = (ee_position_penalty * position_error              # -0.30 * error
        + ee_position_reward * position_exp_error            # +1.2 * exp(-err/0.4)
        + ee_orientation_penalty * orientation_error         # -0.20 * error
        + ee_orientation_reward * orientation_exp_error      # +0.60 * exp(-err/0.7)
        + action_penalty_scale * action_cost                 # -0.001 * sum(a²)
        + velocity_penalty_scale * velocity_cost             # -0.001 * sum(v²)
        + contact_penalty                                    # -0.01 * max(0, F-5N)
        + joint_limit_penalty_scale * joint_violation²       # -0.01 * violation²
        + goal_success_bonus * success_flag)                 # +15.0 if goal reached
```

**Component Breakdown**:

1. **Position Error** (L2 norm in meters):
   - Penalty: Linear term discourages large errors
   - Reward: Exponential term gives dense signal
   - Scale: 0.4m = ~action_scale * decimation * dt

2. **Orientation Error** (quaternion magnitude in radians):
   - Same exp/penalty structure
   - Scale: 0.7 rad (more forgiving than position)

3. **Action Penalties**:
   - Sum of squared action magnitudes
   - Encourages energy-efficient movements

4. **Velocity Penalties**:
   - Sum of squared joint velocities
   - Rewards stillness when goal is reached

5. **Contact Penalties** (force-based):
   - Threshold 5N (soft penalty region)
   - Beyond threshold: linear penalty at -0.01 per Newton
   - Termination threshold 10N (no early episode end, just penalty)

6. **Joint Limit Penalties**:
   - Triggers when normalization approaches ±90% of limits
   - Quadratic penalty to avoid hard limits

7. **Goal Success Bonus**:
   - +15.0 when pos < 2cm AND ori < 0.1rad for 60 consecutive frames
   - Activates goal resampling logic

### **V3 Modifications** (v3 has split action penalties)
```python
action_penalty_scale = -0.001          # position increments
velocity_action_penalty_scale = -0.001  # velocity targets
velocity_penalty_scale = -0.001         # actual joint velocities
```

### **V4 Modifications** (stricter velocity penalty)
```python
action_penalty_scale = -0.01
velocity_action_penalty_scale = -0.05  # 5× stricter for velocity actions
velocity_penalty_scale = -0.01
position_exp_scale = 0.05              # 4× stricter (exp decays faster)
```

---

## 6. Training Configurations

### Euler Cluster Setup ([euler/](euler/))

**Infrastructure**:
- **Container**: Singularity/Apptainer (`isaac_euler_salziegl.sif`)
- **Job Scheduler**: SLURM (ETH Zurich Euler cluster)
- **Logging**: Weights & Biases (WandB) for metrics
- **Git**: Source tracked in repo for reproducibility

**Main Scripts**:
- `train_euler.sh`: Single job submission template
- `generate_sweep.py`: Cartesian product sweep generator
- `sweep_*.yaml`: Sweep configuration files

### Sweep Configuration Example ([sweep_config.yaml](euler/sweep_config.yaml#L1))

**Dimensions** (3 × 2 × 2 × 2 = 24 runs):

1. **Actuators** (stiffness/damping variations):
   - `current`: K={shoulder:200, wrist:80}, D={shoulder:35, wrist:15}
   - `mid`: K={shoulder:500, wrist:215}, D={shoulder:48, wrist:25}
   - `high`: K={shoulder:800, wrist:350}, D={shoulder:60, wrist:35}

2. **Domain Randomization**:
   - `current`: Defaults (minimal randomization)
   - `high`: 2-2.5× noise/delay (action_noise_std=0.025, delay=[0,5], obs_noise_std_pos=0.005, etc.)

3. **Network Architecture**:
   - `ext4`: [512, 256, 128, 64] hidden layers
   - `ext3`: [512, 256, 128]

4. **Action Rate** (decimation/action scaling variations): Custom per sweep

**Typical SLURM Settings**:
```yaml
slurm:
  time: "07:00:00"        # 7 hours wall-time
  gpus: "rtx_4090:1"
  cpus_per_task: 3
  mem_per_cpu: 4000
sequential_per_job: 4    # Chain 4 runs per job (~5h each)
```

### Training Entry Point ([scripts/rsl_rl/train.py](scripts/rsl_rl/train.py#L10))

**Framework**: RSL-RL (Robotics Sim-to-Real Library)
- **Policy Type**: ActorCritic (PPO) or ActorCriticRecurrent (LSTM)
- **Training Duration**: 2500 iterations typical (~1h15m per run)
- **Environment**: 4092 parallel envs per GPU

**Key CLI Arguments**:
```bash
python scripts/rsl_rl/train.py \
  --task=WWSim-Pose-Orientation-Sim2Real-Direct-v3 \
  --agent=rsl_rl_cfg_entry_point \
  --num_envs=4092 \
  --max_iterations=2500 \
  env.domain_randomization.enabled=True \
  agent.max_iterations=2500
```

---

## 7. Pretrained Models Directory

### Location & Structure
[pretrained_models/pose_orientation_two_robots/2026-02-17_16-54-39/](pretrained_models/pose_orientation_two_robots/2026-02-17_16-54-39/):

```
2026-02-17_16-54-39/
├── exported/          # TorchScript models ready for deployment
│   ├── policy.pt
│   └── critic.pt
├── git/               # Git snapshot (code reproducibility)
│   └── [source code at training time]
└── params/            # Training checkpoints
    ├── model_0.pt
    ├── model_1000.pt
    └── ...
```

**Export Format**: TorchScript JIT (platform-independent, no Python dependency)

**Usage in Deployment**:
```python
model = torch.jit.load("policy.pt", map_location="cuda")
with torch.no_grad():
    actions = model(observations)  # Direct inference
```

---

## 8. Additional Task Variants

### [GraspingSingleRobot](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/grasping_single_robot/)
**Purpose**: Gripper grasping + block manipulation task.
- **Observation** (27D): 7 joint pos, 7 joint vel, to_target (3), block_pos (3), last_actions
- **Actions** (7D): 6 arm joints + 1 gripper (prismatic)
- **Object**: Wooden block (0.025×0.1×0.05m cube)
- **Reward**: Grasping success, block position tracking

### [PoseOrientationGripperRobot](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_gripper_robot/)
**Purpose**: Single gripper robot pose reaching (simpler dual-arm concept).
- **Observation** (36D): ee_pos, ee_quat, ee_linvel, ee_angvel, goal_pos, goal_quat, joint_pos, joint_vel
- **Actions** (8D): 8 joints (6+2 gripper)
- **Control**: Joint space control (no TCP command)

### [PoseOrientationTwoRobots](source/Woodworking_Simulation/Woodworking_Simulation/tasks/direct/pose_orientation_two_robots/)
**Purpose**: Dual-arm coordination (gripper + screwdriver).
- **Observation** (92D): ee1_pos/quat/vel, ee2_pos/quat/vel, relational_ee, goals, contacts, last_actions
- **Actions** (15D): 8 (gripper) + 7 (screwdriver) arm joints
- **Centralized Policy**: Single policy controls both arms
- **Use Case**: Complex multi-robot tasks (assembly, screwing)

---

## 9. Key Scientific Insights & Design Choices

### Evolution Strategy: V1→V6 Progression

| Aspect | V1 | V2 | V3 | V4 | V5 | V6 |
|--------|----|----|----|----|----|----|
| Obs Space | 26D | 24D | 24D | 24D | 72D | 24D |
| Action Space | 6D | 6D | 12D | 12D | 12D | 12D |
| Robot | No gripper | " | " | Gripper | No gripper | No gripper |
| DR Scope | Act/Obs | " | " | + Physics | Same as V3 | Disabled |
| Policy Type | MLP | MLP | MLP | MLP | MLP | LSTM |

### Domain Randomization Progression
- **V1-V2**: Communication layer (action/obs delays, noise) - sim latencies
- **V3**: Action-space asymmetry (velocity more noise-sensitive)
- **V4**: Physics layer (mass, inertia, CoM) - model uncertainties
- **Curriculum**: DR often enabled only in later training phases (>15k steps)

### Gripper Integration (V4)
- **Passive gripper** (fixed position) provides:
  - Collision realism (affects robot dynamics)
  - Visual sim-to-real fidelity (matches real setup)
  - No policy complexity (only 6 controllable joints)
- Trade-off: Increases simulation cost without control benefit

### Temporal Handling Strategies
- **V5 (Stacking)**: MLP-friendly, larger state, no hidden state
- **V6 (LSTM)**: Compact observation, learned recurrence, requires `reset()` logic
- **Choice**: Problem-dependent; V6 preferred if computational budget allows LSTM

### Reward Shaping Philosophy
- **Dense exponential terms**: Enable continuous improvement signal
- **Penalty structure**: Multi-objective (position + orientation + energy + contact)
- **Goal logic**: Hysteresis (60-frame confirmation) prevents goal "chatter"
- **Adaptive thresholds**: Different scales for position (0.4m) vs orientation (0.7rad)

---

## 10. README Files & Documentation

### Main [README.md](README.md)
- Isaac Lab installation guide
- Project structure explanation
- Asset import instructions (table, gripper, screwdriver)
- IDE setup with Python path configuration
- Quick start: task listing, training commands

### [Sim2Real Deployment README](scripts/sim2real/README.md) (French + English)
- Observation architecture (19D → 24D evolution)
- ROS2 topic mapping
- Policy loading and inference
- Example goal publisher usage
- Normalization constants reference

### [Euler Training Guide](euler/GUIDE.md)
- Cluster setup (VPN, file transfer)
- WandB configuration
- Single job vs hyperparameter sweep submission
- Monitoring (squeue, WandB dashboard)
- Sweep file structure and customization

---

## Summary: Evolution to Production-Ready Sim2Real

| Phase | Focus | Outcome |
|-------|-------|---------|
| V0-V1 | Foundation | Basic pose+ori reaching, initial DR |
| V2 | Observation design | Normalized 24D obs, goal resampling |
| V3 | Control sophistication | Velocity feedforward, dual-head policy |
| V4 | Physical realism | Gripper fidelity, mass randomization |
| V5 | Temporal context | Frame stacking for implicit latency handling |
| V6 | Recurrent learning | LSTM policy, compact state, no manual stacking |

**Final Sim2Real Pipeline**:
1. Train policy on Isaac Lab (V3-V6 tasks with DR)
2. Export as TorchScript
3. Deploy via ROS2 node with observation builder
4. Real robot runs policy loop at controller frequency (~125 Hz)
5. LSTM reset on goal events (success/timeout)

This systematic evolution demonstrates *principled scientific progression*, each version addressing specific sim-to-real gap and control challenge.

---------------------------------------

Voici mon analyse complète de ton repo et les pistes de questions scientifiques que j'en tire.

---

## Contexte de ton travail

Ton projet couvre une chaîne complète **simulation → transfert → robot réel** pour un UR5e en contexte de travail du bois, avec :

- **6 versions d'environnement** (V1→V6) avec complexité croissante
- Un **contrôleur à impédance custom** en URScript (500 Hz) avec filtrage dual (position + vitesse)
- Du **domain randomization** multi-couche (communication, actionneurs, physique)
- Des architectures de politique **MLP vs LSTM**
- Un canal de **velocity feedforward** explicite (V3+)
- Du **benchmarking sim2real** structuré

---

## Pistes de questions scientifiques

### 1. **Le rôle du velocity feedforward dans le sim2real transfer**

> *Dans quelle mesure l'ajout d'un canal de commande en vitesse explicite (actions 12D = 6 position + 6 vitesse) améliore-t-il la qualité du transfert sim-to-real par rapport à une commande en position seule (6D) ?*

**Angle manuscrit :** Tu compares V2 (position only) vs V3 (position + velocity). Tu as le contrôleur à impédance en deux versions (single-filter vs dual-filter). C'est une contribution concrète et mesurable : le découplage position/vitesse dans l'espace d'actions permet au policy de commander le feedforward sans devoir l'encoder implicitement dans des incréments de position. Tu peux montrer que ça réduit l'erreur en régime transitoire et améliore la stabilité.

---

### 2. **L'impact du design du contrôleur bas-niveau sur l'efficacité du RL**

> *Comment le choix des paramètres du contrôleur à impédance (raideur, amortissement, constantes de temps des filtres) influence-t-il la performance et la robustesse d'une politique apprise par RL après transfert sur robot réel ?*

**Angle manuscrit :** Ton contrôleur URScript à 500 Hz avec filtrage asymétrique ($\tau_{pos} = 0.04\text{s}$, $\tau_{vel} = 0.012\text{s}$) est un choix de design non-trivial. La question est : est-ce que la politique RL et le low-level controller sont **co-dépendants** ? Tu peux étudier l'interaction entre les gains PD du contrôleur et les performances du policy appris en simulation. C'est un angle rarement traité dans la littérature sim2real (la plupart supposent un contrôleur parfait).

---

### 3. **Domain randomization progressif : quelle couche de randomisation compte le plus ?**

> *Parmi les trois couches de domain randomization — communication (délai/bruit d'actions et observations), actionneurs (raideur/amortissement), et physique (masse/inertie/CoM) — laquelle contribue le plus à la robustesse du transfert sim2real pour un manipulateur industriel ?*

**Angle manuscrit :** Tu as une progression claire V1→V4 qui ajoute des couches de DR. En désactivant chaque couche individuellement (ablation), tu peux quantifier leur contribution respective. C'est particulièrement intéressant car ton sweep Euler a déjà des runs avec différents niveaux de DR. La littérature manque d'études d'ablation systématiques sur les types de DR pour des manipulateurs rigides industriels (la plupart des travaux portent sur des mains dextères ou de la locomotion).

---

### 4. **MLP vs LSTM pour la gestion des latences dans le sim2real**

> *Une politique récurrente (LSTM) peut-elle apprendre implicitement à compenser les délais de communication et les dynamiques non-modélisées, rendant le domain randomization superflu ?*

**Angle manuscrit :** V6 (LSTM) désactive le domain randomization, pariant que la mémoire récurrente suffit à gérer les incertitudes temporelles. C'est une hypothèse forte et testable. Tu compares : MLP + DR fort (V3/V4) vs LSTM sans DR (V6) vs LSTM + DR. Si le LSTM seul rivalise, c'est un résultat significatif pour la communauté sim2real.

---

### 5. **Conception d'un pipeline sim2real end-to-end pour la robotique industrielle**

> *Comment concevoir un pipeline sim-to-real complet — de l'environnement de simulation à l'impédance controller sur robot réel — qui soit modulaire, reproductible, et adapté aux contraintes industrielles (UR5e, boucle de contrôle fixe, sécurité) ?*

**Angle manuscrit :** C'est un angle plus "systems paper" / ingénierie. Ton manuscrit documente la conception de bout en bout : choix d'observation normalisée, contrôleur à impédance avec saturation multi-couche, protocole RTDE, thread de lecture découplé, gestion des frames (base robot → centre table). C'est utile comme contribution car la plupart des papiers sim2real montrent des résultats mais détaillent peu le pipeline de déploiement.

---

### 6. **L'effet du gripper passif sur la fidélité du transfert**

> *L'inclusion d'un gripper non-actionné dans la simulation (V4) améliore-t-elle le transfert par rapport à un modèle sans gripper, malgré l'absence de contrôle explicite sur celui-ci ?*

**Angle manuscrit :** V4 ajoute le gripper pour le réalisme des collisions et de l'inertie, sans le contrôler. C'est une question sur la **fidélité du modèle** : est-ce que modéliser les parties passives du robot (masse additionnelle, collisions) vaut le coût computationnel ? Combiné avec la randomisation de masse (V4), ça teste si la fidélité géométrique/inertielle est plus importante que la randomisation.

---

## Ma recommandation

La question la plus **originale et publiable** serait une combinaison des axes **1 + 2** :

> **"Co-design of learned policies and impedance controllers for sim-to-real transfer: the role of explicit velocity feedforward in bridging the sim-to-real gap for industrial manipulators"**

Pourquoi :
- C'est **l'intersection unique** de ton travail (peu de thèses combinent RL policy design + impedance controller design pour le sim2real)
- Tu as **les données comparatives** : V2 (sans velocity FF) vs V3 (avec), contrôleur single-filter vs dual-filter
- Ça touche à un **vrai problème ouvert** : la plupart des travaux sim2real traitent le low-level controller comme une boîte noire
- C'est **mesurable** : erreur de tracking, temps de convergence, robustesse aux perturbations

Le manuscrit s'organiserait alors en :
1. **Intro** : sim2real gap pour manipulateurs industriels
2. **Related work** : RL pour manipulation, sim2real (DR, system identification), impedance control
3. **Méthode** : ton pipeline (simulation, DR, policy architecture, impedance controller)
4. **Expériences** : ablation velocity FF, ablation DR layers, MLP vs LSTM, sim vs real
5. **Résultats** : comparatifs quantitatifs sur le robot réel
6. **Discussion** : co-dépendance policy/controller, limites, généralisation