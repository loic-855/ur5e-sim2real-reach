# Impedance Controller Tuning – UR5e

## Overview

This document describes the tools and procedures for systematically tuning the impedance controller running on the UR5e robot. The controller runs on the robot at 500 Hz as a URScript program and receives commands from a Python host at 60 Hz via the RTDE interface.

The torque law implemented on the robot is:

```
τ = Kp · (q_filt − q) + Kd · (qdot_filt − q̇)
```

where `q_filt` and `qdot_filt` are first-order filtered versions of the RTDE-received `q_des` and `qdot_des`, providing smooth 60 → 500 Hz interpolation.

The tunable parameters are:

| Parameter   | Role                                           | Typical range (shoulder / wrist) |
|-------------|------------------------------------------------|----------------------------------|
| `Kp`        | Proportional gain (stiffness)                  | 100–1000 / 20–80                 |
| `Kd`        | Derivative gain (damping)                      | Computed as `2·ζ·√Kp`           |
| `ζ` (zeta)  | Damping ratio (1.0 = critical)                 | 0.4 – 1.2                       |
| `alpha_pos` | Position low-pass filter coefficient           | 0.05 – 0.20                     |
| `alpha_vel` | Velocity low-pass filter coefficient           | 0.10 – 0.30                     |

---

## File inventory

### URScript controllers

| File | Description |
|------|-------------|
| `URscript/impedance_control_v3.script` | Production controller with **hardcoded** Kp, Kd, alpha. Used for policy deployment. |
| `URscript/impedance_control_tuning.script` | Tuning controller that **reads Kp from RTDE registers 37–42** and **ζ from register 43** every tick, computing `Kd = 2·ζ·√Kp`. Also handles a `go_home` signal (register 65) to perform a `movej` back to home between trials without restarting the script. |

### RTDE configs

| File | Description |
|------|-------------|
| `URscript/rtde_input_v3.xml` | Recipes for production: `q_des` (regs 24–29), `qdot_des` (regs 30–35), `stop` (bit 64), `control_rate_info` (int 36). |
| `URscript/rtde_input_tuning.xml` | Extends v3 with: `kp_gains` (regs 37–42), `damping_ratio` (reg 43), `go_home` (bit 65). |

### Python tuning scripts

| File | Purpose |
|------|---------|
| `impedance_tuner.py` | Manual sinusoidal excitation for visual tracking analysis. |
| `auto_tuner.py` | Automatic Kp sweep to find minimum stiffness meeting an RMS target. |
| `step_tuner.py` | Step-response ζ sweep for damping ratio characterisation. |

All scripts are in `scripts/sim2real/`. Logs (plots + data) are saved to `logs/impedance_tuner/`.

---

## Step 1 – `impedance_tuner.py`: Manual sinusoidal tracking

### What it does

Sends a sinusoidal joint-angle command at 60 Hz to the robot and records the desired vs. actual position. Produces overlay plots of position tracking, position error (with RMS), and velocity tracking.

### Physical justification

A sinusoid is the canonical test signal for characterising a control loop's **frequency response**. At a given frequency, the plots reveal:

- **Phase lag**: how much the actual position trails behind the command, primarily caused by the low-pass filter (`alpha_pos`). A filter with coefficient α at sample rate fs has a time constant τ = (1−α)/(α·fs). Lower α means more smoothing but more lag.
- **Amplitude attenuation**: whether Kp is high enough to track the full amplitude.
- **Velocity feedforward benefit**: enabling `--with-velocity-ff` sends the analytic derivative as `qdot_des`, which the Kd term uses to anticipate the trajectory and reduce phase lag.
- **Cross-coupling**: exciting one joint reveals whether neighbouring joints move due to inertial coupling (indicates Kp is too low on the neighbours).

This script uses `impedance_control_v3.script` (hardcoded gains), so you must manually edit the URScript between runs to test different parameters.

### Usage

```bash
cd scripts/sim2real

# Basic: excite shoulder_lift at 0.5 Hz, ±0.15 rad
python impedance_tuner.py --joints 1 --freq 0.5 --amplitude 0.15

# With velocity feedforward (recommended)
python impedance_tuner.py --joints 1 --freq 0.5 --amplitude 0.15 --with-velocity-ff

# Chirp to sweep frequency bandwidth
python impedance_tuner.py --joints 1 --chirp --freq-start 0.1 --freq-end 2.0 --duration 20

# Multiple joints simultaneously
python impedance_tuner.py --joints 0 1 2 --freq 0.3 --amplitude 0.10
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--joints` | `1` | Joint indices to excite (0–5) |
| `--freq` | `0.5` | Sine frequency in Hz |
| `--amplitude` | `0.15` | Sine amplitude in rad |
| `--duration` | `10` | Excitation duration in seconds |
| `--chirp` | off | Linear frequency sweep mode |
| `--with-velocity-ff` | off | Send analytic velocity derivative as `qdot_des` |
| `--rate` | `60` | Command rate in Hz (should match policy rate) |

---

## Step 2 – `auto_tuner.py`: Automatic Kp sweep

### What it does

Uses `impedance_control_tuning.script` which reads Kp from RTDE registers at runtime. For each joint, sweeps a grid of Kp values (sorted ascending) and measures the RMS tracking error on a sinusoidal excitation. Selects the **minimum Kp** that achieves < `--rms-target` degrees of RMS error, maximising compliance.

### Physical justification

The goal of an impedance controller is to provide **compliant** (soft) behaviour while maintaining adequate tracking. Unlike a stiff position controller, lower Kp means the robot yields more to external forces – essential for safe human-robot interaction and contact tasks.

However, too-low Kp causes:
- **Position tracking error**: the controller cannot generate enough torque to follow the desired trajectory.
- **Cross-coupling drift**: heavy joints (shoulder, elbow) pull on each other via inertial coupling.
- **Gravity sensitivity**: the UR5e's built-in gravity compensation is imperfect; Kp must overcome residual errors.

The auto-tuner finds the sweet spot: the lowest stiffness that still achieves acceptable tracking. During the sweep, `Kd = 2·√Kp` (ζ = 1, critical damping) is used as a neutral baseline so that only the stiffness effect is measured.

The joints are tuned **sequentially** (not all-at-once) to isolate each joint's behaviour. A final validation run excites all joints simultaneously to check for coupling degradation.

### Usage

```bash
cd scripts/sim2real

# Dry run: see the plan without connecting
python auto_tuner.py --dry-run

# Full sweep on all 6 joints (~36 trials, ~7 min)
python auto_tuner.py

# Shoulder joints only with stricter target
python auto_tuner.py --joints 0 1 2 --rms-target 1.5

# Test all Kp values (no early stopping)
python auto_tuner.py --no-early-stop
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--joints` | `0 1 2 3 4 5` | Joints to tune |
| `--rms-target` | `2.0` | Target RMS error in degrees |
| `--freq` | `0.5` | Test sinusoid frequency |
| `--amplitude` | `0.15` | Test sinusoid amplitude |
| `--duration` | `8.0` | Excitation duration per trial |
| `--early-stop` | on | Stop testing higher Kp once target is met |
| `--dry-run` | off | Print plan only, no robot connection |

### Output

- Console summary table: joint × (Kp, Kd, RMS, pass/fail)
- JSON file with all trial results
- Bar chart: RMS vs Kp per joint with target threshold line
- Copy-paste ready `kp = [...]` and `kd = [...]` for `impedance_control_v3.script`

---

## Step 3 – `step_tuner.py`: Damping ratio (ζ) characterisation

### What it does

Applies position step commands (e.g. +5.7° = 0.1 rad) to individual joints and records the transient response. Sweeps the damping ratio ζ while keeping Kp fixed (from Step 2 results). Computes standard step-response metrics: rise time, overshoot, settling time, steady-state error.

Before each trial, a `go_home` signal triggers a `movej` on the robot to ensure every trial starts from exactly the same home position, making all curves directly comparable.

### Physical justification

The damping ratio ζ controls **how the controller dissipates energy** during transients:

- **ζ > 1 (over-damped)**: no overshoot but sluggish response. The system approaches the target monotonically.
- **ζ = 1 (critically damped)**: theoretically the fastest convergence without overshoot. In practice, due to filtering delay and actuator friction, the real system may still overshoot.
- **ζ < 1 (under-damped)**: faster rise time with oscillatory overshoot. The oscillations act as a natural *dithering* signal that helps overcome static friction (stiction) in the actuators.

Real robot actuators have **significant friction** that a simple spring-damper model does not capture. This friction absorbs energy and means that:
- At **intermediate ζ** (0.6–0.8), the system has neither enough oscillation to overcome friction nor enough sustained torque to converge accurately, leading to larger steady-state errors.
- At **low ζ** (0.4–0.5), oscillations shake the joint past friction → small SS error, but large overshoot.
- At **high ζ** (0.9–1.0), the slow monotonic approach allows the error integral to build enough torque to converge, but slowly.

The step test reveals this friction-dependent U-curve in steady-state error and helps select the best ζ for each joint.

A **sinusoidal test** (Step 1–2) is better for measuring **tracking bandwidth** (how well the controller follows a continuous trajectory), while a **step test** is better for measuring **transient behaviour** (how fast and cleanly it reaches a new setpoint). Both are needed for complete characterisation.

### Usage

```bash
cd scripts/sim2real

# Dry run
python step_tuner.py --dry-run

# Sweep ζ on shoulder_lift (Kp=900 from auto_tuner)
python step_tuner.py --joints 1

# Test specific ζ values
python step_tuner.py --joints 1 --zeta 0.5 0.7 0.8 0.9 1.0

# Custom step size
python step_tuner.py --joints 1 --step-size 0.15

# Override Kp for a single joint
python step_tuner.py --joints 1 --kp 800

# All joints (~42 trials, ~7 min)
python step_tuner.py
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--joints` | `0 1 2 3 4 5` | Joints to test |
| `--zeta` | `0.4 0.5 0.6 0.7 0.8 0.9 1.0` | Damping ratios to sweep |
| `--step-size` | `0.10` | Step magnitude in rad (~5.7°) |
| `--kp` | per-joint default | Override Kp (single joint only) |
| `--record` | `3.0` | Recording time per step (seconds) |
| `--settle` | `2.0` | Settle time before step (seconds) |
| `--dry-run` | off | Print plan only |

### Output

- Console table: joint × ζ × (rise time, overshoot%, settling time, SS error)
- JSON file with all metrics
- Plot per joint: step-up/down response overlaid for all ζ, plus metrics bar chart

---

## Recommended tuning workflow

```
1. impedance_tuner.py   →   Get initial feel for tracking quality.
                             Identify phase lag (alpha_pos), cross-coupling (Kp too low),
                             velocity feedforward benefit.

2. auto_tuner.py        →   Systematic Kp sweep per joint.
                             Find minimum Kp for RMS < 2° at 0.5 Hz sinusoid.
                             Result: optimal Kp vector.

3. step_tuner.py        →   Damping ratio sweep with Kp from step 2.
                             Find ζ that best balances speed, overshoot, and
                             friction-induced steady-state error.
                             Result: optimal ζ (→ Kd = 2·ζ·√Kp).

4. impedance_tuner.py   →   Final validation with the tuned Kp/Kd.
   (again)                   Update impedance_control_v3.script with final values.
                             Verify tracking at the policy's operating frequency.
```

---

## RTDE register map (tuning mode)

| Register | Type | Direction | Usage |
|----------|------|-----------|-------|
| `input_double_register_24–29` | DOUBLE | PC → Robot | `q_des` (6 joint positions) |
| `input_double_register_30–35` | DOUBLE | PC → Robot | `qdot_des` (6 joint velocities) |
| `input_int_register_36` | INT32 | PC → Robot | Policy control rate (Hz) |
| `input_double_register_37–42` | DOUBLE | PC → Robot | `Kp` gains (6 joints) |
| `input_double_register_43` | DOUBLE | PC → Robot | Damping ratio `ζ` |
| `input_bit_register_64` | BOOL | PC → Robot | Stop signal (robot returns to home and exits) |
| `input_bit_register_65` | BOOL | PC → Robot | Go-home signal (`movej` to home, resume after) |
| `actual_q` | VECTOR6D | Robot → PC | Current joint positions |
| `actual_qd` | VECTOR6D | Robot → PC | Current joint velocities |
| `actual_TCP_pose` | VECTOR6D | Robot → PC | TCP pose (position + rotation vector) |
| `actual_TCP_speed` | VECTOR6D | Robot → PC | TCP velocity (linear + angular) |
