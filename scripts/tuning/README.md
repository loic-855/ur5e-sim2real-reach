# Impedance Controller Tuning

This directory contains all tools for characterising and tuning the UR5e impedance controller gains.
The URScript files that run on the robot live in `scripts/sim2real/URscript/`.

## File Overview

| File | Purpose |
|------|---------|
| `impedance_tuner.py` | Real-robot sinusoidal/chirp excitation using RTDE; exports CSV for `plot_sim_gain_tuner_csv.py` |
| `auto_tuner.py` | Automated Kp stiffness sweep — finds the minimum Kp per joint achieving RMS tracking < target |
| `step_tuner.py` | Step-response damping sweep — finds optimal ζ per joint (rise time, overshoot, settling time) |
| `sim_gain_tuner_logger.py` | Isaac Sim Script-Editor logger; records commanded + observed joints to CSV during simulation |
| `plot_sim_gain_tuner_csv.py` | Plots commanded vs observed joint tracking from CSV exports; can overlay sim and real datasets |

## Three-Step Tuning Workflow

The correct order matters: damping must be set before searching for stiffness, because an over-damped controller will never reach targets quickly enough to pass the RMS threshold used by the stiffness sweep.

### Step 1 — Sinusoidal excitation with `impedance_tuner.py`

Get a qualitative baseline of how the robot tracks sinusoidal commands with the naive (fixed) gains.
Exports a CSV that can be plotted with `plot_sim_gain_tuner_csv.py` and compared later against tuned results.

```bash
python scripts/tuning/impedance_tuner.py
python scripts/tuning/impedance_tuner.py --joints 0 1 2 --excitation chirp --duration 15
```

### Step 2 — Damping sweep with `step_tuner.py`

Find the optimal ζ per joint using step responses **before** searching for Kp.
ζ = 1.0 is critical damping, ζ < 1 is under-damped (faster, some overshoot), ζ > 1 is over-damped (slower).
Over-damped gains will prevent the robot from reaching targets fast enough for the stiffness sweep to converge.

```bash
python scripts/tuning/step_tuner.py --joints 1
python scripts/tuning/step_tuner.py --joints 1 --zeta 0.5 0.7 0.8 0.9 1.0
python scripts/tuning/step_tuner.py --dry-run
```

### Step 3 — Stiffness sweep with `auto_tuner.py`

With damping fixed, find the minimum Kp per joint such that RMS tracking error on a 0.5 Hz sinusoid stays below `--rms-target` (default 2°).
Kd is derived automatically as `Kd = 2·ζ·√Kp`, using the ζ chosen in Step 2.

```bash
python scripts/tuning/auto_tuner.py
python scripts/tuning/auto_tuner.py --joints 0 1 2 --rms-target 1.5
python scripts/tuning/auto_tuner.py --dry-run   # print the sweep plan, no robot
```

Results are saved as JSON and a bar-chart per joint under `logs/impedance_tuner/`.

## Simulation Gain Logger

`sim_gain_tuner_logger.py` must be run from the Isaac Sim **Script Editor** (not from the terminal).
It records commanded and observed joint positions/velocities during a simulation run and exports a CSV under
`~/Woodworking_Simulation/logs/sim_gain_tuner/`.

## Comparing Sim and Real Tracking

`plot_sim_gain_tuner_csv.py` can overlay simulation and real-robot CSVs on the same figure
to visually evaluate how well the simulation controller matches the real hardware.

```bash
# Single file
python scripts/tuning/plot_sim_gain_tuner_csv.py --file logs/sim_gain_tuner/<run>.csv

# Overlay sim and real
python scripts/tuning/plot_sim_gain_tuner_csv.py \
    --file logs/sim_gain_tuner/sim_run.csv \
    --file-real logs/impedance_tuner/runs_csv/real_run.csv
```

## Related Files

- `scripts/sim2real/URscript/` — the URScript controllers loaded by the tuning scripts:
  - `impedance_control_naive.script` — used by `impedance_tuner.py` (position-only, compatible with v1 RTDE recipe)
  - `impedance_control_tuning_zeta.script` — used by `auto_tuner.py` (fixed per-joint ζ, Kp from RTDE regs)
  - `impedance_control_tuning.script` — used by `step_tuner.py` (Kp + scalar ζ both from RTDE regs)
  - `rtde_input.xml` — RTDE recipe used by `impedance_tuner.py`
  - `rtde_input_tuning.xml` — RTDE recipe used by `auto_tuner.py` and `step_tuner.py` (adds kp, ζ and go_home registers)
