#!/usr/bin/env python3
"""
Step-Response Damping Tuner for UR5e Impedance Controller
==========================================================

Sends step commands to one joint at a time through the RTDE interface and
records the transient response.  Sweeps the damping ratio ζ (zeta) to find
the optimal value that balances speed and overshoot.

The UR-side impedance controller (``impedance_control_tuning.script``)
computes:
    Kd = 2 · ζ · √Kp

So ζ = 1.0 is critical damping, ζ < 1 is under-damped (faster, some
overshoot), ζ > 1 is over-damped (slower, no overshoot).

For each (joint, ζ) trial the script:
    1. Holds the current position for ``settle_time``.
    2. Applies a position step of ``step_size`` rad.
    3. Records the response for ``record_time`` seconds.
    4. Computes step-response metrics:
       - Rise time  (10% → 90%)
       - Overshoot  (%)
       - Settling time (to ±2% of step)
       - Steady-state error
    5. Steps back to the original position, records that too.

Usage:
    # Sweep ζ on joint 1 (shoulder_lift)
    python3 step_tuner.py --joints 1

    # Test specific ζ values
    python3 step_tuner.py --joints 1 --zeta 0.5 0.7 0.8 0.9 1.0

    # Custom step size and Kp
    python3 step_tuner.py --joints 1 --step-size 0.1 --kp 800

    # All joints
    python3 step_tuner.py

    # Dry run
    python3 step_tuner.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# ============================================================================
# Paths & defaults
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
RTDE_CONFIG_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "rtde_input_tuning.xml")
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_tuning.script")
LOG_DIR = REPO_ROOT / "logs" / "impedance_tuner"

ROBOT_HOST = "192.168.1.101"
ROBOT_PORT = 30004
ROBOT_PRIMARY_PORT = 30001

HOME_Q = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Best Kp from auto_tuner (first run)
DEFAULT_KP = np.array([350.0, 800.0, 350.0, 50.0, 50.0, 50.0])

# Default ζ sweep
DEFAULT_ZETA_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ============================================================================
# RTDE link (reuses tuning register layout: Kp on 37-42, ζ on 43)
# ============================================================================

class RTDETuningLink:
    """RTDE connection with Kp + ζ registers for step-response tuning."""

    def __init__(
        self,
        robot_host: str = ROBOT_HOST,
        robot_port: int = ROBOT_PORT,
        primary_port: int = ROBOT_PRIMARY_PORT,
        config_file: str = RTDE_CONFIG_FILE,
        urscript_file: str = URSCRIPT_FILE,
        rtde_frequency: float = 125.0,
        control_rate: float = 60.0,
    ):
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.primary_port = primary_port
        self.config_file = config_file
        self.urscript_file = urscript_file
        self.rtde_frequency = rtde_frequency
        self.control_rate = control_rate

        self.con: Optional[rtde.RTDE] = None
        self.setp = None
        self.setp_vel = None
        self.setp_kp = None
        self.setp_zeta = None
        self.stop_reg = None
        self.go_home_reg = None
        self.control_rate_reg = None
        self.connected = False

        self._lock = Lock()
        self._q: Optional[np.ndarray] = None
        self._qd: Optional[np.ndarray] = None
        self._reader_running = False
        self._reader_thread: Optional[Thread] = None

    def connect(self):
        conf = rtde_config.ConfigFile(self.config_file)
        out_names, out_types = conf.get_recipe("out")
        q_names, q_types = conf.get_recipe("q_des")
        qd_names, qd_types = conf.get_recipe("qdot_des")
        kp_names, kp_types = conf.get_recipe("kp_gains")
        zeta_name, zeta_type = conf.get_recipe("damping_ratio")
        stop_name, stop_type = conf.get_recipe("stop")
        go_home_name, go_home_type = conf.get_recipe("go_home")
        cr_name, cr_type = conf.get_recipe("control_rate_info")

        self.con = rtde.RTDE(self.robot_host, self.robot_port)
        self.con.connect()

        if not self.con.send_output_setup(out_names, out_types,
                                           frequency=self.rtde_frequency):
            raise RuntimeError("RTDE: output setup failed")

        self.setp = self.con.send_input_setup(q_names, q_types)
        self.setp_vel = self.con.send_input_setup(qd_names, qd_types)
        self.setp_kp = self.con.send_input_setup(kp_names, kp_types)
        self.setp_zeta = self.con.send_input_setup(zeta_name, zeta_type)
        self.stop_reg = self.con.send_input_setup(stop_name, stop_type)
        self.go_home_reg = self.con.send_input_setup(go_home_name, go_home_type)
        self.control_rate_reg = self.con.send_input_setup(cr_name, cr_type)

        for name, item in [("q_des", self.setp), ("qdot_des", self.setp_vel),
                           ("kp_gains", self.setp_kp), ("damping_ratio", self.setp_zeta),
                           ("stop", self.stop_reg), ("go_home", self.go_home_reg),
                           ("control_rate", self.control_rate_reg)]:
            if item is None:
                raise RuntimeError(f"RTDE: {name} input setup failed")

        if not self.con.send_start():
            raise RuntimeError("RTDE: start synchronisation failed")

        self.connected = True
        print("[RTDE] Connected and synchronised (step tuning mode)")

    def send_urscript(self):
        self.stop_reg.input_bit_register_64 = False
        self.con.send(self.stop_reg)
        self.go_home_reg.input_bit_register_65 = False
        self.con.send(self.go_home_reg)
        self.control_rate_reg.input_int_register_36 = int(self.control_rate)
        self.con.send(self.control_rate_reg)
        self._write_q(HOME_Q.tolist())
        self._write_qd([0.0] * 6)
        self.write_kp(DEFAULT_KP)
        self.write_zeta(1.0)

        print(f"[RTDE] Uploading URScript: {self.urscript_file}")
        with open(self.urscript_file, "r") as f:
            program = f.read()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.robot_host, self.primary_port))
        s.sendall((program + "\n").encode("utf-8"))
        s.close()
        time.sleep(3.0)
        print("[RTDE] URScript running (step tuning mode)")

    # ---- register writes --------------------------------------------------

    def _write_q(self, q):
        for i in range(6):
            self.setp.__dict__[f"input_double_register_{24 + i}"] = float(q[i])
        self.con.send(self.setp)

    def _write_qd(self, qd):
        for i in range(6):
            self.setp_vel.__dict__[f"input_double_register_{30 + i}"] = float(qd[i])
        self.con.send(self.setp_vel)

    def write_kp(self, kp: np.ndarray):
        for i in range(6):
            self.setp_kp.__dict__[f"input_double_register_{37 + i}"] = float(kp[i])
        self.con.send(self.setp_kp)

    def write_zeta(self, zeta: float):
        self.setp_zeta.__dict__["input_double_register_43"] = float(zeta)
        self.con.send(self.setp_zeta)

    def go_home(self, timeout: float = 8.0):
        """Trigger movej to home on the robot and wait until done."""
        # Set go_home flag
        self.go_home_reg.input_bit_register_65 = True
        self.con.send(self.go_home_reg)
        print("[RTDE] go_home triggered, waiting for movej…", end="", flush=True)

        # Wait for robot to reach home
        t0 = time.monotonic()
        time.sleep(1.0)  # give movej time to start
        while time.monotonic() - t0 < timeout:
            q, _ = self.get_state()
            if q is not None:
                err = np.max(np.abs(q - HOME_Q))
                if err < 0.01:  # within ~0.6°
                    break
            time.sleep(0.05)
        print(" done")

        # Clear go_home flag so URScript resumes impedance control
        self.go_home_reg.input_bit_register_65 = False
        self.con.send(self.go_home_reg)
        time.sleep(0.5)  # let URScript resume

    def send_targets(self, q_des: np.ndarray, qdot_des: np.ndarray):
        self._write_q(q_des.tolist())
        self._write_qd(qdot_des.tolist())

    # ---- background reader ------------------------------------------------

    def start_reader(self):
        if self._reader_thread is not None:
            return
        self._reader_running = True
        self._reader_thread = Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop_reader(self):
        self._reader_running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

    def _reader_loop(self):
        period = 1.0 / self.rtde_frequency
        while self._reader_running and self.connected:
            t0 = time.monotonic()
            try:
                state = self.con.receive()
                if state is not None:
                    with self._lock:
                        self._q = np.array(state.actual_q, dtype=np.float64)
                        self._qd = np.array(state.actual_qd, dtype=np.float64)
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def get_state(self):
        with self._lock:
            if self._q is None:
                return None, None
            return self._q.copy(), self._qd.copy()

    # ---- shutdown ---------------------------------------------------------

    def stop_robot(self):
        if not self.connected:
            return
        try:
            self._write_qd([0.0] * 6)
            self.stop_reg.input_bit_register_64 = True
            self.con.send(self.stop_reg)
            time.sleep(2.0)
            print("[RTDE] Stop signal sent – robot returning to home")
        except Exception as e:
            print(f"[RTDE] Stop error: {e}")

    def disconnect(self):
        self.stop_reader()
        if self.con is not None:
            try:
                self.con.send_pause()
                self.con.disconnect()
            except Exception:
                pass
        self.connected = False
        print("[RTDE] Disconnected")


# ============================================================================
# Step-response metrics
# ============================================================================

def compute_step_metrics(
    t: np.ndarray,
    q_act: np.ndarray,
    q_start: float,
    q_target: float,
    tolerance_pct: float = 2.0,
) -> dict:
    """
    Compute standard step-response metrics.

    Args:
        t: time array (seconds, starting from step instant)
        q_act: actual joint position array
        q_start: position before the step
        q_target: target position after the step
        tolerance_pct: settling tolerance in % of step size

    Returns:
        dict with rise_time, overshoot_pct, settling_time, steady_state_error
    """
    step_size = q_target - q_start
    if abs(step_size) < 1e-8:
        return {"rise_time": np.nan, "overshoot_pct": 0.0,
                "settling_time": np.nan, "steady_state_error": 0.0}

    # Normalised response: 0 at start, 1 at target
    y = (q_act - q_start) / step_size

    # Rise time: 10% → 90%
    rise_time = np.nan
    t_10 = np.nan
    t_90 = np.nan
    for i, val in enumerate(y):
        if np.isnan(t_10) and val >= 0.1:
            t_10 = t[i]
        if np.isnan(t_10):
            continue
        if np.isnan(t_90) and val >= 0.9:
            t_90 = t[i]
            break
    if not np.isnan(t_10) and not np.isnan(t_90):
        rise_time = t_90 - t_10

    # Overshoot
    if step_size > 0:
        peak = np.nanmax(y)
    else:
        peak = np.nanmin(y)
        # For negative step, overshoot means going below target (y < 1)
        # but in normalised space, peak > 1 means overshoot of absolute value
        peak = np.nanmax(y)  # still use max since normalised

    overshoot_pct = max(0.0, (peak - 1.0) * 100.0)

    # Settling time (last time |y - 1| > tolerance)
    tol = tolerance_pct / 100.0
    settled_mask = np.abs(y - 1.0) <= tol
    settling_time = np.nan
    if np.any(settled_mask):
        # Find last index where NOT settled
        not_settled = np.where(~settled_mask)[0]
        if len(not_settled) > 0:
            last_not_settled = not_settled[-1]
            if last_not_settled < len(t) - 1:
                settling_time = t[last_not_settled + 1] - t[0]
        else:
            settling_time = 0.0  # always within tolerance

    # Steady-state error (average of last 20% of recording)
    n_tail = max(1, int(0.2 * len(q_act)))
    ss_value = np.nanmean(q_act[-n_tail:])
    steady_state_error = q_target - ss_value  # in rad

    return {
        "rise_time": float(rise_time) if not np.isnan(rise_time) else None,
        "overshoot_pct": round(float(overshoot_pct), 2),
        "settling_time": float(settling_time) if not np.isnan(settling_time) else None,
        "steady_state_error_deg": round(float(np.degrees(steady_state_error)), 3),
    }


# ============================================================================
# Single step trial
# ============================================================================

def run_step_trial(
    link: RTDETuningLink,
    joint_idx: int,
    kp_values: np.ndarray,
    zeta: float,
    step_size: float,
    rate: float,
    settle_time: float = 2.0,
    record_time: float = 3.0,
) -> dict:
    """
    Run a step-up then step-down trial on one joint.

    Returns dict with metrics and logged data for both step-up and step-down.
    """
    dt = 1.0 / rate

    # Set gains
    link.write_kp(kp_values)
    link.write_zeta(zeta)

    # Go home first to ensure consistent starting position
    link.go_home()

    # Base position is always HOME
    q_base = HOME_Q.copy()

    # ---- Phase 1: settle at base position ----
    n_settle = int(settle_time * rate)
    for _ in range(n_settle):
        t0 = time.monotonic()
        link.send_targets(q_base, np.zeros(6))
        remaining = dt - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # ---- Phase 2: step UP ----
    q_target_up = q_base.copy()
    q_target_up[joint_idx] += step_size

    n_record = int(record_time * rate)
    t_up = np.zeros(n_record)
    q_up = np.zeros((n_record, 6))

    t_start = time.monotonic()
    for step in range(n_record):
        loop_t0 = time.monotonic()
        t_up[step] = loop_t0 - t_start
        # Send step target (no velocity FF — pure step)
        link.send_targets(q_target_up, np.zeros(6))
        q_act, _ = link.get_state()
        q_up[step] = q_act if q_act is not None else np.full(6, np.nan)
        remaining = dt - (time.monotonic() - loop_t0)
        if remaining > 0:
            time.sleep(remaining)

    # ---- Phase 3: settle at step position ----
    for _ in range(n_settle):
        t0 = time.monotonic()
        link.send_targets(q_target_up, np.zeros(6))
        remaining = dt - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # ---- Phase 4: step DOWN (back to base) ----
    t_down = np.zeros(n_record)
    q_down = np.zeros((n_record, 6))

    t_start = time.monotonic()
    for step in range(n_record):
        loop_t0 = time.monotonic()
        t_down[step] = loop_t0 - t_start
        link.send_targets(q_base, np.zeros(6))
        q_act, _ = link.get_state()
        q_down[step] = q_act if q_act is not None else np.full(6, np.nan)
        remaining = dt - (time.monotonic() - loop_t0)
        if remaining > 0:
            time.sleep(remaining)

    # ---- Compute metrics ----
    j = joint_idx
    metrics_up = compute_step_metrics(
        t_up, q_up[:, j], q_base[j], q_target_up[j]
    )
    metrics_down = compute_step_metrics(
        t_down, q_down[:, j], q_target_up[j], q_base[j]
    )

    kd = 2.0 * zeta * math.sqrt(kp_values[j])

    return {
        "joint": j,
        "joint_name": JOINT_NAMES[j],
        "kp": float(kp_values[j]),
        "kd": round(kd, 2),
        "zeta": zeta,
        "step_size_deg": round(float(np.degrees(step_size)), 2),
        "step_up": metrics_up,
        "step_down": metrics_down,
        # Raw data for plotting
        "_t_up": t_up,
        "_q_up": q_up[:, j],
        "_t_down": t_down,
        "_q_down": q_down[:, j],
        "_q_start": float(q_base[j]),
        "_q_target": float(q_target_up[j]),
    }


# ============================================================================
# Main sweep
# ============================================================================

def run_step_tuner(args: argparse.Namespace):
    joints = args.joints
    zeta_grid = args.zeta
    step_size = args.step_size

    # Per-joint Kp (can be overridden with --kp for single-joint tests)
    kp_vec = DEFAULT_KP.copy()
    if args.kp is not None and len(joints) == 1:
        kp_vec[joints[0]] = args.kp

    total_trials = len(joints) * len(zeta_grid)
    trial_time = args.settle * 2 + args.record * 2  # settle+step_up+settle+step_down
    total_est = total_trials * trial_time

    print("=" * 60)
    print("  Step-Response Damping Tuner – UR5e")
    print("  Kd = 2·ζ·√Kp")
    print("=" * 60)
    print(f"  Joints:       {[f'{j} ({JOINT_NAMES[j]})' for j in joints]}")
    print(f"  Kp:           {[kp_vec[j] for j in joints]}")
    print(f"  ζ values:     {zeta_grid}")
    print(f"  Step size:    {step_size:.3f} rad ({np.degrees(step_size):.1f}°)")
    print(f"  Record time:  {args.record:.1f}s per step")
    print(f"  Settle time:  {args.settle:.1f}s")
    print(f"  Total trials: {total_trials}")
    print(f"  Estimated:    {total_est / 60:.1f} min")
    print()
    for j in joints:
        kp = kp_vec[j]
        print(f"  Joint {j} ({JOINT_NAMES[j]}):  Kp = {kp:.0f}")
        for z in zeta_grid:
            kd = 2.0 * z * math.sqrt(kp)
            print(f"    ζ={z:.2f}  →  Kd={kd:.1f}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] No robot connection. Exiting.")
        return

    input("\nPress ENTER to start (ensure robot is in remote-control mode)…")

    link = RTDETuningLink(
        robot_host=args.robot_ip,
        rtde_frequency=125.0,
        control_rate=args.rate,
    )

    all_results: dict[int, list[dict]] = {j: [] for j in joints}

    try:
        link.connect()
        link.send_urscript()
        link.start_reader()

        # Wait for state
        print("[StepTuner] Waiting for robot state…")
        for _ in range(200):
            q, _ = link.get_state()
            if q is not None:
                break
            time.sleep(0.02)
        if q is None:
            print("[StepTuner] ERROR: no robot state – aborting.")
            return

        print(f"[StepTuner] Robot ready: {np.round(q, 3).tolist()}\n")

        trial_num = 0
        for j in joints:
            print(f"\n{'─' * 55}")
            print(f"  Step test: joint {j} ({JOINT_NAMES[j]}), Kp={kp_vec[j]:.0f}")
            print(f"{'─' * 55}")

            for zeta in zeta_grid:
                trial_num += 1
                kd = 2.0 * zeta * math.sqrt(kp_vec[j])
                print(
                    f"  [{trial_num}/{total_trials}] ζ={zeta:.2f}, "
                    f"Kd={kd:.1f} … ",
                    end="", flush=True,
                )

                result = run_step_trial(
                    link,
                    joint_idx=j,
                    kp_values=kp_vec,
                    zeta=zeta,
                    step_size=step_size,
                    rate=args.rate,
                    settle_time=args.settle,
                    record_time=args.record,
                )

                up = result["step_up"]
                dn = result["step_down"]
                print(
                    f"UP: rise={_fmt(up['rise_time'], 's')}, "
                    f"OS={up['overshoot_pct']:.1f}%, "
                    f"settle={_fmt(up['settling_time'], 's')}, "
                    f"ss_err={up['steady_state_error_deg']:.2f}°  |  "
                    f"DOWN: OS={dn['overshoot_pct']:.1f}%, "
                    f"ss_err={dn['steady_state_error_deg']:.2f}°"
                )

                # Remove raw data from stored results (keep for plotting)
                result_clean = {k: v for k, v in result.items() if not k.startswith("_")}
                all_results[j].append(result)

        # Stop robot
        link.stop_robot()

        # ================================================================
        # Summary
        # ================================================================
        print(f"\n{'═' * 70}")
        print("  RESULTS SUMMARY")
        print(f"{'═' * 70}")
        print(
            f"  {'Joint':<16s}  {'ζ':>5s}  {'Kp':>6s}  {'Kd':>6s}  "
            f"{'Rise':>6s}  {'OS%':>5s}  {'Settle':>7s}  {'SS err':>7s}"
        )
        print(f"  {'─' * 16}  {'─' * 5}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 5}  {'─' * 7}  {'─' * 7}")

        for j in joints:
            for r in all_results[j]:
                up = r["step_up"]
                print(
                    f"  {JOINT_NAMES[j]:<16s}  {r['zeta']:5.2f}  {r['kp']:6.0f}  {r['kd']:6.1f}  "
                    f"{_fmt(up['rise_time'], 's'):>6s}  {up['overshoot_pct']:5.1f}  "
                    f"{_fmt(up['settling_time'], 's'):>7s}  {up['steady_state_error_deg']:>6.2f}°"
                )
        print(f"{'═' * 70}")

        # ================================================================
        # Save & plot
        # ================================================================
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # JSON (without numpy arrays)
        json_path = LOG_DIR / f"step_tuner_{stamp}.json"
        json_results = {}
        for j in joints:
            json_results[str(j)] = [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in all_results[j]
            ]
        with open(json_path, "w") as f:
            json.dump({
                "timestamp": stamp,
                "config": {
                    "step_size_rad": step_size,
                    "record_time": args.record,
                    "settle_time": args.settle,
                    "rate": args.rate,
                },
                "kp_vector": kp_vec.tolist(),
                "results": json_results,
            }, f, indent=2)
        print(f"[StepTuner] Results saved: {json_path}")

        # Plot
        plot_step_results(all_results, joints, stamp)

    except KeyboardInterrupt:
        print("\n[StepTuner] Interrupted – stopping robot…")
        link.stop_robot()
    finally:
        link.disconnect()


def _fmt(val, unit: str) -> str:
    """Format a possibly-None value."""
    if val is None:
        return "N/A"
    return f"{val:.3f}{unit}"


# ============================================================================
# Plotting
# ============================================================================

def plot_step_results(
    all_results: dict[int, list[dict]],
    joints: list[int],
    stamp: str,
):
    """
    For each joint: overlay step-up responses for all ζ values.
    Second row: bar chart of overshoot vs ζ and settling time vs ζ.
    """
    n_joints = len(joints)
    fig, axes = plt.subplots(
        3, n_joints, figsize=(6 * n_joints, 12), squeeze=False
    )

    for col, j in enumerate(joints):
        results = all_results[j]
        q_start = results[0]["_q_start"]
        q_target = results[0]["_q_target"]
        step_deg = np.degrees(q_target - q_start)

        # --- Row 1: step-up response overlay ---
        ax = axes[0, col]
        for r in results:
            zeta = r["zeta"]
            t = r["_t_up"]
            q = np.degrees(r["_q_up"])
            ax.plot(t, q, label=f"ζ={zeta:.2f}", linewidth=1.2)

        ax.axhline(np.degrees(q_target), color="k", linestyle="--",
                    linewidth=0.8, label="target")
        ax.axhline(np.degrees(q_start), color="gray", linestyle=":",
                    linewidth=0.6)
        # ±2% settling band
        band = abs(step_deg) * 0.02
        ax.axhspan(np.degrees(q_target) - band, np.degrees(q_target) + band,
                    color="green", alpha=0.08, label="±2% band")
        ax.set_ylabel("Position [deg]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Joint {j} – {JOINT_NAMES[j]} – Step UP")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

        # --- Row 2: step-down response overlay ---
        ax = axes[1, col]
        for r in results:
            zeta = r["zeta"]
            t = r["_t_down"]
            q = np.degrees(r["_q_down"])
            ax.plot(t, q, label=f"ζ={zeta:.2f}", linewidth=1.2)

        ax.axhline(np.degrees(q_start), color="k", linestyle="--",
                    linewidth=0.8, label="target")
        ax.axhspan(np.degrees(q_start) - band, np.degrees(q_start) + band,
                    color="green", alpha=0.08, label="±2% band")
        ax.set_ylabel("Position [deg]")
        ax.set_xlabel("Time [s]")
        ax.set_title(f"Joint {j} – Step DOWN")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # --- Row 3: metrics bar chart ---
        ax = axes[2, col]
        zetas = [r["zeta"] for r in results]
        overshoots = [r["step_up"]["overshoot_pct"] for r in results]
        settle_times = [
            r["step_up"]["settling_time"] if r["step_up"]["settling_time"] is not None else 0
            for r in results
        ]
        ss_errors = [abs(r["step_up"]["steady_state_error_deg"]) for r in results]

        x = np.arange(len(zetas))
        w = 0.25
        ax.bar(x - w, overshoots, w, label="Overshoot [%]", color="tab:orange", alpha=0.8)
        ax.bar(x, [st * 10 for st in settle_times], w,
               label="Settle time [×10 s]", color="tab:blue", alpha=0.8)
        ax.bar(x + w, [e * 10 for e in ss_errors], w,
               label="SS error [×10 °]", color="tab:red", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"ζ={z:.2f}" for z in zetas], rotation=45)
        ax.set_title("Step-up metrics")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Step-Response Damping Tuner  (Kd = 2·ζ·√Kp)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    fig_path = LOG_DIR / f"step_tuner_{stamp}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    print(f"[StepTuner] Figure saved: {fig_path}")
    plt.show()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Step-response damping tuner for UR5e impedance controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--joints", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                    help="Joints to test (default: all)")
    p.add_argument("--zeta", type=float, nargs="+", default=DEFAULT_ZETA_GRID,
                    help=f"Damping ratios to sweep (default: {DEFAULT_ZETA_GRID})")
    p.add_argument("--step-size", type=float, default=0.20,
                    help="Step size [rad] (default: 0.20 ≈ 11.5°)")
    p.add_argument("--kp", type=float, default=None,
                    help="Override Kp for the tested joint (single-joint only)")
    p.add_argument("--record", type=float, default=3.0,
                    help="Recording time per step [s] (default: 3.0)")
    p.add_argument("--settle", type=float, default=2.0,
                    help="Settle time before each step [s] (default: 2.0)")
    p.add_argument("--rate", type=float, default=60.0,
                    help="Command rate [Hz] (default: 60)")
    p.add_argument("--dry-run", action="store_true",
                    help="Print plan without connecting to robot")
    p.add_argument("--robot-ip", type=str, default=ROBOT_HOST,
                    help=f"Robot IP (default: {ROBOT_HOST})")
    return p.parse_args()


def main():
    args = parse_args()
    for j in args.joints:
        if j < 0 or j > 5:
            print(f"ERROR: joint {j} out of range [0, 5]")
            sys.exit(1)
    run_step_tuner(args)


if __name__ == "__main__":
    main()
