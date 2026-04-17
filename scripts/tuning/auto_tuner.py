#!/usr/bin/env python3
"""
Automatic Impedance-Controller Gain Tuner for UR5e
===================================================

Sweeps Kp values per joint via RTDE while the URScript
``impedance_control_tuning_zeta.script`` runs continuously.  Kd is
computed as Kd = 2·ζ·√Kp, with ζ configurable per joint.

For each joint the script searches for the **minimum Kp** that achieves
RMS tracking error < ``--rms-target`` (default 2°) on a 0.5 Hz sinusoid.

The sweep is done **one joint at a time** around a configurable centre
value (based on prior manual tuning).

Results are printed as a summary table, saved as JSON, and a bar-chart
of RMS vs Kp is generated for each joint.

Usage:
    python3 auto_tuner.py
    python3 auto_tuner.py --joints 0 1 2 --rms-target 1.5
    python3 auto_tuner.py --dry-run          # no robot, just prints the plan
    python3 auto_tuner.py --robot-ip 192.168.1.102
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
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_tuning_zeta.script") #choose the URScript with or without zeta control
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

# Default Kp centres (from prior manual testing)
DEFAULT_KP = np.array([200.0, 800.0, 200.0, 40.0, 40.0, 40.0])
DEFAULT_ZETA = np.array([0.5, 1.0, 0.5, 0.2, 0.2, 0.2])

# Search grids per joint (centred around best-known values)
# Sorted ascending so we find the minimum Kp first (early stopping)
DEFAULT_KP_GRIDS = {
    0: [600, 800, 1000],     # shoulder_pan
    1: [800, 1000, 1200, 1400],     # shoulder_lift  (heavy joint)
    2: [600, 800, 1000],     # elbow
    3: [100, 200, 300, 400],           # wrist_1
    4: [100, 200, 300, 400],        # wrist_2
    5: [100, 200, 300, 400],            # wrist_3
}


# ============================================================================
# RTDE link for tuning (reads/writes Kp via registers 37-42)
# ============================================================================

class RTDETuningLink:
    """RTDE connection with Kp gain registers for auto-tuning."""

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
        self.setp = None          # q_des (registers 24-29)
        self.setp_vel = None      # qdot_des (registers 30-35)
        self.setp_kp = None       # kp_gains (registers 37-42)
        self.setp_zeta = None     # damping_ratio (register 43)
        self.stop_reg = None
        self.go_home_reg = None
        self.control_rate_reg = None
        self.connected = False

        # Cached state
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
        print("[RTDE] Connected and synchronised (tuning mode)")

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
        print("[RTDE] URScript running (tuning mode)")

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
        """Write 6 Kp values to RTDE registers 37-42."""
        for i in range(6):
            self.setp_kp.__dict__[f"input_double_register_{37 + i}"] = float(kp[i])
        self.con.send(self.setp_kp)

    def write_zeta(self, zeta: float):
        """Write damping ratio to RTDE register 43."""
        self.setp_zeta.__dict__["input_double_register_43"] = float(zeta)
        self.con.send(self.setp_zeta)

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
# Signal generator
# ============================================================================

def sinusoid(t: float, amplitude: float, freq: float, offset: float):
    pos = offset + amplitude * math.sin(2.0 * math.pi * freq * t)
    vel = amplitude * 2.0 * math.pi * freq * math.cos(2.0 * math.pi * freq * t)
    return pos, vel


def resolve_zeta_vector(zeta_values: Optional[list[float]]) -> np.ndarray:
    if zeta_values is None:
        return DEFAULT_ZETA.copy()

    if len(zeta_values) == 1:
        return np.full(6, float(zeta_values[0]), dtype=np.float64)

    if len(zeta_values) != 6:
        raise ValueError("--zeta expects either 1 value or 6 values")

    return np.asarray(zeta_values, dtype=np.float64)


def compute_kd_vector(kp: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    return 2.0 * zeta * np.sqrt(kp)


# ============================================================================
# Single trial: excite one joint, return RMS error
# ============================================================================

def run_trial(
    link: RTDETuningLink,
    joint_idx: int,
    kp_values: np.ndarray,
    freq: float,
    amplitude: float,
    duration: float,
    rate: float,
    settle_time: float = 2.0,
    skip_transient: float = 2.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single sinusoidal trial on one joint.

    Returns:
        (rms_error_deg, timestamps, q_des_log, qd_des_log, q_act_log, qd_act_log)
    """
    dt = 1.0 / rate

    # 1. Write new Kp gains
    link.write_kp(kp_values)

    # 2. Get current position (= home after previous trial)
    q_centre = HOME_Q.copy()
    q_cur, _ = link.get_state()
    if q_cur is not None:
        q_centre = q_cur.copy()

    # 3. Settle: hold centre position
    n_settle = int(settle_time * rate)
    for _ in range(n_settle):
        t0 = time.monotonic()
        link.send_targets(q_centre, np.zeros(6))
        remaining = dt - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # 4. Sinusoidal excitation
    n_steps = int(duration * rate)
    timestamps = np.zeros(n_steps)
    q_des_log = np.zeros((n_steps, 6))
    qd_des_log = np.zeros((n_steps, 6))
    q_act_log = np.zeros((n_steps, 6))
    qd_act_log = np.zeros((n_steps, 6))

    t_start = time.monotonic()
    for step in range(n_steps):
        loop_t0 = time.monotonic()
        t = loop_t0 - t_start

        q_des = q_centre.copy()
        qd_des = np.zeros(6)

        pos, vel = sinusoid(t, amplitude, freq, q_centre[joint_idx])
        q_des[joint_idx] = pos
        qd_des[joint_idx] = vel  # always send velocity FF for tuning

        link.send_targets(q_des, qd_des)

        q_act, qd_act = link.get_state()
        if q_act is None:
            q_act = np.full(6, np.nan)
            qd_act = np.full(6, np.nan)

        timestamps[step] = t
        q_des_log[step] = q_des
        qd_des_log[step] = qd_des
        q_act_log[step] = q_act
        qd_act_log[step] = qd_act

        remaining = dt - (time.monotonic() - loop_t0)
        if remaining > 0:
            time.sleep(remaining)

    # 5. Hold centre briefly after excitation
    for _ in range(int(0.5 * rate)):
        t0 = time.monotonic()
        link.send_targets(q_centre, np.zeros(6))
        remaining = dt - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # 6. Compute RMS error (skip transient)
    skip_samples = int(skip_transient * rate)
    error_rad = q_des_log[skip_samples:, joint_idx] - q_act_log[skip_samples:, joint_idx]
    error_deg = np.degrees(error_rad)
    rms_deg = float(np.sqrt(np.nanmean(error_deg ** 2)))

    return rms_deg, timestamps, q_des_log, qd_des_log, q_act_log, qd_act_log


# ============================================================================
# Main auto-tuning sweep
# ============================================================================

def run_auto_tuner(args: argparse.Namespace):
    joints_to_tune = args.joints
    rms_target = args.rms_target
    zeta_vector = resolve_zeta_vector(args.zeta)

    if np.any(zeta_vector <= 0.0):
        raise ValueError("All zeta values must be strictly positive")

    # Build Kp grids
    kp_grids = {}
    for j in joints_to_tune:
        kp_grids[j] = DEFAULT_KP_GRIDS[j]

    # Summary of plan
    total_trials = sum(len(kp_grids[j]) for j in joints_to_tune)
    trial_time = args.settle + args.duration + 0.5  # settle + excitation + hold
    total_time = total_trials * trial_time

    print("=" * 65)
    print("  Automatic Impedance Controller Gain Tuner")
    print("  Kd = 2·ζ·√Kp")
    print("=" * 65)
    print(f"  Joints to tune:  {[f'{j} ({JOINT_NAMES[j]})' for j in joints_to_tune]}")
    print(f"  RMS target:      < {rms_target}°")
    print(f"  Sine:            {args.freq} Hz, ±{args.amplitude} rad ({np.degrees(args.amplitude):.1f}°)")
    print(f"  Zeta vector:     {zeta_vector.tolist()}")
    print(f"  Trial duration:  {args.duration}s excitation + {args.settle}s settle")
    print(f"  Total trials:    {total_trials}")
    print(f"  Estimated time:  {total_time / 60:.1f} min")
    print()
    for j in joints_to_tune:
        grid = kp_grids[j]
        print(f"  Joint {j} ({JOINT_NAMES[j]}):  Kp = {grid}")
        print(f"    {'':>22s}  Kd = {[round(2 * zeta_vector[j] * math.sqrt(kp), 1) for kp in grid]}")
    print("=" * 65)

    if args.dry_run:
        print("\n[DRY RUN] No robot connection. Exiting.")
        return

    input("\nPress ENTER to start (ensure robot is in remote-control mode)…")

    link = RTDETuningLink(
        robot_host=args.robot_ip,
        rtde_frequency=125.0,
        control_rate=args.rate,
    )

    # Results storage
    all_results: dict[int, list[dict]] = {j: [] for j in joints_to_tune}
    best_kp = DEFAULT_KP.copy()
    best_results: dict[int, dict] = {}

    try:
        link.connect()
        link.send_urscript()
        link.start_reader()

        # Wait for state
        print("[Tuner] Waiting for robot state…")
        for _ in range(200):
            q, _ = link.get_state()
            if q is not None:
                break
            time.sleep(0.02)
        if q is None:
            print("[Tuner] ERROR: no robot state – aborting.")
            return

        print(f"[Tuner] Robot state OK: {np.round(q, 3).tolist()}\n")

        trial_num = 0
        for j in joints_to_tune:
            print(f"\n{'─' * 55}")
            print(f"  Tuning joint {j} ({JOINT_NAMES[j]})")
            print(f"{'─' * 55}")

            found_best = False
            for kp_candidate in kp_grids[j]:
                trial_num += 1
                kd_candidate = 2.0 * zeta_vector[j] * math.sqrt(kp_candidate)

                # Build Kp vector: use best found so far for other joints
                kp_vec = best_kp.copy()
                kp_vec[j] = kp_candidate

                print(
                    f"  [{trial_num}/{total_trials}] Joint {j}: "
                    f"Kp={kp_candidate:.0f}, Kd={kd_candidate:.1f} … ",
                    end="", flush=True,
                )

                rms, t_log, q_des_log, qd_des_log, q_act_log, qd_act_log = run_trial(
                    link,
                    joint_idx=j,
                    kp_values=kp_vec,
                    freq=args.freq,
                    amplitude=args.amplitude,
                    duration=args.duration,
                    rate=args.rate,
                    settle_time=args.settle,
                    skip_transient=args.skip_transient,
                )

                status = "✓ PASS" if rms < rms_target else "✗ FAIL"
                print(f"RMS = {rms:.2f}°  {status}")

                result = {
                    "joint": j,
                    "joint_name": JOINT_NAMES[j],
                    "kp": kp_candidate,
                    "kd": round(kd_candidate, 2),
                    "rms_deg": round(rms, 3),
                    "passed": rms < rms_target,
                }
                all_results[j].append(result)

                # If this is the first Kp that passes (sorted ascending),
                # it's the minimum compliant Kp → select it
                if rms < rms_target and not found_best:
                    best_kp[j] = kp_candidate
                    best_results[j] = result
                    found_best = True
                    print(f"  → Selected Kp={kp_candidate:.0f} for joint {j} (minimum compliant)")

                    if args.early_stop:
                        print(f"  → Early stopping for joint {j}")
                        break

            if not found_best:
                # No Kp passed → keep the one with lowest RMS
                lowest = min(all_results[j], key=lambda r: r["rms_deg"])
                best_kp[j] = lowest["kp"]
                best_results[j] = lowest
                print(
                    f"  ⚠ No Kp achieved RMS < {rms_target}° for joint {j}. "
                    f"Best: Kp={lowest['kp']:.0f} → RMS={lowest['rms_deg']:.2f}°"
                )

        # ================================================================
        # Validation: run all joints simultaneously with best gains
        # ================================================================
        print(f"\n{'═' * 55}")
        print("  VALIDATION: all joints with best gains")
        print(f"{'═' * 55}")
        print(f"  Best Kp: {best_kp.tolist()}")
        best_kd = compute_kd_vector(best_kp, zeta_vector)
        print(f"  Best Kd: {best_kd.round(1).tolist()}")

        link.write_kp(best_kp)
        dt = 1.0 / args.rate

        # Hold for settle
        for _ in range(int(args.settle * args.rate)):
            t0 = time.monotonic()
            link.send_targets(HOME_Q, np.zeros(6))
            remaining = dt - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

        # Excite ALL tuned joints simultaneously
        n_steps = int(args.duration * args.rate)
        val_q_des = np.zeros((n_steps, 6))
        val_q_act = np.zeros((n_steps, 6))
        val_t = np.zeros(n_steps)

        t_start = time.monotonic()
        for step in range(n_steps):
            loop_t0 = time.monotonic()
            t = loop_t0 - t_start
            q_des = HOME_Q.copy()
            qd_des = np.zeros(6)
            for jj in joints_to_tune:
                pos, vel = sinusoid(t, args.amplitude, args.freq, HOME_Q[jj])
                q_des[jj] = pos
                qd_des[jj] = vel
            link.send_targets(q_des, qd_des)
            q_act, _ = link.get_state()
            if q_act is None:
                q_act = np.full(6, np.nan)
            val_t[step] = t
            val_q_des[step] = q_des
            val_q_act[step] = q_act
            remaining = dt - (time.monotonic() - loop_t0)
            if remaining > 0:
                time.sleep(remaining)

        # Compute validation RMS per joint
        skip = int(args.skip_transient * args.rate)
        print()
        for jj in joints_to_tune:
            err = np.degrees(val_q_des[skip:, jj] - val_q_act[skip:, jj])
            rms_v = float(np.sqrt(np.nanmean(err ** 2)))
            status = "✓" if rms_v < rms_target else "✗"
            print(f"  Joint {jj} ({JOINT_NAMES[jj]}): RMS = {rms_v:.2f}°  {status}")

        # Stop robot
        link.stop_robot()

        # ================================================================
        # Save results
        # ================================================================
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # --- Summary table ---
        print(f"\n{'═' * 65}")
        print("  RESULTS SUMMARY")
        print(f"{'═' * 65}")
        print(f"  {'Joint':<20s}  {'Kp':>6s}  {'Kd':>6s}  {'RMS':>7s}  {'Status'}")
        print(f"  {'─' * 20}  {'─' * 6}  {'─' * 6}  {'─' * 7}  {'─' * 6}")
        for j in joints_to_tune:
            r = best_results[j]
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(f"  {JOINT_NAMES[j]:<20s}  {r['kp']:6.0f}  {r['kd']:6.1f}  {r['rms_deg']:6.2f}°  {status}")
        print(f"{'═' * 65}")
        print()
        print("  Copy-paste for impedance_control_v3.script:")
        print(f"    kp = {[best_kp[i] for i in range(6)]}")
        kd_final = best_kd.round(1).tolist()
        print(f"    kd = {kd_final}")
        print()

        # --- JSON ---
        json_path = LOG_DIR / f"auto_tuner_{stamp}.json"
        output = {
            "timestamp": stamp,
            "config": {
                "freq": args.freq,
                "amplitude": args.amplitude,
                "duration": args.duration,
                "rate": args.rate,
                "rms_target": rms_target,
                "zeta": zeta_vector.tolist(),
            },
            "best_kp": best_kp.tolist(),
            "best_kd": kd_final,
            "per_joint": {
                str(j): all_results[j] for j in joints_to_tune
            },
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[Tuner] Results saved: {json_path}")

        # --- Plot: RMS vs Kp per joint ---
        plot_sweep_results(all_results, joints_to_tune, rms_target, zeta_vector, stamp)

    except KeyboardInterrupt:
        print("\n[Tuner] Interrupted – stopping robot…")
        link.stop_robot()
    finally:
        link.disconnect()


# ============================================================================
# Plotting
# ============================================================================

def plot_sweep_results(
    all_results: dict[int, list[dict]],
    joints: list[int],
    rms_target: float,
    zeta_vector: np.ndarray,
    stamp: str,
):
    """Plot RMS error vs Kp for each joint."""
    n = len(joints)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for col, j in enumerate(joints):
        ax = axes[0, col]
        results = all_results[j]
        kps = [r["kp"] for r in results]
        rmss = [r["rms_deg"] for r in results]

        colors = ["tab:green" if r["passed"] else "tab:red" for r in results]
        ax.bar(range(len(kps)), rmss, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(kps)))
        ax.set_xticklabels([f"{kp:.0f}" for kp in kps], rotation=45)
        ax.axhline(rms_target, color="k", linestyle="--", linewidth=1, label=f"Target {rms_target}°")
        ax.set_xlabel("Kp")
        ax.set_ylabel("RMS Error [°]")
        ax.set_title(f"Joint {j} – {JOINT_NAMES[j]}  (ζ={zeta_vector[j]:.2f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Auto-Tuner: RMS Error vs Kp  (Kd = 2·ζ·√Kp)", fontsize=12)
    fig.tight_layout()

    fig_path = LOG_DIR / f"auto_tuner_{stamp}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    print(f"[Tuner] Figure saved: {fig_path}")
    plt.show()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Automatic impedance-controller gain tuner for UR5e",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--joints", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                    help="Joints to tune (default: all)")
    p.add_argument("--rms-target", type=float, default=2.0,
                    help="Target RMS error [deg] (default: 2.0)")
    p.add_argument("--freq", type=float, default=0.5,
                    help="Sinusoid frequency [Hz] (default: 0.5)")
    p.add_argument("--amplitude", type=float, default=0.3,
                    help="Sinusoid amplitude [rad] (default: 0.3)")
    p.add_argument("--zeta", type=float, nargs="+", default=None,
                    help="One zeta value or 6 per-joint zeta values (default: built-in vector)")
    p.add_argument("--duration", type=float, default=8.0,
                    help="Excitation duration per trial [s] (default: 8)")
    p.add_argument("--settle", type=float, default=2.0,
                    help="Settle time before each trial [s] (default: 2)")
    p.add_argument("--skip-transient", type=float, default=2.0,
                    help="Skip first N seconds for RMS computation (default: 2)")
    p.add_argument("--rate", type=float, default=60.0,
                    help="Command rate [Hz] (default: 60)")
    p.add_argument("--early-stop", action="store_true", default=True,
                    help="Stop testing higher Kp once RMS target is met (default: True)")
    p.add_argument("--no-early-stop", action="store_false", dest="early_stop",
                    help="Test all Kp values even after target is met")
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
    run_auto_tuner(args)


if __name__ == "__main__":
    main()
