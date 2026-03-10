#!/usr/bin/env python3
"""
Impedance Controller Gain Tuner for UR5e via RTDE
==================================================

Sends sinusoidal joint-position commands at 60 Hz through the RTDE interface
to the UR5e running ``impedance_control_v3.script``.  Records desired vs.
actual joint positions (and velocities) and produces tracking plots to
support systematic tuning of Kp, Kd, alpha_pos, alpha_vel.

Inspired by the *Gain Tuner* extension in Isaac Sim, but for the real robot.

Workflow:
    1.  Connect to the robot via RTDE.
    2.  Upload the impedance controller URScript.
    3.  Ramp smoothly from the current joint position to the sinusoid centre.
    4.  Run sinusoidal excitation for ``--duration`` seconds at ``--rate`` Hz.
    5.  Send stop signal; robot returns to home.
    6.  Plot & optionally save the results.

Usage examples:
    # Excite joint 1 (shoulder_lift) at 0.5 Hz ± 0.15 rad around current q
    python3 impedance_tuner.py --joints 1 --freq 0.5 --amplitude 0.15

    # Excite joints 0 and 2 simultaneously
    python3 impedance_tuner.py --joints 0 2 --freq 0.3 --amplitude 0.10

    # Multi-sine chirp from 0.1 to 2.0 Hz to sweep bandwidth
    python3 impedance_tuner.py --joints 1 --chirp --freq-start 0.1 --freq-end 2.0

    # Pass velocity feedforward as analytic derivative of the sinusoid
    python3 impedance_tuner.py --joints 1 --freq 0.5 --amplitude 0.15 --with-velocity-ff

    # Use custom robot IP
    python3 impedance_tuner.py --joints 1 --robot-ip 192.168.1.102
"""

from __future__ import annotations

import argparse
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
RTDE_CONFIG_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "rtde_input_v3.xml")
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_v3.script")
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


# ============================================================================
# Lightweight RTDE helper (stripped-down from RTDEController in sim2real_node)
# ============================================================================

class RTDELink:
    """Minimal RTDE connection: sends q_des + qdot_des, reads actual_q/qd."""

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
        self.stop_reg = None
        self.control_rate_reg = None
        self.connected = False

        # Cached state (updated by reader thread)
        self._lock = Lock()
        self._q: Optional[np.ndarray] = None
        self._qd: Optional[np.ndarray] = None
        self._reader_running = False
        self._reader_thread: Optional[Thread] = None

    # ---- connect ----------------------------------------------------------

    def connect(self):
        conf = rtde_config.ConfigFile(self.config_file)
        out_names, out_types = conf.get_recipe("out")
        q_names, q_types = conf.get_recipe("q_des")
        qd_names, qd_types = conf.get_recipe("qdot_des")
        stop_name, stop_type = conf.get_recipe("stop")
        cr_name, cr_type = conf.get_recipe("control_rate_info")

        self.con = rtde.RTDE(self.robot_host, self.robot_port)
        self.con.connect()

        if not self.con.send_output_setup(out_names, out_types,
                                           frequency=self.rtde_frequency):
            raise RuntimeError("RTDE: output setup failed")

        self.setp = self.con.send_input_setup(q_names, q_types)
        self.setp_vel = self.con.send_input_setup(qd_names, qd_types)
        self.stop_reg = self.con.send_input_setup(stop_name, stop_type)
        self.control_rate_reg = self.con.send_input_setup(cr_name, cr_type)

        for item in (self.setp, self.setp_vel, self.stop_reg, self.control_rate_reg):
            if item is None:
                raise RuntimeError("RTDE: input setup failed")

        if not self.con.send_start():
            raise RuntimeError("RTDE: start synchronisation failed")

        self.connected = True
        print("[RTDE] Connected and synchronised")

    # ---- URScript ---------------------------------------------------------

    def send_urscript(self):
        self.stop_reg.input_bit_register_64 = False
        self.con.send(self.stop_reg)
        self.control_rate_reg.input_int_register_36 = int(self.control_rate)
        self.con.send(self.control_rate_reg)
        self._write_q(HOME_Q.tolist())
        #self._write_q([0.0, -2.07, -1.00, -1.57, 0.0, 0.0])
        self._write_qd([0.0] * 6)

        print(f"[RTDE] Uploading URScript: {self.urscript_file}")
        with open(self.urscript_file, "r") as f:
            program = f.read()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.robot_host, self.primary_port))
        s.sendall((program + "\n").encode("utf-8"))
        s.close()
        time.sleep(3.0)
        print("[RTDE] URScript running")

    # ---- low-level register writes ----------------------------------------

    def _write_q(self, q):
        for i in range(6):
            self.setp.__dict__[f"input_double_register_{24 + i}"] = float(q[i])
        self.con.send(self.setp)

    def _write_qd(self, qd):
        for i in range(6):
            self.setp_vel.__dict__[f"input_double_register_{30 + i}"] = float(qd[i])
        self.con.send(self.setp_vel)

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
        print(f"[RTDE] Reader thread running at {self.rtde_frequency} Hz")

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
# Signal generators
# ============================================================================

def sinusoid(t: float, amplitude: float, freq: float, offset: float,
             phase: float = 0.0):
    """Returns (position, velocity) of a sinusoidal signal."""
    pos = offset + amplitude * math.sin(2.0 * math.pi * freq * t + phase)
    vel = amplitude * 2.0 * math.pi * freq * math.cos(
        2.0 * math.pi * freq * t + phase
    )
    return pos, vel


def chirp(t: float, amplitude: float, f_start: float, f_end: float,
          duration: float, offset: float):
    """Linear frequency sweep (chirp).  Returns (position, velocity)."""
    k = (f_end - f_start) / duration
    inst_freq = f_start + k * t
    phase = 2.0 * math.pi * (f_start * t + 0.5 * k * t * t)
    pos = offset + amplitude * math.sin(phase)
    vel = amplitude * 2.0 * math.pi * inst_freq * math.cos(phase)
    return pos, vel


# ============================================================================
# Main tuning routine
# ============================================================================

def run_tuner(args: argparse.Namespace):
    link = RTDELink(
        robot_host=args.robot_ip,
        rtde_frequency=125.0,
        control_rate=args.rate,
    )

    try:
        link.connect()
        link.send_urscript()
        link.start_reader()

        # Wait for first valid state
        print("[Tuner] Waiting for robot state…")
        q_cur = None
        for _ in range(200):
            q_cur, _ = link.get_state()
            if q_cur is not None:
                break
            time.sleep(0.02)
        if q_cur is None:
            print("[Tuner] ERROR: no robot state received – aborting.")
            return

        print(f"[Tuner] Current q: {np.round(q_cur, 4).tolist()}")

        # Centre of the sinusoid = current joint position
        q_centre = q_cur.copy()

        # ====== Phase 1: ramp to centre (hold current position) ==========
        # Send current position for a short time so the impedance controller
        # locks onto it before we start the sinusoid.
        print("[Tuner] Holding current position for 2 s …")
        ramp_duration = 2.0
        ramp_steps = int(ramp_duration * args.rate)
        dt = 1.0 / args.rate
        for _ in range(ramp_steps):
            t0 = time.monotonic()
            link.send_targets(q_centre, np.zeros(6))
            elapsed = time.monotonic() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # ====== Phase 2: sinusoidal excitation ============================
        duration = args.duration
        n_steps = int(duration * args.rate)
        excited_joints = args.joints  # list of joint indices

        # Data logging arrays
        timestamps = np.zeros(n_steps)
        q_des_log = np.zeros((n_steps, 6))
        qd_des_log = np.zeros((n_steps, 6))
        q_act_log = np.zeros((n_steps, 6))
        qd_act_log = np.zeros((n_steps, 6))

        print(
            f"[Tuner] Starting sinusoidal excitation: "
            f"joints={excited_joints}, "
            f"{'chirp' if args.chirp else f'freq={args.freq} Hz'}, "
            f"amp={args.amplitude} rad, "
            f"duration={duration} s, "
            f"rate={args.rate} Hz, "
            f"velocity_ff={'ON' if args.with_velocity_ff else 'OFF'}"
        )

        t_start = time.monotonic()
        for step in range(n_steps):
            loop_t0 = time.monotonic()
            t = loop_t0 - t_start

            q_des = q_centre.copy()
            qd_des = np.zeros(6)

            for j in excited_joints:
                if args.chirp:
                    pos, vel = chirp(
                        t, args.amplitude, args.freq_start, args.freq_end,
                        duration, q_centre[j],
                    )
                else:
                    pos, vel = sinusoid(t, args.amplitude, args.freq, q_centre[j])
                q_des[j] = pos
                if args.with_velocity_ff:
                    qd_des[j] = vel

            link.send_targets(q_des, qd_des)

            # Read actual state
            q_act, qd_act = link.get_state()
            if q_act is None:
                q_act = np.full(6, np.nan)
                qd_act = np.full(6, np.nan)

            # Log
            timestamps[step] = t
            q_des_log[step] = q_des
            qd_des_log[step] = qd_des
            q_act_log[step] = q_act
            qd_act_log[step] = qd_act

            elapsed = time.monotonic() - loop_t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

        actual_duration = time.monotonic() - t_start
        print(
            f"[Tuner] Excitation done: {n_steps} steps in {actual_duration:.2f} s "
            f"(target {duration:.2f} s)"
        )

        # ====== Phase 3: hold centre for 1 s, then stop =================
        print("[Tuner] Holding centre position for 1 s …")
        for _ in range(int(1.0 * args.rate)):
            t0 = time.monotonic()
            link.send_targets(q_centre, np.zeros(6))
            elapsed = time.monotonic() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

        link.stop_robot()

        # ====== Phase 4: plot & save ======================================
        plot_results(
            timestamps, q_des_log, qd_des_log, q_act_log, qd_act_log,
            excited_joints, args,
        )

    except KeyboardInterrupt:
        print("\n[Tuner] Interrupted – stopping robot…")
        link.stop_robot()
    finally:
        link.disconnect()


# ============================================================================
# Plotting
# ============================================================================

def plot_results(
    t: np.ndarray,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    q_act: np.ndarray,
    qd_act: np.ndarray,
    joints: list[int],
    args: argparse.Namespace,
):
    """Create tracking-quality plots for each excited joint."""

    n_joints = len(joints)
    fig, axes = plt.subplots(
        3, n_joints, figsize=(6 * n_joints, 10), squeeze=False, sharex=True
    )

    for col, j in enumerate(joints):
        # --- Row 1: position tracking ---
        ax = axes[0, col]
        ax.plot(t, np.degrees(q_des[:, j]), label="q_des", linewidth=1.2)
        ax.plot(t, np.degrees(q_act[:, j]), label="q_act", linewidth=1.2, alpha=0.85)
        ax.set_ylabel("Position [deg]")
        ax.set_title(f"Joint {j} – {JOINT_NAMES[j]}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Row 2: position error ---
        ax = axes[1, col]
        error_deg = np.degrees(q_des[:, j] - q_act[:, j])
        ax.plot(t, error_deg, color="tab:red", linewidth=1.0)
        rms = np.sqrt(np.nanmean(error_deg ** 2))
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel("Error [deg]")
        ax.set_title(f"RMS error: {rms:.3f}°")
        ax.grid(True, alpha=0.3)

        # --- Row 3: velocity tracking ---
        ax = axes[2, col]
        ax.plot(
            t, np.degrees(qd_des[:, j]),
            label="qdot_des", linewidth=1.0, linestyle="--",
        )
        ax.plot(
            t, np.degrees(qd_act[:, j]),
            label="qdot_act", linewidth=1.0, alpha=0.85,
        )
        ax.set_ylabel("Velocity [deg/s]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    mode = "chirp" if args.chirp else f"sine_{args.freq}Hz"
    vel_tag = "velFF" if args.with_velocity_ff else "noFF"
    title = (
        f"Impedance Tuner – {mode} – amp={args.amplitude:.3f} rad – "
        f"{vel_tag} – rate={args.rate} Hz"
    )
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()

    # Save
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    joints_tag = "_".join(str(j) for j in joints)
    basename = f"tuner_j{joints_tag}_{mode}_{vel_tag}_{stamp}"

    fig_path = LOG_DIR / f"{basename}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    print(f"[Tuner] Figure saved: {fig_path}")

    # Save raw data as .npz for later analysis
    npz_path = LOG_DIR / f"{basename}.npz"
    np.savez_compressed(
        str(npz_path),
        t=t,
        q_des=q_des,
        qd_des=qd_des,
        q_act=q_act,
        qd_act=qd_act,
        joints=np.array(joints),
        freq=args.freq,
        amplitude=args.amplitude,
        rate=args.rate,
        chirp=args.chirp,
        with_velocity_ff=args.with_velocity_ff,
    )
    print(f"[Tuner] Data saved:   {npz_path}")

    plt.show()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Impedance-controller gain tuner for UR5e (sinusoidal excitation via RTDE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 impedance_tuner.py --joints 1 --freq 0.5 --amplitude 0.15\n"
            "  python3 impedance_tuner.py --joints 0 2 --chirp --freq-start 0.1 --freq-end 2.0\n"
            "  python3 impedance_tuner.py --joints 1 --freq 0.5 --amplitude 0.15 --with-velocity-ff\n"
        ),
    )

    # Joint selection
    p.add_argument(
        "--joints", type=int, nargs="+", default=[1],
        help="Joint indices to excite (0-5). Default: 1 (shoulder_lift).",
    )

    # Sinusoid parameters
    p.add_argument("--freq", type=float, default=0.5, help="Sine frequency [Hz] (default: 0.5)")
    p.add_argument("--amplitude", type=float, default=0.15, help="Sine amplitude [rad] (default: 0.15)")
    p.add_argument("--duration", type=float, default=10.0, help="Excitation duration [s] (default: 10)")

    # Chirp mode
    p.add_argument("--chirp", action="store_true", help="Use linear frequency sweep instead of fixed sine")
    p.add_argument("--freq-start", type=float, default=0.1, help="Chirp start frequency [Hz]")
    p.add_argument("--freq-end", type=float, default=2.0, help="Chirp end frequency [Hz]")

    # Velocity feedforward
    p.add_argument(
        "--with-velocity-ff", action="store_true",
        help="Send analytic velocity derivative as qdot_des (enables velocity feedforward in impedance controller)",
    )

    # Control rate
    p.add_argument("--rate", type=float, default=60.0, help="Command rate [Hz] (default: 60, matches policy rate)")

    # Robot connection
    p.add_argument("--robot-ip", type=str, default=ROBOT_HOST, help=f"Robot IP (default: {ROBOT_HOST})")

    return p.parse_args()


def main():
    args = parse_args()

    # Validate joints
    for j in args.joints:
        if j < 0 or j > 5:
            print(f"ERROR: joint index {j} out of range [0, 5]")
            sys.exit(1)

    print("=" * 60)
    print("  Impedance Controller Gain Tuner – UR5e")
    print("=" * 60)
    print(f"  Joints:        {[f'{j} ({JOINT_NAMES[j]})' for j in args.joints]}")
    if args.chirp:
        print(f"  Mode:          Chirp {args.freq_start:.2f} → {args.freq_end:.2f} Hz")
    else:
        print(f"  Mode:          Sine @ {args.freq:.2f} Hz")
    print(f"  Amplitude:     {args.amplitude:.3f} rad ({np.degrees(args.amplitude):.1f}°)")
    print(f"  Duration:      {args.duration:.1f} s")
    print(f"  Command rate:  {args.rate:.0f} Hz")
    print(f"  Velocity FF:   {'ON' if args.with_velocity_ff else 'OFF'}")
    print(f"  Robot IP:      {args.robot_ip}")
    print("=" * 60)
    print()

    input("Press ENTER to start (ensure robot is in remote-control mode)…")
    run_tuner(args)


if __name__ == "__main__":
    main()
