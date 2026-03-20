#!/usr/bin/env python3
"""
Impedance Controller Gain Tuner for UR5e via RTDE
==================================================

Real-robot counterpart of Isaac Sim's Gain Tuner.

This script sends sinusoidal or chirp joint commands through RTDE to a UR5e
running a compatible impedance URScript, records commanded and observed joint
positions/velocities, and exports a CSV using the same schema as
`scripts/utils/sim_gain_tuner_logger.py` so it can be read directly by
`scripts/utils/plot_sim_gain_tuner_csv.py`.

Workflow:
    1. Connect to the robot via RTDE.
    2. Upload `impedance_control_test.script` by default.
    3. Hold the current joint configuration before excitation.
    4. Run the requested excitation for `--duration` seconds at `--rate` Hz.
    5. Hold the centre position after excitation, stop the robot, export CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Optional

import numpy as np

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RTDE_CONFIG_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "rtde_input_v3.xml")
URSCRIPT_FILE = str(REPO_ROOT / "scripts" / "sim2real" / "URscript" / "impedance_control_test.script")
LOG_DIR = REPO_ROOT / "logs" / "impedance_tuner"

ROBOT_HOST = "192.168.1.101"
ROBOT_PORT = 30004
ROBOT_PRIMARY_PORT = 30001

HOME_Q = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0], dtype=np.float64)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
CSV_JOINT_NAMES = [f"{name}_joint" for name in JOINT_NAMES]
PRE_HOLD_DURATION_S = 2.0
POST_HOLD_DURATION_S = 1.0


class RTDELink:
    """Minimal RTDE connection: sends `q_des` + `qdot_des`, reads `actual_q`/`actual_qd`."""

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
        self.setp: Any = None
        self.setp_vel: Any = None
        self.stop_reg: Any = None
        self.control_rate_reg: Any = None
        self.connected = False

        self._lock = Lock()
        self._q: Optional[np.ndarray] = None
        self._qd: Optional[np.ndarray] = None
        self._reader_running = False
        self._reader_thread: Optional[Thread] = None

    def connect(self) -> None:
        conf = rtde_config.ConfigFile(self.config_file)
        out_names, out_types = conf.get_recipe("out")
        q_names, q_types = conf.get_recipe("q_des")
        qd_names, qd_types = conf.get_recipe("qdot_des")
        stop_name, stop_type = conf.get_recipe("stop")
        cr_name, cr_type = conf.get_recipe("control_rate_info")

        self.con = rtde.RTDE(self.robot_host, self.robot_port)
        self.con.connect()

        if not self.con.send_output_setup(out_names, out_types, frequency=int(self.rtde_frequency)):
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

    def send_urscript(self) -> None:
        if self.con is None or self.stop_reg is None or self.control_rate_reg is None:
            raise RuntimeError("RTDE link is not fully initialised")

        self.stop_reg.input_bit_register_64 = False
        self.con.send(self.stop_reg)
        self.control_rate_reg.input_int_register_36 = int(self.control_rate)
        self.con.send(self.control_rate_reg)
        self._write_q(HOME_Q.tolist())
        self._write_qd([0.0] * 6)

        print(f"[RTDE] Uploading URScript: {self.urscript_file}")
        with open(self.urscript_file, "r") as file_handle:
            program = file_handle.read()
        socket_handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_handle.connect((self.robot_host, self.primary_port))
        socket_handle.sendall((program + "\n").encode("utf-8"))
        socket_handle.close()
        time.sleep(3.0)
        print("[RTDE] URScript running")

    def _write_q(self, q: list[float]) -> None:
        if self.con is None or self.setp is None:
            raise RuntimeError("RTDE position register is not initialised")
        for i in range(6):
            self.setp.__dict__[f"input_double_register_{24 + i}"] = float(q[i])
        self.con.send(self.setp)

    def _write_qd(self, qd: list[float]) -> None:
        if self.con is None or self.setp_vel is None:
            raise RuntimeError("RTDE velocity register is not initialised")
        for i in range(6):
            self.setp_vel.__dict__[f"input_double_register_{30 + i}"] = float(qd[i])
        self.con.send(self.setp_vel)

    def send_targets(self, q_des: np.ndarray, qdot_des: np.ndarray) -> None:
        self._write_q(q_des.tolist())
        self._write_qd(qdot_des.tolist())

    def start_reader(self) -> None:
        if self._reader_thread is not None:
            return
        self._reader_running = True
        self._reader_thread = Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        print(f"[RTDE] Reader thread running at {self.rtde_frequency} Hz")

    def stop_reader(self) -> None:
        self._reader_running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

    def _reader_loop(self) -> None:
        period = 1.0 / self.rtde_frequency
        while self._reader_running and self.connected:
            t0 = time.monotonic()
            try:
                if self.con is None:
                    break
                state = self.con.receive()
                if state is not None:
                    with self._lock:
                        self._q = np.array(state.actual_q, dtype=np.float64)
                        self._qd = np.array(state.actual_qd, dtype=np.float64)
            except Exception:
                pass
            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def get_state(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if self._q is None or self._qd is None:
                return None, None
            return self._q.copy(), self._qd.copy()

    def stop_robot(self) -> None:
        if not self.connected:
            return
        try:
            if self.con is None or self.stop_reg is None:
                raise RuntimeError("RTDE stop register is not initialised")
            self._write_qd([0.0] * 6)
            self.stop_reg.input_bit_register_64 = True
            self.con.send(self.stop_reg)
            time.sleep(2.0)
            print("[RTDE] Stop signal sent – robot returning to home")
        except Exception as exc:
            print(f"[RTDE] Stop error: {exc}")

    def disconnect(self) -> None:
        self.stop_reader()
        if self.con is not None:
            try:
                self.con.send_pause()
                self.con.disconnect()
            except Exception:
                pass
        self.connected = False
        print("[RTDE] Disconnected")


def sinusoid(t: float, amplitude: float, freq: float, offset: float, phase: float = 0.0) -> tuple[float, float]:
    pos = offset + amplitude * math.sin(2.0 * math.pi * freq * t + phase)
    vel = amplitude * 2.0 * math.pi * freq * math.cos(2.0 * math.pi * freq * t + phase)
    return pos, vel


def chirp(t: float, amplitude: float, f_start: float, f_end: float, duration: float, offset: float) -> tuple[float, float]:
    k = (f_end - f_start) / duration
    inst_freq = f_start + k * t
    phase = 2.0 * math.pi * (f_start * t + 0.5 * k * t * t)
    pos = offset + amplitude * math.sin(phase)
    vel = amplitude * 2.0 * math.pi * inst_freq * math.cos(phase)
    return pos, vel


def build_csv_header(joints: list[int]) -> list[str]:
    header = ["time"]
    for joint_idx in joints:
        name = CSV_JOINT_NAMES[joint_idx]
        header.extend([f"{name}_pos_cmd", f"{name}_pos_obs", f"{name}_vel_cmd", f"{name}_vel_obs"])
    return header


def build_csv_row(
    time_s: float,
    q_des: np.ndarray,
    q_act: np.ndarray,
    qd_des: np.ndarray,
    qd_act: np.ndarray,
    joints: list[int],
) -> list[float]:
    row = [float(time_s)]
    for joint_idx in joints:
        row.extend([
            float(q_des[joint_idx]),
            float(q_act[joint_idx]),
            float(qd_des[joint_idx]),
            float(qd_act[joint_idx]),
        ])
    return row


def read_state_or_nan(link: RTDELink) -> tuple[np.ndarray, np.ndarray]:
    q_act, qd_act = link.get_state()
    if q_act is None or qd_act is None:
        return np.full(6, np.nan, dtype=np.float64), np.full(6, np.nan, dtype=np.float64)
    return q_act, qd_act


def append_sample(
    rows: list[list[float]],
    start_time: float,
    link: RTDELink,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    joints: list[int],
) -> None:
    q_act, qd_act = read_state_or_nan(link)
    rows.append(build_csv_row(time.monotonic() - start_time, q_des, q_act, qd_des, qd_act, joints))


def build_log_basename(args: argparse.Namespace, joints: list[int], stamp: str) -> str:
    mode = "chirp" if args.chirp else f"sine_{args.freq}Hz"
    vel_tag = "velFF" if args.vel_ff else "noFF"
    joints_tag = "_".join(str(j) for j in joints)
    return f"{stamp}_tuner_j{joints_tag}_{mode}_{vel_tag}_gain_tuner_plot"


def export_csv(rows: list[list[float]], joints: list[int], args: argparse.Namespace) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = LOG_DIR / f"{build_log_basename(args, joints, stamp)}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_csv_header(joints))
        writer.writerows(rows)
    return csv_path


def run_tuner(args: argparse.Namespace) -> None:
    link = RTDELink(robot_host=args.robot_ip, urscript_file=args.urscript_file, rtde_frequency=125.0, control_rate=args.rate)

    try:
        link.connect()
        link.send_urscript()
        link.start_reader()

        print("[Tuner] Waiting for robot state…")
        q_cur: Optional[np.ndarray] = None
        for _ in range(200):
            q_cur, _ = link.get_state()
            if q_cur is not None:
                break
            time.sleep(0.02)
        if q_cur is None:
            print("[Tuner] ERROR: no robot state received – aborting.")
            return

        print(f"[Tuner] Current q: {np.round(q_cur, 4).tolist()}")

        q_centre = q_cur.copy()
        excited_joints = args.joints
        dt = 1.0 / args.rate
        log_rows: list[list[float]] = []
        log_start_time = time.monotonic()

        print(f"[Tuner] Holding current position for {PRE_HOLD_DURATION_S:.1f} s …")
        for _ in range(int(PRE_HOLD_DURATION_S * args.rate)):
            loop_t0 = time.monotonic()
            q_des = q_centre.copy()
            qd_des = np.zeros(6, dtype=np.float64)
            link.send_targets(q_des, qd_des)
            append_sample(log_rows, log_start_time, link, q_des, qd_des, excited_joints)
            remaining = dt - (time.monotonic() - loop_t0)
            if remaining > 0:
                time.sleep(remaining)

        duration = args.duration
        n_steps = int(duration * args.rate)
        total_steps = n_steps * len(excited_joints)
        print(
            f"[Tuner] Starting sequential excitation: joints={excited_joints}, "
            f"{'chirp' if args.chirp else f'freq={args.freq} Hz'}, amp={args.amplitude} rad, "
            f"duration={duration} s per joint, rate={args.rate} Hz, velocity_ff={'ON' if args.vel_ff else 'OFF'}"
        )

        excitation_start_time = time.monotonic()
        for active_joint in excited_joints:
            print(f"[Tuner] Exciting joint {active_joint} ({JOINT_NAMES[active_joint]}) …")
            joint_start_time = time.monotonic()
            for _ in range(n_steps):
                loop_t0 = time.monotonic()
                t = loop_t0 - joint_start_time

                q_des = q_centre.copy()
                qd_des = np.zeros(6, dtype=np.float64)
                offset = float(q_centre[active_joint])
                if args.chirp:
                    pos, vel = chirp(t, args.amplitude, args.freq_start, args.freq_end, duration, offset)
                else:
                    pos, vel = sinusoid(t, args.amplitude, args.freq, offset)
                q_des[active_joint] = pos
                if args.vel_ff:
                    qd_des[active_joint] = vel

                link.send_targets(q_des, qd_des)
                append_sample(log_rows, log_start_time, link, q_des, qd_des, excited_joints)
                remaining = dt - (time.monotonic() - loop_t0)
                if remaining > 0:
                    time.sleep(remaining)

        actual_duration = time.monotonic() - excitation_start_time
        print(
            f"[Tuner] Excitation done: {total_steps} steps in {actual_duration:.2f} s "
            f"(target {duration * len(excited_joints):.2f} s)"
        )

        print(f"[Tuner] Holding centre position for {POST_HOLD_DURATION_S:.1f} s …")
        for _ in range(int(POST_HOLD_DURATION_S * args.rate)):
            loop_t0 = time.monotonic()
            q_des = q_centre.copy()
            qd_des = np.zeros(6, dtype=np.float64)
            link.send_targets(q_des, qd_des)
            append_sample(log_rows, log_start_time, link, q_des, qd_des, excited_joints)
            remaining = dt - (time.monotonic() - loop_t0)
            if remaining > 0:
                time.sleep(remaining)

        link.stop_robot()

        csv_path = export_csv(log_rows, excited_joints, args)
        print(f"[Tuner] CSV saved:    {csv_path}")
        print("[Tuner] Use scripts/utils/plot_sim_gain_tuner_csv.py --file <csv_path> to generate the plots.")

    except KeyboardInterrupt:
        print("\n[Tuner] Interrupted – stopping robot…")
        link.stop_robot()
    finally:
        link.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Impedance-controller gain tuner for UR5e (sinusoidal excitation via RTDE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--joints", type=int, nargs="+", default=[1], help="Joint indices to excite (0-5). Default: 1.")
    parser.add_argument("--freq", type=float, default=0.5, help="Sine frequency [Hz] (default: 0.5)")
    parser.add_argument("--amplitude", type=float, default=0.3, help="Sine amplitude [rad] (default: 0.3)")
    parser.add_argument("--duration", type=float, default=10.0, help="Excitation duration [s] (default: 10)")
    parser.add_argument("--chirp", action="store_true", help="Use linear frequency sweep instead of fixed sine")
    parser.add_argument("--freq-start", type=float, default=0.1, help="Chirp start frequency [Hz]")
    parser.add_argument("--freq-end", type=float, default=2.0, help="Chirp end frequency [Hz]")
    parser.add_argument("--vel-ff", action="store_true", help="Send analytic velocity derivative as qdot_des")
    parser.add_argument("--rate", type=float, default=60.0, help="Command rate [Hz] (default: 60)")
    parser.add_argument("--robot-ip", type=str, default=ROBOT_HOST, help=f"Robot IP (default: {ROBOT_HOST})")
    parser.add_argument("--urscript-file", type=str, default=URSCRIPT_FILE, help=f"URScript file (default: {URSCRIPT_FILE})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    print(f"  Velocity FF:   {'ON' if args.vel_ff else 'OFF'}")
    print(f"  Robot IP:      {args.robot_ip}")
    print(f"  URScript:      {args.urscript_file}")
    print("=" * 60)
    print()

    input("Press ENTER to start (ensure robot is in remote-control mode)…")
    run_tuner(args)


if __name__ == "__main__":
    main()
