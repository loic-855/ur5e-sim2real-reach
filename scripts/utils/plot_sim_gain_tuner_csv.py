#!/usr/bin/env python3
"""
Plot joint tracking performance from a gain-tuner CSV export.

For each of the 6 UR5e arm joints, the script:
- extracts the 10 s excitation window from the full sequential run,
- adds 0.5 s of context before and after that excitation window,
- reconstructs a local time axis from the CSV sampling period,
- plots commanded vs observed joint position in degrees,
- plots the tracking error in degrees with RMS error in the title,
- saves one figure per joint and displays them.

Usage:
    python3 scripts/utils/plot_sim_gain_tuner_csv.py --file logs/sim_gain_tuner/run.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
except Exception:
    print("matplotlib is required to run this script. Install with: pip install matplotlib")
    raise


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
WINDOW_DURATION_S = 10.0
WINDOW_PADDING_S = 0.5
DISPLAY_WINDOW_S = WINDOW_DURATION_S + 2.0 * WINDOW_PADDING_S
GRID_STEP_S = 2.0
POSITION_LIMIT_DEG = 20.0
ERROR_LIMIT_DEG = 5.0
RAD_TO_DEG = 180.0 / math.pi
ACTIVITY_POS_WEIGHT = 1.0
ACTIVITY_VEL_WEIGHT = 0.2
ACTIVITY_SMOOTHING_S = 0.2
ACTIVITY_THRESHOLD_RATIO = 0.15
ACTIVITY_THRESHOLD_MIN = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gain-tuner CSV tracking data for the 6 UR5e joints.")
    parser.add_argument("--file", type=Path, required=True, help="Path to the gain-tuner CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG figures will be saved. Defaults to <csv_stem>_plots next to the CSV.",
    )
    return parser.parse_args()


def get_csv_base_name(csv_path: Path) -> str:
    return csv_path.stem


def get_default_output_dir(csv_path: Path) -> Path:
    return csv_path.parent / f"{get_csv_base_name(csv_path)}_plots"


def load_csv_columns(file_path: Path) -> dict[str, np.ndarray]:
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with file_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV file has no header: {file_path}")

        columns: dict[str, list[float]] = {field: [] for field in fieldnames}
        for row_idx, row in enumerate(reader, start=2):
            for field in fieldnames:
                raw_value = row.get(field)
                if raw_value is None or raw_value == "":
                    raise ValueError(f"Missing value for column '{field}' at CSV line {row_idx}")
                try:
                    columns[field].append(float(raw_value))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid float for column '{field}' at CSV line {row_idx}: {raw_value}"
                    ) from exc

    numpy_columns = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    if not numpy_columns[fieldnames[0]].size:
        raise ValueError(f"CSV file contains no data rows: {file_path}")
    return numpy_columns


def infer_sample_period(time_values: np.ndarray) -> float:
    if time_values.ndim != 1 or time_values.size < 2:
        raise ValueError("The time column must contain at least two samples.")

    diffs = np.diff(time_values)
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size == 0:
        raise ValueError("Unable to infer a positive sampling period from the time column.")

    return float(np.median(positive_diffs))


def clean_joint_name(joint_name: str) -> str:
    return joint_name.removesuffix("_joint")


def require_joint_columns(columns: dict[str, np.ndarray], joint_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_columns = [
        f"{joint_name}_pos_cmd",
        f"{joint_name}_pos_obs",
        f"{joint_name}_vel_cmd",
        f"{joint_name}_vel_obs",
    ]
    missing = [column_name for column_name in required_columns if column_name not in columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing required columns for joint '{joint_name}': {missing_str}")

    return (
        columns[required_columns[0]],
        columns[required_columns[1]],
        columns[required_columns[2]],
        columns[required_columns[3]],
    )


def compute_activity_metric(
    cmd_pos_rad: np.ndarray,
    obs_pos_rad: np.ndarray,
    cmd_vel_rad_s: np.ndarray,
    obs_vel_rad_s: np.ndarray,
) -> np.ndarray:
    cmd_center = float(np.median(cmd_pos_rad))
    obs_center = float(np.median(obs_pos_rad))
    pos_activity = np.abs(cmd_pos_rad - cmd_center) + 0.25 * np.abs(obs_pos_rad - obs_center)
    vel_activity = np.abs(cmd_vel_rad_s) + 0.25 * np.abs(obs_vel_rad_s)
    return ACTIVITY_POS_WEIGHT * pos_activity + ACTIVITY_VEL_WEIGHT * vel_activity


def find_best_window_start(activity_metric: np.ndarray, window_size: int) -> int:
    if activity_metric.ndim != 1 or activity_metric.size == 0:
        raise ValueError("Activity metric must be a non-empty 1D array.")
    if window_size <= 0:
        raise ValueError("Window size must be strictly positive.")
    if activity_metric.size <= window_size:
        return 0

    kernel = np.ones(window_size, dtype=np.float64)
    window_scores = np.convolve(activity_metric, kernel, mode="valid")
    return int(np.argmax(window_scores))


def smooth_activity(activity_metric: np.ndarray, dt: float) -> np.ndarray:
    smoothing_samples = max(3, int(round(ACTIVITY_SMOOTHING_S / dt)))
    kernel = np.ones(smoothing_samples, dtype=np.float64) / smoothing_samples
    return np.convolve(activity_metric, kernel, mode="same")


def find_excitation_start(activity_metric: np.ndarray, dt: float, window_size: int) -> int:
    smoothed_activity = smooth_activity(activity_metric, dt)
    peak_activity = float(np.max(smoothed_activity))
    if peak_activity <= 0.0:
        return find_best_window_start(activity_metric, window_size)

    threshold = max(ACTIVITY_THRESHOLD_RATIO * peak_activity, ACTIVITY_THRESHOLD_MIN)
    active_mask = smoothed_activity >= threshold
    active_indices = np.flatnonzero(active_mask)

    if active_indices.size == 0:
        return find_best_window_start(activity_metric, window_size)

    split_points = np.where(np.diff(active_indices) > 1)[0] + 1
    segments = np.split(active_indices, split_points)
    best_segment = max(segments, key=lambda seg: float(np.sum(smoothed_activity[seg])))
    return int(best_segment[0])


def extract_joint_window(
    cmd_pos_rad: np.ndarray,
    obs_pos_rad: np.ndarray,
    cmd_vel_rad_s: np.ndarray,
    obs_vel_rad_s: np.ndarray,
    dt: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_samples = cmd_pos_rad.size
    samples_per_window = max(2, int(round(duration_s / dt)))
    samples_per_window = min(samples_per_window, total_samples)
    padding_samples = max(1, int(round(WINDOW_PADDING_S / dt)))

    activity_metric = compute_activity_metric(cmd_pos_rad, obs_pos_rad, cmd_vel_rad_s, obs_vel_rad_s)
    start_idx = find_excitation_start(activity_metric, dt, samples_per_window)
    end_idx = min(total_samples, start_idx + samples_per_window)

    padded_start_idx = max(0, start_idx - padding_samples)
    padded_end_idx = min(total_samples, end_idx + padding_samples)

    pre_padding_samples = start_idx - padded_start_idx

    cmd_window_deg = cmd_pos_rad[padded_start_idx:padded_end_idx] * RAD_TO_DEG
    obs_window_deg = obs_pos_rad[padded_start_idx:padded_end_idx] * RAD_TO_DEG
    local_time_s = (np.arange(padded_end_idx - padded_start_idx, dtype=np.float64) - pre_padding_samples) * dt
    return local_time_s, cmd_window_deg, obs_window_deg


def make_joint_figure(
    joint_index: int,
    joint_name: str,
    local_time_s: np.ndarray,
    cmd_deg: np.ndarray,
    obs_deg: np.ndarray,
    output_dir: Path,
    csv_stem: str,
) -> Path:
    error_deg = cmd_deg - obs_deg
    rms_error_deg = float(np.sqrt(np.mean(np.square(error_deg))))
    display_name = clean_joint_name(joint_name)
    position_reference_deg = float(np.median(cmd_deg))
    cmd_plot_deg = cmd_deg - position_reference_deg
    obs_plot_deg = obs_deg - position_reference_deg

    fig, (ax_pos, ax_err) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    ax_pos.plot(local_time_s, cmd_plot_deg, label="Command position", linewidth=2.0)
    ax_pos.plot(local_time_s, obs_plot_deg, label="Observed position", linewidth=1.8)
    ax_pos.set_title(f"Simulation: joint {joint_index + 1} - {display_name}")
    ax_pos.set_ylabel("Position [deg]")
    ax_pos.set_xlim(-WINDOW_PADDING_S, WINDOW_DURATION_S + WINDOW_PADDING_S)
    ax_pos.set_ylim(-POSITION_LIMIT_DEG, POSITION_LIMIT_DEG)
    ax_pos.legend(loc="upper right")

    ax_err.plot(local_time_s, error_deg, color="tab:red", linewidth=1.8)
    ax_err.set_title(f"RMS error: {rms_error_deg:.2f}°")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [deg]")
    ax_err.set_xlim(-WINDOW_PADDING_S, WINDOW_DURATION_S + WINDOW_PADDING_S)
    ax_err.set_ylim(-ERROR_LIMIT_DEG, ERROR_LIMIT_DEG)

    for axis in (ax_pos, ax_err):
        axis.xaxis.set_major_locator(MultipleLocator(GRID_STEP_S))
        axis.grid(True, which="major", axis="x", linestyle="--", alpha=0.6)

    output_path = output_dir / f"{csv_stem}_joint_{joint_index + 1}_{display_name}.png"
    fig.savefig(str(output_path), dpi=160)
    return output_path


def main() -> None:
    args = parse_args()
    csv_path = args.file.expanduser().resolve()
    csv_base_name = get_csv_base_name(csv_path)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else get_default_output_dir(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        columns = load_csv_columns(csv_path)
        if "time" not in columns:
            raise KeyError("Missing required 'time' column in the CSV file.")
        dt = infer_sample_period(columns["time"])
    except Exception as exc:
        print(f"Failed to load CSV data: {exc}")
        sys.exit(1)

    figure_paths: list[Path] = []
    print(f"Loaded CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Detected sampling period: {dt:.6f} s ({1.0 / dt:.3f} Hz)")
    print(
        f"Using per-joint local window: {WINDOW_DURATION_S:.1f} s + "
        f"{WINDOW_PADDING_S:.1f} s before/after ({DISPLAY_WINDOW_S:.1f} s displayed)"
    )

    for joint_index, joint_name in enumerate(ARM_JOINTS):
        try:
            cmd_pos_rad, obs_pos_rad, cmd_vel_rad_s, obs_vel_rad_s = require_joint_columns(columns, joint_name)
            local_time_s, cmd_deg, obs_deg = extract_joint_window(
                cmd_pos_rad,
                obs_pos_rad,
                cmd_vel_rad_s,
                obs_vel_rad_s,
                dt,
                WINDOW_DURATION_S,
            )
            figure_paths.append(
                make_joint_figure(joint_index, joint_name, local_time_s, cmd_deg, obs_deg, output_dir, csv_base_name)
            )
        except Exception as exc:
            print(f"Skipping joint '{joint_name}': {exc}")

    if not figure_paths:
        print("No figures were generated.")
        sys.exit(1)

    for figure_path in figure_paths:
        print(f"Saved: {figure_path}")

    plt.show()


if __name__ == "__main__":
    main()
