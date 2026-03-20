#!/usr/bin/env python3
"""
Plot joint tracking performance from gain-tuner CSV exports.

For each of the 6 UR5e arm joints, the script:
- extracts the 10 s excitation window from the full sequential run,
- adds 0.5 s of context before and after that excitation window,
- reconstructs a local time axis from the CSV sampling period,
- plots commanded vs observed joint position in degrees,
- optionally overlays simulation and real-robot CSV exports,
- plots the tracking error in degrees with RMS error in the title,
- saves one figure per joint and displays them.

Usage:
    python3 scripts/utils/plot_sim_gain_tuner_csv.py --file logs/sim_gain_tuner/run.csv
    python3 scripts/utils/plot_sim_gain_tuner_csv.py --file sim.csv --file-real real.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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

plt.rcParams["path.simplify"] = False
plt.rcParams["path.simplify_threshold"] = 0.0


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
WINDOW_DURATION_S = 6.0
WINDOW_PADDING_S = 0.5
DISPLAY_WINDOW_S = WINDOW_DURATION_S + 2.0 * WINDOW_PADDING_S
GRID_STEP_S = 2.0
POSITION_LIMIT_DEG = 20.0
ERROR_LIMIT_DEG = 7.0
RAD_TO_DEG = 180.0 / math.pi
ACTIVITY_POS_WEIGHT = 1.0
ACTIVITY_VEL_WEIGHT = 0.2
ACTIVITY_SMOOTHING_S = 0.2
ACTIVITY_THRESHOLD_RATIO = 0.15
ACTIVITY_THRESHOLD_MIN = 1e-4
SIM_COLOR = "tab:blue"
REAL_COLOR = "tab:red"
DEBUG_COMMAND_COLOR = "black"


@dataclass(frozen=True)
class JointWindow:
    local_time_s: np.ndarray
    cmd_deg: np.ndarray
    obs_deg: np.ndarray
    core_slice: slice


@dataclass(frozen=True)
class DatasetInfo:
    label: str
    csv_path: Path
    columns: dict[str, np.ndarray]
    dt: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gain-tuner CSV tracking data for the 6 UR5e joints.")
    parser.add_argument("--file", type=Path, default=None, help="Path to the simulation gain-tuner CSV file.")
    parser.add_argument(
        "--file-real",
        type=Path,
        default=None,
        help="Optional path to the real-robot gain-tuner CSV file to overlay with the simulation one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG figures will be saved. Defaults to <csv_stem>_plots next to the CSV.",
    )
    parser.add_argument(
        "--plot-real-command-only",
        action="store_true",
        help="In dual-file mode, plot only the real-robot command curve and keep both observed curves.",
    )
    return parser.parse_args()


def get_csv_base_name(csv_path: Path) -> str:
    return csv_path.stem


def get_output_base_name(sim_csv_path: Path | None = None, real_csv_path: Path | None = None) -> str:
    if sim_csv_path is not None and real_csv_path is not None:
        return f"{get_csv_base_name(sim_csv_path)}__vs__{get_csv_base_name(real_csv_path)}"
    if sim_csv_path is not None:
        return get_csv_base_name(sim_csv_path)
    if real_csv_path is not None:
        return get_csv_base_name(real_csv_path)
    raise ValueError("At least one CSV path must be provided.")


def get_default_output_dir(sim_csv_path: Path | None = None, real_csv_path: Path | None = None) -> Path:
    base_path = sim_csv_path if sim_csv_path is not None else real_csv_path
    if base_path is None:
        raise ValueError("At least one CSV path must be provided.")
    return base_path.parent / f"{get_output_base_name(sim_csv_path, real_csv_path)}_plots"


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


def load_dataset(csv_path: Path, label: str) -> DatasetInfo:
    columns = load_csv_columns(csv_path)
    if "time" not in columns:
        raise KeyError(f"Missing required 'time' column in the {label} CSV file.")

    return DatasetInfo(label=label, csv_path=csv_path, columns=columns, dt=infer_sample_period(columns["time"]))


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


def has_joint_columns(columns: dict[str, np.ndarray], joint_name: str) -> bool:
    required_columns = [
        f"{joint_name}_pos_cmd",
        f"{joint_name}_pos_obs",
        f"{joint_name}_vel_cmd",
        f"{joint_name}_vel_obs",
    ]
    return all(column_name in columns for column_name in required_columns)


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


def compute_command_activity_metric(cmd_pos_rad: np.ndarray, cmd_vel_rad_s: np.ndarray) -> np.ndarray:
    cmd_center = float(np.median(cmd_pos_rad))
    pos_activity = np.abs(cmd_pos_rad - cmd_center)
    vel_activity = np.abs(cmd_vel_rad_s)
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
) -> JointWindow:
    total_samples = cmd_pos_rad.size
    samples_per_window = max(2, int(round(duration_s / dt)))
    samples_per_window = min(samples_per_window, total_samples)
    padding_samples = max(1, int(round(WINDOW_PADDING_S / dt)))

    command_activity_metric = compute_command_activity_metric(cmd_pos_rad, cmd_vel_rad_s)
    start_idx = find_excitation_start(command_activity_metric, dt, samples_per_window)
    end_idx = min(total_samples, start_idx + samples_per_window)

    padded_start_idx = max(0, start_idx - padding_samples)
    padded_end_idx = min(total_samples, end_idx + padding_samples)

    pre_padding_samples = start_idx - padded_start_idx

    cmd_window_deg = cmd_pos_rad[padded_start_idx:padded_end_idx] * RAD_TO_DEG
    obs_window_deg = obs_pos_rad[padded_start_idx:padded_end_idx] * RAD_TO_DEG
    local_time_s = (np.arange(padded_end_idx - padded_start_idx, dtype=np.float64) - pre_padding_samples) * dt
    core_start = start_idx - padded_start_idx
    core_end = core_start + (end_idx - start_idx)

    return JointWindow(
        local_time_s=local_time_s,
        cmd_deg=cmd_window_deg,
        obs_deg=obs_window_deg,
        core_slice=slice(core_start, core_end),
    )


def compute_rms_error_deg(window: JointWindow) -> float:
    core_error_deg = window.cmd_deg[window.core_slice] - window.obs_deg[window.core_slice]
    return float(np.sqrt(np.mean(np.square(core_error_deg))))


def build_joint_window(dataset: DatasetInfo, joint_name: str) -> JointWindow:
    cmd_pos_rad, obs_pos_rad, cmd_vel_rad_s, obs_vel_rad_s = require_joint_columns(dataset.columns, joint_name)
    return extract_joint_window(
        cmd_pos_rad,
        obs_pos_rad,
        cmd_vel_rad_s,
        obs_vel_rad_s,
        dataset.dt,
        WINDOW_DURATION_S,
    )


def make_joint_figure(
    joint_index: int,
    joint_name: str,
    sim_window: JointWindow | None,
    output_dir: Path,
    csv_stem: str,
    real_window: JointWindow | None = None,
    plot_real_command_only: bool = False,
) -> Path:
    display_name = clean_joint_name(joint_name)
    if sim_window is None and real_window is None:
        raise ValueError("At least one dataset must be provided for plotting.")

    sim_cmd_plot_deg = None
    sim_obs_plot_deg = None
    sim_error_deg = None
    sim_rms_error_deg = None
    if sim_window is not None:
        sim_position_reference_deg = float(np.median(sim_window.cmd_deg))
        sim_cmd_plot_deg = sim_window.cmd_deg - sim_position_reference_deg
        sim_obs_plot_deg = sim_window.obs_deg - sim_position_reference_deg
        sim_error_deg = sim_window.cmd_deg - sim_window.obs_deg
        sim_rms_error_deg = compute_rms_error_deg(sim_window)

    real_cmd_plot_deg = None
    real_obs_plot_deg = None
    real_error_deg = None
    real_rms_error_deg = None
    if real_window is not None:
        real_position_reference_deg = float(np.median(real_window.cmd_deg))
        real_cmd_plot_deg = real_window.cmd_deg - real_position_reference_deg
        real_obs_plot_deg = real_window.obs_deg - real_position_reference_deg
        real_error_deg = real_window.cmd_deg - real_window.obs_deg
        real_rms_error_deg = compute_rms_error_deg(real_window)

    fig, (ax_pos, ax_err) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    if sim_window is not None and sim_cmd_plot_deg is not None and sim_obs_plot_deg is not None and not (plot_real_command_only and real_window is not None):
        ax_pos.plot(
            sim_window.local_time_s,
            sim_cmd_plot_deg,
            color=SIM_COLOR,
            linestyle="--",
            linewidth=2.0,
            antialiased=False,
            label="Simulation command",
        )
        ax_pos.plot(
            sim_window.local_time_s,
            sim_obs_plot_deg,
            color=SIM_COLOR,
            linestyle="-",
            linewidth=1.8,
            antialiased=False,
            label="Simulation observed",
        )

    if real_window is not None and real_cmd_plot_deg is not None and real_obs_plot_deg is not None:
        ax_pos.plot(
            real_window.local_time_s,
            real_cmd_plot_deg,
            color=DEBUG_COMMAND_COLOR if plot_real_command_only else REAL_COLOR,
            linestyle="--",
            linewidth=2.0,
            antialiased=False,
            label="Command" if plot_real_command_only else "Real command",
        )
        ax_pos.plot(
            real_window.local_time_s,
            real_obs_plot_deg,
            color=REAL_COLOR,
            linestyle="-",
            linewidth=1.8,
            antialiased=False,
            label="Real observed",
        )

    ax_pos.set_title(f"Joint {joint_index + 1} - {display_name}")
    ax_pos.set_ylabel("Position [deg]")
    ax_pos.set_xlim(-WINDOW_PADDING_S, WINDOW_DURATION_S + WINDOW_PADDING_S)
    ax_pos.set_ylim(-POSITION_LIMIT_DEG, POSITION_LIMIT_DEG)
    ax_pos.legend(loc="upper right")

    if sim_window is not None and sim_error_deg is not None:
        ax_err.plot(
            sim_window.local_time_s,
            sim_error_deg,
            color=SIM_COLOR,
            linestyle="-",
            linewidth=1.8,
            antialiased=False,
            label="Simulation error",
        )
    if real_window is not None and real_error_deg is not None:
        ax_err.plot(
            real_window.local_time_s,
            real_error_deg,
            color=REAL_COLOR,
            linestyle="-",
            linewidth=1.8,
            antialiased=False,
            label="Real error",
        )

    if sim_rms_error_deg is not None and real_rms_error_deg is None:
        ax_err.set_title(f"RMS error: sim {sim_rms_error_deg:.2f}°")
    elif sim_rms_error_deg is None and real_rms_error_deg is not None:
        ax_err.set_title(f"RMS error: real {real_rms_error_deg:.2f}°")
    else:
        ax_err.set_title(f"RMS error: sim {sim_rms_error_deg:.2f}° | real {real_rms_error_deg:.2f}°")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [deg]")
    ax_err.set_xlim(-WINDOW_PADDING_S, WINDOW_DURATION_S + WINDOW_PADDING_S)
    ax_err.set_ylim(-ERROR_LIMIT_DEG, ERROR_LIMIT_DEG)
    ax_err.legend(loc="upper right")

    for axis in (ax_pos, ax_err):
        axis.xaxis.set_major_locator(MultipleLocator(GRID_STEP_S))
        axis.grid(True, which="major", axis="x", linestyle="--", alpha=0.6)

    output_path = output_dir / f"{csv_stem}_joint_{joint_index + 1}_{display_name}.png"
    fig.savefig(str(output_path), dpi=160)
    return output_path


def main() -> None:
    args = parse_args()
    sim_csv_path = args.file.expanduser().resolve() if args.file else None
    real_csv_path = args.file_real.expanduser().resolve() if args.file_real else None

    if sim_csv_path is None and real_csv_path is None:
        print("Provide at least one input CSV with --file and/or --file-real.")
        sys.exit(1)

    if args.plot_real_command_only and real_csv_path is None:
        print("The --plot-real-command-only option requires --file-real.")
        sys.exit(1)

    csv_base_name = get_output_base_name(sim_csv_path, real_csv_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else get_default_output_dir(sim_csv_path, real_csv_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sim_dataset = load_dataset(sim_csv_path, "simulation") if sim_csv_path is not None else None
        real_dataset = load_dataset(real_csv_path, "real") if real_csv_path is not None else None
    except Exception as exc:
        print(f"Failed to load CSV data: {exc}")
        sys.exit(1)

    figure_paths: list[Path] = []
    if sim_dataset is not None:
        print(f"Loaded simulation CSV: {sim_dataset.csv_path}")
        print(f"Detected simulation sampling period: {sim_dataset.dt:.6f} s ({1.0 / sim_dataset.dt:.3f} Hz)")
    if real_dataset is not None:
        print(f"Loaded real CSV: {real_dataset.csv_path}")
        print(f"Detected real sampling period: {real_dataset.dt:.6f} s ({1.0 / real_dataset.dt:.3f} Hz)")
    print(f"Output directory: {output_dir}")
    print(
        f"Using per-joint local window: {WINDOW_DURATION_S:.1f} s + "
        f"{WINDOW_PADDING_S:.1f} s before/after ({DISPLAY_WINDOW_S:.1f} s displayed)"
    )

    for joint_index, joint_name in enumerate(ARM_JOINTS):
        try:
            if sim_dataset is not None and real_dataset is not None:
                if not has_joint_columns(sim_dataset.columns, joint_name) or not has_joint_columns(real_dataset.columns, joint_name):
                    print(f"Skipping joint '{joint_name}': missing in one of the two CSV files.")
                    continue
            if sim_dataset is None and real_dataset is not None and not has_joint_columns(real_dataset.columns, joint_name):
                print(f"Skipping joint '{joint_name}': missing in the real CSV file.")
                continue
            if real_dataset is None and sim_dataset is not None and not has_joint_columns(sim_dataset.columns, joint_name):
                print(f"Skipping joint '{joint_name}': missing in the simulation CSV file.")
                continue

            sim_window = build_joint_window(sim_dataset, joint_name) if sim_dataset is not None else None
            real_window = build_joint_window(real_dataset, joint_name) if real_dataset is not None else None
            figure_paths.append(
                make_joint_figure(
                    joint_index,
                    joint_name,
                    sim_window,
                    output_dir,
                    csv_base_name,
                    real_window=real_window,
                    plot_real_command_only=args.plot_real_command_only,
                )
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
