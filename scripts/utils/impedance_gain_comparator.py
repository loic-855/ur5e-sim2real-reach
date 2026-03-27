#!/usr/bin/env python3
"""Compare naive vs tuned impedance gains from benchmark YAML runs.

The script scans a folder for YAML files, groups runs by metadata.model_path,
compares the naive baseline against tuned runs for:

- summary.mean_time_to_area_s
- summary.mean_pos_err_area_m
- summary.mean_rot_err_area_rad

It writes:

- a YAML report with per-model-path and global comparisons
- one PNG plot per metric showing naive vs tuned means by model path

Example:
    python3 scripts/utils/impedance_gain_comparator.py \
        --input-dir logs/benchmarks/sim_pose_real \
        --output-dir logs/benchmarks/sim_pose_real/impedance_gain_comparison
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import error path is user-facing
    raise SystemExit(
        "matplotlib is required to run this script. Install it with: pip install matplotlib"
    ) from exc

try:
    import yaml
except Exception as exc:  # pragma: no cover - import error path is user-facing
    raise SystemExit("PyYAML is required to run this script. Install it with: pip install pyyaml") from exc


METRIC_KEYS = [
    ("mean_time_to_area_s", "Mean time to area [s]"),
    ("mean_pos_err_area_m", "Mean position error in area [m]"),
    ("mean_rot_err_area_rad", "Mean rotation error in area [rad]"),
]


@dataclass
class RunEntry:
    file_path: Path
    model_path: str
    model_key: str
    robot_gain: str
    action_scale: float | None
    metrics: dict[str, float | None]


@dataclass
class GainAggregate:
    runs: list[RunEntry] = field(default_factory=list)

    def add(self, run: RunEntry) -> None:
        self.runs.append(run)

    def count(self) -> int:
        return len(self.runs)

    def action_scales(self) -> list[float]:
        return [run.action_scale for run in self.runs if run.action_scale is not None]

    def metric_values(self, metric_key: str) -> list[float]:
        values: list[float] = []
        for run in self.runs:
            value = run.metrics.get(metric_key)
            if value is not None:
                values.append(value)
        return values

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count(),
            "action_scales": self.action_scales(),
            "metrics": {
                metric_key: summarize_values(self.metric_values(metric_key)) for metric_key, _ in METRIC_KEYS
            },
            "runs": [
                {
                    "file": str(run.file_path),
                    "model_path": run.model_path,
                    "robot_gain": run.robot_gain,
                    "action_scale": run.action_scale,
                    "metrics": run.metrics,
                }
                for run in self.runs
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare naive and tuned impedance gains across benchmark YAML runs.")
    parser.add_argument("input_dir", type=Path, help="Folder containing benchmark YAML files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the comparison report and plots. Defaults to <input_dir>/impedance_gain_comparison.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search for YAML files recursively under the input folder.",
    )
    return parser.parse_args()


def collect_yaml_files(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    patterns = ["*.yaml", "*.yml"]
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))

    return sorted({path.resolve() for path in files})


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    if len(values) == 1:
        value = float(values[0])
        return {"count": 1, "mean": value, "std": 0.0, "min": value, "max": value}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def default_model_label(model_key: str) -> str:
    model = Path(model_key)
    tail = model.parts[-2:] if len(model.parts) >= 2 else model.parts
    label = "/".join(tail) if tail else model_key
    if len(label) <= 42:
        return label
    return "..." + label[-39:]


def prompt_model_labels(grouped_runs: dict[str, dict[str, GainAggregate]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not grouped_runs:
        return labels

    print("\nModel naming prompt")
    print("Enter a display name for each model path. Press Enter to accept the default name.")

    for model_key in sorted(grouped_runs.keys()):
        default_label = default_model_label(model_key)
        print(f"\nModel path: {model_key}")
        print(f"Default label: {default_label}")
        try:
            user_label = input("Display name: ").strip()
        except EOFError:
            user_label = ""

        labels[model_key] = user_label or default_label
        print(f"Using label: {labels[model_key]}")

    return labels


def load_run_entry(file_path: Path) -> RunEntry | None:
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        print(f"[WARN] Skipping unreadable YAML file {file_path}: {exc}")
        return None

    if not isinstance(data, dict):
        print(f"[WARN] Skipping non-mapping YAML file {file_path}")
        return None

    metadata = data.get("metadata") or {}
    summary = data.get("summary") or {}

    model_path = metadata.get("model_path")
    robot_gain = metadata.get("robot_gain")
    if not model_path or not robot_gain:
        print(f"[WARN] Skipping {file_path}: missing metadata.model_path or metadata.robot_gain")
        return None

    if robot_gain not in {"naive", "tuned"}:
        print(f"[WARN] Skipping {file_path}: unsupported robot_gain '{robot_gain}'")
        return None

    model_path_str = str(model_path)
    action_scale = safe_float(metadata.get("action_scale"))

    metrics = {
        metric_key: safe_float(summary.get(metric_key))
        for metric_key, _ in METRIC_KEYS
    }

    return RunEntry(
        file_path=file_path,
        model_path=model_path_str,
        model_key=model_path_str,
        robot_gain=str(robot_gain),
        action_scale=action_scale,
        metrics=metrics,
    )


def group_runs(run_entries: list[RunEntry]) -> dict[str, dict[str, GainAggregate]]:
    grouped: dict[str, dict[str, GainAggregate]] = {}
    for entry in run_entries:
        model_group = grouped.setdefault(entry.model_key, {"naive": GainAggregate(), "tuned": GainAggregate()})
        model_group[entry.robot_gain].add(entry)
    return grouped


def build_comparison(baseline: GainAggregate, tuned: GainAggregate) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "baseline": baseline.summary(),
        "tuned": tuned.summary(),
        "metrics": {},
    }

    for metric_key, _ in METRIC_KEYS:
        baseline_values = baseline.metric_values(metric_key)
        tuned_values = tuned.metric_values(metric_key)
        baseline_summary = summarize_values(baseline_values)
        tuned_summary = summarize_values(tuned_values)

        baseline_mean = baseline_summary["mean"]
        tuned_mean = tuned_summary["mean"]
        delta = None if baseline_mean is None or tuned_mean is None else float(tuned_mean - baseline_mean)
        pct_change = None
        if baseline_mean not in (None, 0.0) and delta is not None:
            pct_change = float((delta / float(baseline_mean)) * 100.0)

        comparison["metrics"][metric_key] = {
            "baseline": baseline_summary,
            "tuned": tuned_summary,
            "delta": delta,
            "pct_change": pct_change,
            "direction": "lower_is_better",
        }

    return comparison


def plot_metric_comparison(
    grouped_runs: dict[str, dict[str, GainAggregate]],
    model_labels: dict[str, str],
    metric_key: str,
    metric_label: str,
    output_dir: Path,
) -> None:
    model_keys = sorted(grouped_runs.keys())
    labels = [model_labels.get(model_key, default_model_label(model_key)) for model_key in model_keys]

    baseline_means = []
    tuned_means = []
    baseline_stds = []
    tuned_stds = []
    x_positions = list(range(len(model_keys)))

    for model_key in model_keys:
        naive_summary = summarize_values(grouped_runs[model_key]["naive"].metric_values(metric_key))
        tuned_summary = summarize_values(grouped_runs[model_key]["tuned"].metric_values(metric_key))
        baseline_means.append(naive_summary["mean"])
        tuned_means.append(tuned_summary["mean"])
        baseline_stds.append(naive_summary["std"])
        tuned_stds.append(tuned_summary["std"])

    if not any(value is not None for value in baseline_means + tuned_means):
        print(f"[WARN] Skipping plot for {metric_key}: no values available")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(10, len(model_keys) * 1.6), 6.5))

    bar_width = 0.36
    naive_x = [position - bar_width / 2 for position in x_positions]
    tuned_x = [position + bar_width / 2 for position in x_positions]

    naive_plot = [value if value is not None else 0.0 for value in baseline_means]
    tuned_plot = [value if value is not None else 0.0 for value in tuned_means]
    naive_err = [value if value is not None else 0.0 for value in baseline_stds]
    tuned_err = [value if value is not None else 0.0 for value in tuned_stds]

    ax.bar(naive_x, naive_plot, width=bar_width, label="naive", color="#4C78A8", yerr=naive_err, capsize=4)
    ax.bar(tuned_x, tuned_plot, width=bar_width, label="tuned", color="#F58518", yerr=tuned_err, capsize=4)

    ax.set_title(f"Naive vs tuned comparison: {metric_label}")
    ax.set_ylabel(metric_label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.margins(x=0.02)
    fig.tight_layout()

    output_path = output_dir / f"{metric_key}_comparison.png"
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved plot: {output_path}")


def build_report(
    grouped_runs: dict[str, dict[str, GainAggregate]],
    model_labels: dict[str, str],
    source_files: list[Path],
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
) -> dict[str, Any]:
    per_model: dict[str, Any] = {}
    all_naive = GainAggregate()
    all_tuned = GainAggregate()

    for model_key in sorted(grouped_runs.keys()):
        naive_runs = grouped_runs[model_key]["naive"]
        tuned_runs = grouped_runs[model_key]["tuned"]
        all_naive.runs.extend(naive_runs.runs)
        all_tuned.runs.extend(tuned_runs.runs)
        per_model[model_key] = {
            "label": model_labels.get(model_key, default_model_label(model_key)),
            "naive": naive_runs.summary(),
            "tuned": tuned_runs.summary(),
            "comparison": build_comparison(naive_runs, tuned_runs)["metrics"],
        }

    global_comparison = build_comparison(all_naive, all_tuned)

    return {
        "metadata": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "recursive": recursive,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_file_count": len(source_files),
            "model_path_count": len(grouped_runs),
        },
        "summary": {
            "naive": all_naive.summary(),
            "tuned": all_tuned.summary(),
            "comparison": global_comparison["metrics"],
        },
        "model_paths": per_model,
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "impedance_gain_comparison")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = collect_yaml_files(input_dir, recursive=bool(args.recursive))
    if not source_files:
        raise SystemExit(f"No YAML files found under {input_dir}")

    run_entries: list[RunEntry] = []
    for file_path in source_files:
        entry = load_run_entry(file_path)
        if entry is not None:
            run_entries.append(entry)

    if not run_entries:
        raise SystemExit("No valid benchmark YAML runs were found.")

    grouped_runs = group_runs(run_entries)
    model_labels = prompt_model_labels(grouped_runs)
    report = build_report(grouped_runs, model_labels, source_files, input_dir, output_dir, bool(args.recursive))

    report_path = output_dir / "impedance_gain_comparison.yaml"
    with report_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, default_flow_style=False)

    print(f"[INFO] Saved report: {report_path}")

    for metric_key, metric_label in METRIC_KEYS:
        plot_metric_comparison(grouped_runs, model_labels, metric_key, metric_label, output_dir)

    print(f"[INFO] Processed {len(run_entries)} valid runs from {len(source_files)} YAML files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())