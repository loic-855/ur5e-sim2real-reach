#!/usr/bin/env python3
"""
plot_obs.py
-----------

Read the latest `logs/sim2real/obs_*.bin` (or use `--file`) and for each of the
36 dimensions (29 obs + 7 action) plot a histogram with an overlaid normal PDF
fitted by mean/std of the recorded samples. Saves or shows the figure.

Usage:
    python3 scripts/sim2real/plot_obs.py [--file path/to/file.bin] [--save out.png] [--no-show]
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import math
import sys

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib is required to run this script. Install with: pip install matplotlib")
    raise


DEFAULT_LOG_DIR = Path("logs/sim2real")
SAMPLE_SIZE = 36  # 29 obs + 7 action


def find_latest_file(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    files = sorted(log_dir.glob("*.bin"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def build_labels() -> list[str]:
    labels = []
    # Observation labels (29D)
    for i in range(8):
        labels.append(f"joint_pos_{i}")
    for i in range(8):
        labels.append(f"joint_vel_{i}")
    labels += ["to_target_x", "to_target_y", "to_target_z"]
    labels += ["goal_pos_x", "goal_pos_y", "goal_pos_z"]
    for i in range(7):
        labels.append(f"last_action_{i}")

    # Appended actions (7D) -> total 36 dims
    for i in range(7):
        labels.append(f"action_{i}")

    assert len(labels) == SAMPLE_SIZE
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None, help="Path to .bin file to read")
    parser.add_argument("--save", type=Path, default=None, help="If provided, save figure to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    args = parser.parse_args()

    filepath = args.file
    if filepath is None:
        filepath = find_latest_file(DEFAULT_LOG_DIR)
        if filepath is None:
            print(f"No log files found in {DEFAULT_LOG_DIR}")
            sys.exit(1)

    data = np.fromfile(filepath, dtype=np.float32)
    if data.size == 0 or data.size % SAMPLE_SIZE != 0:
        print(f"File {filepath} has unexpected size: {data.size}")
        sys.exit(1)
    data = data.reshape(-1, SAMPLE_SIZE)

    labels = build_labels()

    ncols = 6
    nrows = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= SAMPLE_SIZE:
            ax.axis("off")
            continue
        arr = data[:, i]
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        sigma = max(sigma, 1e-6)

        # Histogram
        ax.hist(arr, bins=50, density=True, alpha=0.6, color="C0")

        # Fitted normal PDF
        xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        pdf = (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        ax.plot(xs, pdf, "r-", linewidth=1)

        ax.set_title(labels[i])

    fig.suptitle(f"Obs+Action distributions — file: {filepath.name}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if args.save is not None:
        fig.savefig(args.save)
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
