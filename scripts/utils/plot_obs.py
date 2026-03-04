#!/usr/bin/env python3
"""
plot_obs.py
-----------

Read the latest `logs/obs_*.bin` (or use `--file`) and plot histograms
with overlaid normal PDFs. Configuration (obs_size, action_size) is read from
the accompanying JSON metadata file.

Usage:
    python3 scripts/plot_obs.py [--file path/to/file.bin] [--save out.png] [--no-show]
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import math
import sys

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib is required to run this script. Install with: pip install matplotlib")
    raise


DEFAULT_LOG_DIR = Path("logs/obs")


def find_latest_file(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    files = sorted(log_dir.glob("*.bin"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_metadata(bin_path: Path) -> dict:
    """Load metadata JSON adjacent to bin file."""
    meta_path = bin_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def build_labels(obs_size: int, action_size: int) -> list[str]:
    """Generate generic labels for observations and actions."""
    labels = []
    for i in range(obs_size):
        labels.append(f"obs_{i}")
    for i in range(action_size):
        labels.append(f"action_{i}")
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

    # Load metadata
    try:
        meta = load_metadata(filepath)
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        sys.exit(1)

    obs_size = meta.get("obs_size")
    action_size = meta.get("action_size")
    sample_size = obs_size + action_size

    # Load data
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size == 0 or data.size % sample_size != 0:
        print(f"File {filepath} has unexpected size: {data.size} (expected multiple of {sample_size})")
        sys.exit(1)
    data = data.reshape(-1, sample_size)

    labels = build_labels(obs_size, action_size)

    # Calculate grid
    ncols = 6
    nrows = (sample_size + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))
    axes = axes.flatten()

    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= sample_size:
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

        # Ensure ticks -1 and +1 are always shown on x-axis
        xmin, xmax = ax.get_xlim()
        xmin = min(xmin, -1.0)
        xmax = max(xmax, 1.0)
        ax.set_xlim(xmin, xmax)
        ticks = np.unique(np.concatenate((ax.get_xticks(), [-1.0, 1.0])))
        ax.set_xticks(ticks)

    fig.suptitle(f"Obs+Action distributions — file: {filepath.name}\nobs_size={obs_size}, action_size={action_size}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if args.save is not None:
        fig.savefig(args.save)
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
