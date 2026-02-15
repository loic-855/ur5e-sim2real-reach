#!/usr/bin/env python3
"""Generate sweep_runs.txt from sweep_config.yaml.

Each output line has the format:
    run_name|hydra_override_1 hydra_override_2 ...

Usage:
    python euler/generate_sweep.py                  # default config
    python euler/generate_sweep.py --config my.yaml # custom config
    python euler/generate_sweep.py --dry-run        # preview only, don't write
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required.  pip install pyyaml")
    sys.exit(1)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_runs(cfg: dict) -> list[tuple[str, str]]:
    """Return a list of (run_name, hydra_overrides_string) from the config."""
    base_overrides: list[str] = cfg.get("base_overrides", [])
    dimensions: dict[str, dict[str, list[str]]] = cfg["dimensions"]

    # Preserve dimension ordering (Python 3.7+ guarantees dict order)
    dim_names = list(dimensions.keys())
    dim_presets: list[list[tuple[str, list[str]]]] = []
    for dim_name in dim_names:
        presets = dimensions[dim_name]
        dim_presets.append(
            [(name, overrides or []) for name, overrides in presets.items()]
        )

    combinations = list(itertools.product(*dim_presets))

    runs: list[tuple[str, str]] = []
    for combo in combinations:
        # Build a descriptive run name from the preset names
        tag_parts = [f"{dim_names[i]}-{combo[i][0]}" for i in range(len(combo))]
        run_name = "_".join(tag_parts)

        # Merge base + per-dimension overrides
        all_overrides = list(base_overrides)
        for _, overrides in combo:
            all_overrides.extend(overrides)

        override_str = " ".join(all_overrides)
        runs.append((run_name, override_str))

    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Generate sweep_runs.txt for Euler sweeps."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "sweep_config.yaml"),
        help="Path to sweep config YAML (default: euler/sweep_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the sweep without writing sweep_runs.txt",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    cfg = load_config(config_path)
    runs = build_runs(cfg)

    task_name = cfg["task_name"]
    seq_per_job = cfg.get("sequential_per_job", 3)
    num_jobs = -(-len(runs) // seq_per_job)  # ceiling division

    # --- Summary ---
    print("=" * 64)
    print("  SWEEP CONFIGURATION SUMMARY")
    print("=" * 64)
    print(f"  Task:             {task_name}")
    print(f"  Total runs:       {len(runs)}")
    print(f"  Sequential/job:   {seq_per_job}")
    print(f"  SLURM array jobs: {num_jobs}  (--array=0-{num_jobs - 1})")
    print(f"  Est. time/run:    ~1h")
    print(f"  Est. wall-time:   ~{seq_per_job}h per GPU")
    print()

    # --- Dimension breakdown ---
    dimensions = cfg["dimensions"]
    for dim_name, presets in dimensions.items():
        preset_names = list(presets.keys())
        print(f"  {dim_name}: {preset_names}")
    print()

    # --- Run list ---
    print("  Runs:")
    for i, (run_name, overrides) in enumerate(runs):
        job_id = i // seq_per_job
        pos_in_job = i % seq_per_job
        print(f"    [{i:2d}] (job {job_id}, run {pos_in_job}) {run_name}")
        if overrides.strip():
            # Show overrides on a wrapped line for readability
            parts = overrides.split()
            for part in parts:
                print(f"           {part}")
        else:
            print(f"           (defaults)")
    print()

    # --- Write output ---
    if args.dry_run:
        print("  [DRY RUN] sweep_runs.txt NOT written.")
    else:
        output_path = config_path.parent / "sweep_runs.txt"
        with open(output_path, "w", newline="\n") as f:
            for run_name, overrides in runs:
                f.write(f"{run_name}|{overrides}\n")
        print(f"  Written: {output_path}")
        print(f"  Launch:  cd <project_root> && bash euler/launch_sweep.sh")

    print("=" * 64)


if __name__ == "__main__":
    main()
