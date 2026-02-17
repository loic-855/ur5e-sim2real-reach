#!/usr/bin/env python3
"""Generate sweep_runs.txt from a YAML sweep config and optionally submit to SLURM.

Output file (sweep_runs.txt) contains a metadata header followed by run lines:
    # META task_name=...
    # META sequential_per_job=...
    run_name|hydra_override_1 hydra_override_2 ...

Usage:
    python euler/generate_sweep.py --config euler/sweep_position_orientation.yaml
    python euler/generate_sweep.py --config euler/sweep_config.yaml --submit
    python euler/generate_sweep.py --config euler/sweep_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from datetime import datetime
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
        tag_parts = [f"{dim_names[i]}-{combo[i][0]}" for i in range(len(combo))]
        run_name = "_".join(tag_parts)

        all_overrides = list(base_overrides)
        for _, overrides in combo:
            all_overrides.extend(overrides)

        override_str = " ".join(all_overrides)
        runs.append((run_name, override_str))

    return runs


def write_sweep_file(
    output_path: Path,
    runs: list[tuple[str, str]],
    cfg: dict,
    config_name: str,
) -> None:
    """Write sweep_runs.txt with metadata header."""
    task_name = cfg["task_name"]
    seq_per_job = cfg.get("sequential_per_job", 3)

    with open(output_path, "w", newline="\n") as f:
        f.write(f"# META task_name={task_name}\n")
        f.write(f"# META sequential_per_job={seq_per_job}\n")
        f.write(f"# META total_runs={len(runs)}\n")
        f.write(f"# META config={config_name}\n")
        f.write(f"# META generated={datetime.now().isoformat(timespec='seconds')}\n")
        for run_name, overrides in runs:
            f.write(f"{run_name}|{overrides}\n")


def submit_sweep(cfg: dict, num_jobs: int, script_dir: Path) -> None:
    """Submit sweep_euler.sh via sbatch with SLURM settings from the YAML."""
    slurm = cfg.get("slurm", {})
    sweep_script = script_dir / "sweep_euler.sh"

    if not sweep_script.exists():
        print(f"  Error: {sweep_script} not found")
        sys.exit(1)

    array_spec = f"0-{num_jobs - 1}" if num_jobs > 1 else "0"

    cmd = ["sbatch", f"--array={array_spec}"]
    if "time" in slurm:
        cmd.append(f"--time={slurm['time']}")
    if "gpus" in slurm:
        cmd.append(f"--gpus={slurm['gpus']}")
    if "cpus_per_task" in slurm:
        cmd.append(f"--cpus-per-task={slurm['cpus_per_task']}")
    if "mem_per_cpu" in slurm:
        cmd.append(f"--mem-per-cpu={slurm['mem_per_cpu']}")
    cmd.append(str(sweep_script))

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  sbatch failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sweep_runs.txt and optionally submit to SLURM."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to sweep config YAML (e.g. euler/sweep_position_orientation.yaml)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the sweep to SLURM via sbatch after generating sweep_runs.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the sweep without writing sweep_runs.txt or submitting",
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
    slurm = cfg.get("slurm", {})

    # --- Summary ---
    print("=" * 64)
    print("  SWEEP SUMMARY")
    print("=" * 64)
    print(f"  Config:           {config_path.name}")
    print(f"  Task:             {task_name}")
    print(f"  Total runs:       {len(runs)}")
    print(f"  Sequential/job:   {seq_per_job}")
    print(f"  SLURM array jobs: {num_jobs}  (--array=0-{num_jobs - 1})")
    if "time" in slurm:
        print(f"  Wall-time/job:    {slurm['time']}")
    if "gpus" in slurm:
        print(f"  GPU:              {slurm['gpus']}")
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
            parts = overrides.split()
            for part in parts:
                print(f"           {part}")
        else:
            print(f"           (defaults)")
    print()

    # --- Write output ---
    if args.dry_run:
        print("  [DRY RUN] No files written, no jobs submitted.")
        print("=" * 64)
        return

    output_path = config_path.parent / "sweep_runs.txt"
    write_sweep_file(output_path, runs, cfg, config_path.name)
    print(f"  Written: {output_path}")

    # --- Submit ---
    if args.submit:
        print()
        print("  Submitting to SLURM...")
        submit_sweep(cfg, num_jobs, config_path.parent)
    else:
        print(
            f"  To submit:  python euler/generate_sweep.py --config {config_path} --submit"
        )

    print()
    print("  Monitor:  squeue -u $USER")
    print("  Logs:     logs/sweep_<JOB_ID>_<ARRAY_ID>.out")
    print("  WandB:    https://wandb.ai  (project: isaaclab_euler)")
    print("=" * 64)


if __name__ == "__main__":
    main()
