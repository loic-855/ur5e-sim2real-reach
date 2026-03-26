#!/usr/bin/env python3
"""Generate random goals and write them to a JSON file.

Each goal is a list: [x, y, z, qw, qx, qy, qz]

Usage:
  python generate_goals.py output.json         # writes 10 goals
  python generate_goals.py output.json -n 20   # writes 20 goals
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple


def random_quaternion() -> Tuple[float, float, float, float]:
    """Return a uniformly random unit quaternion as (qw, qx, qy, qz)."""
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()

    qx = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    qy = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return qw, qx, qy, qz


def generate_goals(num_goals: int = 10) -> List[List[float]]:
    """Generate a list of goals.

    Position ranges (meters):
      x in [-0.6, 0.4]
      y in [-0.4, 0.4]
      z in [0.03, 0.6]
    Orientation is a random unit quaternion.
    """
    goals: List[List[float]] = []
    for _ in range(num_goals):
        x = random.uniform(-0.6, 0.4)
        y = random.uniform(-0.4, 0.4)
        z = random.uniform(0.03, 0.6)
        qw, qx, qy, qz = random_quaternion()
        goals.append([x, y, z, qw, qx, qy, qz])
    return goals


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random goals and save to JSON.")
    parser.add_argument("--filename", nargs="?", help="Output JSON filename (written into this script's directory)")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of goals to generate (default: 10)")
    args = parser.parse_args()

    if not args.filename:
        parser.error("An output filename must be provided.")

    out_name = args.filename
    if not out_name.lower().endswith(".json"):
        out_name += ".json"

    out_path = Path(__file__).parent / out_name

    goals = generate_goals(num_goals=args.num)

    # Round each component to 2 decimal places and write each goal on a single line
    rounded_goals = [[round(float(x), 2) for x in goal] for goal in goals]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for i, goal in enumerate(rounded_goals):
            line = json.dumps(goal)
            sep = "," if i < len(rounded_goals) - 1 else ""
            f.write(f"  {line}{sep}\n")
        f.write("]\n")

    print(f"Wrote {len(rounded_goals)} goals to: {out_path}")


if __name__ == "__main__":
    main()
