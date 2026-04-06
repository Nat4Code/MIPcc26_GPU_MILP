#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from common import cartesian_grid


def main():
    if len(sys.argv) not in {4, 5}:
        print("Usage: test_heuristic.py <config.json> <instance.mps> <method> [max_runs]")
        sys.exit(1)

    config_path = sys.argv[1]
    instance_path = sys.argv[2]
    method = sys.argv[3]
    max_runs = int(sys.argv[4]) if len(sys.argv) == 5 else 3

    with open(config_path, "r") as f:
        cfg = json.load(f)

    from feature_extract import extract_features
    features = extract_features(instance_path)

    grid = cartesian_grid(cfg["grids"][method])[:max_runs]
    module = importlib.import_module(f"heuristics.{method}")
    run_heuristic = module.run_heuristic

    best = None
    for i, params in enumerate(grid):
        out = run_heuristic(instance_path, params, cfg["global"]["time_limit_sec"], features=features)
        print(f"[{i}] {json.dumps(out, indent=2)}")
        if out["feasible"] and (best is None or out["objective"] < best["objective"]):
            best = out

    print("\nBEST")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
