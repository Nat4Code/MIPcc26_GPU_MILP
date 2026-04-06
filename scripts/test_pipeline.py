#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) != 3:
        print("Usage: test_pipeline.py <config.json> <instance.mps>")
        sys.exit(1)

    config_path = sys.argv[1]
    instance_path = sys.argv[2]

    root = Path(".")
    tmp = root / "tmp"
    results = root / "results"
    if tmp.exists():
        shutil.rmtree(tmp)
    if results.exists():
        shutil.rmtree(results)
    tmp.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    feats = tmp / "features.json"
    plan = tmp / "plan.json"
    summary = tmp / "summary.json"

    run(["python3", "scripts/feature_extract.py", instance_path, str(feats)])
    run(["python3", "scripts/make_plan.py", config_path, str(feats), instance_path, str(plan)])

    with open(plan, "r") as f:
        p = json.load(f)

    for task in p["tasks"]:
        run(["python3", "scripts/run_task.py", str(plan), str(task["task_id"])])

    run(["python3", "scripts/merge_results.py", "results", str(summary)])


if __name__ == "__main__":
    main()
