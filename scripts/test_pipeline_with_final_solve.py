#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("instance_path")
    ap.add_argument("--work-dir", default="results/test_run")
    ap.add_argument("--farm-seconds", type=float, default=10.0)
    ap.add_argument("--final-seconds", type=float, default=50.0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    features_json = work / "features.json"
    plan_json = work / "plan.json"
    results_dir = work / "farm_results"
    merged_json = work / "merged.json"
    final_json = work / "final_gurobi.json"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Match CURRENT CLI:
    # feature_extract.py <instance.mps> <features.json>
    run([
        sys.executable,
        "scripts/feature_extract.py",
        args.instance_path,
        str(features_json),
    ])

    # Match CURRENT CLI:
    # make_plan.py <config.json> <features.json> <instance.mps> <plan.json>
    run([
        sys.executable,
        "scripts/make_plan.py",
        args.config_path,
        str(features_json),
        args.instance_path,
        str(plan_json),
    ])

    plan = load_json(plan_json)
    tasks = plan["tasks"]
    per_task_time = max(0.25, float(args.farm_seconds) / max(1, len(tasks)))

    # This assumes your patched run_task.py supports:
    # run_task.py <instance_path> <plan_json> <task_id> --out <out.json> --time-limit <sec>
    #
    # If your local run_task.py still uses the OLD placeholder interface,
    # then this will be the next place that fails, and we should patch that file next.
    for task in tasks:
        task_id = int(task["task_id"])
        out_file = results_dir / f"task_{task_id:03d}.json"
        run([
            sys.executable,
            "scripts/run_task.py",
            args.instance_path,
            str(plan_json),
            str(task_id),
            "--out",
            str(out_file),
            "--time-limit",
            str(per_task_time),
        ])

    # Match CURRENT CLI:
    # merge_results.py <results_dir> <summary.json>
    run([
        sys.executable,
        "scripts/merge_results.py",
        str(results_dir),
        "--out",
        str(merged_json),
    ])

    # final_gurobi_solve.py uses argparse-style flags from the version I gave you.
    run([
        sys.executable,
        "scripts/final_gurobi_solve.py",
        args.instance_path,
        str(merged_json),
        "--out",
        str(final_json),
        "--time-limit",
        str(args.final_seconds),
        "--threads",
        str(args.threads),
        "--seed",
        "0",
        "--mip-focus",
        "1",
        "--heuristics",
        "0.05",
        "--start-node-limit",
        "500",
        "--start-time-limit",
        "2.0",
        "--log-to-console",
    ])

    print("\nDone.")
    print(f"Features:      {features_json}")
    print(f"Plan:          {plan_json}")
    print(f"Farm results:  {results_dir}")
    print(f"Merged:        {merged_json}")
    print(f"Final solve:   {final_json}")


if __name__ == "__main__":
    raise SystemExit(main())