#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def task_ids(plan, phase):
    return [int(t["task_id"]) for t in plan.get("tasks", []) if str(t.get("phase", "phase1")) == phase]


def run_wave(instance_path, plan_json, ids, results_dir, log_dir, per_task_time, warmstart_json=None):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    for task_id in ids:
        out_file = Path(results_dir) / f"task_{task_id:03d}.json"
        cmd = [sys.executable, "scripts/run_task.py", instance_path, str(plan_json), str(task_id), "--out", str(out_file), "--time-limit", str(per_task_time)]
        if warmstart_json:
            cmd.extend(["--warmstart-json", str(warmstart_json)])
        run(cmd)


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
    phase1_dir = work / "phase1_results"
    phase2_dir = work / "phase2_results"
    merged_phase1_json = work / "merged_phase1.json"
    merged_phase2_json = work / "merged_phase2.json"
    merged_json = work / "merged.json"
    final_json = work / "final_gurobi.json"

    run([sys.executable, "scripts/feature_extract.py", args.instance_path, str(features_json)])
    run([sys.executable, "scripts/make_plan.py", args.config_path, str(features_json), args.instance_path, str(plan_json)])

    plan = load_json(plan_json)
    phase1_ids = task_ids(plan, "phase1")
    phase2_ids = task_ids(plan, "phase2")
    p1_time = max(0.25, float(args.farm_seconds) / max(1, len(phase1_ids)))
    p2_time = max(0.25, float(args.farm_seconds) / max(1, len(phase2_ids)))

    run_wave(args.instance_path, plan_json, phase1_ids, phase1_dir, work / "logs_phase1", p1_time)
    run([sys.executable, "scripts/merge_results.py", str(phase1_dir), "--out", str(merged_phase1_json)])

    if phase2_ids:
        run_wave(args.instance_path, plan_json, phase2_ids, phase2_dir, work / "logs_phase2", p2_time, merged_phase1_json)
        run([sys.executable, "scripts/merge_results.py", str(phase2_dir), "--out", str(merged_phase2_json)])
    else:
        merged_phase2_json.write_text(json.dumps({"status": "ok", "num_task_files": 0, "all_results": []}, indent=2))

    run([sys.executable, "scripts/merge_results.py", str(phase1_dir), str(phase2_dir), "--out", str(merged_json)])
    run([sys.executable, "scripts/final_gurobi_solve.py", args.instance_path, str(merged_json), "--out", str(final_json), "--time-limit", str(args.final_seconds), "--threads", str(args.threads), "--seed", "0", "--mip-focus", "1", "--heuristics", "0.05", "--start-node-limit", "500", "--start-time-limit", "2.0", "--log-to-console"])

    print("\nDone.")
    print(f"Features:      {features_json}")
    print(f"Plan:          {plan_json}")
    print(f"Phase1 merge:  {merged_phase1_json}")
    print(f"Phase2 merge:  {merged_phase2_json}")
    print(f"Merged:        {merged_json}")
    print(f"Final solve:   {final_json}")


if __name__ == "__main__":
    raise SystemExit(main())
