#!/usr/bin/env python3
import glob
import json
import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: merge_results.py <results_dir> <summary.json>")
        sys.exit(1)

    results_dir = sys.argv[1]
    out_path = sys.argv[2]

    files = sorted(glob.glob(os.path.join(results_dir, "task_*.json")))
    all_results = []
    best = None

    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)
        all_results.append(data)

        for cand in data.get("candidates", []):
            if not cand.get("feasible", False):
                continue
            obj = cand.get("objective")
            if obj is None:
                continue
            if best is None or obj < best["objective"]:
                best = {
                    "objective": obj,
                    "method": cand.get("method"),
                    "params": cand.get("params"),
                    "task_id": data.get("task_id")
                }

    summary = {
        "num_task_files": len(files),
        "best_candidate": best,
        "tasks": [
            {
                "task_id": r.get("task_id"),
                "method": r.get("method"),
                "status": r.get("status"),
                "elapsed_sec": r.get("elapsed_sec")
            }
            for r in all_results
        ]
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()