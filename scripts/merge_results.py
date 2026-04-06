#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def better(a, b, sense="min"):
    if a is None:
        return False
    if b is None:
        return True
    return a < b if sense == "min" else a > b


def extract_candidate_objective(cand: Dict[str, Any]) -> Optional[float]:
    obj = cand.get("objective")
    if obj is not None:
        try:
            return float(obj)
        except Exception:
            pass

    incumbent = cand.get("incumbent")
    if isinstance(incumbent, dict):
        obj = incumbent.get("objective")
        if obj is not None:
            try:
                return float(obj)
            except Exception:
                pass

    diagnostics = cand.get("diagnostics")
    if isinstance(diagnostics, dict):
        obj = diagnostics.get("incumbent_obj")
        if obj is not None:
            try:
                return float(obj)
            except Exception:
                pass

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sense", choices=["min", "max"], default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "task_*.json")))
    all_results = []

    inferred_sense = None
    best_result = None
    best_obj = None

    for fp in files:
        try:
            data = load_json(fp)
        except Exception as exc:
            all_results.append({
                "file": fp,
                "status": "read_error",
                "error": repr(exc),
            })
            continue

        all_results.append(data)

        if inferred_sense is None:
            inferred_sense = data.get("objective_sense")

        for cand in data.get("candidates", []):
            if not cand.get("feasible", False):
                continue

            obj = extract_candidate_objective(cand)
            if obj is None:
                continue

            sense = args.sense or inferred_sense or "min"

            if better(obj, best_obj, sense):
                best_obj = obj
                best_result = {
                    "objective": obj,
                    "method": cand.get("method"),
                    "params": cand.get("params"),
                    "task_id": data.get("task_id"),
                    "status": cand.get("status"),
                    "runtime_sec": cand.get("runtime_sec"),
                    "incumbent": cand.get("incumbent"),
                    "diagnostics": cand.get("diagnostics"),
                    "notes": cand.get("notes"),
                    "_source_file": fp,
                }

    final_sense = args.sense or inferred_sense or "min"

    summary = {
        "results_dir": args.results_dir,
        "num_task_files": len(files),
        "num_loaded_results": len(all_results),
        "sense": final_sense,
        "best_objective": best_obj,
        "best_candidate": best_result,
        "best_result": best_result,
        "tasks": [
            {
                "task_id": r.get("task_id"),
                "method": r.get("method"),
                "status": r.get("status"),
                "elapsed_sec": r.get("elapsed_sec"),
                "num_candidates": r.get("num_candidates"),
                "num_errors": r.get("num_errors"),
                "objective_sense": r.get("objective_sense"),
                "best_effective_objective": r.get("best_effective_objective"),
            }
            for r in all_results if isinstance(r, dict)
        ],
        "all_results": all_results,
    }

    dump_json(summary, args.out)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()