#!/usr/bin/env python3
"""
scripts/merge_results.py

Robust merge for heuristic result JSON files.

Accepts one or more files/directories and writes a merged incumbent result.

Usage:
  python3 -m scripts.merge_results DIR_OR_FILE [DIR_OR_FILE ...] --out merged.json

This version is deliberately tolerant of result schemas from:
  - make_result(...) outputs
  - lp_seed_solve.py outputs
  - older heuristic outputs
  - nested {"repair": {"incumbent": ...}}
  - nested {"incumbent": {"objective": ..., "values": ...}}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y) or math.isinf(y):
        return None
    return y


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def iter_json_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(x for x in p.glob("*.json") if x.is_file()))
        elif p.is_file() and p.suffix.lower() == ".json":
            files.append(p)
    # de-dupe while preserving order
    seen = set()
    out = []
    for f in files:
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def detect_sense(objs: List[Dict[str, Any]], default: str = "min") -> str:
    for obj in objs:
        for key in ("objective_sense", "sense"):
            v = obj.get(key)
            if isinstance(v, str):
                s = v.lower()
                if s.startswith("max"):
                    return "max"
                if s.startswith("min"):
                    return "min"

        for parent in ("diagnostics", "lp_meta", "model", "features"):
            sub = obj.get(parent)
            if isinstance(sub, dict):
                for key in ("objective_sense", "sense"):
                    v = sub.get(key)
                    if isinstance(v, str):
                        s = v.lower()
                        if s.startswith("max"):
                            return "max"
                        if s.startswith("min"):
                            return "min"

    return default


def better(a: float, b: float, sense: str) -> bool:
    return a > b if sense == "max" else a < b


def better_bound(a: float, b: float, sense: str) -> bool:
    # For minimization, a larger lower bound is tighter. For maximization,
    # a smaller upper bound is tighter.
    return a < b if sense == "max" else a > b


def valid_bound_for_incumbent(bound: float, incumbent: float, sense: str, tol: float = 1e-7) -> bool:
    if sense == "max":
        return bound >= incumbent - tol
    return bound <= incumbent + tol


def gap_from_bound(incumbent: Optional[float], bound: Optional[float], sense: str) -> Tuple[Optional[float], Optional[float]]:
    if incumbent is None or bound is None:
        return None, None
    if not valid_bound_for_incumbent(float(bound), float(incumbent), sense):
        return None, None
    abs_gap = abs(float(incumbent) - float(bound))
    rel_gap = abs_gap / max(1.0, abs(float(incumbent)))
    return abs_gap, rel_gap


def candidate_obj_from_dict(d: Dict[str, Any]) -> Optional[float]:
    for key in (
        "objective",
        "best_objective",
        "best_obj",
        "incumbent_obj",
        "obj",
        "solution_objective",
    ):
        v = safe_float(d.get(key))
        if v is not None:
            return v

    # Nested common structures.
    for parent in ("incumbent", "solution", "best", "best_incumbent"):
        sub = d.get(parent)
        if isinstance(sub, dict):
            v = candidate_obj_from_dict(sub)
            if v is not None:
                return v

    # LP seed repair output.
    repair = d.get("repair")
    if isinstance(repair, dict):
        v = safe_float(repair.get("objective"))
        if v is not None:
            return v
        inc = repair.get("incumbent")
        if isinstance(inc, dict):
            v = candidate_obj_from_dict(inc)
            if v is not None:
                return v

    # Some diagnostics carry repair nested under diagnostics.
    diag = d.get("diagnostics")
    if isinstance(diag, dict):
        repair = diag.get("repair")
        if isinstance(repair, dict):
            v = safe_float(repair.get("objective"))
            if v is not None:
                return v
            inc = repair.get("incumbent")
            if isinstance(inc, dict):
                v = candidate_obj_from_dict(inc)
                if v is not None:
                    return v

    return None


def direct_bound_from_dict(d: Dict[str, Any]) -> Optional[float]:
    for key in (
        "best_bound",
        "bound",
        "obj_bound",
        "objbound",
        "dual_bound",
        "lower_bound",
        "upper_bound",
    ):
        v = safe_float(d.get(key))
        if v is not None:
            return v
    return None


def lp_relaxation_bound_from_dict(d: Dict[str, Any]) -> Optional[float]:
    diag = d.get("diagnostics")
    if not isinstance(diag, dict):
        return None

    lp_meta = diag.get("lp_meta")
    if isinstance(lp_meta, dict):
        status_name = str(lp_meta.get("status_name", "")).upper()
        status_code = safe_float(lp_meta.get("status"))
        # Gurobi OPTIMAL is 2. Only use LP objective as a global bound when the
        # relaxation itself was solved to optimality.
        lp_solved = status_name == "OPTIMAL" or status_code == 2
        if lp_solved:
            for key in ("lp_obj", "objective", "obj", "objval"):
                v = safe_float(diag.get(key)) if key == "lp_obj" else safe_float(lp_meta.get(key))
                if v is not None:
                    return v

        v = safe_float(lp_meta.get("objbound"))
        if v is not None:
            return v

    v = safe_float(diag.get("lp_obj"))
    if v is not None and not isinstance(lp_meta, dict):
        return v

    return None


def collect_bound_candidates(path: Path, obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def visit(x: Any, label: str) -> None:
        if isinstance(x, dict):
            b = direct_bound_from_dict(x)
            if b is not None:
                out.append({"file": str(path), "source": label, "bound": b})

            lp_b = lp_relaxation_bound_from_dict(x)
            if lp_b is not None:
                out.append({"file": str(path), "source": label + ".lp_relaxation", "bound": lp_b})

            for k, v in x.items():
                if k in {"values", "solution", "x", "vars", "variables"} and isinstance(v, dict):
                    continue
                visit(v, label + "." + str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                visit(v, label + f"[{i}]")

    visit(obj, "root")
    return out


def candidate_incumbent_from_dict(d: Dict[str, Any], obj: Optional[float]) -> Optional[Dict[str, Any]]:
    for key in ("incumbent", "solution", "best", "best_incumbent"):
        sub = d.get(key)
        if isinstance(sub, dict):
            vals = sub.get("values") or sub.get("solution") or sub.get("x")
            sub_obj = candidate_obj_from_dict(sub)
            if vals is not None or sub_obj is not None:
                out = dict(sub)
                if "objective" not in out and sub_obj is not None:
                    out["objective"] = sub_obj
                elif "objective" not in out and obj is not None:
                    out["objective"] = obj
                return out

    repair = d.get("repair")
    if isinstance(repair, dict):
        inc = repair.get("incumbent")
        if isinstance(inc, dict):
            out = dict(inc)
            if "objective" not in out and safe_float(repair.get("objective")) is not None:
                out["objective"] = float(repair["objective"])
            elif "objective" not in out and obj is not None:
                out["objective"] = obj
            return out

    diag = d.get("diagnostics")
    if isinstance(diag, dict):
        repair = diag.get("repair")
        if isinstance(repair, dict):
            inc = repair.get("incumbent")
            if isinstance(inc, dict):
                out = dict(inc)
                if "objective" not in out and safe_float(repair.get("objective")) is not None:
                    out["objective"] = float(repair["objective"])
                elif "objective" not in out and obj is not None:
                    out["objective"] = obj
                return out

    # If no values were present, still return objective-only incumbent.
    if obj is not None:
        return {"objective": obj, "values": {}}

    return None


def extract_candidate(path: Path, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cand_obj = candidate_obj_from_dict(obj)
    if cand_obj is None:
        return None

    incumbent = candidate_incumbent_from_dict(obj, cand_obj)
    phase = str(obj.get("phase", "unknown"))
    method = str(obj.get("method", obj.get("name", "unknown")))
    runtime = safe_float(obj.get("runtime")) or safe_float(obj.get("runtime_sec")) or safe_float(obj.get("wall_runtime")) or safe_float(obj.get("total_wall_time"))

    return {
        "file": str(path),
        "phase": phase,
        "method": method,
        "objective": cand_obj,
        "runtime": runtime,
        "incumbent": incumbent,
        "raw": obj,
    }


def phase_counts(objs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for obj in objs:
        ph = str(obj.get("phase", "unknown"))
        counts[ph] = counts.get(ph, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Result JSON files or directories.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sense", default=None, choices=["min", "max"])
    args = ap.parse_args()

    files = iter_json_files(args.inputs)
    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    for f in files:
        obj = load_json(f)
        if obj is not None:
            loaded.append((f, obj))

    objs = [o for _, o in loaded]
    sense = args.sense or detect_sense(objs, default="min")

    candidates = []
    for f, obj in loaded:
        cand = extract_candidate(f, obj)
        if cand is not None:
            candidates.append(cand)

    best = None
    for cand in candidates:
        if best is None or better(float(cand["objective"]), float(best["objective"]), sense):
            best = cand

    bound_candidates = []
    for f, obj in loaded:
        bound_candidates.extend(collect_bound_candidates(f, obj))

    incumbent_obj = None if best is None else safe_float(best["objective"])
    usable_bounds = []
    for b in bound_candidates:
        bf = safe_float(b.get("bound"))
        if bf is None:
            continue
        if incumbent_obj is not None and not valid_bound_for_incumbent(bf, incumbent_obj, sense):
            continue
        usable_bounds.append({**b, "bound": bf})

    best_bound = None
    best_bound_candidate = None
    for b in usable_bounds:
        if best_bound is None or better_bound(float(b["bound"]), float(best_bound), sense):
            best_bound = float(b["bound"])
            best_bound_candidate = b

    absolute_gap, relative_gap = gap_from_bound(incumbent_obj, best_bound, sense)

    out_obj: Dict[str, Any] = {
        "out": str(Path(args.out)),
        "num_task_files": len(loaded),
        "num_candidate_files": len(candidates),
        "sense": sense,
        "best_objective": None if best is None else best["objective"],
        "objective": None if best is None else best["objective"],
        "best_obj": None if best is None else best["objective"],
        "best_bound": best_bound,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "best_bound_source": None if best_bound_candidate is None else best_bound_candidate,
        "num_bound_candidates": len(bound_candidates),
        "phase_task_counts": phase_counts(objs),
        "candidates": [
            {
                "file": c["file"],
                "phase": c["phase"],
                "method": c["method"],
                "objective": c["objective"],
                "runtime": c["runtime"],
            }
            for c in candidates
        ],
        "best_file": None if best is None else best["file"],
        "best_method": None if best is None else best["method"],
        "best_phase": None if best is None else best["phase"],
        "incumbent": None if best is None else best["incumbent"],
    }

    # Compatibility field for final_gurobi_solve variants.
    if best is not None:
        out_obj["solution"] = best["incumbent"]
        out_obj["found"] = True
        out_obj["found_incumbent"] = True
    else:
        out_obj["solution"] = None
        out_obj["found"] = False
        out_obj["found_incumbent"] = False

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, allow_nan=False)

    print(json.dumps({
        "out": str(out_path),
        "num_task_files": len(loaded),
        "num_candidate_files": len(candidates),
        "sense": sense,
        "best_objective": out_obj["best_objective"],
        "best_bound": out_obj["best_bound"],
        "relative_gap": out_obj["relative_gap"],
        "phase_task_counts": out_obj["phase_task_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
