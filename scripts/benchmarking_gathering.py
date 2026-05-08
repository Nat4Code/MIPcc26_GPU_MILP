#!/usr/bin/env python3
"""
scripts/benchmarking_gathering.py

Container-side benchmark parser/plotter.

Run this INSIDE the Apptainer container, after the host-side script has created
a raw benchmark manifest:

  python3 scripts/benchmarking_gathering.py \
    --manifest results/benchmarking_raw/<timestamp>/manifest.json

This script may use matplotlib and newer Python because it is intended for the
container environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Event = Tuple[float, float]
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def find_key_recursive(obj: Any, names: Sequence[str]) -> Any:
    lowered = {n.lower() for n in names}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in lowered:
                return v
        for v in obj.values():
            out = find_key_recursive(v, names)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for v in obj:
            out = find_key_recursive(v, names)
            if out is not None:
                return out
    return None


def detect_sense(run_dir: Path, fallback: str = "min") -> str:
    for p in [
        run_dir / "results" / "features.json",
        run_dir / "results" / "plan.json",
        run_dir / "results" / "merged.json",
        run_dir / "results" / "merged_phase1.json",
    ]:
        if not p.exists():
            continue
        try:
            data = load_json(p)
        except Exception:
            continue
        val = find_key_recursive(data, ["objective_sense", "sense", "model_sense"])
        if isinstance(val, str):
            s = val.lower()
            if s in {"min", "minimize", "minimization"}:
                return "min"
            if s in {"max", "maximize", "maximization"}:
                return "max"
        if isinstance(val, (int, float)):
            if int(val) == 1:
                return "min"
            if int(val) == -1:
                return "max"
    return fallback


def dedupe_events(events: Iterable[Event]) -> List[Event]:
    seen = set()
    out: List[Event] = []
    for t, obj in events:
        tf = safe_float(t)
        of = safe_float(obj)
        if tf is None or of is None:
            continue
        key = (round(float(tf), 6), round(float(of), 9))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(tf), float(of)))
    out.sort(key=lambda z: (z[0], z[1]))
    return out


def best_objective(events: Sequence[Event], sense: str) -> Optional[float]:
    vals = [obj for _, obj in events if safe_float(obj) is not None]
    if not vals:
        return None
    return min(vals) if sense == "min" else max(vals)


def objective_winner(h: Optional[float], g: Optional[float], sense: str, tol: float = 1e-8) -> str:
    if h is None and g is None:
        return "tie"
    if h is None:
        return "gurobi"
    if g is None:
        return "heuristic"
    if sense == "min":
        if h < g - tol:
            return "heuristic"
        if g < h - tol:
            return "gurobi"
    else:
        if h > g + tol:
            return "heuristic"
        if g > h + tol:
            return "gurobi"
    return "tie"


def make_best_curve(events: Sequence[Event], sense: str) -> List[Event]:
    clean = [(max(0.0, float(t)), float(o)) for t, o in events
             if safe_float(t) is not None and safe_float(o) is not None]
    clean.sort(key=lambda z: (z[0], z[1]))

    out: List[Event] = []
    best: Optional[float] = None
    for t, obj in clean:
        if best is None:
            best = obj
            out.append((t, obj))
        elif sense == "min" and obj < best - 1e-9:
            best = obj
            out.append((t, obj))
        elif sense == "max" and obj > best + 1e-9:
            best = obj
            out.append((t, obj))
    return out


def csv_events(path: Path) -> List[Event]:
    out: List[Event] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return out
            lower = {c.lower(): c for c in reader.fieldnames}

            time_col = None
            for name in ["time", "time_sec", "elapsed", "elapsed_sec", "runtime", "runtime_sec",
                         "t", "wall_time", "wall_time_sec", "seconds"]:
                if name in lower:
                    time_col = lower[name]
                    break

            obj_col = None
            for name in ["objective", "obj", "best_obj", "best_objective", "incumbent_obj",
                         "incumbent_objective", "solution_objective"]:
                if name in lower:
                    obj_col = lower[name]
                    break

            if obj_col is None:
                return out

            for i, row in enumerate(reader, start=1):
                obj = safe_float(row.get(obj_col))
                if obj is None:
                    continue
                t = safe_float(row.get(time_col)) if time_col else None
                if t is None:
                    t = float(i)
                out.append((float(t), float(obj)))
    except Exception:
        return []
    return out


def obj_from_dict(d: Dict[str, Any]) -> Optional[float]:
    for k in ["objective", "obj", "best_obj", "best_objective", "incumbent_obj",
              "incumbent_objective", "solution_objective"]:
        if k in d:
            v = safe_float(d.get(k))
            if v is not None:
                return v

    for parent in ["incumbent", "best", "solution", "best_incumbent"]:
        sub = d.get(parent)
        if isinstance(sub, dict):
            for k in ["objective", "obj", "best_obj", "incumbent_obj"]:
                v = safe_float(sub.get(k))
                if v is not None:
                    return v
    return None


def time_from_dict(d: Dict[str, Any]) -> Optional[float]:
    for k in ["time", "time_sec", "elapsed", "elapsed_sec", "runtime", "runtime_sec",
              "total_runtime", "wall_time", "wall_time_sec", "t", "solve_time"]:
        if k in d:
            v = safe_float(d.get(k))
            if v is not None:
                return v
    return None


def collect_json_events(obj: Any, inherited_time: Optional[float] = None) -> List[Event]:
    out: List[Event] = []
    if isinstance(obj, dict):
        t = time_from_dict(obj)
        if t is None:
            t = inherited_time
        o = obj_from_dict(obj)
        if t is not None and o is not None:
            out.append((float(t), float(o)))
        for v in obj.values():
            out.extend(collect_json_events(v, t))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_json_events(v, inherited_time))
    return out


def json_events(path: Path) -> List[Event]:
    if not path.exists():
        return []
    try:
        return collect_json_events(load_json(path))
    except Exception:
        return []


def offset_events(events: Sequence[Event], offset: float) -> List[Event]:
    return [(max(0.0, float(offset)) + max(0.0, float(t)), float(obj)) for t, obj in events]


def parse_heuristic_events(run_dir: Path) -> List[Event]:
    results_dir = run_dir / "results"
    events: List[Event] = []

    for name in ["heuristic_log.csv", "heuristic_phase1_log.csv", "heuristic_phase2_log.csv"]:
        p = results_dir / name
        if p.exists():
            events.extend(csv_events(p))

    lp_events = dedupe_events(json_events(results_dir / "lp_seed.json"))
    lp_end = max([t for t, _ in lp_events], default=0.0)

    phase1_local: List[Event] = []

    d1 = results_dir / "phase1_results"
    if d1.exists():
        for p in sorted(d1.glob("*.json")):
            if p.name == "task_lp_seed.json":
                if not lp_events:
                    lp_events.extend(json_events(p))
                    lp_events = dedupe_events(lp_events)
                    lp_end = max([t for t, _ in lp_events], default=0.0)
                continue
            phase1_local.extend(json_events(p))

    if not phase1_local:
        phase1_local.extend(json_events(results_dir / "merged_phase1.json"))

    phase1_local = dedupe_events(phase1_local)
    phase1_events = offset_events(phase1_local, lp_end)
    phase1_end = lp_end + max([t for t, _ in phase1_local], default=0.0)

    phase2_local: List[Event] = []

    d2 = results_dir / "phase2_results"
    if d2.exists():
        for p in sorted(d2.glob("*.json")):
            phase2_local.extend(json_events(p))

    if not phase2_local:
        phase2_local.extend(json_events(results_dir / "merged_phase2.json"))

    phase2_local = dedupe_events(phase2_local)
    phase2_events = offset_events(phase2_local, phase1_end)
    phase2_end = phase1_end + max([t for t, _ in phase2_local], default=0.0)

    bound_probe_trace = dedupe_events(csv_events(results_dir / "bound_probe_incumbents.csv"))
    bound_probe_events = offset_events(bound_probe_trace, lp_end)

    final_trace = dedupe_events(csv_events(results_dir / "final_gurobi_incumbents.csv"))
    if final_trace:
        final_events = offset_events(final_trace, phase2_end)
        final_events.extend(offset_events(json_events(results_dir / "final_gurobi.json"), phase2_end))
    else:
        final_events = offset_events(json_events(results_dir / "final_gurobi.json"), phase2_end)

    events.extend(lp_events)
    events.extend(bound_probe_events)
    events.extend(phase1_events)
    events.extend(phase2_events)
    events.extend(final_events)
    return dedupe_events(events)


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_baseline_events(log_path: Path, horizon: float,
                          event_log_path: Optional[Path] = None,
                          result_json_path: Optional[Path] = None) -> List[Event]:
    events: List[Event] = []

    if event_log_path is not None and event_log_path.exists():
        events.extend(csv_events(event_log_path))

    if result_json_path is not None and result_json_path.exists():
        events.extend(json_events(result_json_path))

    # New benchmark runs write callback CSVs. Keep log parsing as a fallback for
    # older compiled-baseline runs that only have Gurobi text output.
    if events or not log_path.exists():
        return dedupe_events(events)

    gurobi_incumbent_re = re.compile(
        rf"^\s*[H\*]\s*\d+\s+\d+\s+"
        rf"(?P<incumbent>{FLOAT_RE})\s+"
        rf"(?P<best_bound>{FLOAT_RE})\s+"
        rf"\S+\s+\S+\s+"
        rf"(?P<time>{FLOAT_RE})s\s*$"
    )
    time_re = re.compile(rf"({FLOAT_RE})s\b")
    found_heur_re = re.compile(rf"Found heuristic solution:\s*objective\s+({FLOAT_RE})", re.I)
    best_obj_re = re.compile(rf"Best objective\s+({FLOAT_RE})", re.I)

    last_time = 0.0
    final_obj: Optional[float] = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tm = list(time_re.finditer(line))
            if tm:
                v = safe_float(tm[-1].group(1))
                if v is not None:
                    last_time = float(v)

            js = parse_json_line(line)
            if js is not None:
                for k in ["best_obj", "objective"]:
                    v = safe_float(js.get(k))
                    if v is not None:
                        final_obj = float(v)
                        break
                rt = safe_float(js.get("grb_runtime_sec")) or safe_float(js.get("wall_runtime_sec"))
                if final_obj is not None:
                    events.append((float(rt if rt is not None else horizon), final_obj))
                continue

            m = gurobi_incumbent_re.search(line)
            if m:
                obj = safe_float(m.group("incumbent"))
                t = safe_float(m.group("time"))
                if obj is not None and t is not None:
                    events.append((float(t), obj))
                continue

            m = found_heur_re.search(line)
            if m:
                obj = safe_float(m.group(1))
                if obj is not None:
                    events.append((last_time, obj))
                continue

            m = best_obj_re.search(line)
            if m:
                obj = safe_float(m.group(1))
                if obj is not None:
                    final_obj = obj

    if final_obj is not None:
        events.append((horizon, final_obj))

    return dedupe_events(events)


def reference_best(h_events: Sequence[Event], g_events: Sequence[Event], sense: str) -> Optional[float]:
    vals = [obj for _, obj in list(h_events) + list(g_events) if safe_float(obj) is not None]
    if not vals:
        return None
    return min(vals) if sense == "min" else max(vals)


def gap_value(obj: Optional[float], ref: float, sense: str) -> float:
    denom = max(1.0, abs(ref))
    if obj is None:
        return 1.0e6
    if sense == "min":
        return max(0.0, (obj - ref) / denom)
    return max(0.0, (ref - obj) / denom)


def primal_integral(events: Sequence[Event], horizon: float, sense: str, ref: Optional[float]) -> Optional[float]:
    if horizon <= 0.0 or ref is None:
        return None

    curve = make_best_curve(events, sense)
    if not curve:
        return None

    area = 0.0
    last_t = 0.0
    last_obj: Optional[float] = None

    for t, obj in curve:
        t = max(0.0, min(horizon, float(t)))
        if t > last_t:
            area += (t - last_t) * gap_value(last_obj, ref, sense)
        last_t = t
        last_obj = obj

    if last_t < horizon:
        area += (horizon - last_t) * gap_value(last_obj, ref, sense)

    return area


def pi_winner(h_pi: Optional[float], g_pi: Optional[float], tol: float = 1e-9) -> str:
    if h_pi is None and g_pi is None:
        return "tie"
    if h_pi is None:
        return "gurobi"
    if g_pi is None:
        return "heuristic"
    if h_pi < g_pi - tol:
        return "heuristic"
    if g_pi < h_pi - tol:
        return "gurobi"
    return "tie"


def plot_instance(iid: str, sense: str, h_events: Sequence[Event], g_events: Sequence[Event],
                  horizon: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    any_plot = False
    for label, events in [("Heuristic battery", h_events), ("Gurobi baseline", g_events)]:
        curve = make_best_curve(events, sense)
        if not curve:
            continue
        xs = [0.0]
        ys = [curve[0][1]]
        for t, obj in curve:
            xs.append(min(horizon, t))
            ys.append(obj)
        if xs[-1] < horizon:
            xs.append(horizon)
            ys.append(ys[-1])
        plt.step(xs, ys, where="post", label=label)
        any_plot = True

    plt.title(f"{iid}: incumbent objective over time ({sense})")
    plt.xlabel("time (seconds)")
    plt.ylabel("best incumbent objective")
    plt.xlim(0, horizon)
    plt.grid(True, alpha=0.3)
    if any_plot:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No incumbent events parsed", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_bar(title: str, counts: Dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Heuristic", "Gurobi", "Tie/no decision"]
    vals = [counts.get("heuristic", 0), counts.get("gurobi", 0), counts.get("tie", 0)]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.title(title)
    plt.ylabel("number of instances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "instance", "sense",
        "heuristic_best_obj", "gurobi_best_obj", "best_objective_winner",
        "heuristic_primal_integral", "gurobi_primal_integral", "primal_integral_winner",
        "heuristic_event_count", "gurobi_event_count",
        "heuristic_run_dir", "baseline_log", "objective_plot",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="Default: <manifest_dir>/gathered")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (manifest_path.parent / "gathered").resolve()
    plots_dir = out_dir / "per_instance"
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = float(manifest.get("seconds", 300.0))
    rows_in = manifest.get("rows", [])

    best_counts = {"heuristic": 0, "gurobi": 0, "tie": 0}
    pi_counts = {"heuristic": 0, "gurobi": 0, "tie": 0}
    rows_out: List[Dict[str, Any]] = []

    print(f"Manifest:  {manifest_path}")
    print(f"Output:    {out_dir}")
    print(f"Horizon:   {horizon}")
    print(f"Instances: {len(rows_in)}")

    for item in rows_in:
        iid = item["instance"]
        run_dir = Path(item["heuristic_run_dir"])
        base_log = Path(item["baseline_log"])

        sense = detect_sense(run_dir, fallback="min")
        h_events = parse_heuristic_events(run_dir)
        base_event_raw = item.get("baseline_event_log")
        base_result_raw = item.get("baseline_result_json")
        base_event_log = Path(base_event_raw) if base_event_raw else None
        base_result_json = Path(base_result_raw) if base_result_raw else None
        g_events = parse_baseline_events(
            base_log,
            horizon=horizon,
            event_log_path=base_event_log,
            result_json_path=base_result_json,
        )

        h_best = best_objective(h_events, sense)
        g_best = best_objective(g_events, sense)
        best_win = objective_winner(h_best, g_best, sense)
        best_counts[best_win] += 1

        ref = reference_best(h_events, g_events, sense)
        h_pi = primal_integral(h_events, horizon, sense, ref)
        g_pi = primal_integral(g_events, horizon, sense, ref)
        pi_win = pi_winner(h_pi, g_pi)
        pi_counts[pi_win] += 1

        plot_path = plots_dir / f"{iid}_objective_trace.png"
        plot_instance(iid, sense, h_events, g_events, horizon, plot_path)

        row = {
            "instance": iid,
            "sense": sense,
            "heuristic_best_obj": h_best,
            "gurobi_best_obj": g_best,
            "best_objective_winner": best_win,
            "heuristic_primal_integral": h_pi,
            "gurobi_primal_integral": g_pi,
            "primal_integral_winner": pi_win,
            "heuristic_event_count": len(h_events),
            "gurobi_event_count": len(g_events),
            "heuristic_run_dir": str(run_dir),
            "baseline_log": str(base_log),
            "objective_plot": str(plot_path),
        }
        rows_out.append(row)

        print(f"{iid}: sense={sense}, h_best={h_best}, g_best={g_best}, "
              f"best_win={best_win}, h_pi={h_pi}, g_pi={g_pi}, pi_win={pi_win}")

    plot_bar("Best incumbent objective wins", best_counts, out_dir / "aggregate_best_objective_wins.png")
    plot_bar("Primal integral wins", pi_counts, out_dir / "aggregate_primal_integral_wins.png")
    write_summary_csv(rows_out, out_dir / "benchmark_summary.csv")

    summary = {
        "manifest": str(manifest_path),
        "seconds": horizon,
        "best_objective_win_counts": best_counts,
        "primal_integral_win_counts": pi_counts,
        "note": (
            "Primal integral here is a normalized incumbent-gap integral against the best "
            "objective found by either method on the instance. Lower is better."
        ),
        "rows": rows_out,
    }
    with (out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print("\nDone.")
    print(f"Summary CSV:    {out_dir / 'benchmark_summary.csv'}")
    print(f"Best wins plot: {out_dir / 'aggregate_best_objective_wins.png'}")
    print(f"PI wins plot:   {out_dir / 'aggregate_primal_integral_wins.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
