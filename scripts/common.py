from __future__ import annotations

import itertools
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

METHODS = ["greedy", "rens", "local_search_lp", "dive_fix"]
INTEGER_VTYPES = {"B", "I"}


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def host_info() -> str:
    return socket.gethostname()


def cartesian_grid(grid_spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def split_evenly(items: List[Any], num_buckets: int) -> List[List[Any]]:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    buckets = [[] for _ in range(num_buckets)]
    for i, item in enumerate(items):
        buckets[i % num_buckets].append(item)
    return buckets


def normalize_allocation(raw_alloc: Dict[str, int], total_tasks: int) -> Dict[str, int]:
    alloc = {k: max(1, int(v)) for k, v in raw_alloc.items()}
    while sum(alloc.values()) < total_tasks:
        k = min(alloc, key=lambda x: (alloc[x], x))
        alloc[k] += 1
    while sum(alloc.values()) > total_tasks:
        candidates = [k for k, v in alloc.items() if v > 1]
        if not candidates:
            break
        k = max(candidates, key=lambda x: (alloc[x], x))
        alloc[k] -= 1
    return alloc


def objective_better(a: Optional[float], b: Optional[float], sense: str = "min") -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a < b if sense == "min" else a > b


def summarize_feature_vector(features: Dict[str, Any]) -> Dict[str, float]:
    return {
        "num_vars": float(features.get("num_vars_est", 0)),
        "num_bin": float(features.get("num_bin_est", 0)),
        "num_int": float(features.get("num_int_est", 0)),
        "num_constr": float(features.get("num_constr_est", 0)),
        "density": float(features.get("density_est", 0.0)),
        "bin_ratio": float(features.get("bin_ratio_est", 0.0)),
        "integrality_ratio": float(features.get("integrality_ratio_est", 0.0)),
    }


def require_gurobi() -> Tuple[Any, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
        return gp, GRB
    except Exception as exc:
        raise RuntimeError(
            "gurobipy is required for heuristic execution. Install Gurobi Python and ensure the license is visible."
        ) from exc


def make_env(log_to_console: bool = False):
    gp, _ = require_gurobi()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if log_to_console else 0)
    env.start()
    return env


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def read_model(instance_path: str, env=None, log_to_console: bool = False):
    gp, _ = require_gurobi()
    if env is None:
        env = make_env(log_to_console=log_to_console)
    model = gp.read(instance_path, env=env)
    model.update()
    return model


def integer_vars(model) -> List[Any]:
    return [v for v in model.getVars() if v.VType in INTEGER_VTYPES]


def binary_vars(model) -> List[Any]:
    return [v for v in model.getVars() if v.VType == "B"]


def is_binary_var(v) -> bool:
    return getattr(v, "VType", None) == "B"


def max_remaining_time(deadline: float) -> float:
    return max(0.0, deadline - time.time())


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def configure_model_for_heuristic(model, *, time_limit: float, seed: int, threads: int = 1,
                                  mip_focus: int = 1, heuristics: float = 0.35,
                                  submip_nodes: Optional[int] = None,
                                  start_node_limit: Optional[int] = None,
                                  start_time_limit: Optional[float] = None,
                                  improve_start_time: Optional[float] = None,
                                  no_rel_heur_time: Optional[float] = None,
                                  pump_passes: Optional[int] = None,
                                  rins: Optional[int] = None,
                                  presolve: Optional[int] = None,
                                  cuts: Optional[int] = None,
                                  node_limit: Optional[int] = None,
                                  solution_limit: Optional[int] = None,
                                  partition_place: Optional[int] = None,
                                  symmetry: Optional[int] = None) -> None:
    model.Params.TimeLimit = max(0.01, float(time_limit))
    model.Params.Threads = max(1, int(threads))
    model.Params.Seed = int(seed)
    model.Params.MIPFocus = int(mip_focus)
    model.Params.Heuristics = float(heuristics)
    if submip_nodes is not None:
        model.Params.SubMIPNodes = int(submip_nodes)
    if start_node_limit is not None:
        model.Params.StartNodeLimit = int(start_node_limit)
    if start_time_limit is not None:
        model.Params.StartTimeLimit = float(start_time_limit)
    if improve_start_time is not None:
        model.Params.ImproveStartTime = float(improve_start_time)
    if no_rel_heur_time is not None:
        model.Params.NoRelHeurTime = float(no_rel_heur_time)
    if pump_passes is not None:
        model.Params.PumpPasses = int(pump_passes)
    if rins is not None:
        model.Params.RINS = int(rins)
    if presolve is not None:
        model.Params.Presolve = int(presolve)
    if cuts is not None:
        model.Params.Cuts = int(cuts)
    if node_limit is not None:
        model.Params.NodeLimit = int(node_limit)
    if solution_limit is not None:
        model.Params.SolutionLimit = int(solution_limit)
    if partition_place is not None:
        model.Params.PartitionPlace = int(partition_place)
    if symmetry is not None:
        model.Params.Symmetry = int(symmetry)


def describe_model(model) -> Dict[str, Any]:
    return {
        "model_name": getattr(model, "ModelName", ""),
        "is_mip": int(getattr(model, "IsMIP", 0)),
        "is_qp": int(getattr(model, "IsQP", 0)),
        "is_qcp": int(getattr(model, "IsQCP", 0)),
        "num_vars": int(getattr(model, "NumVars", 0)),
        "num_bin": int(getattr(model, "NumBinVars", 0)),
        "num_int": int(getattr(model, "NumIntVars", 0)),
        "num_constrs": int(getattr(model, "NumConstrs", 0)),
        "num_qconstrs": int(getattr(model, "NumQConstrs", 0)),
        "sense": int(getattr(model, "ModelSense", 1)),
    }


def _safe_model_attr(model, attr: str):
    try:
        return model.getAttr(attr)
    except Exception:
        return None


def _status_name(status: int) -> str:
    _, GRB = require_gurobi()
    table = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        getattr(GRB, "USER_OBJ_LIMIT", 15): "USER_OBJ_LIMIT",
    }
    return table.get(int(status), f"STATUS_{status}")


def _extract_lp_solution(relax, integer_names: set[str]) -> Tuple[Optional[float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Any]]:
    status = int(relax.Status)
    solcount = int(relax.SolCount)
    meta = {
        "status": status,
        "status_name": _status_name(status),
        "solcount": solcount,
        "objbound": _safe_model_attr(relax, "ObjBound"),
        "runtime": _safe_model_attr(relax, "Runtime"),
        "itercount": _safe_model_attr(relax, "IterCount"),
        "baritercount": _safe_model_attr(relax, "BarIterCount"),
        "nodecount": _safe_model_attr(relax, "NodeCount"),
    }
    if solcount <= 0:
        return None, {}, {}, {}, meta

    xvals: Dict[str, float] = {}
    rc: Dict[str, float] = {}
    fracs: Dict[str, float] = {}
    for rv in relax.getVars():
        try:
            x = float(rv.X)
        except Exception:
            continue
        xvals[rv.VarName] = x
        try:
            rc[rv.VarName] = float(rv.RC)
        except Exception:
            rc[rv.VarName] = 0.0
        fracs[rv.VarName] = abs(x - round(x)) if rv.VarName in integer_names else 0.0

    try:
        lp_obj = float(relax.ObjVal)
    except Exception:
        lp_obj = None
    return lp_obj, xvals, rc, fracs, meta

def solve_lp_relaxation(model, *, time_limit: float, method: int = -1):
    """
    Robust LP relaxation solve with fallback strategies.
    Returns (lp_obj, xvals, rc, fracs, meta).
    """
    integer_names = {v.VarName for v in integer_vars(model)}
    total = max(0.02, float(time_limit))
    attempts = []

    candidates = []
    if method != -1:
        candidates.append((int(method), None, 0))
    candidates.extend([
        (1, None, 0),  # dual simplex
        (0, None, 0),  # primal simplex
        (2, 0, 0),     # barrier, no crossover
    ])

    best = (
        None,
        {},
        {},
        {},
        {
            "status": None,
            "status_name": "NOT_RUN",
            "solcount": 0,
            "attempts": [],
        },
    )

    per = max(0.02, total / max(1, len(candidates)))

    for meth, crossover, dualreductions in candidates:
        relax = model.relax()
        relax.Params.OutputFlag = 0
        relax.Params.TimeLimit = per
        relax.Params.Method = meth
        relax.Params.Presolve = 2
        try:
            relax.Params.DualReductions = dualreductions
        except Exception:
            pass
        if crossover is not None:
            try:
                relax.Params.Crossover = crossover
            except Exception:
                pass
        try:
            relax.Params.NumericFocus = 1
        except Exception:
            pass

        relax.optimize()
        lp_obj, xvals, rc, fracs, meta = _extract_lp_solution(relax, integer_names)
        meta.update({
            "method": meth,
            "crossover": crossover,
            "dualreductions": dualreductions,
        })

        # store a COPY, not the live meta object
        attempts.append(dict(meta))

        if xvals:
            final_meta = dict(meta)
            final_meta["attempts"] = list(attempts)
            return lp_obj, xvals, rc, fracs, final_meta

        best = (
            lp_obj,
            xvals,
            rc,
            fracs,
            {
                **dict(meta),
                "attempts": list(attempts),
            },
        )

    if any(a.get("status_name") == "INF_OR_UNBD" for a in attempts):
        relax = model.relax()
        relax.Params.OutputFlag = 0
        relax.Params.TimeLimit = max(0.02, min(0.25, total))
        relax.Params.Method = 1
        relax.Params.Presolve = 2
        relax.Params.DualReductions = 0
        relax.optimize()

        lp_obj, xvals, rc, fracs, meta = _extract_lp_solution(relax, integer_names)
        meta.update({
            "method": 1,
            "crossover": None,
            "dualreductions": 0,
            "disambiguation": True,
        })

        attempts.append(dict(meta))

        if xvals:
            final_meta = dict(meta)
            final_meta["attempts"] = list(attempts)
            return lp_obj, xvals, rc, fracs, final_meta

        best = (
            lp_obj,
            xvals,
            rc,
            fracs,
            {
                **dict(meta),
                "attempts": list(attempts),
            },
        )

    return best

def extract_lock_counts(model) -> Dict[str, Dict[str, int]]:
    counts = {v.VarName: {"up": 0, "down": 0} for v in integer_vars(model)}
    for c in model.getConstrs():
        row = model.getRow(c)
        sense = c.Sense
        for i in range(row.size()):
            v = row.getVar(i)
            if v.VType not in INTEGER_VTYPES:
                continue
            a = row.getCoeff(i)
            if sense == '<':
                if a > 0:
                    counts[v.VarName]["up"] += 1
                elif a < 0:
                    counts[v.VarName]["down"] += 1
            elif sense == '>':
                if a > 0:
                    counts[v.VarName]["down"] += 1
                elif a < 0:
                    counts[v.VarName]["up"] += 1
            else:
                counts[v.VarName]["up"] += 1
                counts[v.VarName]["down"] += 1
    return counts


def rounded_value(v, x: float, prefer_obj: bool = True) -> float:
    if v.VType == "B":
        return float(clamp(round(x), 0.0, 1.0))
    if v.VType == "I":
        return float(clamp(round(x), v.LB, v.UB))
    return float(clamp(x, v.LB, v.UB))


def choose_round_direction(v, x: float, rc: float = 0.0, obj: Optional[float] = None, sense: int = 1) -> float:
    lo = max(v.LB, math.floor(x))
    hi = min(v.UB, math.ceil(x))
    if abs(x - round(x)) <= 1e-9:
        return float(round(x))
    if v.VType == "B":
        lo, hi = 0.0, 1.0

    score_hi = abs(hi - x) - 0.05 * rc
    score_lo = abs(x - lo) + 0.05 * rc
    if obj is not None:
        score_hi += 0.02 * abs(obj) * abs(hi)
        score_lo += 0.02 * abs(obj) * abs(lo)
    return float(hi if score_hi <= score_lo else lo)


def seed_from_lp_values(model, x_lp: Dict[str, float], rc: Optional[Dict[str, float]] = None,
                        lock_counts: Optional[Dict[str, Dict[str, int]]] = None,
                        fractionality_bias: float = 0.15) -> Dict[str, float]:
    rc = rc or {}
    lock_counts = lock_counts or {}
    start_values: Dict[str, float] = {}
    sense = int(getattr(model, "ModelSense", 1))

    for v in integer_vars(model):
        x = x_lp.get(v.VarName)
        if x is None:
            continue

        if abs(x - round(x)) <= 1e-8:
            start_values[v.VarName] = clamp(round(x), v.LB, v.UB)
            continue

        locks = lock_counts.get(v.VarName, {"up": 0, "down": 0})
        lock_push = 0.02 * (locks.get("down", 0) - locks.get("up", 0))
        rc_push = 0.05 * rc.get(v.VarName, 0.0)
        obj_push = 0.01 * sense * float(v.Obj)

        target = x - rc_push - obj_push + lock_push
        if abs(x - 0.5) < fractionality_bias and v.VType == "B":
            target += math.copysign(0.1, 0.5 - x)

        start_values[v.VarName] = choose_round_direction(
            v, target, rc.get(v.VarName, 0.0), float(v.Obj), sense
        )
    return start_values


def try_root_seed(model, *, time_limit: float, seed: int = 0) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    """
    Fallback when no LP solution is available:
    run a very short primal-oriented root MIP and steal the incumbent if one appears.
    """
    work = model.copy()
    configure_model_for_heuristic(
        work,
        time_limit=max(0.02, time_limit),
        seed=seed,
        threads=1,
        mip_focus=1,
        heuristics=0.8,
        submip_nodes=64,
        start_node_limit=64,
        no_rel_heur_time=min(0.2, max(0.02, time_limit / 2.0)),
        pump_passes=2,
        rins=10,
        presolve=2,
        cuts=0,
        node_limit=0,
        solution_limit=1,
    )
    work.optimize()
    info = {
        "status": int(work.Status),
        "status_name": _status_name(int(work.Status)),
        "solcount": int(work.SolCount),
        "runtime": float(work.Runtime),
    }
    if work.SolCount <= 0:
        return None, info

    vals = {v.VarName: float(v.X) for v in work.getVars() if v.VType in INTEGER_VTYPES}
    info["objective"] = float(work.ObjVal)
    return vals, info


def apply_partial_start(model, start_values: Dict[str, float]) -> None:
    _, GRB = require_gurobi()
    for v in model.getVars():
        if v.VarName in start_values:
            v.Start = float(clamp(start_values[v.VarName], v.LB, v.UB))
        else:
            v.Start = GRB.UNDEFINED
    model.update()


def collect_solution(model) -> Optional[Dict[str, Any]]:
    if int(getattr(model, "SolCount", 0)) <= 0:
        return None

    values = {}
    for v in model.getVars():
        if v.VType not in INTEGER_VTYPES:
            continue
        try:
            values[v.VarName] = float(v.X)
        except Exception:
            continue

    out = {
        "objective": None,
        "solution_count": int(model.SolCount),
        "runtime_sec": float(getattr(model, "Runtime", 0.0)),
        "mip_gap": None,
        "node_count": None,
        "bound": None,
        "status_code": int(model.Status),
        "values": values,
    }
    try:
        out["objective"] = float(model.ObjVal)
    except Exception:
        pass

    if int(getattr(model, "IsMIP", 0)) != 0:
        out["mip_gap"] = float(getattr(model, "MIPGap", 0.0)) if out["objective"] is not None else None
        out["node_count"] = float(getattr(model, "NodeCount", 0.0))
        try:
            out["bound"] = float(getattr(model, "ObjBound", out["objective"])) if out["objective"] is not None else None
        except Exception:
            out["bound"] = None

    return out


def make_result(method: str, params: Dict[str, Any], elapsed: float, feasible: bool,
                incumbent: Optional[Dict[str, Any]], notes: List[str],
                diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": method,
        "params": params,
        "status": "ok" if feasible else "no_solution",
        "feasible": feasible,
        "objective": None if incumbent is None else incumbent.get("objective"),
        "runtime_sec": float(elapsed),
        "incumbent": incumbent,
        "notes": notes,
        "diagnostics": diagnostics,
    }