import time
from typing import Any, Dict, Optional

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    make_result,
    read_model,
    seed_from_lp_values,
    solve_lp_relaxation,
    try_root_seed,
)


def _attempt(instance_path: str, seed: int, time_limit_sec: float, pump_passes: int,
             features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    model = read_model(instance_path)
    model_stats = describe_model(model)
    lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(model, time_limit=max(0.03, min(0.20, 0.20 * time_limit_sec)))
    seed_meta: Optional[Dict[str, Any]] = None
    if x_lp:
        start_values = seed_from_lp_values(model, x_lp, rc)
        apply_partial_start(model, start_values)
    else:
        seed_vals, seed_meta = try_root_seed(model, time_limit=max(0.03, min(0.15, 0.20 * time_limit_sec)), seed=seed)
        if seed_vals:
            apply_partial_start(model, seed_vals)

    remain = max(0.03, time_limit_sec)
    configure_model_for_heuristic(
        model,
        time_limit=remain,
        seed=seed,
        threads=1,
        mip_focus=1,
        heuristics=0.95,
        submip_nodes=300,
        start_node_limit=200,
        start_time_limit=min(0.25, max(0.03, remain / 2.0)),
        improve_start_time=min(0.20, max(0.03, remain / 3.0)),
        no_rel_heur_time=min(0.25, max(0.03, remain / 3.0)),
        pump_passes=pump_passes,
        rins=10,
        presolve=2,
        cuts=0,
        symmetry=-1,
    )
    model.optimize()
    incumbent = collect_solution(model)
    return {
        "seed": seed,
        "model": model_stats,
        "lp_obj": lp_obj,
        "lp_meta": lp_meta,
        "seed_meta": seed_meta,
        "solver_status": int(model.Status),
        "incumbent": incumbent,
    }


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    base_seed = int(params.get("seed", 0))
    restarts = int(params.get("restarts", 2))
    pump_passes = int(params.get("pump_passes", 8))
    per_try = max(0.05, time_limit_sec / max(1, restarts))

    attempts = []
    incumbent = None
    best_obj = None
    sense = "min" if (features or {}).get("objective_sense", "min") == "min" else "max"
    for k in range(restarts):
        meta = _attempt(instance_path, base_seed + k, per_try, pump_passes, features=features)
        attempts.append({
            "seed": meta["seed"],
            "solver_status": meta["solver_status"],
            "objective": None if meta["incumbent"] is None else meta["incumbent"].get("objective"),
            "lp_obj": meta["lp_obj"],
        })
        cand = meta["incumbent"]
        if cand is None or cand.get("objective") is None:
            continue
        obj = float(cand["objective"])
        if incumbent is None or (sense == "min" and obj < best_obj) or (sense == "max" and obj > best_obj):
            incumbent = cand
            best_obj = obj

    return make_result(
        "feasibility_pump",
        params,
        time.time() - t0,
        incumbent is not None,
        incumbent,
        [
            "Multi-start feasibility-pump style seed-and-repair heuristic.",
            "Subdivides naturally by assigning different random seeds to different shards.",
        ],
        {
            "attempts": attempts,
            "restarts": restarts,
            "pump_passes": pump_passes,
        },
    )
