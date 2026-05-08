import math
import time
from typing import Any, Dict

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    make_result,
    incumbent_from_features,
    read_model,
    seed_from_lp_values,
    solve_lp_relaxation,
    try_root_seed,
)


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get('seed', 0))
    improvement = float(params.get('improvement', 1e-4))
    sense = (features or {}).get('objective_sense', 'min')
    model = read_model(instance_path)
    model_stats = describe_model(model)
    warm_vals, warm_obj, warm_method = incumbent_from_features(features)
    lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(model, time_limit=max(0.03, min(0.20, 0.20 * time_limit_sec)))
    incumbent_obj = warm_obj
    if warm_vals:
        apply_partial_start(model, warm_vals)
        lp_meta = {**lp_meta, "warmstart_source": warm_method, "warmstart_objective": warm_obj}
    elif x_lp:
        start_values = seed_from_lp_values(model, x_lp, rc)
        apply_partial_start(model, start_values)
        # get a quick incumbent first
        configure_model_for_heuristic(model, time_limit=max(0.03, min(0.20, 0.30 * time_limit_sec)), seed=seed,
                                      threads=1, mip_focus=1, heuristics=0.7, submip_nodes=300,
                                      start_node_limit=200, start_time_limit=0.10, improve_start_time=0.05,
                                      rins=10, presolve=2, cuts=0, symmetry=-1)
        model.optimize()
        quick = collect_solution(model)
        incumbent_obj = None if quick is None else quick.get('objective')
    else:
        seed_vals, seed_meta = try_root_seed(model, time_limit=max(0.03, min(0.20, 0.25 * time_limit_sec)), seed=seed)
        if seed_vals:
            apply_partial_start(model, seed_vals)
        lp_meta = {**lp_meta, 'root_seed_meta': seed_meta}
        configure_model_for_heuristic(model, time_limit=max(0.03, min(0.20, 0.30 * time_limit_sec)), seed=seed,
                                      threads=1, mip_focus=1, heuristics=0.8, submip_nodes=300,
                                      start_node_limit=200, start_time_limit=0.10, improve_start_time=0.05,
                                      rins=10, presolve=2, cuts=0, symmetry=-1)
        model.optimize()
        quick = collect_solution(model)
        incumbent_obj = None if quick is None else quick.get('objective')

    # exact improvement neighborhood with a cutoff if we have an incumbent
    if incumbent_obj is not None:
        cutoff = float(incumbent_obj) - abs(improvement) if sense == 'min' else float(incumbent_obj) + abs(improvement)
        model = read_model(instance_path)
        if warm_vals:
            apply_partial_start(model, warm_vals)
        elif x_lp:
            apply_partial_start(model, seed_from_lp_values(model, x_lp, rc))
        try:
            model.Params.Cutoff = cutoff
        except Exception:
            pass
    remain = max(0.05, time_limit_sec - (time.time() - t0))
    configure_model_for_heuristic(model, time_limit=remain, seed=seed + 1, threads=1,
                                  mip_focus=1, heuristics=0.35, submip_nodes=int(params.get('submip_nodes', 1500)),
                                  start_node_limit=300, start_time_limit=min(0.30, max(0.05, remain / 2.0)),
                                  improve_start_time=min(0.25, max(0.05, remain / 3.0)), rins=30,
                                  presolve=2, cuts=-1, symmetry=-1)
    model.optimize()
    incumbent = collect_solution(model)
    diag = {'model': model_stats, 'lp_obj': lp_obj, 'lp_meta': lp_meta, 'initial_incumbent_obj': incumbent_obj,
            'requested_improvement': improvement, 'solver_status': int(model.Status)}
    if incumbent is not None:
        diag['incumbent_obj'] = incumbent['objective']
    return make_result('objective_bound_search', params, time.time() - t0, incumbent is not None, incumbent,
                       ['Objective-bounded improvement search using a cutoff against the seed incumbent.',
                        'Subdivide by different improvement targets and seeds.'], diag)
