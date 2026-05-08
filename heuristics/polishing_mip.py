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
    model = read_model(instance_path)
    model_stats = describe_model(model)
    warm_vals, warm_obj, warm_method = incumbent_from_features(features)
    lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(model, time_limit=max(0.03, min(0.20, 0.20 * time_limit_sec)))
    if warm_vals:
        apply_partial_start(model, warm_vals)
        lp_meta = {**lp_meta, "warmstart_source": warm_method, "warmstart_objective": warm_obj}
    elif x_lp:
        apply_partial_start(model, seed_from_lp_values(model, x_lp, rc))
    else:
        seed_vals, seed_meta = try_root_seed(model, time_limit=max(0.03, min(0.20, 0.25 * time_limit_sec)), seed=seed)
        if seed_vals:
            apply_partial_start(model, seed_vals)
        lp_meta = {**lp_meta, 'root_seed_meta': seed_meta}

    remain = max(0.05, time_limit_sec - (time.time() - t0))
    configure_model_for_heuristic(model, time_limit=remain, seed=seed, threads=1,
                                  mip_focus=1, heuristics=0.9, submip_nodes=int(params.get('submip_nodes', 1000)),
                                  start_node_limit=int(params.get('start_node_limit', 500)),
                                  start_time_limit=float(params.get('start_time_limit', 0.0)),
                                  improve_start_time=float(params.get('improve_start_time', 0.0)),
                                  rins=int(params.get('rins', 50)), presolve=2, cuts=-1, symmetry=-1)
    try:
        model.Params.ImproveStartTime = 0.0
    except Exception:
        pass
    try:
        model.Params.ImproveStartNodes = 0
    except Exception:
        pass
    model.optimize()
    incumbent = collect_solution(model)
    diag = {'model': model_stats, 'lp_obj': lp_obj, 'lp_meta': lp_meta, 'solver_status': int(model.Status)}
    if incumbent is not None:
        diag['incumbent_obj'] = incumbent['objective']
    return make_result('polishing_mip', params, time.time() - t0, incumbent is not None, incumbent,
                       ['Aggressive incumbent-improvement pass using Gurobi improvement-focused settings.',
                        'Subdivide by seeds and parameter sets for start-improvement emphasis.'], diag)
