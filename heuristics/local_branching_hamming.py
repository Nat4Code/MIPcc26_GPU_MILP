import time
from typing import Any, Dict, List, Tuple

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    integer_vars,
    is_binary_var,
    make_result,
    incumbent_from_features,
    read_model,
    seed_from_lp_values,
    solve_lp_relaxation,
    try_root_seed,
)


def _make_start(model, x_lp: Dict[str, float]):
    start = {}
    ranked: List[Tuple[float, Any]] = []
    for v in integer_vars(model):
        x = x_lp.get(v.VarName)
        if x is None:
            continue
        val = 1.0 if (v.VType == 'B' and x >= 0.5) else round(x)
        val = max(v.LB, min(v.UB, val))
        start[v.VarName] = float(val)
        ranked.append((abs(x - round(x)), v))
    ranked.sort(key=lambda t: (-t[0], t[1].VarName))
    return start, [v for _, v in ranked]


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get('seed', 0))
    k = int(params.get('k', 16))
    model = read_model(instance_path)
    model_stats = describe_model(model)
    warm_vals, warm_obj, warm_method = incumbent_from_features(features)
    lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(model, time_limit=max(0.03, min(0.20, 0.20 * time_limit_sec)))
    if warm_vals:
        start_values = dict(warm_vals)
        ranked_vars = [v for v in integer_vars(model) if v.VarName in start_values]
        lp_meta = {**lp_meta, "warmstart_source": warm_method, "warmstart_objective": warm_obj}
    elif x_lp:
        start_values, ranked_vars = _make_start(model, x_lp)
        start_values.update(seed_from_lp_values(model, x_lp, rc))
    else:
        seed_vals, seed_meta = try_root_seed(model, time_limit=max(0.03, min(0.20, 0.25 * time_limit_sec)), seed=seed)
        if not seed_vals:
            return make_result('local_branching_hamming', params, time.time() - t0, False, None,
                               ['Local branching over a seeded incumbent.'],
                               {'model': model_stats, 'lp_meta': lp_meta, 'root_seed_meta': seed_meta})
        start_values = seed_vals
        ranked_vars = [v for v in integer_vars(model)]
        lp_meta = {**lp_meta, 'root_seed_meta': seed_meta}

    apply_partial_start(model, start_values)
    expr = 0.0
    used = 0
    for v in ranked_vars:
        if not is_binary_var(v) or v.VarName not in start_values:
            continue
        s = int(round(start_values[v.VarName]))
        expr += (1 - v) if s == 1 else v
        used += 1
    if used > 0:
        model.addConstr(expr <= int(k), name=f'hamming_ball_{int(k)}')
        model.update()

    remain = max(0.05, time_limit_sec - (time.time() - t0))
    configure_model_for_heuristic(model, time_limit=remain, seed=seed, threads=1,
                                  mip_focus=1, heuristics=0.45, submip_nodes=int(params.get('submip_nodes', 1500)),
                                  start_node_limit=300, start_time_limit=min(0.35, max(0.05, remain / 2.0)),
                                  improve_start_time=min(0.30, max(0.05, remain / 3.0)), rins=20,
                                  presolve=2, cuts=-1, symmetry=-1)
    model.optimize()
    incumbent = collect_solution(model)
    diag = {'model': model_stats, 'lp_obj': lp_obj, 'lp_meta': lp_meta, 'k': k, 'used_binary_vars': used,
            'solver_status': int(model.Status)}
    if incumbent is not None:
        diag['incumbent_obj'] = incumbent['objective']
    return make_result('local_branching_hamming', params, time.time() - t0, incumbent is not None, incumbent,
                       ['Local Branching style hamming-ball neighborhood around the seed.',
                        'Subdivide by assigning different radii k and seeds to different shards.'], diag)
