import time
from typing import Any, Dict, List, Tuple

from scripts.common import (
    apply_partial_start,
    collect_solution,
    configure_model_for_heuristic,
    describe_model,
    integer_vars,
    make_result,
    incumbent_from_features,
    read_model,
    seed_from_lp_values,
    solve_lp_relaxation,
    try_root_seed,
)


def _rank_vars(model, x_lp: Dict[str, float]) -> List[Any]:
    items: List[Tuple[float, Any]] = []
    for v in integer_vars(model):
        x = x_lp.get(v.VarName)
        frac = 0.0 if x is None else abs(x - round(x))
        score = 0.8 * frac + 0.2 * abs(float(v.Obj))
        items.append((score, v))
    items.sort(key=lambda t: (-t[0], t[1].VarName))
    return [v for _, v in items]


def run_heuristic(instance_path: str, params: Dict[str, Any], time_limit_sec: float,
                  features: Dict[str, Any] | None = None) -> Dict[str, Any]:
    t0 = time.time()
    seed = int(params.get('seed', 0))
    block_size = int(params.get('block_size', 32))
    block_id = int(params.get('block_id', 0))
    model = read_model(instance_path)
    model_stats = describe_model(model)
    warm_vals, warm_obj, warm_method = incumbent_from_features(features)
    lp_obj, x_lp, rc, _, lp_meta = solve_lp_relaxation(model, time_limit=max(0.03, min(0.20, 0.20 * time_limit_sec)))
    if warm_vals:
        start_values = dict(warm_vals)
        ranked = [v for v in integer_vars(model) if v.VarName in start_values]
        lp_meta = {**lp_meta, "warmstart_source": warm_method, "warmstart_objective": warm_obj}
    elif x_lp:
        start_values = seed_from_lp_values(model, x_lp, rc)
        ranked = _rank_vars(model, x_lp)
    else:
        seed_vals, seed_meta = try_root_seed(model, time_limit=max(0.03, min(0.20, 0.25 * time_limit_sec)), seed=seed)
        if not seed_vals:
            return make_result('lns_fix_optimize', params, time.time() - t0, False, None,
                               ['Fix-and-optimize block neighborhood search.'],
                               {'model': model_stats, 'lp_meta': lp_meta, 'root_seed_meta': seed_meta})
        start_values = seed_vals
        ranked = [v for v in integer_vars(model)]
        lp_meta = {**lp_meta, 'root_seed_meta': seed_meta}

    apply_partial_start(model, start_values)
    if ranked:
        start_idx = (block_id * block_size) % len(ranked)
        free_names = {ranked[(start_idx + i) % len(ranked)].VarName for i in range(min(block_size, len(ranked)))}
    else:
        free_names = set()
    fixed_count = 0
    for v in integer_vars(model):
        if v.VarName in free_names or v.VarName not in start_values:
            continue
        val = start_values[v.VarName]
        v.LB = val
        v.UB = val
        fixed_count += 1
    model.update()

    remain = max(0.05, time_limit_sec - (time.time() - t0))
    configure_model_for_heuristic(model, time_limit=remain, seed=seed, threads=1,
                                  mip_focus=1, heuristics=0.40, submip_nodes=int(params.get('submip_nodes', 2000)),
                                  start_node_limit=300, start_time_limit=min(0.35, max(0.05, remain / 2.0)),
                                  improve_start_time=min(0.30, max(0.05, remain / 3.0)), rins=20,
                                  presolve=2, cuts=-1, symmetry=-1)
    model.optimize()
    incumbent = collect_solution(model)
    diag = {'model': model_stats, 'lp_obj': lp_obj, 'lp_meta': lp_meta, 'block_size': block_size,
            'block_id': block_id, 'fixed_count': fixed_count, 'free_count': len(free_names),
            'solver_status': int(model.Status)}
    if incumbent is not None:
        diag['incumbent_obj'] = incumbent['objective']
    return make_result('lns_fix_optimize', params, time.time() - t0, incumbent is not None, incumbent,
                       ['Fix-and-optimize large-neighborhood search around a seed.',
                        'Subdivide directly by disjoint or cyclic variable blocks across shards.'], diag)
