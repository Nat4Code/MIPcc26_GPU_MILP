import gurobipy as gp
from gurobipy import GRB
import random
import time

def run_heuristic(instance_path: str, params: dict, time_limit_sec: float) -> dict:
    """runs the dive and fix heuristic

    Args:
        instance_path (str): path to instance
        params (dict): _description_
        time_limit_sec (float): _description_

    Returns:
        dict: _description_
    """
    t0 = time.time()
    
    model = gp.read(instance_path)
    relax = model.relax()
    
    while (time.time() -t0) < time_limit_sec:
        relax.optimize()
        
        if relax.Status != GRB.OPTIMAL:
            break
        fractional_vars = [
            v for v in relax.getVars() 
            if v.VType != GRB.CONTINUOUS and abs(v.X - round(v.X)) > 1e-6
        ]
        if not fractional_vars:
            return{
            "method": "dive_fix",
            "params": params,
            "status": "ok",
            "objective": model.get_objective(),
            "feasible": True,
            "runtime_sec": time.time() - t0
            }
        var_to_fix = min(fractional_vars, key=lambda v: abs(v.X - round(v.X)))
        fixed_value = round(var_to_fix.X)
        var_to_fix.LB = fixed_value
        var_to_fix.UB = fixed_value
    return {"status": "timeout_or_infeasible", "feasible": False}

    # random.seed(params.get("seed", 0))
    # fake_obj = 960.0 - 120.0 * random.random()

    # time.sleep(min(0.04, time_limit_sec))

    # return {
    #     "method": "dive_fix",
    #     "params": params,
    #     "status": "ok",
    #     "objective": fake_obj,
    #     "feasible": True,
    #     "runtime_sec": time.time() - t0
    # }