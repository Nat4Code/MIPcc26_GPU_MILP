import random
import time

def run_heuristic(instance_path: str, params: dict, time_limit_sec: float) -> dict:
    t0 = time.time()

    random.seed(params.get("seed", 0))
    fake_obj = 960.0 - 120.0 * random.random()

    time.sleep(min(0.04, time_limit_sec))

    return {
        "method": "dive_fix",
        "params": params,
        "status": "ok",
        "objective": fake_obj,
        "feasible": True,
        "runtime_sec": time.time() - t0
    }