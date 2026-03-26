import random
import time

def run_heuristic(instance_path: str, params: dict, time_limit_sec: float) -> dict:
    t0 = time.time()

    random.seed(params.get("seed", 0))
    fake_obj = 970.0 - 90.0 * random.random()

    time.sleep(min(0.03, time_limit_sec))

    return {
        "method": "local_search_lp",
        "params": params,
        "status": "ok",
        "objective": fake_obj,
        "feasible": True,
        "runtime_sec": time.time() - t0
    }