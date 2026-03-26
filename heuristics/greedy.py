import random
import time

def run_heuristic(instance_path: str, params: dict, time_limit_sec: float) -> dict:
    t0 = time.time()

    # Placeholder greedy logic.
    random.seed(params.get("seed", 0))
    fake_obj = 1000.0 - 100.0 * random.random()

    time.sleep(min(0.02, time_limit_sec))

    return {
        "method": "greedy",
        "params": params,
        "status": "ok",
        "objective": fake_obj,
        "feasible": True,
        "runtime_sec": time.time() - t0
    }