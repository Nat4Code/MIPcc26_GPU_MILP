#!/usr/bin/env python3
import json
import os
import sys

import gurobipy as gp


def extract_features(instance_path: str) -> dict:
    model = gp.read(instance_path)
    model.update()

    num_vars = int(model.NumVars)
    num_bin = int(model.NumBinVars)
    num_int_total = int(model.NumIntVars)
    num_int_nonbin = max(0, num_int_total - num_bin)
    num_constr = int(model.NumConstrs)

    try:
        num_nz = int(model.NumNZs)
    except Exception:
        num_nz = 0

    density = 0.0
    if num_vars > 0 and num_constr > 0:
        density = num_nz / float(num_vars * num_constr)

    sense = "min" if int(model.ModelSense) == 1 else "max"

    return {
        "instance_path": instance_path,
        "file_size_bytes": os.path.getsize(instance_path),
        "num_vars_est": num_vars,
        "num_bin_est": num_bin,
        "num_int_est": num_int_nonbin,
        "num_constr_est": num_constr,
        "num_nz_est": num_nz,
        "density_est": density,
        "bin_ratio_est": (num_bin / num_vars) if num_vars > 0 else 0.0,
        "integrality_ratio_est": (num_int_total / num_vars) if num_vars > 0 else 0.0,
        "objective_sense": sense,
        "gurobi_model_sense": int(model.ModelSense),
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: feature_extract.py <instance.mps> <features.json>")
        sys.exit(1)

    instance_path = sys.argv[1]
    out_path = sys.argv[2]

    feats = extract_features(instance_path)

    with open(out_path, "w") as f:
        json.dump(feats, f, indent=2)

    print(f"Wrote features to {out_path}")
    print(f"Objective sense: {feats['objective_sense']}")


if __name__ == "__main__":
    main()