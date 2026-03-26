#!/usr/bin/env python3
import json
import os
import sys

def extract_features(instance_path: str) -> dict:
    # Placeholder feature extraction.
    # Replace with actual MPS/LP parser or solver-based model introspection.
    #
    # Good future features:
    # - number of vars, binaries, integers, continuous
    # - number of constraints
    # - density / nnz
    # - ratio of binaries
    # - objective sparsity
    # - row type counts
    # - LP relaxation gap estimate
    # - graph/community statistics
    # - presolve reductions
    #
    # For now: dummy features keyed off file size.
    file_size = os.path.getsize(instance_path)

    return {
        "instance_path": instance_path,
        "file_size_bytes": file_size,
        "num_vars_est": max(100, file_size // 50),
        "num_bin_est": max(20, file_size // 200),
        "num_constr_est": max(50, file_size // 100),
        "density_est": 0.02,
        "bin_ratio_est": 0.25
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

if __name__ == "__main__":
    main()