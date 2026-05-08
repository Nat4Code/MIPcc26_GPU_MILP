#!/usr/bin/env python3
"""
gurobi_baseline_basic.py

Minimal Gurobi baseline:
  - reads one MPS/LP model
  - runs Gurobi for 300 seconds by default
  - uses 16 threads by default
  - prints normal Gurobi output to the terminal
  - prints a final summary

Usage inside Apptainer:
  python3 scripts/gurobi_baseline_basic.py tests/instance_03.original.mps

Optional:
  python3 scripts/gurobi_baseline_basic.py tests/instance_03.original.mps --time-limit 300 --threads 16 --seed 0
"""

from __future__ import print_function

import argparse
import sys

import gurobipy as gp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="Path to .mps/.lp model file")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("===== GUROBI BASELINE BASIC =====")
    print("Instance:   {}".format(args.instance))
    print("TimeLimit:  {}".format(args.time_limit))
    print("Threads:    {}".format(args.threads))
    print("Seed:       {}".format(args.seed))
    print("=================================")
    print("")

    model = gp.read(args.instance)

    model.Params.TimeLimit = float(args.time_limit)
    model.Params.Threads = int(args.threads)
    model.Params.Seed = int(args.seed)
    model.Params.OutputFlag = 1

    model.optimize()

    print("")
    print("===== FINAL BASELINE RESULT =====")
    print("Status code: {}".format(model.Status))
    print("Runtime:     {:.6f} sec".format(float(model.Runtime)))
    print("SolCount:    {}".format(int(model.SolCount)))

    if model.SolCount > 0:
        print("Best objective: {:.12g}".format(float(model.ObjVal)))
        try:
            print("Best bound:     {:.12g}".format(float(model.ObjBound)))
        except Exception:
            pass
        try:
            print("MIP gap:        {:.12g}".format(float(model.MIPGap)))
        except Exception:
            pass
    else:
        print("No incumbent solution found.")

    print("=================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())