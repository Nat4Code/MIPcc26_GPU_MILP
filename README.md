# MIPcc26_GPU_MILP
Applying multi-processing primal heuristic algorithms to Mixed Integer Linear Programs (MILPs) on the MIPcc26 problem instances.

Enter Gurobi Apptainer:
======================================================================
Note: put your licence file one directory up from the pwd.

apptainer shell \
  --bind "$PWD:/work" \
  --bind "$PWD/../gurobi.lic:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  gurobi.sif

To run baseline on an individual case:
./baseline tests/instance_01.original.mps

Workflow:
=======================================
instance.mps
   ↓
feature_extract.py
   ↓
make_plan.py
   ↓
plan.json  (allocation + shards)
   ↓
sbatch --array=0-(K-1) run_array_task.sh
   ↓
run_task.py → heuristic stub
   ↓
JSON result per shard
   ↓
merge_results.py
   ↓
best incumbent summary

How to Run:
=========================================
chmod +x slurm/submit_workflow.sh
chmod +x slurm/run_array_task.sh
chmod +x scripts/feature_extract.py
chmod +x scripts/make_plan.py
chmod +x scripts/run_task.py
chmod +x scripts/merge_results.py

bash slurm/submit_workflow.sh inputs/example_instance.mps

python3 scripts/merge_results.py results results/summary.json

Greedy
- Use:
* LP reduced-cost guidance
* objective coefficient ranking
* pseudo-cost style prioritization
* lock counts / constraint activity

RENS
- Use:
* solve LP relaxation
* fix near-integral vars
* define a reduced neighborhood
* solve sub-MIP for a short cap

Local search off LP
- Use:
* start from rounded LP or repair solution
* 1-flip / 2-flip neighborhoods
* ruin-and-recreate
* feasibility repair penalties

Dive-and-fix
* Use:
* repeated fixing based on fractionality / reduced cost / inference
* shallow backtracking
* short subproblem solves

Research question we will attempt to answer:
Given a MILP instance and a small primal-heuristic budget (roughly 0.1–5 seconds), can instance-structure features predict an allocation of parallel compute effort across multiple primal heuristics that improves incumbent quality relative to static allocations?