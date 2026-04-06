# MIPcc26_GPU_MILP

Applying parallel primal heuristic algorithms to Mixed Integer Linear Programs (MILPs) on the MIPcc26 problem instances, with a final incumbent handoff into Gurobi for exact solving.

## Enter Gurobi Apptainer

Note: put your academic license file one directory up from the current working directory.

```bash
apptainer shell \
  --bind "$PWD:/work" \
  --bind "$PWD/../gurobi.lic:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  gurobi.sif
```

## To run baseline on an individual case

```bash
./baseline tests/instance_01.original.mps
```

---

## Current workflow

```text
instance.mps
   ↓
feature_extract.py
   ↓
make_plan.py
   ↓
plan.json  (allocation + shards + objective sense)
   ↓
SLURM array / run_array_task.sh
   ↓
run_task.py → one heuristic shard
   ↓
JSON result per shard
   ↓
merge_results.py
   ↓
best incumbent summary
   ↓
final_gurobi_solve.py
   ↓
best farm incumbent passed to Gurobi as a MIP start
   ↓
final exact solve under a capped time budget
```

This is now a **two-phase workflow**.

### Phase 1: Parallel primal heuristic farm

A portfolio of primal heuristics is launched across a fixed number of array tasks.

Each heuristic family receives:
- some number of tasks
- a parameter grid
- a subdivision of that grid into shards
- one SLURM array task per shard

Each shard runs independently and writes a JSON result file containing:
- candidate heuristic results
- the best incumbent found in that shard
- diagnostics and timing

### Phase 2: Final Gurobi solve

After all heuristic shards finish:
- results are merged
- the best incumbent found by the farm is extracted
- that incumbent is passed to Gurobi as a MIP start
- Gurobi then performs a final exact solve under a separate time cap

---

## High-level research goal

We are studying whether a MILP instance’s structure can predict how to allocate a small primal heuristic budget across multiple heuristic families in parallel. And is this process scalable?

### Main research question

Given a MILP instance and a small primal-heuristic budget, can instance-structure features predict an allocation of parallel compute effort across multiple primal heuristics that improves incumbent quality relative to static allocations?

### Two sub-problems

#### A. Allocation problem

How many tasks or cores should go to each heuristic family?

#### B. Intra-heuristic partition problem

Given a method’s budget, how should its local search or parameter grid be subdivided?

---

## Heuristic families

### 1) Greedy
- solves the LP relaxation
- ranks integer variables using reduced costs, objective coefficients, pseudo-cost-style scoring proxies, and lock counts
- builds a rounded start vector
- hands the start to a short Gurobi improvement run

### 2) RENS
- solves the LP relaxation
- fixes near-integral variables
- restricts remaining integer variables to a tiny relaxation-induced neighborhood
- solves the resulting sub-MIP under a hard cap

### 3) Local search off LP
- starts from the rounded LP point
- either frees a small set of promising variables or adds a local-branching-style neighborhood around the start
- solves the neighborhood under a short cap
- currently this family is the one most sensitive to how much time is available per parameter point

### 4) Dive-and-fix
- repeatedly solves the LP relaxation of the current partially fixed model
- fixes a small batch of variables using fractionality, reduced-cost, or inference-style dive scores
- allows shallow backtracking
- finishes with a short capped MIP solve

---

## Current structure of the pipeline

### `scripts/feature_extract.py`
- reads the MPS with Gurobi
- extracts lightweight structure features
- records the model objective sense (`min` or `max`)
- writes `features.json`

### `scripts/make_plan.py`
- builds the parallel execution plan
- chooses how many array tasks go to each heuristic family
- splits each family’s parameter grid into shards
- writes `plan.json`

### `scripts/run_task.py`
- executes one shard from `plan.json`
- runs each parameter point in that shard
- tracks the best effective objective found
- writes one task JSON file

### `scripts/merge_results.py`
- collects all task JSON files
- merges them into one summary
- picks the best incumbent across the whole heuristic farm

### `scripts/final_gurobi_solve.py`
- reads the merged heuristic summary
- extracts the best incumbent
- applies it to the original MILP as a Gurobi MIP start
- runs a final exact solve under a separate cap

### `scripts/test_heuristic.py`
- sanity-checks one heuristic family at a time

### `scripts/test_pipeline_with_final_solve.py`
- local end-to-end test of the full two-phase workflow:
  - feature extraction
  - plan building
  - shard execution
  - merge
  - final Gurobi solve

---

## Current budget model

The pipeline is currently built around:
- a **farm budget** for the heuristic array phase
- a **final solve budget** for the Gurobi exact phase

For example:
- `FARM_SECONDS=160`
- `FINAL_SECONDS=50`

With 16 array tasks, a farm budget of 160 seconds means:
- about **10 seconds per array task**
- with array tasks running in parallel, this is about **10 seconds of wall-clock** for the farm phase
- then about **50 seconds** for the final Gurobi solve
- plus overhead for startup, file I/O, merge, and scheduling

So actual wall-clock runtime is usually on the order of:
- **~60–75 seconds**, not counting queue delay

### Important note on per-candidate time

Each task may contain multiple parameter points.

`run_task.py` now splits the task budget across the candidates in that shard, so:
- task budget is honest at the shard level
- per-candidate time depends on shard size

This matters especially for `local_search_lp`, which often needs more time per parameter point than greedy or RENS.

---

## Apptainer + SLURM setup

The current SLURM scripts assume:
- `gurobi.sif` is the Apptainer image
- submission is done from the **host shell**
- Python execution is done **inside Apptainer**
- outputs are written to a writable run directory, not assumed to be the project root

### Default queued run

```bash
sbatch slurm/submit_workflow.sh tests/instance_03.original.mps config/default_config.json
```

### Example override

```bash
FARM_SECONDS=160 FINAL_SECONDS=50 APPTAINER_IMAGE=/path/to/gurobi.sif \
sbatch slurm/submit_workflow.sh tests/instance_03.original.mps config/default_config.json
```

---

## Sanity tests

### Per-family sanity test

```bash
python3 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps greedy 3
python3 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps rens 3
python3 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps local_search_lp 3
python3 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps dive_fix 3
```

### End-to-end local run with final Gurobi solve

```bash
python3 scripts/test_pipeline_with_final_solve.py \
  config/default_config.json \
  tests/instance_01.original.mps \
  --farm-seconds 160 \
  --final-seconds 50
```

### Cluster run

```bash
sbatch slurm/submit_workflow.sh tests/instance_01.original.mps config/default_config.json
```

---

## Current scope

This is simply a **research scaffold**, not a industry-standard solver replacement.

The code is designed for:
- short primal heuristic budgets
- controlled parallel budget allocation
- solver-backed experimentation with Gurobi (industry standard)
- later replacement of the hand-coded allocator with a learned allocator

This framework is intended to answer:
- whether different instance structures prefer different heuristic families
- whether static allocations can be beaten by structure-aware allocations
- how best to subdivide a family’s budget internally
- how these heuristical methods will react with scaling

---

## Requirements

- Python 3.10+
- `gurobipy`
- a working Gurobi license visible to the job environment
- Apptainer / Singularity on the cluster
- SLURM submission from the host shell

---

## Research references behind the scaffold

- RENS: T. Berthold, *RENS – The Relaxation Enforced Neighborhood Search*, ZIB Report 07-28, 2008.
- RINS neighborhood family: E. Danna, E. Rothberg, C. Le Pape, *Exploring relaxation induced neighborhoods to improve MIP solutions*, Mathematical Programming, 2005.
- Local branching: M. Fischetti, A. Lodi, *Local Branching*, Mathematical Programming, 2003.
- Feasibility pump background for rounded-start repair: M. Fischetti, F. Glover, A. Lodi, *The Feasibility Pump*, Mathematical Programming, 2005.
- Diving heuristics overview: SCIP heuristic framework and later conflict-driven diving work.
- Fix-and-dive implementation pattern: Gurobi example `fixanddive.py`.