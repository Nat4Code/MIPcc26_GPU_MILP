# MIPcc26

Current two-phase heuristic workflow, deterministic per-instance run directories, incumbent merging, final Gurobi warmstart solve, and benchmark support.

This repository is a research scaffold for **parallel primal heuristics for MILP**, followed by a **final exact Gurobi solve** using the best farm incumbent as a MIP start.

## Current workflow

```text
instance.mps
   ↓
feature_extract.py
   ↓
make_plan.py
   ↓
plan.json  (two-phase allocation + shards + objective sense)
   ↓
Phase 1 parallel farm
   ↓
merge / incumbent handoff
   ↓
Phase 2 warmstart heuristic battery
   ↓
merge / incumbent handoff
   ↓
final_gurobi_solve.py
   ↓
final exact solve under a capped time budget
   ↓
run outputs + incumbent logs + summary artifacts
```

## What is now authoritative

The following **phase-1 heuristic implementations** are the working reference versions and should not be overwritten by placeholder scaffolds:

- `heuristics/greedy.py`
- `heuristics/rens.py`
- `heuristics/local_search_lp.py`
- `heuristics/dive_fix.py`
- `heuristics/feasibility_pump.py`

These are LP-guided or LP-seeded primal heuristics that already match the desired task/result format used by the pipeline.

## Output layout

Run outputs should be written to:

```text
milp_runs/<instance_stem>/
```

For example:

```text
milp_runs/instance_01/
```

instead of timestamp-only directories such as `run_20260407T043653Z`.

A typical per-instance output directory contains:

- `features.json`
- `plan.json`
- `lp_seed.json` when the shared LP seed stage is enabled
- `phase1_results/*.json`
- `phase2_results/*.json`
- `merged_phase1.json`
- `merged_phase2.json`
- `merged.json`
- `final_gurobi.json`
- `bound_probe_incumbents.csv` when the concurrent proof probe is enabled
- `final_gurobi_incumbents.csv`
- benchmark plots when run through the benchmarking driver

## Heuristic logging

Incumbent-time logs are now written as callback/event traces where possible:

### 1. Heuristic method trace

`heuristic_log.csv`, when present, should record the best incumbent found over time during the farm phases. The current benchmark gatherer can also derive incumbent events directly from the phase JSON files, so CSV logs are useful but not required for the active workflow. At minimum:

- wall-clock seconds from start of heuristic phase
- method name
- shard/task id
- incumbent objective
- whether the incumbent improved the global farm best

This can be assembled from shard result JSON plus the merged summary.

### 2. Exact Gurobi callback traces

`final_gurobi_incumbents.csv` records final exact-solve incumbents from a Gurobi `MIPSOL` callback. If the bound probe is enabled, `bound_probe_incumbents.csv` records the same kind of callback trace for the one-thread proof job. Benchmark baseline runs also write `<instance>_baseline_incumbents.csv` next to the captured baseline log.

Core columns:

- `time_sec`
- `objective`
- `incumbent_objective`
- `method`
- `phase`
- `source`

The benchmark gatherer prefers these CSV traces when present and falls back to parsing Gurobi text logs for older runs.

## Benchmarking script

A benchmark driver should run over all instances in `tests/`, compare the full pipeline against a plain Gurobi baseline, and produce both per-instance traces and aggregate plots.

### Default benchmark parameters

- compute units: `16`
- pipeline heuristic budget: `300` seconds by default unless overridden
- final exact phase: configurable
- Gurobi baseline: same total wall-clock budget, or a clearly documented alternative budget rule

### Expected benchmark outputs

#### 1. Incumbent-vs-time line plot

For each instance, plot:

- **our method**: incumbent objective vs time
- **Gurobi baseline**: incumbent objective vs time

Use step plots, not interpolated smooth curves.

#### 2. Aggregate primal-integral bar graph

Compute primal integral over the granted solve horizon for:

- pipeline method
- plain Gurobi baseline

For minimization problems, **smaller primal integral is better**.

#### 3. Aggregate best-incumbent winner bar graph

Count, over the test set, which method achieved the best incumbent within the granted solve time:

- pipeline method
- plain Gurobi baseline

Optional but useful:

- fastest-to-best-final bar graph
- mean final incumbent gap bar graph
- per-family contribution counts for the farm methods

## Two-phase heuristic battery

## Phase 1: broad primal farm

Current authoritative methods:

### Greedy
- solve LP relaxation
- rank integer variables using fractionality, reduced costs, objective information, and lock counts
- construct a start vector
- pass the start to a short improvement solve

### RENS
- solve LP relaxation
- fix near-integral variables
- leave a restricted neighborhood around the fractional core
- solve a sub-MIP under a cap

### Local search off LP
- seed from rounded LP values
- search either a local-branching neighborhood or a restricted free-variable neighborhood
- solve a short neighborhood sub-MIP

### Dive-and-fix
- repeated LP guidance
- batch variable fixing according to fractionality/reduced-cost/objective signals
- shallow backtracking
- short capped sub-MIP solves

### Added first-pass family: Feasibility Pump
- randomized LP-rounding / projection style restarts
- multiple seeds and restart policies subdivide naturally across workers
- best used as a diversification-heavy feasible-solution generator

## Phase 2: incumbent-centered warmstart battery

Intended methods:

### RINS warmstart
Apply a RINS-style neighborhood around the current best incumbent and LP relaxation agreement.

### Local Branching via Hamming ball
Search a neighborhood defined by a Hamming-distance style constraint around the incumbent.

### LNS Fix & Optimize / block neighborhoods
Fix most variables and free a block or partition at a time.

### Polishing / incumbent improvement mode
Use incumbent-focused Gurobi settings to improve the current solution rather than spend budget on broad exploration.

### Objective-bounded neighborhood search
Add an incumbent-improvement threshold and search only neighborhoods that can beat the current objective by a specified amount or fraction.

### Feasibility Pump with random seeds
A second-wave diversification pass can still be useful after phase 1 if incumbent quality is weak or feasibility was hard to obtain.

## Subdividing methods across `n` compute units

The clean design is to give each heuristic family a parameter grid and then shard that grid across workers.

### Good subdivision knobs by method

#### Greedy
- seed
- weighting of fractionality / reduced cost / objective / locks
- repair effort
- short improvement cap

#### RENS
- integral tolerance
- free radius
- restrict band
- sub-MIP node cap
- seed

#### Local Search off LP
- neighborhood type
- neighborhood size
- move policy
- seed

#### Dive-and-Fix
- branch score policy
- fix batch size
- backtrack depth
- iteration cap
- seed

#### Feasibility Pump
- random seed
- restart count
- perturbation strength
- repair cap

#### RINS warmstart
- agreement threshold
- sub-MIP node cap
- neighborhood tightness
- seed

#### Local Branching
- Hamming radius `k`
- node cap
- seed

#### LNS Fix & Optimize
- block id
- block size
- variable ordering policy
- seed

#### Polishing
- improvement-only parameter bundles
- node cap
- seed

#### Objective-bounded search
- target improvement ratio
- neighborhood size
- seed

## Structure-aware planning

`make_plan.py` uses a **structure-aware allocator**, not a flat static split.

It uses lightweight instance features from `feature_extract.py`, including model size, binary share, general-integer share, integrality ratio, matrix density, and average row/column nonzeros. The generated `plan.json` records:

- `structure_profile`
- `phase1_method_weights`
- `phase2_method_weights`
- `allocation_diagnostics`

The current rule-based allocator biases:

- large models toward cheaper starts, feasibility pump, LNS, and polishing
- binary-heavy models toward RENS, LP local search, feasibility pump, and Hamming-ball local branching
- general-integer/no-binary models away from Hamming-ball local branching and toward dive/fix, RINS, LNS, and polishing
- sparse models toward greedy, dive/fix, and block neighborhoods
- dense or highly coupled models toward restricted sub-MIPs, polishing, and objective-bounded search

Honest status:
- a **rule-based structure-aware planner** is a realistic immediate step
- a true **reinforcement-learning allocator** is a later step and needs training data from many solved benchmark runs

### Practical RL path

A credible RL / learning path is:
1. collect per-instance features
2. collect per-method performance traces under fixed budgets
3. train a policy or contextual bandit to predict worker allocations
4. compare learned allocation against static and rule-based baselines

Until then, use a deterministic feature-based allocator.

## GPU LP phase

The intended long-term structure is:

1. read model and extract features
2. run a fast LP phase
3. launch phase-1 heuristics
4. launch phase-2 warmstart heuristics
5. hand best incumbent to final exact Gurobi solve

For the **GPU LP** component, the most credible near-term direction is **NVIDIA cuOpt**, which documents GPU-accelerated LP capabilities including PDLP-based methods and a barrier LP solver. Gurobi itself remains the exact CPU-parallel MILP engine in this workflow, so GPU LP should be treated as an optional front-end acceleration stage rather than a replacement for the exact MILP backend.

Recommendation:
- keep GPU LP as an optional module behind a config switch
- if GPU LP fails or is unavailable, fall back immediately to the existing CPU LP relaxation path
- do not entangle the rest of the heuristic code with a hard dependency on GPU infrastructure

## Typical local tests

The source uses modern Python syntax. On systems where `python3` is older than 3.7, run these commands with a newer interpreter such as `python3.11`, or run them inside the Apptainer image.

## Enter Gurobi Apptainer

Note: put your academic license file one directory up from the current working directory.

If `apptainer` is not visible in the batch environment, either load the site module before running or set `CONTAINER_RUNTIME=/usr/bin/apptainer` / `CONTAINER_RUNTIME=/usr/bin/singularity`. The workflow scripts will also try those common paths automatically.

```bash
apptainer shell \
  --bind "$PWD:/work" \
  --bind "$PWD/../gurobi.lic:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  gurobi.sif
```

## To run baseline on an individual case

```bash
./baseline tests/instance_01.original.mps 300 16
```

## To run a test on an individual case
```bash
PHASE1_COMPUTE_UNITS=16 \
PHASE2_COMPUTE_UNITS=16 \
PHASE1_NUM_TASKS=8 \
PHASE2_NUM_TASKS=8 \
PHASE1_THREADS_PER_TASK=2 \
PHASE2_THREADS_PER_TASK=2 \
PHASE1_MIN_TASK_SECONDS=45 \
PHASE2_MIN_TASK_SECONDS=75 \
LP_SEED_ENABLE=1 \
LP_SEED_SECONDS=30 \
LP_THREADS=16 \
LP_METHOD=1 \
LP_REPAIR_SECONDS=5 \
HEURISTIC_WALL_SECONDS=150 \
PHASE1_WALL_SECONDS=75 \
PHASE2_WALL_SECONDS=75 \
FINAL_SECONDS=150 \
FINAL_THREADS=16 \
FINAL_FOCUS_MODE=prove \
FINAL_HEURISTICS=0.02 \
FINAL_CUTS=2 \
FINAL_PRESOLVE=2 \
PHASE2_SKIP_REL_GAP=0.01 \
ALLOW_PARTIAL_PHASE_FAILURES=1 \
bash slurm/submit_workflow.sh tests/instance_03.original.mps config/default_config.json
```

`FINAL_FOCUS_MODE=prove` maps to a bound/gap-closing final solve. It keeps the best heuristic incumbent as a MIP start, but shifts Gurobi toward proof work (`MIPFocus=3` by default), stronger cuts, and lower heuristic effort. You can override individual settings with `FINAL_MIP_FOCUS`, `FINAL_CUTS`, `FINAL_CUT_PASSES`, `FINAL_PRESOLVE`, `FINAL_MIP_GAP`, and `FINAL_MIP_GAP_ABS`.

`PHASE2_SKIP_REL_GAP` is optional. When set, the workflow merges phase 1, checks the best incumbent against the best valid bound reported by phase-1 artifacts, and skips phase 2 only if the relative gap is at or below that threshold. `PHASE2_SKIP_ABS_GAP` provides the same guard using an absolute gap threshold. If no usable bound is available, phase 2 runs normally.

When phase 2 is skipped, `REALLOCATE_SKIPPED_PHASE2_TO_FINAL=1` is the default. This adds the skipped phase-2 wall budget onto `FINAL_SECONDS`, so a benchmark run still spends the intended total method time on the instance.

`BOUND_PROBE_ENABLE=1` optionally launches a separate one-thread Gurobi proof job while the heuristic phases run. It uses the LP seed incumbent if available, otherwise it runs without a start. Its output is written to `bound_probe.json` and is merged before the final exact solve, so any incumbent it finds can be used and its bound can improve reported gap estimates. Set `BOUND_PROBE_USE_FOR_PHASE1_GAP=1` when you want the workflow to wait for this probe immediately after phase 1, create `merged_phase1_with_bound_probe.json`, and use that combined incumbent/bound pair for the phase-2 skip decision. This job consumes an additional CPU while it overlaps the heuristic farm; reserve a core for it if you need a hard total-core cap.

## To run the benchmarker for all instances

```bash
python3 scripts/benchmarking_submit.py \
  --tests-dir tests \
  --config config/default_config.json \
  --seconds 300 \
  --compute-units 16

apptainer shell \
  --bind "$PWD:/work" \
  --bind "$PWD/../gurobi.lic:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  gurobi.sif

python3 scripts/benchmarking_gathering.py \
  --manifest results/benchmarking_raw/<timestamp>/manifest.json
```
---

### Per-family sanity tests

```bash
python3.11 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps greedy 3
python3.11 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps rens 3
python3.11 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps local_search_lp 3
python3.11 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps dive_fix 3
python3.11 scripts/test_heuristic.py config/default_config.json tests/instance_01.original.mps feasibility_pump 3
```

### Full local run

```bash
python3.11 scripts/test_pipeline_with_final_solve.py \
  config/default_config.json \
  tests/instance_01.original.mps \
  --farm-seconds 300 \
  --final-seconds 300
```

### Syntax check

```bash
python3.11 -B -m py_compile scripts/*.py heuristics/*.py
```

## Cluster execution

```bash
sbatch slurm/submit_workflow.sh tests/instance_01.original.mps config/default_config.json
```

## Scope

This code is a **research framework**, not a finished production solver.

The near-term objective is straightforward:
- improve incumbent quality quickly with a parallel primal farm
- use deterministic logging and benchmark plots
- learn which MILP structures favor which heuristic families
- pass the best farm incumbent into a final exact Gurobi solve

## References

1. Emilie Danna, Edward Rothberg, Claude Le Pape. **Exploring relaxation induced neighborhoods to improve MIP solutions**. *Mathematical Programming*, 2005. RINS is the canonical incumbent-centered relaxation neighborhood method. https://link.springer.com/article/10.1007/s10107-004-0518-7

2. Matteo Fischetti, Andrea Lodi. **Local Branching**. *Mathematical Programming*, 2003. Classic Hamming-ball neighborhood search for MIP incumbents. See bibliographic confirmation in the cited survey and scholar records. https://scholar.google.com/citations?hl=vi&user=5rOdaqAAAAAJ

3. Matteo Fischetti, Fred Glover, Andrea Lodi. **The Feasibility Pump**. *Mathematical Programming*, 2005. Foundational primal heuristic for finding feasible MILP solutions quickly. Bibliographic confirmation: https://scholar.google.com/citations?hl=vi&user=5rOdaqAAAAAJ

4. Thorsten Berthold. **RENS – The Relaxation Enforced Neighborhood Search**. ZIB report / SCIP-related line of work. RENS is a standard LP-neighborhood primal heuristic family and is widely cited in later SCIP/ZIB work. A later ZIB document explicitly references the introduction of RENS. https://webdoc.sub.gwdg.de/ebook/serien/ah/ZIB/14_14.pdf

5. Thorsten Berthold. **Primal Heuristics for Mixed Integer Programs**. Dissertation / ZIB-related work surveying practical primal heuristics including large-neighborhood and improvement methods. https://opus4.kobv.de/opus4-zib/files/1029/Berthold_Primal_Heuristics_For_Mixed_Integer_Programs.pdf

6. Gurobi documentation / support on incumbent-focused improvement settings. Gurobi documents solution-improvement behavior and polishing-style incumbent improvement through parameters such as `ImproveStartTime` and related settings. https://support.gurobi.com/hc/en-us/articles/360013420711-Does-Gurobi-have-a-solution-polishing-algorithm

7. NVIDIA cuOpt documentation. NVIDIA documents GPU-accelerated LP solving, including PDLP-based LP methods and barrier LP support, which makes cuOpt the most credible candidate for an optional GPU LP front-end in this workflow. https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html

8. Lodi-style feasibility pump follow-up work and later refinements remain relevant when building randomized multi-seed pump variants. Example: **A New Approach to the Feasibility Pump in Mixed Integer Programming**. SIAM Journal on Optimization. https://epubs.siam.org/doi/10.1137/110823596

## Disclosure Note:

AI tools were used to scrape and pull research implementations repositories for the heuristic integrations and was sub-contracted to draft some portions of the code more rapidly.