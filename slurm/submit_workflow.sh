#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# slurm/submit_workflow.sh
#
# Host-side two-wave MILP heuristic workflow for Slurm + Apptainer.
#
# What it does:
#   1. Extract features
#   2. Build two-phase heuristic plan
#   3. Optional shared LP seed solve:
#        - CPU-parallel Gurobi LP relaxation
#        - dual simplex by default so basis can be exported
#        - optional short repair MIP
#        - lp_seed.json is copied into phase1_results so it competes in merge
#   4. Submit phase-1 Slurm array and wait
#   5. Merge phase-1 JSONs into merged_phase1.json
#   6. Submit phase-2 Slurm array with merged_phase1.json as warmstart and wait
#   7. Merge phase-2 JSONs into merged_phase2.json
#   8. Merge both phases into merged.json
#   9. Run final exact Gurobi solve with the merged incumbent as MIP start
#
# Important:
#   - LP basis is used for LP relaxations / LP seed artifacts only.
#   - The final MIP solve receives an incumbent/MIP start, not VBasis/CBasis.
#
# Example:
#   LP_SEED_ENABLE=1 \
#   LP_SEED_SECONDS=30 \
#   LP_THREADS=16 \
#   HEURISTIC_WALL_SECONDS=150 \
#   PHASE1_WALL_SECONDS=60 \
#   PHASE2_WALL_SECONDS=60 \
#   FINAL_SECONDS=150 \
#   FINAL_THREADS=16 \
#   ALLOW_PARTIAL_PHASE_FAILURES=1 \
#   bash slurm/submit_workflow.sh tests/instance_01.original.mps config/default_config.json
# =============================================================================

INSTANCE_PATH="${1:?need instance path}"
CONFIG_PATH="${2:?need config path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Clear inherited bind settings that can break Apptainer on some nodes.
unset APPTAINER_BIND || true
unset APPTAINER_BINDPATH || true
unset SINGULARITY_BIND || true
unset SINGULARITY_BINDPATH || true

# -----------------------------------------------------------------------------
# User-configurable runtime knobs
# -----------------------------------------------------------------------------

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/gurobi.sif}"
LICENSE_FILE="${LICENSE_FILE:-$(cd "${PROJECT_ROOT}/.." && pwd)/gurobi.lic}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-${APPTAINER_CMD:-}}"

# Total heuristic wall budget, excluding final exact Gurobi.
# Recommended for 300s benchmark: HEURISTIC_WALL_SECONDS=150 and FINAL_SECONDS=150.
HEURISTIC_WALL_SECONDS="${HEURISTIC_WALL_SECONDS:-${HEURISTIC_SECONDS:-150}}"

# If not explicitly set, split heuristic wall budget evenly across the two waves.
PHASE1_FRACTION="${PHASE1_FRACTION:-0.50}"
PHASE2_FRACTION="${PHASE2_FRACTION:-0.50}"
PHASE1_WALL_SECONDS="${PHASE1_WALL_SECONDS:-}"
PHASE2_WALL_SECONDS="${PHASE2_WALL_SECONDS:-}"
PHASE1_MIN_TASK_SECONDS="${PHASE1_MIN_TASK_SECONDS:-30}"
PHASE2_MIN_TASK_SECONDS="${PHASE2_MIN_TASK_SECONDS:-45}"
PHASE2_SKIP_REL_GAP="${PHASE2_SKIP_REL_GAP:-}"
PHASE2_SKIP_ABS_GAP="${PHASE2_SKIP_ABS_GAP:-}"
REALLOCATE_SKIPPED_PHASE2_TO_FINAL="${REALLOCATE_SKIPPED_PHASE2_TO_FINAL:-1}"

FINAL_SECONDS="${FINAL_SECONDS:-150}"
FINAL_THREADS="${FINAL_THREADS:-16}"
FINAL_FOCUS_MODE="${FINAL_FOCUS_MODE:-prove}"  # balanced, incumbent, or prove.
FINAL_MIP_FOCUS="${FINAL_MIP_FOCUS:-}"         # Optional explicit Gurobi MIPFocus override.
FINAL_HEURISTICS="${FINAL_HEURISTICS:-0.02}"
FINAL_CUTS="${FINAL_CUTS:-2}"
FINAL_CUT_PASSES="${FINAL_CUT_PASSES:-}"
FINAL_PRESOLVE="${FINAL_PRESOLVE:-2}"
FINAL_MIP_GAP="${FINAL_MIP_GAP:-}"
FINAL_MIP_GAP_ABS="${FINAL_MIP_GAP_ABS:-}"
FINAL_START_NODE_LIMIT="${FINAL_START_NODE_LIMIT:-500}"
FINAL_START_TIME_LIMIT="${FINAL_START_TIME_LIMIT:-2.0}"

# Optional one-thread proof job that runs alongside the heuristic phases.
# It cannot receive new incumbents mid-solve, but it can improve our bound view
# and any incumbent it finds is merged before the final exact solve.
BOUND_PROBE_ENABLE="${BOUND_PROBE_ENABLE:-0}"
BOUND_PROBE_SECONDS="${BOUND_PROBE_SECONDS:-}"
BOUND_PROBE_THREADS="${BOUND_PROBE_THREADS:-1}"
BOUND_PROBE_USE_FOR_PHASE1_GAP="${BOUND_PROBE_USE_FOR_PHASE1_GAP:-0}"
BOUND_PROBE_FOCUS_MODE="${BOUND_PROBE_FOCUS_MODE:-prove}"
BOUND_PROBE_MIP_FOCUS="${BOUND_PROBE_MIP_FOCUS:-}"
BOUND_PROBE_HEURISTICS="${BOUND_PROBE_HEURISTICS:-0.01}"
BOUND_PROBE_CUTS="${BOUND_PROBE_CUTS:-2}"
BOUND_PROBE_CUT_PASSES="${BOUND_PROBE_CUT_PASSES:-}"
BOUND_PROBE_PRESOLVE="${BOUND_PROBE_PRESOLVE:-2}"
BOUND_PROBE_START_NODE_LIMIT="${BOUND_PROBE_START_NODE_LIMIT:-100}"
BOUND_PROBE_START_TIME_LIMIT="${BOUND_PROBE_START_TIME_LIMIT:-1.0}"

# Shared LP seed stage.
LP_SEED_ENABLE="${LP_SEED_ENABLE:-1}"
LP_SEED_SECONDS="${LP_SEED_SECONDS:-30}"
LP_THREADS="${LP_THREADS:-16}"
LP_METHOD="${LP_METHOD:-1}"              # 1 = dual simplex, basis-friendly.
LP_WARM_START="${LP_WARM_START:-2}"      # Use presolve-friendly warmstart behavior.
LP_REPAIR_SECONDS="${LP_REPAIR_SECONDS:-5}"
LP_REPAIR_THREADS="${LP_REPAIR_THREADS:-1}"

# Slurm/account controls.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-ISAAC-UTK0448}"
SLURM_PARTITION="${SLURM_PARTITION:-short}"
SLURM_QOS="${SLURM_QOS:-short}"
SLURM_MEM="${SLURM_MEM:-2G}"

# Add slack to each array task's Slurm time limit beyond solver time.
SLURM_TASK_OVERHEAD_SECONDS="${SLURM_TASK_OVERHEAD_SECONDS:-180}"

# Waiting / robustness controls.
POLL_SECONDS="${POLL_SECONDS:-10}"
ALLOW_PARTIAL_PHASE_FAILURES="${ALLOW_PARTIAL_PHASE_FAILURES:-1}"

# -----------------------------------------------------------------------------
# Resolve paths and output layout
# -----------------------------------------------------------------------------

if [[ -n "${RUN_BASE_DIR:-}" ]]; then
  BASE_DIR="${RUN_BASE_DIR}"
elif [[ -n "${SCRATCH:-}" && -w "${SCRATCH}" ]]; then
  BASE_DIR="${SCRATCH}"
else
  BASE_DIR="${PROJECT_ROOT}"
fi

if [[ -f "${INSTANCE_PATH}" ]]; then
  INSTANCE_ABS="$(cd "$(dirname "${INSTANCE_PATH}")" && pwd)/$(basename "${INSTANCE_PATH}")"
elif [[ -f "${PROJECT_ROOT}/${INSTANCE_PATH}" ]]; then
  INSTANCE_ABS="$(cd "$(dirname "${PROJECT_ROOT}/${INSTANCE_PATH}")" && pwd)/$(basename "${INSTANCE_PATH}")"
else
  echo "ERROR: instance file not found: ${INSTANCE_PATH}"
  exit 1
fi

if [[ -f "${CONFIG_PATH}" ]]; then
  CONFIG_ABS="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
elif [[ -f "${PROJECT_ROOT}/${CONFIG_PATH}" ]]; then
  CONFIG_ABS="$(cd "$(dirname "${PROJECT_ROOT}/${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
else
  echo "ERROR: config file not found: ${CONFIG_PATH}"
  exit 1
fi

INSTANCE_BASE="$(basename "${INSTANCE_ABS}")"
if [[ "${INSTANCE_BASE}" =~ (instance_[0-9]+) ]]; then
  INSTANCE_ID="${BASH_REMATCH[1]}"
else
  INSTANCE_ID="${INSTANCE_BASE%.original.mps}"
  INSTANCE_ID="${INSTANCE_ID%.mps}"
fi

RUN_ROOT="${BASE_DIR}/milp_runs"
RUN_DIR="${RUN_ROOT}/${INSTANCE_ID}"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"
PHASE1_RESULTS_DIR="${RESULTS_DIR}/phase1_results"
PHASE2_RESULTS_DIR="${RESULTS_DIR}/phase2_results"

FEATURES_JSON="${RESULTS_DIR}/features.json"
PLAN_JSON="${RESULTS_DIR}/plan.json"
LP_SEED_JSON="${RESULTS_DIR}/lp_seed.json"
MERGED_PHASE1_JSON="${RESULTS_DIR}/merged_phase1.json"
MERGED_PHASE1_BOUND_JSON="${RESULTS_DIR}/merged_phase1_with_bound_probe.json"
MERGED_PHASE2_JSON="${RESULTS_DIR}/merged_phase2.json"
MERGED_JSON="${RESULTS_DIR}/merged.json"
BOUND_PROBE_JSON="${RESULTS_DIR}/bound_probe.json"
FINAL_JSON="${RESULTS_DIR}/final_gurobi.json"
BOUND_PROBE_EVENT_LOG="${RESULTS_DIR}/bound_probe_incumbents.csv"
FINAL_EVENT_LOG="${RESULTS_DIR}/final_gurobi_incumbents.csv"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${PHASE1_RESULTS_DIR}" "${PHASE2_RESULTS_DIR}"

# Clear stale task output files for repeatable reruns of same instance.
# Keep logs; overwrite JSON result artifacts.
rm -f "${PHASE1_RESULTS_DIR}"/*.json "${PHASE2_RESULTS_DIR}"/*.json
rm -f "${MERGED_PHASE1_JSON}" "${MERGED_PHASE1_BOUND_JSON}" "${MERGED_PHASE2_JSON}" "${MERGED_JSON}" "${BOUND_PROBE_JSON}" "${FINAL_JSON}" "${LP_SEED_JSON}" "${BOUND_PROBE_EVENT_LOG}" "${FINAL_EVENT_LOG}"

# -----------------------------------------------------------------------------
# Preconditions
# -----------------------------------------------------------------------------

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: Apptainer image not found: ${APPTAINER_IMAGE}"
  exit 1
fi

if [[ ! -f "${LICENSE_FILE}" ]]; then
  echo "ERROR: Gurobi license file not found: ${LICENSE_FILE}"
  exit 1
fi

if [[ -z "${CONTAINER_RUNTIME}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME="$(command -v apptainer)"
  elif [[ -x /usr/bin/apptainer ]]; then
    CONTAINER_RUNTIME="/usr/bin/apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME="$(command -v singularity)"
  elif [[ -x /usr/bin/singularity ]]; then
    CONTAINER_RUNTIME="/usr/bin/singularity"
  fi
elif command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v "${CONTAINER_RUNTIME}")"
fi

if [[ -z "${CONTAINER_RUNTIME}" || ! -x "${CONTAINER_RUNTIME}" ]]; then
  echo "ERROR: container runtime not found."
  echo "ERROR: Set CONTAINER_RUNTIME=/path/to/apptainer, or load the apptainer/singularity module before running."
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch command not found on host PATH"
  exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "ERROR: squeue command not found on host PATH"
  exit 1
fi

# sacct is useful but not always enabled. We tolerate absence and fall back.
HAS_SACCT=1
if ! command -v sacct >/dev/null 2>&1; then
  HAS_SACCT=0
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

apptainer_py() {
  "${CONTAINER_RUNTIME}" exec --cleanenv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${BASE_DIR}:${BASE_DIR}" \
    --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
    --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
    --env PYTHONPATH="${PROJECT_ROOT}" \
    "${APPTAINER_IMAGE}" \
    python3 "$@"
}

fmt_slurm_time() {
  python3 - "$1" <<'PY'
import math
import sys
sec = int(math.ceil(float(sys.argv[1])))
sec = max(sec, 60)
h = sec // 3600
m = (sec % 3600) // 60
s = sec % 60
print("%02d:%02d:%02d" % (h, m, s))
PY
}

phase_count() {
  python3 - "${PLAN_JSON}" "$1" <<'PY'
import json
import sys

plan_path = sys.argv[1]
phase = sys.argv[2].lower()

with open(plan_path) as f:
    data = json.load(f)

# Preferred schema: {"phase1_tasks": [...], "phase2_tasks": [...]}
val = data.get(phase + "_tasks")
if isinstance(val, list):
    print(len(val))
    raise SystemExit

# Alternative schema: {"tasks": [{"phase": "phase1", ...}, ...]}
tasks = data.get("tasks")
if isinstance(tasks, list):
    print(sum(1 for t in tasks if str(t.get("phase", "")).lower() == phase))
    raise SystemExit

# Alternative schema: {"tasks_by_phase": {"phase1": [...], "phase2": [...]}}
tbp = data.get("tasks_by_phase")
if isinstance(tbp, dict) and isinstance(tbp.get(phase), list):
    print(len(tbp[phase]))
    raise SystemExit

print(0)
PY
}

phase_threads() {
  python3 - "${PLAN_JSON}" "$1" <<'PY'
import json
import sys

plan_path = sys.argv[1]
phase = sys.argv[2].lower()

with open(plan_path) as f:
    data = json.load(f)

for key in (phase + "_threads_per_task", phase + "_task_threads"):
    try:
        val = int(data.get(key, 0))
    except Exception:
        val = 0
    if val > 0:
        print(val)
        raise SystemExit

tasks = data.get(phase + "_tasks")
if isinstance(tasks, list) and tasks:
    try:
        val = int(tasks[0].get("threads_per_task", 0))
    except Exception:
        val = 0
    if val > 0:
        print(val)
        raise SystemExit

print(1)
PY
}

count_json_files() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' '
}

phase2_skip_decision() {
  local phase1_gap_json="${1:-${MERGED_PHASE1_JSON}}"
  python3 - "${phase1_gap_json}" "${PHASE2_SKIP_REL_GAP}" "${PHASE2_SKIP_ABS_GAP}" <<'PY'
import json
import math
import sys

merged_path, rel_arg, abs_arg = sys.argv[1:4]

def as_float(text):
    if text is None or str(text).strip() == "":
        return None
    try:
        val = float(text)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val

rel_threshold = as_float(rel_arg)
abs_threshold = as_float(abs_arg)
decision = {
    "skip": False,
    "reason": "no phase-2 skip threshold configured",
    "relative_gap_threshold": rel_threshold,
    "absolute_gap_threshold": abs_threshold,
}

if rel_threshold is None and abs_threshold is None:
    print(json.dumps(decision))
    raise SystemExit(1)

try:
    with open(merged_path, "r", encoding="utf-8", errors="replace") as f:
        merged = json.load(f)
except Exception as exc:
    decision["reason"] = "could not read merged phase-1 JSON: %s" % exc
    print(json.dumps(decision))
    raise SystemExit(1)

rel_gap = as_float(merged.get("relative_gap"))
abs_gap = as_float(merged.get("absolute_gap"))
objective = as_float(merged.get("best_objective"))
bound = as_float(merged.get("best_bound"))

decision.update({
    "objective": objective,
    "best_bound": bound,
    "relative_gap": rel_gap,
    "absolute_gap": abs_gap,
    "bound_source": merged.get("best_bound_source"),
})

if objective is None:
    decision["reason"] = "phase 1 did not produce an incumbent"
elif bound is None or (rel_gap is None and abs_gap is None):
    decision["reason"] = "phase 1 has no usable bound, so gap-based skip is unsafe"
elif rel_threshold is not None and rel_gap is not None and rel_gap <= rel_threshold:
    decision["skip"] = True
    decision["reason"] = "relative gap is below threshold"
elif abs_threshold is not None and abs_gap is not None and abs_gap <= abs_threshold:
    decision["skip"] = True
    decision["reason"] = "absolute gap is below threshold"
else:
    decision["reason"] = "phase-1 gap is above configured skip threshold"

print(json.dumps(decision))
raise SystemExit(0 if decision["skip"] else 1)
PY
}

wait_for_job() {
  local job_id="$1"

  echo "Waiting for Slurm job ${job_id} to finish..."
  while [[ -n "$(squeue -j "${job_id}" -h 2>/dev/null || true)" ]]; do
    sleep "${POLL_SECONDS}"
  done

  if [[ "${HAS_SACCT}" == "0" ]]; then
    echo "sacct not available; assuming job ${job_id} left queue."
    return 0
  fi

  local states
  states="$(sacct -j "${job_id}" --format=State --noheader 2>/dev/null | awk '{print $1}' | sed '/^$/d' || true)"

  echo "Observed states for ${job_id}:"
  if [[ -n "${states}" ]]; then
    echo "${states}" | sed 's/^/  - /'
  else
    echo "  - UNKNOWN"
  fi

  if echo "${states}" | grep -Eiq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED'; then
    return 1
  fi

  return 0
}

handle_phase_status() {
  local phase="$1"
  local job_id="$2"
  local results_dir="$3"

  if wait_for_job "${job_id}"; then
    return 0
  fi

  local nfiles
  nfiles="$(count_json_files "${results_dir}")"

  echo "WARNING: ${phase} array ${job_id} had at least one failed task."
  echo "WARNING: ${phase} produced ${nfiles} JSON result file(s)."

  if [[ "${ALLOW_PARTIAL_PHASE_FAILURES}" == "1" && "${nfiles}" -gt 0 ]]; then
    echo "WARNING: continuing with partial ${phase} results because ALLOW_PARTIAL_PHASE_FAILURES=1."
    return 0
  fi

  echo "ERROR: ${phase} failed and no usable partial results were available."
  echo "ERROR: Set ALLOW_PARTIAL_PHASE_FAILURES=1 to continue when partial JSON results exist."
  return 1
}

submit_phase_array() {
  local phase="$1"
  local phase_count="$2"
  local phase_seconds_total="$3"
  local results_dir="$4"
  local warmstart_json="${5:-}"
  local threads_per_task
  threads_per_task="$(phase_threads "${phase}")"

  local per_task_time
  per_task_time="$(python3 - <<PY
total = float("${phase_seconds_total}")
n = int("${phase_count}")
print(max(1.0, total / max(1, n)))
PY
)"

  local slurm_seconds
  slurm_seconds="$(python3 - <<PY
print(float("${per_task_time}") + float("${SLURM_TASK_OVERHEAD_SECONDS}"))
PY
)"

  local slurm_time
  slurm_time="$(fmt_slurm_time "${slurm_seconds}")"

  local export_vars
  export_vars="ALL,PROJECT_ROOT=${PROJECT_ROOT},APPTAINER_IMAGE=${APPTAINER_IMAGE},LICENSE_FILE=${LICENSE_FILE},CONTAINER_RUNTIME=${CONTAINER_RUNTIME},RUN_DIR=${RUN_DIR},LOG_DIR=${LOG_DIR},PHASE=${phase},PHASE_SECONDS_TOTAL=${phase_seconds_total},ARRAY_TASKS=${phase_count},HEURISTIC_THREADS=${threads_per_task},PYTHONPATH=${PROJECT_ROOT}"

  if [[ -n "${warmstart_json}" ]]; then
    export_vars="${export_vars},WARMSTART_JSON=${warmstart_json}"
  fi

  sbatch --parsable \
    -A "${SLURM_ACCOUNT}" \
    -p "${SLURM_PARTITION}" \
    --qos="${SLURM_QOS}" \
    --array="0-$((phase_count - 1))" \
    --cpus-per-task="${threads_per_task}" \
    --time="${slurm_time}" \
    --mem="${SLURM_MEM}" \
    --job-name="milp_${phase}" \
    --output="${LOG_DIR}/${phase}_%A_%a.out" \
    --error="${LOG_DIR}/${phase}_%A_%a.err" \
    --export="${export_vars}" \
    "${PROJECT_ROOT}/slurm/run_array_task.sh" \
      "${INSTANCE_ABS}" \
      "${PLAN_JSON}" \
      "${results_dir}" \
      "${phase}"
}

submit_bound_probe() {
  local seconds="$1"
  local start_json="$2"

  local slurm_seconds
  slurm_seconds="$(python3 - <<PY
print(float("${seconds}") + float("${SLURM_TASK_OVERHEAD_SECONDS}"))
PY
)"

  local slurm_time
  slurm_time="$(fmt_slurm_time "${slurm_seconds}")"

  local export_vars
  export_vars="ALL,PROJECT_ROOT=${PROJECT_ROOT},APPTAINER_IMAGE=${APPTAINER_IMAGE},LICENSE_FILE=${LICENSE_FILE},CONTAINER_RUNTIME=${CONTAINER_RUNTIME},BOUND_PROBE_THREADS=${BOUND_PROBE_THREADS},BOUND_PROBE_FOCUS_MODE=${BOUND_PROBE_FOCUS_MODE},BOUND_PROBE_MIP_FOCUS=${BOUND_PROBE_MIP_FOCUS},BOUND_PROBE_HEURISTICS=${BOUND_PROBE_HEURISTICS},BOUND_PROBE_CUTS=${BOUND_PROBE_CUTS},BOUND_PROBE_CUT_PASSES=${BOUND_PROBE_CUT_PASSES},BOUND_PROBE_PRESOLVE=${BOUND_PROBE_PRESOLVE},BOUND_PROBE_START_NODE_LIMIT=${BOUND_PROBE_START_NODE_LIMIT},BOUND_PROBE_START_TIME_LIMIT=${BOUND_PROBE_START_TIME_LIMIT},PYTHONPATH=${PROJECT_ROOT}"

  sbatch --parsable \
    -A "${SLURM_ACCOUNT}" \
    -p "${SLURM_PARTITION}" \
    --qos="${SLURM_QOS}" \
    --cpus-per-task="${BOUND_PROBE_THREADS}" \
    --time="${slurm_time}" \
    --mem="${SLURM_MEM}" \
    --job-name="milp_bound_probe" \
    --output="${LOG_DIR}/bound_probe_%j.out" \
    --error="${LOG_DIR}/bound_probe_%j.err" \
    --export="${export_vars}" \
    "${PROJECT_ROOT}/slurm/run_bound_probe.sh" \
      "${INSTANCE_ABS}" \
      "${start_json}" \
      "${BOUND_PROBE_JSON}" \
      "${seconds}" \
      "${BOUND_PROBE_EVENT_LOG}"
}

# -----------------------------------------------------------------------------
# Start workflow
# -----------------------------------------------------------------------------

echo "Project root:        ${PROJECT_ROOT}"
echo "Run dir:             ${RUN_DIR}"
echo "Instance:            ${INSTANCE_ABS}"
echo "Config:              ${CONFIG_ABS}"
echo "Apptainer image:     ${APPTAINER_IMAGE}"
echo "Container runtime:   ${CONTAINER_RUNTIME}"
echo "License file:        ${LICENSE_FILE}"
echo "Heuristic wall sec:  ${HEURISTIC_WALL_SECONDS}"
echo "Final secs:          ${FINAL_SECONDS}"
echo "Final threads:       ${FINAL_THREADS}"
echo "Final focus mode:    ${FINAL_FOCUS_MODE}"
echo "Final heuristics:    ${FINAL_HEURISTICS}"
echo "Final cuts:          ${FINAL_CUTS:-auto}"
echo "Final presolve:      ${FINAL_PRESOLVE:-auto}"
echo "Bound probe enabled: ${BOUND_PROBE_ENABLE}"
echo "Bound probe threads: ${BOUND_PROBE_THREADS}"
echo "Bound probe p1 gap:  ${BOUND_PROBE_USE_FOR_PHASE1_GAP}"
echo "LP seed enabled:     ${LP_SEED_ENABLE}"
echo "LP seed seconds:     ${LP_SEED_SECONDS}"
echo "LP threads:          ${LP_THREADS}"
echo "Partial failures:    ${ALLOW_PARTIAL_PHASE_FAILURES}"
echo

# Feature extraction and planning.
apptainer_py -m scripts.feature_extract "${INSTANCE_ABS}" "${FEATURES_JSON}"
apptainer_py -m scripts.make_plan "${CONFIG_ABS}" "${FEATURES_JSON}" "${INSTANCE_ABS}" "${PLAN_JSON}"

# -----------------------------------------------------------------------------
# Optional shared LP seed stage
# -----------------------------------------------------------------------------

if [[ "${LP_SEED_ENABLE}" == "1" ]]; then
  if "${CONTAINER_RUNTIME}" exec --cleanenv \
      --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      --bind "${BASE_DIR}:${BASE_DIR}" \
      --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
      --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
      --env PYTHONPATH="${PROJECT_ROOT}" \
      "${APPTAINER_IMAGE}" \
      test -f "${PROJECT_ROOT}/scripts/lp_seed_solve.py"; then

    echo
    echo "Running shared LP seed stage..."
    echo "  LP_SEED_SECONDS=${LP_SEED_SECONDS}"
    echo "  LP_THREADS=${LP_THREADS}"
    echo "  LP_METHOD=${LP_METHOD}"
    echo "  LP_WARM_START=${LP_WARM_START}"
    echo "  LP_REPAIR_SECONDS=${LP_REPAIR_SECONDS}"
    echo

    # This is intentionally synchronous. It stops as soon as Gurobi LP reaches
    # its normal stopping point or the LP seed time limit.
    apptainer_py -m scripts.lp_seed_solve \
      "${INSTANCE_ABS}" \
      "${LP_SEED_JSON}" \
      --time-limit "${LP_SEED_SECONDS}" \
      --threads "${LP_THREADS}" \
      --method "${LP_METHOD}" \
      --lp-warm-start "${LP_WARM_START}" \
      --repair-seconds "${LP_REPAIR_SECONDS}" \
      --repair-threads "${LP_REPAIR_THREADS}"

    # Let the LP seed / repaired incumbent compete in the phase-1 merge.
    # If repair did not find an incumbent, merge_results should safely ignore it
    # unless it also accepts non-incumbent metadata.
    if [[ -f "${LP_SEED_JSON}" ]]; then
      cp "${LP_SEED_JSON}" "${PHASE1_RESULTS_DIR}/task_lp_seed.json"
      echo "Copied LP seed artifact into phase-1 results:"
      echo "  ${PHASE1_RESULTS_DIR}/task_lp_seed.json"
    fi
  else
    echo "WARNING: LP_SEED_ENABLE=1 but scripts/lp_seed_solve.py was not found inside container."
    echo "WARNING: continuing without shared LP seed stage."
  fi
fi

# -----------------------------------------------------------------------------
# Compute phase sizes and timing
# -----------------------------------------------------------------------------

PHASE1_COUNT="$(phase_count phase1)"
PHASE2_COUNT="$(phase_count phase2)"

if [[ "${PHASE1_COUNT}" -le 0 && "${PHASE2_COUNT}" -le 0 ]]; then
  echo "ERROR: plan contains no phase1 or phase2 tasks."
  exit 1
fi

if [[ -z "${PHASE1_WALL_SECONDS}" ]]; then
  PHASE1_WALL_SECONDS="$(python3 - <<PY
print(float("${HEURISTIC_WALL_SECONDS}") * float("${PHASE1_FRACTION}"))
PY
)"
fi

if [[ -z "${PHASE2_WALL_SECONDS}" ]]; then
  PHASE2_WALL_SECONDS="$(python3 - <<PY
print(float("${HEURISTIC_WALL_SECONDS}") * float("${PHASE2_FRACTION}"))
PY
)"
fi

PHASE1_WALL_SECONDS="$(python3 - <<PY
print(max(float("${PHASE1_WALL_SECONDS}"), float("${PHASE1_MIN_TASK_SECONDS}")))
PY
)"
PHASE2_WALL_SECONDS="$(python3 - <<PY
print(max(float("${PHASE2_WALL_SECONDS}"), float("${PHASE2_MIN_TASK_SECONDS}")))
PY
)"

# run_array_task.sh divides phase total seconds by number of phase tasks.
# Therefore, total budget = intended wall time * number of tasks.
PHASE1_SECONDS_TOTAL="$(python3 - <<PY
print(float("${PHASE1_WALL_SECONDS}") * max(1, int("${PHASE1_COUNT}")))
PY
)"
PHASE2_SECONDS_TOTAL="$(python3 - <<PY
print(float("${PHASE2_WALL_SECONDS}") * max(1, int("${PHASE2_COUNT}")))
PY
)"

echo
echo "Phase 1 tasks:        ${PHASE1_COUNT}"
echo "Phase 2 tasks:        ${PHASE2_COUNT}"
echo "Phase1 wall target:   ${PHASE1_WALL_SECONDS}"
echo "Phase2 wall target:   ${PHASE2_WALL_SECONDS}"
echo "Phase1 min task sec:  ${PHASE1_MIN_TASK_SECONDS}"
echo "Phase2 min task sec:  ${PHASE2_MIN_TASK_SECONDS}"
echo "Phase2 skip rel gap:  ${PHASE2_SKIP_REL_GAP:-disabled}"
echo "Phase2 skip abs gap:  ${PHASE2_SKIP_ABS_GAP:-disabled}"
echo "Reallocate skipped p2:${REALLOCATE_SKIPPED_PHASE2_TO_FINAL}"
echo "Phase1 total budget:  ${PHASE1_SECONDS_TOTAL}"
echo "Phase2 total budget:  ${PHASE2_SECONDS_TOTAL}"
echo

BOUND_PROBE_JOB_ID=""
if [[ "${BOUND_PROBE_ENABLE}" == "1" ]]; then
  if [[ -z "${BOUND_PROBE_SECONDS}" ]]; then
    BOUND_PROBE_SECONDS="$(python3 - <<PY
if "${BOUND_PROBE_USE_FOR_PHASE1_GAP}" == "1":
    print(float("${PHASE1_WALL_SECONDS}"))
else:
    print(float("${PHASE1_WALL_SECONDS}") + float("${PHASE2_WALL_SECONDS}"))
PY
)"
  fi

  BOUND_PROBE_START_JSON="${LP_SEED_JSON}"
  if [[ ! -f "${BOUND_PROBE_START_JSON}" ]]; then
    BOUND_PROBE_START_JSON="${RESULTS_DIR}/bound_probe_empty_start.json"
    printf '{}\n' > "${BOUND_PROBE_START_JSON}"
  fi

  BOUND_PROBE_JOB_ID="$(submit_bound_probe "${BOUND_PROBE_SECONDS}" "${BOUND_PROBE_START_JSON}")"
  echo "Submitted bound-probe job: ${BOUND_PROBE_JOB_ID}"
  echo "Bound-probe seconds:       ${BOUND_PROBE_SECONDS}"
fi

# -----------------------------------------------------------------------------
# Phase 1 array
# -----------------------------------------------------------------------------

if [[ "${PHASE1_COUNT}" -gt 0 ]]; then
  PHASE1_JOB_ID="$(submit_phase_array phase1 "${PHASE1_COUNT}" "${PHASE1_SECONDS_TOTAL}" "${PHASE1_RESULTS_DIR}")"
  echo "Submitted phase-1 array: ${PHASE1_JOB_ID}"

  handle_phase_status phase1 "${PHASE1_JOB_ID}" "${PHASE1_RESULTS_DIR}"

  PHASE1_JSON_COUNT="$(count_json_files "${PHASE1_RESULTS_DIR}")"
  if [[ "${PHASE1_JSON_COUNT}" -gt 0 ]]; then
    apptainer_py -m scripts.merge_results "${PHASE1_RESULTS_DIR}" --out "${MERGED_PHASE1_JSON}"
  else
    echo "WARNING: no phase-1 JSON files to merge."
  fi
else
  echo "No phase-1 tasks; skipping phase 1."
fi

PHASE1_GAP_JSON="${MERGED_PHASE1_JSON}"
if [[ "${BOUND_PROBE_ENABLE}" == "1" && "${BOUND_PROBE_USE_FOR_PHASE1_GAP}" == "1" && -n "${BOUND_PROBE_JOB_ID}" ]]; then
  echo "Waiting for bound probe so phase-1 gap can use its proof bound..."
  set +e
  wait_for_job "${BOUND_PROBE_JOB_ID}"
  BOUND_PROBE_PHASE1_RC=$?
  set -e
  if [[ "${BOUND_PROBE_PHASE1_RC}" -ne 0 ]]; then
    echo "WARNING: bound-probe job ${BOUND_PROBE_JOB_ID} did not finish cleanly before phase-2 decision."
  fi

  if [[ -f "${MERGED_PHASE1_JSON}" && -f "${BOUND_PROBE_JSON}" ]]; then
    apptainer_py -m scripts.merge_results \
      "${MERGED_PHASE1_JSON}" \
      "${BOUND_PROBE_JSON}" \
      --out "${MERGED_PHASE1_BOUND_JSON}"
    PHASE1_GAP_JSON="${MERGED_PHASE1_BOUND_JSON}"
    echo "Phase-1 incumbent plus bound-probe gap file: ${PHASE1_GAP_JSON}"
  elif [[ ! -f "${BOUND_PROBE_JSON}" ]]; then
    echo "WARNING: bound probe produced no JSON before phase-2 decision; using phase-1 merge only."
  fi
fi

SKIP_PHASE2=0
PHASE2_SKIP_DECISION="{}"
if [[ -f "${PHASE1_GAP_JSON}" ]]; then
  set +e
  PHASE2_SKIP_DECISION="$(phase2_skip_decision "${PHASE1_GAP_JSON}")"
  PHASE2_SKIP_RC=$?
  set -e
  echo "Phase-2 skip decision: ${PHASE2_SKIP_DECISION}"
  if [[ "${PHASE2_SKIP_RC}" -eq 0 ]]; then
    SKIP_PHASE2=1
  fi
fi

# -----------------------------------------------------------------------------
# Phase 2 array with warmstart from phase 1
# -----------------------------------------------------------------------------

if [[ "${PHASE2_COUNT}" -gt 0 && "${SKIP_PHASE2}" != "1" ]]; then
  WARMSTART_ARG=""
  if [[ -f "${PHASE1_GAP_JSON}" ]]; then
    WARMSTART_ARG="${PHASE1_GAP_JSON}"
  fi

  PHASE2_JOB_ID="$(submit_phase_array phase2 "${PHASE2_COUNT}" "${PHASE2_SECONDS_TOTAL}" "${PHASE2_RESULTS_DIR}" "${WARMSTART_ARG}")"
  echo "Submitted phase-2 array: ${PHASE2_JOB_ID}"

  handle_phase_status phase2 "${PHASE2_JOB_ID}" "${PHASE2_RESULTS_DIR}"

  PHASE2_JSON_COUNT="$(count_json_files "${PHASE2_RESULTS_DIR}")"
  if [[ "${PHASE2_JSON_COUNT}" -gt 0 ]]; then
    apptainer_py -m scripts.merge_results "${PHASE2_RESULTS_DIR}" --out "${MERGED_PHASE2_JSON}"
  else
    echo "WARNING: no phase-2 JSON files to merge."
  fi
else
  if [[ "${SKIP_PHASE2}" == "1" ]]; then
    echo "Skipping phase 2 because phase-1 incumbent gap is already below threshold."
    PHASE2_SKIP_DECISION_JSON="${PHASE2_SKIP_DECISION}" python3 - "${MERGED_PHASE2_JSON}" <<'PY'
import json
import os
import sys

out_path = sys.argv[1]
try:
    decision = json.loads(os.environ.get("PHASE2_SKIP_DECISION_JSON", "{}"))
except Exception:
    decision = {}

payload = {
    "status": "skipped",
    "phase": "phase2",
    "reason": "phase1_gap_below_threshold",
    "skip_decision": decision,
    "num_task_files": 0,
    "num_candidate_files": 0,
    "found": False,
    "found_incumbent": False,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, allow_nan=False)
PY
  else
    echo "No phase-2 tasks; skipping phase 2."
  fi
fi

if [[ "${SKIP_PHASE2}" == "1" && "${REALLOCATE_SKIPPED_PHASE2_TO_FINAL}" == "1" ]]; then
  OLD_FINAL_SECONDS="${FINAL_SECONDS}"
  FINAL_SECONDS="$(python3 - <<PY
print(float("${FINAL_SECONDS}") + float("${PHASE2_WALL_SECONDS}"))
PY
)"
  echo "Reallocated skipped phase-2 wall budget to final exact solve:"
  echo "  Final seconds: ${OLD_FINAL_SECONDS} -> ${FINAL_SECONDS}"
fi

# -----------------------------------------------------------------------------
# Combined merge
# -----------------------------------------------------------------------------

if [[ -n "${BOUND_PROBE_JOB_ID}" ]]; then
  set +e
  wait_for_job "${BOUND_PROBE_JOB_ID}"
  BOUND_PROBE_RC=$?
  set -e
  if [[ "${BOUND_PROBE_RC}" -ne 0 ]]; then
    echo "WARNING: bound-probe job ${BOUND_PROBE_JOB_ID} did not finish cleanly."
  fi
  if [[ -f "${BOUND_PROBE_JSON}" ]]; then
    echo "Bound-probe output:   ${BOUND_PROBE_JSON}"
  else
    echo "WARNING: bound probe produced no JSON output."
  fi
fi

MERGE_INPUTS=()
if [[ "$(count_json_files "${PHASE1_RESULTS_DIR}")" -gt 0 ]]; then
  MERGE_INPUTS+=("${PHASE1_RESULTS_DIR}")
fi
if [[ "$(count_json_files "${PHASE2_RESULTS_DIR}")" -gt 0 ]]; then
  MERGE_INPUTS+=("${PHASE2_RESULTS_DIR}")
fi
if [[ -f "${BOUND_PROBE_JSON}" ]]; then
  MERGE_INPUTS+=("${BOUND_PROBE_JSON}")
fi

if [[ "${#MERGE_INPUTS[@]}" -eq 0 ]]; then
  echo "ERROR: no heuristic result JSONs were produced."
  exit 1
fi

apptainer_py -m scripts.merge_results "${MERGE_INPUTS[@]}" --out "${MERGED_JSON}"

# -----------------------------------------------------------------------------
# Final exact solve
# -----------------------------------------------------------------------------

# The final exact solve receives the best incumbent from merged.json.
# We do not pass LP VBasis/CBasis here; those are LP-only warmstart artifacts.
FINAL_SOLVE_ARGS=(
  -m scripts.final_gurobi_solve
  "${INSTANCE_ABS}"
  "${MERGED_JSON}"
  --out "${FINAL_JSON}"
  --time-limit "${FINAL_SECONDS}"
  --threads "${FINAL_THREADS}"
  --seed 0
  --focus-mode "${FINAL_FOCUS_MODE}"
  --heuristics "${FINAL_HEURISTICS}"
  --start-node-limit "${FINAL_START_NODE_LIMIT}"
  --start-time-limit "${FINAL_START_TIME_LIMIT}"
  --event-log "${FINAL_EVENT_LOG}"
  --log-to-console
)

if [[ -n "${FINAL_MIP_FOCUS}" ]]; then
  FINAL_SOLVE_ARGS+=(--mip-focus "${FINAL_MIP_FOCUS}")
fi
if [[ -n "${FINAL_CUTS}" ]]; then
  FINAL_SOLVE_ARGS+=(--cuts "${FINAL_CUTS}")
fi
if [[ -n "${FINAL_CUT_PASSES}" ]]; then
  FINAL_SOLVE_ARGS+=(--cut-passes "${FINAL_CUT_PASSES}")
fi
if [[ -n "${FINAL_PRESOLVE}" ]]; then
  FINAL_SOLVE_ARGS+=(--presolve "${FINAL_PRESOLVE}")
fi
if [[ -n "${FINAL_MIP_GAP}" ]]; then
  FINAL_SOLVE_ARGS+=(--mip-gap "${FINAL_MIP_GAP}")
fi
if [[ -n "${FINAL_MIP_GAP_ABS}" ]]; then
  FINAL_SOLVE_ARGS+=(--mip-gap-abs "${FINAL_MIP_GAP_ABS}")
fi

apptainer_py "${FINAL_SOLVE_ARGS[@]}"

echo
echo "Workflow completed."
echo "Run directory:       ${RUN_DIR}"
echo "Features:            ${FEATURES_JSON}"
echo "Plan:                ${PLAN_JSON}"
echo "LP seed:             ${LP_SEED_JSON}"
echo "Phase1 merged:       ${MERGED_PHASE1_JSON}"
echo "Phase1 + bound:      ${MERGED_PHASE1_BOUND_JSON}"
echo "Phase2 merged:       ${MERGED_PHASE2_JSON}"
echo "Bound probe output:  ${BOUND_PROBE_JSON}"
echo "Combined merged:     ${MERGED_JSON}"
echo "Final solve output:  ${FINAL_JSON}"
echo
echo "Useful logs:"
echo "  ${LOG_DIR}"
