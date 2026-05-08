#!/usr/bin/env bash
#SBATCH --job-name=milp_array
#SBATCH -A ISAAC-UTK0448
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --mem=2G

set -euo pipefail

# Clear inherited bind settings that can break Apptainer.
unset APPTAINER_BIND || true
unset APPTAINER_BINDPATH || true
unset SINGULARITY_BIND || true
unset SINGULARITY_BINDPATH || true
export APPTAINER_NO_MOUNT="${APPTAINER_NO_MOUNT:-bind-paths}"
export SINGULARITY_NO_MOUNT="${SINGULARITY_NO_MOUNT:-bind-paths}"

INSTANCE_PATH="${1:?need instance path}"
PLAN_JSON="${2:?need plan json}"
RESULTS_DIR="${3:?need results dir}"
PHASE_ARG="${4:-${PHASE:-phase1}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/gurobi.sif}"
LICENSE_FILE="${LICENSE_FILE:-$(cd "${PROJECT_ROOT}/.." && pwd)/gurobi.lic}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-${APPTAINER_CMD:-}}"
TASK_ID_LOCAL="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
ARRAY_TASKS="${ARRAY_TASKS:-1}"
PHASE_SECONDS_TOTAL="${PHASE_SECONDS_TOTAL:-${FARM_SECONDS:-160}}"
WARMSTART_JSON="${WARMSTART_JSON:-}"
HEURISTIC_THREADS="${HEURISTIC_THREADS:-1}"

mkdir -p "${RESULTS_DIR}"

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

PER_TASK_TIME="$(python3 - <<PY
print(max(0.25, float("${PHASE_SECONDS_TOTAL}") / max(1, int("${ARRAY_TASKS}"))))
PY
)"

OUT_FILE="${RESULTS_DIR}/task_$(printf "%03d" "${TASK_ID_LOCAL}").json"

APPTAINER_BASE=(
  "${CONTAINER_RUNTIME}" exec
  --cleanenv
  --no-mount bind-paths
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}"
  --bind "$(dirname "${RESULTS_DIR}")":"$(dirname "${RESULTS_DIR}")"
  --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic"
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
  --env PYTHONPATH="${PROJECT_ROOT}"
  --env OMP_NUM_THREADS="${HEURISTIC_THREADS}"
  "${APPTAINER_IMAGE}"
)

# Determine whether scripts.run_task supports the new --phase argument.
RUN_TASK_HELP="$("${APPTAINER_BASE[@]}" python3 -m scripts.run_task --help 2>&1 || true)"

SUPPORTS_PHASE=0
if echo "${RUN_TASK_HELP}" | grep -q -- '--phase'; then
  SUPPORTS_PHASE=1
fi

# If run_task is old/flat and does not support --phase, phase2 task ids need
# to be offset by the number of phase1 tasks in the plan.
TASK_ID_TO_RUN="${TASK_ID_LOCAL}"
if [[ "${SUPPORTS_PHASE}" == "0" && "${PHASE_ARG}" == "phase2" ]]; then
  PHASE1_COUNT="$(
    python3 - "${PLAN_JSON}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if isinstance(data.get("phase1_tasks"), list):
    print(len(data["phase1_tasks"]))
elif isinstance(data.get("tasks"), list):
    print(sum(1 for t in data["tasks"] if str(t.get("phase","")).lower() == "phase1"))
else:
    print(0)
PY
  )"
  TASK_ID_TO_RUN="$((PHASE1_COUNT + TASK_ID_LOCAL))"
fi

echo "Task ID local:       ${TASK_ID_LOCAL}"
echo "Task ID to run:      ${TASK_ID_TO_RUN}"
echo "Phase:               ${PHASE_ARG}"
echo "Supports --phase:    ${SUPPORTS_PHASE}"
echo "Project root:        ${PROJECT_ROOT}"
echo "Instance:            ${INSTANCE_PATH}"
echo "Plan JSON:           ${PLAN_JSON}"
echo "Results dir:         ${RESULTS_DIR}"
echo "Phase seconds total: ${PHASE_SECONDS_TOTAL}"
echo "Array tasks:         ${ARRAY_TASKS}"
echo "Per-task budget:     ${PER_TASK_TIME}"
echo "Heuristic threads:   ${HEURISTIC_THREADS}"
echo "Output:              ${OUT_FILE}"
if [[ -n "${WARMSTART_JSON}" ]]; then
  echo "Warmstart JSON:      ${WARMSTART_JSON}"
fi

CMD=(python3 -m scripts.run_task "${INSTANCE_PATH}" "${PLAN_JSON}" "${TASK_ID_TO_RUN}" --out "${OUT_FILE}" --time-limit "${PER_TASK_TIME}")

if [[ "${SUPPORTS_PHASE}" == "1" ]]; then
  CMD+=(--phase "${PHASE_ARG}")
fi

if [[ -n "${WARMSTART_JSON}" && -f "${WARMSTART_JSON}" ]]; then
  CMD+=(--warmstart-json "${WARMSTART_JSON}")
fi

# Only pass --threads if run_task advertises it. Otherwise the per-method
# config/params remain responsible for Gurobi Threads.
if echo "${RUN_TASK_HELP}" | grep -q -- '--threads'; then
  CMD+=(--threads "${HEURISTIC_THREADS}")
fi

"${APPTAINER_BASE[@]}" "${CMD[@]}"
