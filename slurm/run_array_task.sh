#!/usr/bin/env bash
#SBATCH --job-name=milp_array
#SBATCH -A ISAAC-UTK0448
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --time=00:05:00
#SBATCH --mem=2G

set -euo pipefail

INSTANCE_PATH="${1:?need instance path}"
PLAN_JSON="${2:?need plan json}"
RESULTS_DIR="${3:?need results dir}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/gurobi.sif}"
FARM_SECONDS="${FARM_SECONDS:-160}"
ARRAY_TASKS="${ARRAY_TASKS:-16}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
LOG_DIR="${LOG_DIR:-${PWD}}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: Apptainer image not found: ${APPTAINER_IMAGE}"
  exit 1
fi

PER_TASK_TIME="$(python3 - <<PY
farm = float("${FARM_SECONDS}")
tasks = int("${ARRAY_TASKS}")
print(max(0.25, farm / max(1, tasks)))
PY
)"

echo "Task ID:            ${TASK_ID}"
echo "Project root:       ${PROJECT_ROOT}"
echo "Apptainer image:    ${APPTAINER_IMAGE}"
echo "Instance:           ${INSTANCE_PATH}"
echo "Plan JSON:          ${PLAN_JSON}"
echo "Results dir:        ${RESULTS_DIR}"
echo "Farm seconds total: ${FARM_SECONDS}"
echo "Per-task budget:    ${PER_TASK_TIME}"

apptainer exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  "${APPTAINER_IMAGE}" \
  python3 "${PROJECT_ROOT}/scripts/run_task.py" \
    "${PROJECT_ROOT}/${INSTANCE_PATH}" \
    "${PLAN_JSON}" \
    "${TASK_ID}" \
    --out "${RESULTS_DIR}/task_$(printf "%03d" "${TASK_ID}").json" \
    --time-limit "${PER_TASK_TIME}"