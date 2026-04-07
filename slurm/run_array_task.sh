#!/usr/bin/env bash
#SBATCH --job-name=milp_array
#SBATCH -A ISAAC-UTK0448
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --time=00:05:00
#SBATCH --mem=2G

set -euo pipefail

# Clear inherited bind settings that can break Apptainer on some nodes.
unset APPTAINER_BIND || true
unset APPTAINER_BINDPATH || true
unset SINGULARITY_BIND || true
unset SINGULARITY_BINDPATH || true

INSTANCE_PATH="${1:?need instance path}"
PLAN_JSON="${2:?need plan json}"
RESULTS_DIR="${3:?need results dir}"

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
FARM_SECONDS="${FARM_SECONDS:-160}"
ARRAY_TASKS="${ARRAY_TASKS:-16}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
LOG_DIR="${LOG_DIR:-${PWD}}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: Apptainer image not found: ${APPTAINER_IMAGE}"
  exit 1
fi

if [[ ! -f "${LICENSE_FILE}" ]]; then
  echo "ERROR: Gurobi license file not found: ${LICENSE_FILE}"
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
echo "License file:       ${LICENSE_FILE}"
echo "Instance:           ${INSTANCE_PATH}"
echo "Plan JSON:          ${PLAN_JSON}"
echo "Results dir:        ${RESULTS_DIR}"
echo "Farm seconds total: ${FARM_SECONDS}"
echo "Per-task budget:    ${PER_TASK_TIME}"

apptainer exec --cleanenv \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  "${APPTAINER_IMAGE}" \
  python3 "${PROJECT_ROOT}/scripts/run_task.py" \
    "${PROJECT_ROOT}/${INSTANCE_PATH}" \
    "${PLAN_JSON}" \
    "${TASK_ID}" \
    --out "${RESULTS_DIR}/task_$(printf "%03d" "${TASK_ID}").json" \
    --time-limit "${PER_TASK_TIME}"