#!/usr/bin/env bash
#SBATCH --job-name=milp_submit
#SBATCH -A ISAAC-UTK0448
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

INSTANCE_PATH="${1:?need instance path}"
CONFIG_PATH="${2:?need config path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- baked-in defaults ----
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/gurobi.sif}"
FARM_SECONDS="${FARM_SECONDS:-160}"
FINAL_SECONDS="${FINAL_SECONDS:-50}"
ARRAY_TASKS="${ARRAY_TASKS:-16}"

# Pick a writable base dir for logs/results
if [[ -n "${RUN_BASE_DIR:-}" ]]; then
  BASE_DIR="${RUN_BASE_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -w "${SLURM_SUBMIT_DIR}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SCRATCH:-}" && -w "${SCRATCH}" ]]; then
  BASE_DIR="${SCRATCH}"
else
  BASE_DIR="${HOME}"
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${BASE_DIR}/milp_runs"
RUN_DIR="${RUN_ROOT}/run_${RUN_STAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"
FEATURES_JSON="${RESULTS_DIR}/features.json"
PLAN_JSON="${RESULTS_DIR}/plan.json"
FARM_RESULTS_DIR="${RESULTS_DIR}/farm_results"
MERGED_JSON="${RESULTS_DIR}/merged.json"
FINAL_JSON="${RESULTS_DIR}/final_gurobi.json"

mkdir -p "${LOG_DIR}" "${FARM_RESULTS_DIR}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: Apptainer image not found: ${APPTAINER_IMAGE}"
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer command not found on host PATH"
  exit 1
fi

echo "Project root:      ${PROJECT_ROOT}"
echo "Apptainer image:   ${APPTAINER_IMAGE}"
echo "Instance:          ${INSTANCE_PATH}"
echo "Config:            ${CONFIG_PATH}"
echo "Base dir:          ${BASE_DIR}"
echo "Run dir:           ${RUN_DIR}"
echo "Log dir:           ${LOG_DIR}"
echo "Results dir:       ${RESULTS_DIR}"
echo "Farm seconds:      ${FARM_SECONDS}"
echo "Final seconds:     ${FINAL_SECONDS}"
echo "Array tasks:       ${ARRAY_TASKS}"

# Build features + plan inside container, but write to writable run dir
apptainer exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  "${APPTAINER_IMAGE}" \
  python3 "${PROJECT_ROOT}/scripts/feature_extract.py" \
    "${PROJECT_ROOT}/${INSTANCE_PATH}" \
    "${FEATURES_JSON}"

apptainer exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  "${APPTAINER_IMAGE}" \
  python3 "${PROJECT_ROOT}/scripts/make_plan.py" \
    "${PROJECT_ROOT}/${CONFIG_PATH}" \
    "${FEATURES_JSON}" \
    "${PROJECT_ROOT}/${INSTANCE_PATH}" \
    "${PLAN_JSON}"

ARRAY_JOB_ID=$(
  sbatch --parsable \
    --array=0-$((ARRAY_TASKS - 1)) \
    --time=00:05:00 \
    --mem=2G \
    --output="${LOG_DIR}/array_%A_%a.out" \
    --error="${LOG_DIR}/array_%A_%a.err" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",APPTAINER_IMAGE="${APPTAINER_IMAGE}",FARM_SECONDS="${FARM_SECONDS}",ARRAY_TASKS="${ARRAY_TASKS}",RUN_DIR="${RUN_DIR}",LOG_DIR="${LOG_DIR}" \
    "${PROJECT_ROOT}/slurm/run_array_task.sh" "${INSTANCE_PATH}" "${PLAN_JSON}" "${FARM_RESULTS_DIR}"
)

echo "Submitted heuristic farm array job: ${ARRAY_JOB_ID}"

POST_JOB_ID=$(
  sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --time=00:05:00 \
    --mem=3G \
    --output="${LOG_DIR}/post_%j.out" \
    --error="${LOG_DIR}/post_%j.err" \
    --wrap="apptainer exec --bind '${PROJECT_ROOT}:${PROJECT_ROOT}' '${APPTAINER_IMAGE}' python3 '${PROJECT_ROOT}/scripts/merge_results.py' '${FARM_RESULTS_DIR}' --out '${MERGED_JSON}' && \
            apptainer exec --bind '${PROJECT_ROOT}:${PROJECT_ROOT}' '${APPTAINER_IMAGE}' python3 '${PROJECT_ROOT}/scripts/final_gurobi_solve.py' '${PROJECT_ROOT}/${INSTANCE_PATH}' '${MERGED_JSON}' --out '${FINAL_JSON}' --time-limit '${FINAL_SECONDS}' --threads 1 --seed 0 --mip-focus 1 --heuristics 0.05 --start-node-limit 500 --start-time-limit 2.0 --log-to-console"
)

echo "Submitted merge/final-solve job: ${POST_JOB_ID}"
echo "Run directory:    ${RUN_DIR}"
echo "Merged results:   ${MERGED_JSON}"
echo "Final solve:      ${FINAL_JSON}"