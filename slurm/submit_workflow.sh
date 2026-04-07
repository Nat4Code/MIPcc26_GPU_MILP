#!/usr/bin/env bash
set -euo pipefail

# Clear inherited bind settings that can break Apptainer on some nodes.
unset APPTAINER_BIND || true
unset APPTAINER_BINDPATH || true
unset SINGULARITY_BIND || true
unset SINGULARITY_BINDPATH || true

INSTANCE_PATH="${1:?need instance path}"
CONFIG_PATH="${2:?need config path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/gurobi.sif}"
LICENSE_FILE="${LICENSE_FILE:-$(cd "${PROJECT_ROOT}/.." && pwd)/gurobi.lic}"
FARM_SECONDS="${FARM_SECONDS:-160}"
FINAL_SECONDS="${FINAL_SECONDS:-50}"
ARRAY_TASKS="${ARRAY_TASKS:-16}"

# Choose a writable base dir for run artifacts
if [[ -n "${RUN_BASE_DIR:-}" ]]; then
  BASE_DIR="${RUN_BASE_DIR}"
elif [[ -n "${SCRATCH:-}" && -w "${SCRATCH}" ]]; then
  BASE_DIR="${SCRATCH}"
elif [[ -w "${PROJECT_ROOT}" ]]; then
  BASE_DIR="${PROJECT_ROOT}"
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

if [[ ! -f "${LICENSE_FILE}" ]]; then
  echo "ERROR: Gurobi license file not found: ${LICENSE_FILE}"
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer command not found on host PATH"
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch command not found on host PATH"
  exit 1
fi

echo "Project root:      ${PROJECT_ROOT}"
echo "Apptainer image:   ${APPTAINER_IMAGE}"
echo "License file:      ${LICENSE_FILE}"
echo "Instance:          ${INSTANCE_PATH}"
echo "Config:            ${CONFIG_PATH}"
echo "Base dir:          ${BASE_DIR}"
echo "Run dir:           ${RUN_DIR}"
echo "Log dir:           ${LOG_DIR}"
echo "Results dir:       ${RESULTS_DIR}"
echo "Farm seconds:      ${FARM_SECONDS}"
echo "Final seconds:     ${FINAL_SECONDS}"
echo "Array tasks:       ${ARRAY_TASKS}"

# Build features
apptainer exec --cleanenv \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  "${APPTAINER_IMAGE}" \
  python3 "${PROJECT_ROOT}/scripts/feature_extract.py" \
    "${PROJECT_ROOT}/${INSTANCE_PATH}" \
    "${FEATURES_JSON}"

# Build plan
apptainer exec --cleanenv \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  "${APPTAINER_IMAGE}" \
  python3 "${PROJECT_ROOT}/scripts/make_plan.py" \
    "${PROJECT_ROOT}/${CONFIG_PATH}" \
    "${FEATURES_JSON}" \
    "${PROJECT_ROOT}/${INSTANCE_PATH}" \
    "${PLAN_JSON}"

# Submit array job directly from host shell
ARRAY_JOB_ID="$(
  sbatch --parsable \
    -A ISAAC-UTK0448 \
    -p short \
    --qos=short \
    --array=0-$((ARRAY_TASKS - 1)) \
    --time=00:05:00 \
    --mem=2G \
    --output="${LOG_DIR}/array_%A_%a.out" \
    --error="${LOG_DIR}/array_%A_%a.err" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",APPTAINER_IMAGE="${APPTAINER_IMAGE}",LICENSE_FILE="${LICENSE_FILE}",FARM_SECONDS="${FARM_SECONDS}",ARRAY_TASKS="${ARRAY_TASKS}",RUN_DIR="${RUN_DIR}",LOG_DIR="${LOG_DIR}" \
    "${PROJECT_ROOT}/slurm/run_array_task.sh" "${INSTANCE_PATH}" "${PLAN_JSON}" "${FARM_RESULTS_DIR}"
)"

echo "Submitted heuristic farm array job: ${ARRAY_JOB_ID}"

# Submit dependent merge/final job directly from host shell
POST_JOB_ID="$(
  sbatch --parsable \
    -A ISAAC-UTK0448 \
    -p short \
    --qos=short \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --time=00:05:00 \
    --mem=3G \
    --output="${LOG_DIR}/post_%j.out" \
    --error="${LOG_DIR}/post_%j.err" \
    --wrap="unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH || true; \
            apptainer exec --cleanenv --bind '${PROJECT_ROOT}:${PROJECT_ROOT}' --bind '${LICENSE_FILE}:/opt/gurobi/gurobi.lic' --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic '${APPTAINER_IMAGE}' python3 '${PROJECT_ROOT}/scripts/merge_results.py' '${FARM_RESULTS_DIR}' --out '${MERGED_JSON}' && \
            apptainer exec --cleanenv --bind '${PROJECT_ROOT}:${PROJECT_ROOT}' --bind '${LICENSE_FILE}:/opt/gurobi/gurobi.lic' --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic '${APPTAINER_IMAGE}' python3 '${PROJECT_ROOT}/scripts/final_gurobi_solve.py' '${PROJECT_ROOT}/${INSTANCE_PATH}' '${MERGED_JSON}' --out '${FINAL_JSON}' --time-limit '${FINAL_SECONDS}' --threads 1 --seed 0 --mip-focus 1 --heuristics 0.05 --start-node-limit 500 --start-time-limit 2.0 --log-to-console"
)"

echo "Submitted merge/final-solve job: ${POST_JOB_ID}"
echo "Run directory:    ${RUN_DIR}"
echo "Merged results:   ${MERGED_JSON}"
echo "Final solve:      ${FINAL_JSON}"
echo
echo "Check status with:"
echo "  squeue -j ${ARRAY_JOB_ID},${POST_JOB_ID}"