#!/usr/bin/env bash
set -euo pipefail

INSTANCE_PATH="${1:?need instance path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Clear inherited bind settings that can break Apptainer on some nodes.
unset APPTAINER_BIND || true
unset APPTAINER_BINDPATH || true
unset SINGULARITY_BIND || true
unset SINGULARITY_BINDPATH || true

APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/gurobi.sif}"
LICENSE_FILE="${LICENSE_FILE:-$(cd "${PROJECT_ROOT}/.." && pwd)/gurobi.lic}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-${APPTAINER_CMD:-}}"
BASELINE_EXE="${BASELINE_EXE:-${PROJECT_ROOT}/baseline}"
THREADS="${THREADS:-16}"
TIME_LIMIT="${TIME_LIMIT:-320}"
BASELINE_USE_PYTHON="${BASELINE_USE_PYTHON:-0}"
BASELINE_EVENT_LOG="${BASELINE_EVENT_LOG:-}"
BASELINE_RESULT_JSON="${BASELINE_RESULT_JSON:-}"
BASELINE_SOLVER_LOG="${BASELINE_SOLVER_LOG:-}"

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

if [[ "${BASELINE_USE_PYTHON}" != "1" && -z "${BASELINE_EVENT_LOG}" && ! -x "${BASELINE_EXE}" ]]; then
  echo "ERROR: baseline executable not found or not executable: ${BASELINE_EXE}"
  exit 1
fi

if [[ ! -f "${INSTANCE_PATH}" ]]; then
  if [[ -f "${PROJECT_ROOT}/${INSTANCE_PATH}" ]]; then
    INSTANCE_PATH="${PROJECT_ROOT}/${INSTANCE_PATH}"
  else
    echo "ERROR: instance file not found: ${INSTANCE_PATH}"
    exit 1
  fi
fi

INSTANCE_ABS="$(cd "$(dirname "${INSTANCE_PATH}")" && pwd)/$(basename "${INSTANCE_PATH}")"

echo "Project root:   ${PROJECT_ROOT}"
echo "Apptainer:      ${APPTAINER_IMAGE}"
echo "Runtime:        ${CONTAINER_RUNTIME}"
echo "License file:   ${LICENSE_FILE}"
echo "Baseline exe:   ${BASELINE_EXE}"
echo "Instance:       ${INSTANCE_ABS}"
echo "Threads:        ${THREADS}"
echo "Time limit:     ${TIME_LIMIT}"

if [[ "${BASELINE_USE_PYTHON}" == "1" || -n "${BASELINE_EVENT_LOG}" ]]; then
  if [[ -z "${BASELINE_EVENT_LOG}" ]]; then
    BASELINE_EVENT_LOG="${PROJECT_ROOT}/baseline/$(basename "${INSTANCE_ABS%.*}")_incumbents.csv"
  fi
  if [[ -z "${BASELINE_RESULT_JSON}" ]]; then
    BASELINE_RESULT_JSON="${BASELINE_EVENT_LOG%.csv}.json"
  fi
  if [[ -z "${BASELINE_SOLVER_LOG}" ]]; then
    BASELINE_SOLVER_LOG="${BASELINE_EVENT_LOG%.csv}.solver.log"
  fi
  mkdir -p "$(dirname "${BASELINE_EVENT_LOG}")" "$(dirname "${BASELINE_RESULT_JSON}")" "$(dirname "${BASELINE_SOLVER_LOG}")"
  echo "Baseline mode:  Python callback trace"
  echo "Event log:      ${BASELINE_EVENT_LOG}"
  echo "Result JSON:    ${BASELINE_RESULT_JSON}"
  echo "Solver log:     ${BASELINE_SOLVER_LOG}"
  "${CONTAINER_RUNTIME}" exec --cleanenv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
    --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
    --env OMP_NUM_THREADS="${THREADS}" \
    --env PYTHONPATH="${PROJECT_ROOT}" \
    "${APPTAINER_IMAGE}" \
    python3 -m scripts.gurobi_baseline "${INSTANCE_ABS}" \
      --out "${BASELINE_RESULT_JSON}" \
      --event-log "${BASELINE_EVENT_LOG}" \
      --solver-log "${BASELINE_SOLVER_LOG}" \
      --time-limit "${TIME_LIMIT}" \
      --threads "${THREADS}" \
      --seed 0
else
  echo "Baseline mode:  compiled executable"
  "${CONTAINER_RUNTIME}" exec --cleanenv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
    --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
    --env OMP_NUM_THREADS="${THREADS}" \
    "${APPTAINER_IMAGE}" \
    "${BASELINE_EXE}" "${INSTANCE_ABS}" "${TIME_LIMIT}" "${THREADS}"
fi
