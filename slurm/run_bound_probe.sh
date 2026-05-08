#!/usr/bin/env bash
#SBATCH --job-name=milp_bound_probe
#SBATCH -A ISAAC-UTK0448
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --mem=2G

set -euo pipefail

unset APPTAINER_BIND || true
unset APPTAINER_BINDPATH || true
unset SINGULARITY_BIND || true
unset SINGULARITY_BINDPATH || true
export APPTAINER_NO_MOUNT="${APPTAINER_NO_MOUNT:-bind-paths}"
export SINGULARITY_NO_MOUNT="${SINGULARITY_NO_MOUNT:-bind-paths}"

INSTANCE_PATH="${1:?need instance path}"
START_JSON="${2:?need start json path}"
OUT_JSON="${3:?need output json path}"
TIME_LIMIT="${4:?need time limit seconds}"
EVENT_LOG="${5:-}"

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

BOUND_PROBE_THREADS="${BOUND_PROBE_THREADS:-1}"
BOUND_PROBE_FOCUS_MODE="${BOUND_PROBE_FOCUS_MODE:-prove}"
BOUND_PROBE_MIP_FOCUS="${BOUND_PROBE_MIP_FOCUS:-}"
BOUND_PROBE_HEURISTICS="${BOUND_PROBE_HEURISTICS:-0.01}"
BOUND_PROBE_CUTS="${BOUND_PROBE_CUTS:-2}"
BOUND_PROBE_CUT_PASSES="${BOUND_PROBE_CUT_PASSES:-}"
BOUND_PROBE_PRESOLVE="${BOUND_PROBE_PRESOLVE:-2}"
BOUND_PROBE_START_NODE_LIMIT="${BOUND_PROBE_START_NODE_LIMIT:-100}"
BOUND_PROBE_START_TIME_LIMIT="${BOUND_PROBE_START_TIME_LIMIT:-1.0}"

mkdir -p "$(dirname "${OUT_JSON}")"

if [[ ! -f "${START_JSON}" ]]; then
  START_JSON="$(dirname "${OUT_JSON}")/bound_probe_empty_start.json"
  printf '{}\n' > "${START_JSON}"
fi

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

echo "Project root:       ${PROJECT_ROOT}"
echo "Container runtime:  ${CONTAINER_RUNTIME}"
echo "Instance:           ${INSTANCE_PATH}"
echo "Start JSON:         ${START_JSON}"
echo "Output JSON:        ${OUT_JSON}"
echo "Event log:          ${EVENT_LOG:-none}"
echo "Time limit:         ${TIME_LIMIT}"
echo "Threads:            ${BOUND_PROBE_THREADS}"
echo "Focus mode:         ${BOUND_PROBE_FOCUS_MODE}"
echo "Heuristics:         ${BOUND_PROBE_HEURISTICS}"
echo "Cuts:               ${BOUND_PROBE_CUTS:-auto}"
echo "Presolve:           ${BOUND_PROBE_PRESOLVE:-auto}"

CMD=(
  python3 -m scripts.final_gurobi_solve
  "${INSTANCE_PATH}"
  "${START_JSON}"
  --out "${OUT_JSON}"
  --time-limit "${TIME_LIMIT}"
  --threads "${BOUND_PROBE_THREADS}"
  --seed 17
  --focus-mode "${BOUND_PROBE_FOCUS_MODE}"
  --heuristics "${BOUND_PROBE_HEURISTICS}"
  --start-node-limit "${BOUND_PROBE_START_NODE_LIMIT}"
  --start-time-limit "${BOUND_PROBE_START_TIME_LIMIT}"
  --log-to-console
)

if [[ -n "${EVENT_LOG}" ]]; then
  mkdir -p "$(dirname "${EVENT_LOG}")"
  CMD+=(--event-log "${EVENT_LOG}")
fi

if [[ -n "${BOUND_PROBE_MIP_FOCUS}" ]]; then
  CMD+=(--mip-focus "${BOUND_PROBE_MIP_FOCUS}")
fi
if [[ -n "${BOUND_PROBE_CUTS}" ]]; then
  CMD+=(--cuts "${BOUND_PROBE_CUTS}")
fi
if [[ -n "${BOUND_PROBE_CUT_PASSES}" ]]; then
  CMD+=(--cut-passes "${BOUND_PROBE_CUT_PASSES}")
fi
if [[ -n "${BOUND_PROBE_PRESOLVE}" ]]; then
  CMD+=(--presolve "${BOUND_PROBE_PRESOLVE}")
fi

"${CONTAINER_RUNTIME}" exec --cleanenv \
  --no-mount bind-paths \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "$(dirname "${OUT_JSON}")":"$(dirname "${OUT_JSON}")" \
  --bind "${LICENSE_FILE}:/opt/gurobi/gurobi.lic" \
  --env GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
  --env PYTHONPATH="${PROJECT_ROOT}" \
  --env OMP_NUM_THREADS="${BOUND_PROBE_THREADS}" \
  "${APPTAINER_IMAGE}" \
  "${CMD[@]}"
