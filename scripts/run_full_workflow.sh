#!/usr/bin/env bash
set -euo pipefail

INSTANCE_PATH="${1:?need instance path}"
CONFIG_PATH="${2:?need config path}"

WORKDIR="${3:-results/full_run}"
FARM_RESULTS_DIR="${WORKDIR}/farm_results"
FEATURES_JSON="${WORKDIR}/features.json"
PLAN_JSON="${WORKDIR}/plan.json"
MERGED_JSON="${WORKDIR}/merged.json"
FINAL_JSON="${WORKDIR}/final_gurobi.json"

mkdir -p "${WORKDIR}" "${FARM_RESULTS_DIR}" logs results

# Budget split for testing
FARM_SECONDS="${FARM_SECONDS:-10}"
FINAL_SECONDS="${FINAL_SECONDS:-50}"
TOTAL_SECONDS=$(( FARM_SECONDS + FINAL_SECONDS ))

python3 scripts/feature_extract.py "${INSTANCE_PATH}" --out "${FEATURES_JSON}"
python3 scripts/make_plan.py "${CONFIG_PATH}" "${FEATURES_JSON}" --out "${PLAN_JSON}"

# Submit / run the farm externally before this point in production.
# For local testing, you can loop over tasks yourself or use test_pipeline_with_final_solve.py.

python3 scripts/merge_results.py "${FARM_RESULTS_DIR}" --out "${MERGED_JSON}"

python3 scripts/final_gurobi_solve.py \
  "${INSTANCE_PATH}" \
  "${MERGED_JSON}" \
  --out "${FINAL_JSON}" \
  --time-limit "${FINAL_SECONDS}" \
  --threads 1 \
  --seed 0 \
  --mip-focus 1 \
  --heuristics 0.05 \
  --start-node-limit 500 \
  --start-time-limit 2.0 \
  --log-to-console

echo "Workflow finished."
echo "Merged results: ${MERGED_JSON}"
echo "Final solve:    ${FINAL_JSON}"
echo "Total nominal optimization budget: ${TOTAL_SECONDS}s"