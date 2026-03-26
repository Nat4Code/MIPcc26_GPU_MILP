#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${1:?Usage: submit_workflow.sh <instance.mps> [config.json]}"
CONFIG="${2:-config/default_config.json}"

mkdir -p logs results tmp

FEATURES_JSON="tmp/features.json"
PLAN_JSON="tmp/plan.json"

python3 scripts/feature_extract.py "${INSTANCE}" "${FEATURES_JSON}"
python3 scripts/make_plan.py "${CONFIG}" "${FEATURES_JSON}" "${INSTANCE}" "${PLAN_JSON}"

NUM_TASKS=$(python3 - <<'PY'
import json
with open("tmp/plan.json","r") as f:
    p=json.load(f)
print(len(p["tasks"]))
PY
)

echo "Submitting ${NUM_TASKS} array tasks"

sbatch --array=0-$((NUM_TASKS-1)) slurm/run_array_task.sh "${PLAN_JSON}"