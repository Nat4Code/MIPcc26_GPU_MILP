#!/usr/bin/env bash
#SBATCH --job-name=milp_heur
#SBATCH -A ISAAC-UTK0448
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

set -euo pipefail

PLAN_JSON="${1:?need plan.json}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?missing array task id}"
echo "HOSTNAME=$(hostname)"
echo "START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 scripts/run_task.py "${PLAN_JSON}" "${SLURM_ARRAY_TASK_ID}"

echo "END=$(date -u +%Y-%m-%dT%H:%M:%SZ)"