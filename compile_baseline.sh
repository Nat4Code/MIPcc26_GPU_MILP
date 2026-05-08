#!/usr/bin/env bash
set -euo pipefail

# Compile baseline_solve.c into ./baseline using the FULL Gurobi C library.
# This is intentionally pinned to your full Gurobi install:
#   /nfs/home/nsheets1/MIP/gurobi1301/linux64
#
# Usage:
#   ./compile_baseline_full.sh baseline_solve.c baseline

SRC="${1:-baseline_solve.c}"
OUT="${2:-baseline}"

GUROBI_HOME="${GUROBI_HOME:-/nfs/home/nsheets1/MIP/gurobi1301/linux64}"

if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: source file not found: ${SRC}"
  exit 1
fi

if [[ ! -d "${GUROBI_HOME}/include" ]]; then
  echo "ERROR: Gurobi include dir not found: ${GUROBI_HOME}/include"
  exit 1
fi

if [[ ! -d "${GUROBI_HOME}/lib" ]]; then
  echo "ERROR: Gurobi lib dir not found: ${GUROBI_HOME}/lib"
  exit 1
fi

FULL_LIB="${GUROBI_HOME}/lib/libgurobi130.so"
LIGHT_LIB="${GUROBI_HOME}/lib/libgurobi130_light.so"

if [[ ! -f "${FULL_LIB}" ]]; then
  echo "ERROR: full Gurobi library not found: ${FULL_LIB}"
  echo "Do not compile against the light library; WLS will fail."
  exit 1
fi

if [[ -f "${LIGHT_LIB}" ]]; then
  echo "Found light library too, but intentionally ignoring it:"
  echo "  ${LIGHT_LIB}"
fi

echo "GUROBI_HOME: ${GUROBI_HOME}"
echo "Using library: ${FULL_LIB}"
echo "Source:       ${SRC}"
echo "Output:       ${OUT}"

gcc -O3 -std=c11 -Wall -Wextra -pedantic \
  -I"${GUROBI_HOME}/include" \
  "${SRC}" \
  -L"${GUROBI_HOME}/lib" \
  -Wl,-rpath,"${GUROBI_HOME}/lib" \
  -lgurobi130 \
  -lm \
  -o "${OUT}"

echo
echo "Built ./${OUT}"
echo
echo "Linked libraries:"
ldd "./${OUT}" | grep -E 'gurobi|not found' || true

if ldd "./${OUT}" | grep -q 'libgurobi130_light'; then
  echo
  echo "ERROR: binary linked against light library. This should not happen."
  exit 1
fi

if ! ldd "./${OUT}" | grep -q 'libgurobi130.so'; then
  echo
  echo "WARNING: could not confirm libgurobi130.so in ldd output."
fi