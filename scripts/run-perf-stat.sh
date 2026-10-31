#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# OCI Ampere A1 / Linux ARM: run with hardware cache counters.
set -euo pipefail
cd "$(dirname "$0")/.."
make build/bench
command -v perf >/dev/null || { echo "perf is not installed"; exit 2; }
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
  ./build/bench results/bench-perf.json
python3 scripts/roofline.py
