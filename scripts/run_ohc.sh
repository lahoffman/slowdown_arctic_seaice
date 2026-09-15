#!/usr/bin/env bash
# run_ohc.sh — tomorrow's OHC sequence in one screen: extract OHC100+OHC300 for the cmip6 group from the
# AWS Zarr store (~8 min/member, network-bound), then the two analyses on the result.
#
#   screen -S ohc; export SLOWDOWN_DATA_ROOT=…; conda activate arcticwatch
#   scripts/run_ohc.sh 2>&1 | tee $SLOWDOWN_DATA_ROOT/results/logs/run_ohc_$(date +%Y%m%d_%H%M).log
#   SKIP_EXTRACT=1 scripts/run_ohc.sh        # analyses only (files already there)
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; PY="${PY:-python}"
: "${SLOWDOWN_DATA_ROOT:?set SLOWDOWN_DATA_ROOT}"
step() { local n="$1"; shift; echo; echo "==== $(date)  $n"; echo "     $*"; if "$@"; then echo "     ok"; else echo "     FAILED (exit $?)"; fi; }

[[ -z "${SKIP_EXTRACT:-}" ]] && step extract      $PY -u scripts/01_cesm2le_ohc.py --depth 100 300
step lb22_single   $PY -u scripts/09_ohc_lb22_test.py                              # one held-out block, 20 PCs — the first look
step lb22_cv       $PY -u scripts/09_ohc_lb22_test.py --cv --n-pcs 20 50 --ann      # all blocks, two PC counts, LB22-style MLP
step sie_ledger100 $PY -u scripts/09_ohc_sie_ledger.py --depth 100 --cv --n-pcs 10 20
step sie_ledger300 $PY -u scripts/09_ohc_sie_ledger.py --depth 300 --cv --n-pcs 10 20
echo; echo "==== $(date)  done — read results/ohc/*/summary.md"
