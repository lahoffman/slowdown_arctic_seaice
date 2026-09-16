#!/usr/bin/env bash
# run_ohc2.sh — paper-B compute, second night: OHC700 extraction (~7.5 h, network-bound) and the "does even the
# realised ocean matter" ledgers, plus SST PCs and the tail asymmetry from the residual analysis.
#   screen -S ohc2; export SLOWDOWN_DATA_ROOT=…; conda activate arcticwatch
#   LBL1=$(ls $SLOWDOWN_DATA_ROOT/cesm2le/slowdowns/*_w10_*_off1_*.nc) scripts/run_ohc2.sh 2>&1 | tee $SLOWDOWN_DATA_ROOT/results/logs/run_ohc2_$(date +%Y%m%d_%H%M).log
#   SKIP_EXTRACT=1 … to run the analyses only (OHC700 file already there)
set -u
: "${SLOWDOWN_DATA_ROOT:?set SLOWDOWN_DATA_ROOT}"; : "${LBL1:?set LBL1 (offset labels file)}"
PY=${PY:-python}
step() { local name=$1; shift; echo; echo "==== $(date)  $name"; echo "     $*"; "$@" && echo "     ok" || echo "     FAILED (exit $?)"; }

# cheap analyses first (minutes), so results exist even if the extraction is slow
step residual_pcs   $PY -u scripts/09_residual_analysis.py --labels-file "$LBL1" --sst-pcs 10 20 --pc-mask-north 65
step tail_asym      $PY -u scripts/09_tail_asymmetry.py --key w10_off1
step conc100        $PY -u scripts/09_ohc_ledger.py --depth 100 --cv --n-pcs 10 20 --concurrent --exclude-north 65
step conc300        $PY -u scripts/09_ohc_ledger.py --depth 300 --cv --n-pcs 10 20 --concurrent --exclude-north 65
step conc300a       $PY -u scripts/09_ohc_ledger.py --depth 300 --cv --n-pcs 10 20 --concurrent --exclude-north 65 --months 1 2 3 4 5 6 7 8 9 10 11 12

# OHC700: a fresh pass over the store (the per-member cache is per depth)
[[ -z "${SKIP_EXTRACT:-}" ]] && step extract700 $PY -u scripts/01_cesm2le_ohc.py --depth 700
step sie_ledger700  $PY -u scripts/09_ohc_ledger.py --depth 700 --cv --n-pcs 10 20
step sie_ledger700a $PY -u scripts/09_ohc_ledger.py --depth 700 --cv --n-pcs 10 20 --months 1 2 3 4 5 6 7 8 9 10 11 12
step conc700        $PY -u scripts/09_ohc_ledger.py --depth 700 --cv --n-pcs 10 20 --concurrent --exclude-north 65
step gmt_ledger700  $PY -u scripts/09_ohc_ledger.py --target gmt --depth 700 --cv --n-pcs 10 20
echo; echo "==== $(date)  done — read results/ohc/*/summary.md and results/residual/w10_off1/{residual_summary,tail_asymmetry}.md"
