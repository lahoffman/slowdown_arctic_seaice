#!/usr/bin/env bash
# run_queue.sh — the overnight sequence, one job after another, each logged; a failing step is
# reported and the queue moves on (so a missing input never blocks the rest).
#
#   PID=$(pgrep -f "run_retrain.sh off1" | head -1)          # the training to wait for (empty = start now)
#   nohup scripts/run_queue.sh $PID > $SLOWDOWN_DATA_ROOT/results/logs/queue_$(date +%Y%m%d_%H%M).out 2>&1 &
#   tail -f $SLOWDOWN_DATA_ROOT/results/logs/queue_*.out
#
# Steps (edit the list at the bottom): postprocess off1 + off1_aux → XAI rel_aux, off1_aux → regression
# on the offset target → extra-Arctic splits (--mask-north 50, aux) → train off1_pacific → postprocess →
# XAI off1_pacific → observations with CNN votes → AIES figure build + status table.
# Requires SLOWDOWN_DATA_ROOT and LBL1 in the environment (put both exports in ~/.bashrc).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
PY="${PY:-python}"
: "${SLOWDOWN_DATA_ROOT:?set SLOWDOWN_DATA_ROOT}"; : "${LBL1:?set LBL1 (offset labels file)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
LOGDIR="$SLOWDOWN_DATA_ROOT/results/logs"; mkdir -p "$LOGDIR"
MASK_NORTH="${MASK_NORTH:-50}"

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "$(date)  waiting for pid $WAIT_PID to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
  echo "$(date)  pid $WAIT_PID gone — starting the queue"
fi

step() {   # step <name> <command...>
  local name="$1"; shift
  local log="$LOGDIR/queue_${name}_$(date +%Y%m%d_%H%M).log"
  echo; echo "==== $(date)  $name"; echo "     $*"; echo "     log: $log"
  if "$@" > "$log" 2>&1; then echo "     ok"; else echo "     FAILED (exit $?) — see $log"; fi
}

step postprocess_off1   scripts/run_postprocess.sh off1 off1_aux
step xai_rel_aux        $PY -u scripts/05b_xai_compare.py --tag rel_aux  --split 2 --run 0
step xai_off1_aux       $PY -u scripts/05b_xai_compare.py --tag off1_aux --split 2 --run 0
step regress_off1_aux   $PY -u scripts/04_cesm2le_cnn_regress.py --tag off1_aux --splits 2 5 7 --n-runs 2
step splits_pacific     $PY -u scripts/03_cesm2le_tvt_splits.py --labels-file "$LBL1" --demean group --end-year 2029 \
                            --aux sie_anom --mask-north "$MASK_NORTH" --tag off1_pacific --no-fig
step train_pacific      env LABELS="$LBL1" scripts/run_retrain.sh off1_pacific
step postprocess_pac    scripts/run_postprocess.sh off1_pacific
step xai_pacific        $PY -u scripts/05b_xai_compare.py --tag off1_pacific --split 2 --run 0
step regress_pacific    $PY -u scripts/04_cesm2le_cnn_regress.py --tag off1_pacific --splits 2 5 7 --n-runs 2
step obs_ersst          $PY -u scripts/10_obs_baseline_predict.py --labels-file "$LBL1" --forced linear --tag off1_aux --product ersst
step obs_oisst          $PY -u scripts/10_obs_baseline_predict.py --labels-file "$LBL1" --forced linear --tag off1_aux --product oisst
step event_stats        $PY -u scripts/09_event_stats.py --tag off1_aux --labels-file "$LBL1"
step figures_aies       $PY -u scripts/make_figures_aies.py
echo; echo "==== $(date)  queue finished"
