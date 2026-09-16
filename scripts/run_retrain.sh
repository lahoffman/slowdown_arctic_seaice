#!/usr/bin/env bash
# run_retrain.sh — revision Steps 1.4 + 1.5: train, predict and score the
# retrained CNN configurations, one tag after another, logging everything.
#
#   scripts/run_retrain.sh --smoke                 # 1 split, 1 seed, 2 epochs per tag (a few minutes)
#   scripts/run_retrain.sh                         # full run: rel_base rel_aux rel_lag1 (9 x 5 x 50 epochs each)
#   scripts/run_retrain.sh rel_aux                 # one configuration only
#   nohup scripts/run_retrain.sh > /dev/null 2>&1 &   # detached; progress in results/logs/retrain_<tag>_<stamp>.log
#   STOPPING=auprc scripts/run_retrain.sh rel_concurrent   # Phase-8 early-stopping rule (val AUPRC, min 5 epochs)
#
# Per tag:  04_cesm2le_cnn_train.py --tag T  →  06_cnn_predict_cesm2le.py --tag T
#           →  07_baselines.py --cnn-tag T --tag T --labels-file <relative> --demean group
#              (onset years read from the split file, so rel_lag1's 1991 start is handled)
# Re-running after a crash reuses the models already saved (--skip-existing).
# Requires SLOWDOWN_DATA_ROOT and the split files from 03_cesm2le_tvt_splits.py --tag T.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
: "${SLOWDOWN_DATA_ROOT:?set SLOWDOWN_DATA_ROOT first}"

LABELS="${LABELS:-$SLOWDOWN_DATA_ROOT/cesm2le/slowdowns/cesm2le_sie_slowdown_relative_SEP_w10_s1_group_1990-2100.nc}"
N_BOOT="${N_BOOT:-1000}"

SMOKE=0; TAGS=()
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    *) TAGS+=("$a") ;;
  esac
done
[[ ${#TAGS[@]} -eq 0 ]] && TAGS=(rel_base rel_aux rel_lag1)

TRAIN_OPTS=(--skip-existing)
[[ -n "${STOPPING:-}" ]] && TRAIN_OPTS+=(--stopping "$STOPPING")   # STOPPING=auprc for Phase-8 tags
if [[ $SMOKE -eq 1 ]]; then
  TRAIN_OPTS+=(--splits 0 --n-runs 1 --epochs 2)
  N_BOOT=50
fi

LOGDIR="$SLOWDOWN_DATA_ROOT/results/logs"; mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M)"

for TAG in "${TAGS[@]}"; do
  SUFFIX=""; [[ $SMOKE -eq 1 ]] && SUFFIX="_smoke"
  LOG="$LOGDIR/retrain_${TAG}_${STAMP}${SUFFIX}.log"
  SPLIT0="$SLOWDOWN_DATA_ROOT/results/tvt_splits/$TAG/cesm2le_sst_jja_slowdown_split0.nc"
  if [[ ! -f "$SPLIT0" ]]; then
    echo "!! no splits for tag '$TAG' ($SPLIT0) — run 03_cesm2le_tvt_splits.py --tag $TAG first" | tee -a "$LOG"
    exit 1
  fi
  # onset-year range the splits were built with (e.g. 1990-2030; 1991-2030 for lag1)
  read -r Y0 Y1 < <("$PY" -c "import xarray as xr; a=xr.open_dataset('$SPLIT0').attrs['target_years'].split('-'); print(a[0], a[1])")
  {
    echo "==== $(date)  tag=$TAG  smoke=$SMOKE  host=$(hostname)  target years $Y0-$Y1"
    echo "---- 04 train"
    "$PY" -u "$HERE/scripts/04_cesm2le_cnn_train.py" --tag "$TAG" "${TRAIN_OPTS[@]}"
    echo "---- 06 predict"
    "$PY" -u "$HERE/scripts/06_cnn_predict_cesm2le.py" --tag "$TAG"
    echo "---- 07 baselines (CNN scored against the same baselines, Fig. S5 axes)"
    "$PY" -u "$HERE/scripts/07_baselines.py" --labels-file "$LABELS" --demean group \
        --start-year "$Y0" --end-year "$Y1" --cnn-tag "$TAG" --tag "$TAG" --n-boot "$N_BOOT" --no-fig
    echo "==== $(date)  tag=$TAG  done"
  } 2>&1 | tee -a "$LOG"
  echo ">> $TAG finished — log: $LOG"
done

if [[ $SMOKE -eq 1 ]]; then
  echo
  echo "Smoke test passed. Smoke-test models/metrics/predictions live in models/<tag>, metrics/<tag>,"
  echo "predictions/cesm2le/<tag> and will be OVERWRITTEN by the full run except the model files:"
  echo "delete them first so --skip-existing does not keep the 2-epoch model:"
  for TAG in "${TAGS[@]}"; do echo "  rm $SLOWDOWN_DATA_ROOT/results/models/$TAG/cnn_jja_split0_run0.h5"; done
fi
