#!/usr/bin/env bash
# run_postprocess.sh — everything that needs the trained models but no more training
# (steps 1.7, 5.3, 6.2/6.3), one tag after another, logged.
#
#   scripts/run_postprocess.sh                       # rel_base rel_aux rel_lag1 rel_openwater
#   scripts/run_postprocess.sh rel_aux               # one tag
#   FORCED="ensmean group_smbb" scripts/run_postprocess.sh
#   nohup bash -c "while kill -0 <retrain-pid> 2>/dev/null; do sleep 300; done; scripts/run_postprocess.sh" > .../logs/postprocess_nohup.out 2>&1 &
#
# Per tag:  08_occlusion.py            region-occlusion test (fast)
#           03_obs_test.py + 06_cnn_predict_obs.py   for each product x forced method (fast)
#           05_cesm2le_lrp.py          LRP-z on all 45 models (slow; last so the quick results land first)
# Skips a tag whose models are not all there yet; safe to rerun (LRP files are overwritten, nothing else changes).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
: "${SLOWDOWN_DATA_ROOT:?set SLOWDOWN_DATA_ROOT first}"
PRODUCTS="${PRODUCTS:-ersst oisst}"
FORCED="${FORCED:-ensmean group_smbb group_cmip6 linear}"
OBS_END_YEAR="${OBS_END_YEAR:-2025}"
SKIP_LRP="${SKIP_LRP:-0}"

TAGS=("$@"); [[ ${#TAGS[@]} -eq 0 ]] && TAGS=(rel_base rel_aux rel_lag1 rel_openwater)
LOGDIR="$SLOWDOWN_DATA_ROOT/results/logs"; mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M)"

for TAG in "${TAGS[@]}"; do
  LOG="$LOGDIR/postprocess_${TAG}_${STAMP}.log"
  N_MODELS=$(ls "$SLOWDOWN_DATA_ROOT/results/models/$TAG"/cnn_jja_split*_run*.h5 2>/dev/null | wc -l)
  if [[ $N_MODELS -lt 45 ]]; then
    echo "!! $TAG has $N_MODELS/45 models — skipping (rerun later)" | tee -a "$LOG"; continue
  fi
  {
    echo "==== $(date)  tag=$TAG  host=$(hostname)"
    echo "---- 08 occlusion"
    "$PY" -u "$HERE/scripts/08_occlusion.py" --tag "$TAG"
    for P in $PRODUCTS; do
      for F in $FORCED; do
        echo "---- 03/06 observations  product=$P  forced=$F"
        if "$PY" -u "$HERE/scripts/03_obs_test.py" --product "$P" --forced-method "$F" --tag "$TAG" --end-year "$OBS_END_YEAR"; then
          "$PY" -u "$HERE/scripts/06_cnn_predict_obs.py" --product "$P" --forced-method "$F" --tag "$TAG"
        else
          echo "!! obs input failed for $P/$F (see above) — continuing"
        fi
      done
    done
    if [[ "$SKIP_LRP" != "1" ]]; then
      echo "---- 05 LRP"
      "$PY" -u "$HERE/scripts/05_cesm2le_lrp.py" --tag "$TAG"
    fi
    echo "==== $(date)  tag=$TAG  done"
  } 2>&1 | tee -a "$LOG"
  echo ">> $TAG finished — log: $LOG"
done
