#!/usr/bin/env bash
# X31 FairVGNN configuration-variation cells, GPU 2 only, sequential per lane.
# A cell runs only after its smoke block shows SMOKE PASS.
# usage: [NATIVE=1] x31_fairvgnn_config_stream.sh <dataset:config> ...
#        (one lane; lanes must not share a cell)
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=2
PY=/home/sypark/miniconda3/envs/dev/bin/python
OUT=study/results/x31; LOG=$OUT/logs; mkdir -p "$LOG"
if [ "${NATIVE:-0}" = 1 ]; then
  SMOKE=$OUT/smoke/fairvgnn_config_native_smoke.log; PRE=x31native; FLAG=--native
else
  SMOKE=$OUT/smoke/fairvgnn_config_smoke.log; PRE=x31; FLAG=
fi
for dc in "$@"; do
  d=${dc%%:*}; c=${dc##*:}
  if [ "$(awk -v h="== $d $c" '$0==h{f=1;next} /^== /{f=0} f' "$SMOKE" | grep -c 'SMOKE PASS')" != 1 ]; then
    echo "=== $(date -Is) SKIP $PRE $c $d (smoke did not pass)" >> "$LOG/stream_fairvgnn_config.log"; continue
  fi
  echo "=== $(date -Is) START $PRE $c $d" >> "$LOG/stream_fairvgnn_config.log"
  $PY study/experiments/x31_fairvgnn_config_run.py --dataset "$d" --config "$c" $FLAG \
      --out "$OUT/${PRE}_FairVGNN-${c}_${d}.csv" > "$LOG/${PRE}_FairVGNN-${c}_${d}.log" 2>&1
  rc=$?
  echo "=== $(date -Is) END $PRE $c $d rc=$rc" >> "$LOG/stream_fairvgnn_config.log"
done
