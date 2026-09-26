#!/bin/bash
# X31 SFG controlled cells, GPU 2. german first (smoke passed); bail and credit
# start only after their smoke log shows ALL GATES PASS. Resumable (CellStore).
set -u
cd /home/sypark/workspace/FairGate
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
PY=/home/sypark/miniconda3/envs/dev/bin/python
R=study/results/x31; L=$R/logs; S=$R/smoke/sfg_smoke.log
cell() {
  local ds=$1
  echo "=== $(date -Is) START SFG $ds" >> $L/stream_sfg.log
  $PY study/experiments/x30_run.py --method SFG --dataset $ds --protocol-name x31 \
      --splits 20 21 22 23 24 25 --runs 5 --epochs 200 --out $R/x31_SFG_$ds.csv >> $L/x31_SFG_$ds.log 2>&1
  local rc=$?
  echo "=== $(date -Is) END rc=$rc SFG $ds" >> $L/stream_sfg.log
}
# usage: x31_sfg_stream.sh <dataset> ...   (one lane; lanes must not share a dataset)
for ds in "$@"; do
  if grep -q "SMOKE SFG/$ds: ALL GATES PASS" $S; then cell $ds
  else echo "=== $(date -Is) SFG $ds smoke has not passed: cell not started" >> $L/stream_sfg.log; fi
done
