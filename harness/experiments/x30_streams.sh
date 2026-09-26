#!/bin/bash
# X30 execution streams (protocol section 9). GPU 2 only. Resumable: the runner's
# CellStore skips persisted units. A failed cell is logged and the stream moves on
# (a stopped cell does not block others; protocol section 10).
set -u
cd /home/sypark/workspace/FairGate
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
R=study/results/x30; L=$R/logs
DEV=/home/sypark/miniconda3/envs/dev/bin/python
EDT=/home/sypark/x30_edits_env/bin/python
DGL=/home/sypark/x27_dgl_cuda/bin/python

cell() {  # py method encoder dataset [--native]
  local py=$1 m=$2 enc=$3 ds=$4 nat=${5:-}
  local tag=$m; [ -n "$enc" ] && tag=$m-$enc
  local pre=x30; [ -n "$nat" ] && pre=x30native
  local out=$R/${pre}_${tag}_${ds}.csv log=$L/${pre}_${tag}_${ds}.log
  echo "=== $(date -Is) START $pre $tag $ds" | tee -a $L/stream_$STREAM.log
  $py study/experiments/x30_run.py --method $m ${enc:+--encoder $enc} --dataset $ds \
      --splits 20 21 22 23 24 25 --runs 5 --epochs 200 $nat --out $out >> $log 2>&1
  local rc=$?                       # captured before any $( ) resets it
  echo "=== $(date -Is) END rc=$rc $pre $tag $ds" | tee -a $L/stream_$STREAM.log
}

case "$1" in
  S1) STREAM=S1
      for enc in GCN GIN SAGE; do for ds in german bail credit pokec_z pokec_n; do cell $DEV FairSIN $enc $ds; done; done
      for enc in GCN GIN SAGE; do for ds in german bail credit pokec_z pokec_n; do cell $DEV FairSIN $enc $ds --native; done; done ;;
  S2) STREAM=S2
      for ds in german bail credit; do cell $EDT EDITS "" $ds; done
      for ds in german bail credit; do cell $EDT FairEdit "" $ds; done
      cell $EDT GEAR "" bail ;;
  S3) STREAM=S3
      for ds in bail credit pokec_z; do cell $DGL BeMap "" $ds; done
      for enc in 1pct 10pct; do cell $DGL BIND $enc bail; done ;;
  S4) STREAM=S4   # BIND income: admission smoke first (no timeout), then split-parallel
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
      echo "=== $(date -Is) START smoke BIND-1pct income" | tee -a $L/stream_S4.log
      $DGL study/experiments/x30_run.py --method BIND --encoder 1pct --dataset income \
          --splits 20 --runs 1 --epochs 200 --smoke > $L/smoke_BIND-1pct_income.log 2>&1
      rc=$?; echo "=== $(date -Is) END smoke rc=$rc" | tee -a $L/stream_S4.log
      grep -q "ALL GATES PASS" $L/smoke_BIND-1pct_income.log || { echo "BIND income admission smoke FAILED: cells stopped" | tee -a $L/stream_S4.log; exit 3; }
      for enc in 1pct 10pct; do
        for sp in 20 21 22 23 24 25; do
          ( out=$R/x30_BIND-${enc}_income_s$sp.csv; log=$L/x30_BIND-${enc}_income_s$sp.log
            echo "=== $(date -Is) START BIND-$enc income s$sp" >> $L/stream_S4.log
            $DGL study/experiments/x30_run.py --method BIND --encoder $enc --dataset income \
                --splits $sp --runs 5 --epochs 200 --out $out >> $log 2>&1
            rc=$?; echo "=== $(date -Is) END rc=$rc BIND-$enc income s$sp" >> $L/stream_S4.log ) &
        done
        wait
      done ;;
esac
