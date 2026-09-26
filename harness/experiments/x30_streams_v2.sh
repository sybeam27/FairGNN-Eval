#!/bin/bash
# X30 execution streams, order amendment 1 (user instruction 2026-09-18, before
# any X30 outcome): cells on the core datasets german/bail/credit run first.
# Scientific content unchanged. GPU 2 only. Resumable via CellStore.
# Usage: x30_streams_v2.sh <S1|S3|S4> [pid-to-wait-for]
set -u
cd /home/sypark/workspace/FairGate
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
R=study/results/x30; L=$R/logs
DEV=/home/sypark/miniconda3/envs/dev/bin/python
DGL=/home/sypark/x27_dgl_cuda/bin/python
STREAM=$1; WAITPID=${2:-}
if [ -n "$WAITPID" ]; then
  echo "=== $(date -Is) v2 waiting for running cell pid $WAITPID" | tee -a $L/stream_$STREAM.log
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
fi

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

CORE="german bail credit"; EXT="pokec_z pokec_n"
case "$STREAM" in
  S1) for enc in GCN GIN SAGE; do for ds in $CORE; do cell $DEV FairSIN $enc $ds; done; done
      touch $L/core_done_S1
      for enc in GCN GIN SAGE; do for ds in $CORE; do cell $DEV FairSIN $enc $ds --native; done; done
      for enc in GCN GIN SAGE; do for ds in $EXT; do cell $DEV FairSIN $enc $ds; done; done
      for enc in GCN GIN SAGE; do for ds in $EXT; do cell $DEV FairSIN $enc $ds --native; done; done ;;
  S3) for ds in bail credit; do cell $DGL BeMap "" $ds; done
      for enc in 1pct 10pct; do cell $DGL BIND $enc bail; done
      touch $L/core_done_S3
      cell $DGL BeMap "" pokec_z ;;
  S4) grep -q "ALL GATES PASS" $L/smoke_BIND-1pct_income.log || { echo "=== $(date -Is) BIND income admission smoke did not pass: cells stopped" | tee -a $L/stream_S4.log; exit 3; }
      echo "=== $(date -Is) smoke passed; waiting for core-dataset cells (S1, S3)" | tee -a $L/stream_S4.log
      while [ ! -f $L/core_done_S1 ] || [ ! -f $L/core_done_S3 ]; do sleep 120; done
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
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
