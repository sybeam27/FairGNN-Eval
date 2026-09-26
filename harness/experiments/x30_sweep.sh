#!/bin/bash
# X30 completion sweep: after the streams finish, re-run every admitted cell that
# is not complete (30 units / 61 lines). A cell can be short only for resource
# reasons (e.g. a CUDA OOM while another cell held the GPU); the protocol says to
# wait and retry on GPU 2, never to move GPUs. CellStore resume means a partially
# written cell continues where it stopped. Sequential, with a free-memory check.
set -u
cd /home/sypark/workspace/FairGate
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
R=study/results/x30; L=$R/logs
DEV=/home/sypark/miniconda3/envs/dev/bin/python
EDT=/home/sypark/x30_edits_env/bin/python
DGL=/home/sypark/x27_dgl_cuda/bin/python
NEED_MB=${NEED_MB:-9000}

while pgrep -f "x30_streams(_v2)?\.sh" > /dev/null; do sleep 120; done
echo "=== $(date -Is) sweep start (streams finished)" >> $L/stream_sweep.log

run_cell() {  # py method encoder dataset native_flag
  local py=$1 m=$2 enc=$3 ds=$4 nat=$5
  local tag=$m; [ -n "$enc" ] && tag=$m-$enc
  local pre=x30; [ -n "$nat" ] && pre=x30native
  local out=$R/${pre}_${tag}_${ds}.csv
  local have=0; [ -f "$out" ] && have=$(wc -l < "$out")
  [ "$have" -ge 61 ] && return 0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)
    [ "$free" -ge "$NEED_MB" ] && break
    sleep 180
  done
  echo "=== $(date -Is) SWEEP START $pre $tag $ds (had $have lines)" >> $L/stream_sweep.log
  $py study/experiments/x30_run.py --method $m ${enc:+--encoder $enc} --dataset $ds \
      --splits 20 21 22 23 24 25 --runs 5 --epochs 200 $nat --out $out >> $L/${pre}_${tag}_${ds}.log 2>&1
  local rc=$?
  echo "=== $(date -Is) SWEEP END rc=$rc $pre $tag $ds" >> $L/stream_sweep.log
}

# Amendment 2 (2026-09-18, user instruction, before any X30 outcome): pokec cells
# are paused. DS_SWEEP lists the datasets the sweep may start; the pokec cells
# stay admitted and are run later.
DS_SWEEP=${DS_SWEEP:-"german bail credit"}
for pass in 1 2 3; do
  for enc in GCN GIN SAGE; do
    for ds in $DS_SWEEP; do
      run_cell $DEV FairSIN $enc $ds ""
      run_cell $DEV FairSIN $enc $ds --native
    done
  done
  for ds in german bail credit; do run_cell $EDT EDITS "" $ds ""; run_cell $EDT FairEdit "" $ds ""; done
  run_cell $EDT GEAR "" bail ""
  for ds in bail credit pokec_z; do run_cell $DGL BeMap "" $ds ""; done
  for enc in 1pct 10pct; do run_cell $DGL BIND $enc bail ""; done
  for enc in 1pct 10pct; do
    for sp in 20 21 22 23 24 25; do
      out=$R/x30_BIND-${enc}_income_s$sp.csv
      have=0; [ -f "$out" ] && have=$(wc -l < "$out")
      [ "$have" -ge 11 ] && continue        # 5 units x 2 selector rows + header
      echo "=== $(date -Is) SWEEP START BIND-$enc income s$sp (had $have lines)" >> $L/stream_sweep.log
      $DGL study/experiments/x30_run.py --method BIND --encoder $enc --dataset income \
          --splits $sp --runs 5 --epochs 200 --out $out >> $L/x30_BIND-${enc}_income_s$sp.log 2>&1
      echo "=== $(date -Is) SWEEP END rc=$? BIND-$enc income s$sp" >> $L/stream_sweep.log
    done
  done
  echo "=== $(date -Is) sweep pass $pass done" >> $L/stream_sweep.log
done
