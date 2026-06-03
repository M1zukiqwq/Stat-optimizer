#!/usr/bin/env bash
# Multi-config sparse-feedback K sweep with the v3 prior (MR2).
# Grid matches the published dense batch (postgres_batch_v3):
#   left_shift/right_shift x {100K,200K} rows x 3 seeds = 12 configs,
# swept over feedback budget K in {2,4,6,8,16}. K=16 should reproduce the
# dense batch (within ANALYZE-sampling noise).
set -u
cd "$(dirname "$0")"
PY=../.venv_v3/bin/python
RUN=oasis_torch/run_v3.py
export V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt
OUT=results/exp2_sparse_multiconfig_v3_20260602
mkdir -p "$OUT"
for K in 2 4 6 8 16; do
  echo "========== [$(date +%H:%M:%S)] K=$K =========="
  $PY $RUN pg --batch \
    --batch-drift-families left_shift right_shift \
    --batch-rows 100000 200000 \
    --batch-seeds 20260529 20260530 20260531 \
    --dim-rows-ratio 0.06 --min-dim-rows 5000 \
    --num-feedback $K \
    --output-dir "$OUT/k$K" > "$OUT/k$K.log" 2>&1
  echo "   K=$K rc=$? -> $OUT/k$K"
done
echo "========== ALL DONE [$(date +%H:%M:%S)] =========="
