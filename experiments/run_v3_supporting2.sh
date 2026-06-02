#!/usr/bin/env bash
# Batch 2: feedback budget + noise sweeps with the v3 prior (feed deployment-safety
# + feedback-sensitivity). Reuses existing cached compound_data.
set -u
cd "$(dirname "$0")"            # experiments/
PY=../.venv_v3/bin/python
RUN=oasis_torch/run_v3.py
export V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt
DATA=results/synthetic_paper_suite_rerun_20260529/compound_data
MODEL=results/synthetic_paper_suite_rerun_20260529/models/oasis_k16.json
LOG=results/_v3_supporting_logs
mkdir -p "$LOG"

run() {
  local name="$1"; shift; local target="$1"; shift
  echo "===== [$(date +%H:%M:%S)] START $name ($target) ====="
  $PY $RUN "$target" "$@" > "$LOG/$name.log" 2>&1
  echo "===== [$(date +%H:%M:%S)] END   $name rc=$? ====="
}

run budget budget --output-dir results/feedback_budget_sensitivity_v3 \
    --data-root "$DATA" --model-path "$MODEL" \
    --q-values 5 10 15 20 25 30 --max-cases-per-q 128 --seed 42

run noise noise --output-dir results/feedback_noise_robustness_v3 \
    --data-root "$DATA" --model-path "$MODEL" \
    --q-values 5 10 15 20 25 30 --max-cases-per-q 128 --seed 42

echo "===== BATCH2 DONE [$(date +%H:%M:%S)] ====="
