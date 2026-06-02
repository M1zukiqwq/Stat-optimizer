#!/usr/bin/env bash
# Re-run supporting experiments with the v3 composite-objective prior.
# Reuses the EXISTING cached compound_data (does not regenerate shared data).
set -u
cd "$(dirname "$0")"            # experiments/
PY=../.venv_v3/bin/python
RUN=oasis_torch/run_v3.py
export V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt
DATA=results/synthetic_paper_suite_rerun_20260529/compound_data
MODEL=results/synthetic_paper_suite_rerun_20260529/models/oasis_k16.json
LOG=results/_v3_supporting_logs
mkdir -p "$LOG"

run() {  # name target args...
  local name="$1"; shift; local target="$1"; shift
  echo "===== [$(date +%H:%M:%S)] START $name ($target) ====="
  $PY $RUN "$target" "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "===== [$(date +%H:%M:%S)] END   $name rc=$rc ====="
}

run stage1swap stage1swap --output-dir results/stage1_estimator_swap_v3 \
    --data-root "$DATA" --model-path "$MODEL" \
    --q-values 1 3 5 10 15 20 25 30 --max-cases-per-q 128 --seed 20260531

run ood ood --output-dir results/ood_drift_realism_v3 \
    --model-path "$MODEL" --cases-per-pattern 128 --seed 20260529

run trace trace --output-dir results/trace_grounded_drift_v3 \
    --model-path "$MODEL" --cases-per-trace 96 --seed 20260529

run public public --output-dir results/public_trace_workload_v3 \
    --model-path "$MODEL" --cases-per-column 96 --seed 20260601 --no-download --no-progress

run odp odp --output-dir results/optimizer_decision_proxy_v3 \
    --data-root "$DATA" --model-path "$MODEL" \
    --q-values 5 10 15 20 25 30 --max-cases-per-q 128 --seed 42

echo "===== ALL DONE [$(date +%H:%M:%S)] ====="
