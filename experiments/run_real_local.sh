#!/usr/bin/env bash
# Local real-dataset single-column evaluation (CPU, reuses the pretrained v3 prior).
# Generates sliding-window drift cases from DMV/Census/Forest/Power columns at dense
# (K=16) and sparse (K=2) feedback, then runs the canonical stage1swap evaluator so the
# projection / router / Q-error code and the v3 prior are reused verbatim.
set -u
cd "$(dirname "$0")"                       # experiments/
PY=../.venv_v3/bin/python
export V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt
MODEL=results/synthetic_paper_suite_rerun_20260529/models/oasis_k16.json
LOG=results/_real_local_logs; mkdir -p "$LOG"
INT="0.1 0.3 0.5 1.0"; QV="1 3 5 10"; CPI=128

# dataset  csv  col  sep  skipheader  window  stride
DATASETS=(
  "power|data/real/household_power_consumption.txt|2|;|1|20000|400000"
  "forest|data/real/covtype.data|0|,|0|20000|200000"
  "census|data/real/adult.data|0|,|0|6000|13000"
)

gen() {  # name csv col sep skip window stride K outdir
  local name=$1 csv=$2 col=$3 sep=$4 skip=$5 win=$6 stride=$7 K=$8 out=$9
  local hdr=""; [ "$skip" = "1" ] && hdr="--skip-header"
  $PY make_real_dataset_cases.py --csv "$csv" --col "$col" --sep "$sep" $hdr \
      --name "$name" --window "$win" --stride "$stride" --out "$out" \
      --intensities $INT --cases-per-intensity $CPI --k "$K" --seed 42
}

for spec in "${DATASETS[@]}"; do
  IFS='|' read -r name csv col sep skip win stride <<< "$spec"
  for K in 16 2; do
    tagk=$([ "$K" = "16" ] && echo dense || echo sparse)
    out=data/real_cases/${name}_${tagk}
    echo "===== [$(date +%H:%M:%S)] GEN $name K=$K ($tagk) ====="
    gen "$name" "$csv" "$col" "$sep" "$skip" "$win" "$stride" "$K" "$out" 2>&1 | tail -1
    echo "===== [$(date +%H:%M:%S)] EVAL $name K=$K ====="
    $PY oasis_torch/run_v3.py stage1swap \
        --output-dir results/real_${name}_${tagk} --data-root "$out" \
        --model-path "$MODEL" --q-values $QV --max-cases-per-q $CPI --seed 20260531 \
        > "$LOG/${name}_${tagk}.log" 2>&1
    echo "  rc=$? -> results/real_${name}_${tagk}/estimator_swap_overall.csv"
  done
done
echo "===== ALL DONE [$(date +%H:%M:%S)] ====="
