#!/usr/bin/env bash
# Path A: build a leakage-free real-column TRAINING pool for retraining the OASIS prior.
# Columns come ONLY from Wine Quality + Bike Sharing -- never Power/Forest/Census (the
# held-out real TEST sets). A few columns are reserved as the val (early-stop) split.
set -u
cd "$(dirname "$0")"                       # experiments/
PY=../.venv_v3/bin/python
OUT=data/real_cases/train_pool
INT="0.1 0.3 0.5 1.0"
rm -rf "$OUT"
G() { $PY make_real_dataset_cases.py "$@" --intensities $INT --out "$OUT" --k 16 >/dev/null; }

# wine-red  (N=1599, ';' header) -- 11 numeric cols -> train
for c in 0 1 2 3 4 5 6 7 8 9 10; do
  G --csv data/real/winequality-red.csv --col $c --sep ';' --skip-header \
    --name wred_c$c --window 300 --stride 600 --cases-per-intensity 60 --seed $((100+c)) --split train
done
# wine-white (N=4898, ';' header) -- cols 0-8 train, 9-10 val
for c in 0 1 2 3 4 5 6 7 8; do
  G --csv data/real/winequality-white.csv --col $c --sep ';' --skip-header \
    --name wwhite_c$c --window 800 --stride 2000 --cases-per-intensity 60 --seed $((200+c)) --split train
done
for c in 9 10; do
  G --csv data/real/winequality-white.csv --col $c --sep ';' --skip-header \
    --name wwhite_c$c --window 800 --stride 2000 --cases-per-intensity 30 --seed $((200+c)) --split test
done
# bike hour (N=17379, ',' header) -- temp/atemp/hum/windspeed/casual train; registered/cnt val
for c in 10 11 12 13 14; do
  G --csv data/real/hour.csv --col $c --sep ',' --skip-header \
    --name bike_c$c --window 2000 --stride 8000 --cases-per-intensity 60 --seed $((300+c)) --split train
done
for c in 15 16; do
  G --csv data/real/hour.csv --col $c --sep ',' --skip-header \
    --name bike_c$c --window 2000 --stride 8000 --cases-per-intensity 30 --seed $((300+c)) --split test
done

echo "train cases: $(find "$OUT" -path '*train_q*/*.json' | wc -l | tr -d ' ')"
echo "val cases:   $(find "$OUT" -path '*test_q*/*.json' | wc -l | tr -d ' ')"
