"""
Overhead Latency Test for OASIS

Tests the pure Python/NumPy inference latency of:
1. Teacher (Isotonic Regression)
2. MLP Model (Tensorization + Forward Pass)

This is to demonstrate that the correction step runs well within
the typical 200ms planning budget of the Presto CBO.
"""
import timeit
import json
from pathlib import Path
import random
import numpy as np

from json_histogram_parser import load_feedback_sample
from cdf_teacher import correct_quantiles
from tensorizer import tensorize_sample
from mlp_histogram_model import MlpHistogramModel

def main():
    # 1. Load a realistic test sample (from q=10 ablation data)
    test_dir = Path("ablation_C_work/test_q10")
    if not test_dir.exists():
        print(f"Error: {test_dir} not found. Please check paths.")
        return
        
    sample_files = list(test_dir.glob("*.json"))
    if not sample_files:
        print(f"Error: No json files in {test_dir}")
        return
        
    sample_file = sample_files[0]
    sample_data = load_feedback_sample(str(sample_file))
    
    # 2. Load the trained MLP model
    mlp_path = Path("sensitivity_K_work/k_16/artifacts/mlp_train_q10_20_k1000.json")
    if not mlp_path.exists():
        print(f"Error: {mlp_path} not found.")
        return
    mlp_model = MlpHistogramModel.load(str(mlp_path))
    
    # Define microbenchmarks
    def run_teacher():
        _ = correct_quantiles(sample_data)
        
    def run_mlp():
        record = tensorize_sample(sample_data, max_observations=mlp_model.max_observations, teacher_fn=None, use_time_decay=False)
        ft = record.feature_tensor
        _ = mlp_model.predict([list(ft)])
        
    # Warmup
    for _ in range(10):
        run_teacher()
        run_mlp()
        
    # Measure Teacher Latency
    n_teacher = 1000
    t_teacher = timeit.timeit(run_teacher, number=n_teacher)
    avg_teacher_ms = (t_teacher / n_teacher) * 1000
    
    # Measure MLP Latency
    n_mlp = 1000
    t_mlp = timeit.timeit(run_mlp, number=n_mlp)
    avg_mlp_ms = (t_mlp / n_mlp) * 1000
    
    print("=" * 50)
    print("OASIS Overhead Latency Test (Pure Python/NumPy)")
    print(f"Sample: {sample_file.name} (Observations: {len(sample_data.observations)})")
    print(f"Model: {mlp_path.name}")
    print("=" * 50)
    print(f"Runs per method: {n_teacher}")
    print(f"{'Method':<20} {'Avg Latency (ms)':<15}")
    print("-" * 50)
    print(f"{'Teacher':<20} {avg_teacher_ms:<15.4f}")
    print(f"{'OASIS MLP':<20} {avg_mlp_ms:<15.4f}")
    print("=" * 50)
    
if __name__ == "__main__":
    main()
