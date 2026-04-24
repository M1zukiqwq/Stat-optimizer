#!/usr/bin/env python3
"""
Distribution Generalization Experiment (Q3)
============================================

Test OASIS generalization to different initial data distributions.
Uses existing trained model + existing drift simulator.

Approach:
1. Use existing compound drift model (trained on Gaussian mixtures)
2. Generate test data with different initial distributions:
   - Uniform
   - Highly skewed (power-law)
   - Bimodal
   - Triangular
3. Apply same compound drift process
4. Evaluate Q-Error
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent))

from simulate_memory_kll_dataset import MemoryTable, _draw_observation
from histogram_math import evaluate_piecewise_cdf, clamp01
from baselines import correct_stholes
from modern_baselines import correct_quicksel_h, correct_isomer
from tensorizer import tensorize_sample
# No complex imports needed for simple predictor
from datetime import datetime, timezone, timedelta


@dataclass
class ExperimentResult:
    init_dist: str
    q_error_stale: float
    q_error_stholes: float
    q_error_quicksel: float
    q_error_isomer: float
    q_error_oasis: float


def generate_initial_data_by_distribution(
    rng: random.Random, 
    size: int, 
    dist_type: str
) -> tuple[List[float], int]:
    """Generate initial data with specified distribution."""
    data = []
    
    if dist_type == "gaussian_mixture":
        # Original: 2-4 Gaussian centers
        centers = [rng.uniform(0.1, 0.9) for _ in range(rng.randint(2, 4))]
        for _ in range(size):
            c = rng.choice(centers)
            v = rng.normalvariate(c, 0.1)
            data.append(clamp01(v))
    
    elif dist_type == "uniform":
        # Uniform distribution
        for _ in range(size):
            data.append(rng.uniform(0.0, 1.0))
    
    elif dist_type == "skewed_powerlaw":
        # Power-law (80-20 rule)
        for _ in range(size):
            u = rng.random()
            # Power law: most values near 0, few near 1
            v = 1.0 - (1.0 - u) ** 0.2
            data.append(v)
    
    elif dist_type == "bimodal":
        # Two distinct peaks
        for _ in range(size):
            if rng.random() < 0.5:
                v = rng.normalvariate(0.25, 0.08)
            else:
                v = rng.normalvariate(0.75, 0.08)
            data.append(clamp01(v))
    
    elif dist_type == "triangular":
        # Triangular distribution (peak at 0.5)
        for _ in range(size):
            v = rng.triangular(0.0, 0.5, 1.0)
            data.append(v)
    
    elif dist_type == "exponential":
        # Exponential decay from 0
        for _ in range(size):
            v = 1.0 - np.exp(-rng.expovariate(3.0))
            data.append(min(1.0, v))
    
    null_count = int(size * rng.uniform(0.01, 0.1))
    return data, null_count


def build_test_case_for_distribution(
    dist_type: str,
    q_mods: int,
    rng: random.Random,
    bucket_count: int = 10,
    initial_rows: int = 5000
) -> Dict:
    """Build a test case with specified initial distribution."""
    
    # Generate initial data with specified distribution
    data, null_count = generate_initial_data_by_distribution(rng, initial_rows, dist_type)
    table = MemoryTable(data, null_count)
    
    # Capture prior
    prior_null_frac = table.get_null_fraction()
    quantile_levels = [i / bucket_count for i in range(1, bucket_count)]
    prior_boundaries = table.get_bucket_boundaries(bucket_count)
    prior_quantiles = prior_boundaries[1:-1]
    
    prior_x = prior_boundaries
    prior_p = [i / bucket_count for i in range(bucket_count + 1)]
    
    # Apply compound drift (same as training)
    persistent_center = rng.uniform(0.1, 0.9)
    observations = []
    observation_count = rng.randint(8, 24)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    
    for obs_index in range(observation_count):
        table.apply_drift(rng, q_mods, persistent_center=persistent_center)
        ts = base_time + timedelta(hours=obs_index)
        obs = _draw_observation(rng, table, prior_x, prior_p, prior_null_frac, ts)
        observations.append(obs)
    
    # Capture ground truth
    true_boundaries = table.get_bucket_boundaries(bucket_count)
    true_quantiles = true_boundaries[1:-1]
    
    return {
        "dist_type": dist_type,
        "prior_quantiles": prior_quantiles,
        "prior_null_frac": prior_null_frac,
        "true_quantiles": true_quantiles,
        "observations": observations,
        "min_val": 0.0,
        "max_val": 1.0,
    }


def compute_q_error(pred_quantiles: List[float], true_quantiles: List[float], rng: random.Random = None) -> float:
    """Compute Q-Error."""
    if rng is None:
        rng = random.Random(42)
        
    pred_boundaries = [0.0] + list(pred_quantiles) + [1.0]
    true_boundaries = [0.0] + list(true_quantiles) + [1.0]
    n_buckets = len(pred_boundaries) - 1
    
    pred_cdf_p = [i / n_buckets for i in range(n_buckets + 1)]
    true_cdf_p = pred_cdf_p.copy()
    
    errors = []
    eps = 1e-6
    for _ in range(50):
        v = rng.uniform(0.05, 0.95)
        est = max(evaluate_piecewise_cdf(pred_boundaries, pred_cdf_p, v), eps)
        act = max(evaluate_piecewise_cdf(true_boundaries, true_cdf_p, v), eps)
        errors.append(max(est / act, act / est))
        
    return sum(errors) / len(errors)


class MLPModelPredictor:
    """MLP model predictor using actual model weights from MlpHistogramModelV2."""
    
    def __init__(self, model_path: str):
        with open(model_path) as f:
            self.model = json.load(f)
        self.max_obs = self.model.get("max_observations", 16)
        self.prior_dim = self.model.get("prior_dim", 9)
        self.obs_dim = self.model.get("obs_dim", 12)
        self.num_heads = self.model.get("num_heads", 3)
        
        # Load attention weights (each head has a 12-dim weight vector and scalar bias)
        self.W_attn_heads = [np.array(w) for w in self.model.get("W_attn_heads", [])]
        self.b_attn_heads = self.model.get("b_attn_heads", [])
        
        # Load prior encoder weights (from 'prior_encoder' key)
        prior_enc = self.model.get("prior_encoder", [])
        if prior_enc:
            self.W_prior_enc = np.array(prior_enc[0]['W'])  # (32, 9)
            self.b_prior_enc = np.array(prior_enc[0]['b'])  # (32,)
        else:
            self.W_prior_enc = None
            
        # Load MLP weights (from 'layers' key)
        self.mlp_layers = []
        layers = self.model.get("layers", [])
        for layer in layers:
            self.mlp_layers.append({
                'W': np.array(layer['W']),
                'b': np.array(layer['b'])
            })
    
    def _encode_observation(self, obs: Dict) -> np.ndarray:
        """Encode a single observation into 12-dim vector."""
        # One-hot predicate (6 dims)
        predicates = ["<", "<=", ">", ">=", "=", "BETWEEN"]
        one_hot = [1.0 if obs["predicate_type"] == p else 0.0 for p in predicates]
        
        # Numeric features (6 dims) - must match training exactly
        v = obs["value"]
        v_upper = obs.get("value_upper", 0.0) or 0.0
        est_sel = obs["estimated_sel"]
        act_sel = obs["actual_sel"]
        
        # Check for BETWEEN to set has_upper and span correctly
        has_upper = 1.0 if obs["predicate_type"] == "BETWEEN" and v_upper > v else 0.0
        span = (v_upper - v) if has_upper > 0 else 0.0
        
        # Note: No time_decay in this version (matches model.obs_dim=12)
        return np.array(one_hot + [v, v_upper, est_sel, act_sel, has_upper, span])
    
    def _softmax_masked(self, scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute softmax with mask."""
        scores = scores * mask - 1e9 * (1 - mask)  # Masked positions get large negative
        exp_x = np.exp(scores - np.max(scores))
        return exp_x / (np.sum(exp_x) + 1e-12)
    
    def predict_quantiles(self, test_case: Dict) -> List[float]:
        """Predict using MLP model."""
        prior = np.array(test_case["prior_quantiles"])
        
        # Encode observations
        obs_list = test_case["observations"][-self.max_obs:]
        n_valid = len(obs_list)
        obs_encoded = [self._encode_observation(obs) for obs in obs_list]
        
        # Pad with zeros
        while len(obs_encoded) < self.max_obs:
            obs_encoded.append(np.zeros(self.obs_dim))
        
        obs_matrix = np.stack(obs_encoded)  # (max_obs, obs_dim)
        
        # Create mask for valid observations
        mask = np.zeros(self.max_obs)
        mask[:n_valid] = 1.0
        
        # Multi-head attention pooling
        pooled_list = []
        for h in range(self.num_heads):
            if h < len(self.W_attn_heads):
                # Compute attention scores: (16, 12) dot (12,) = (16,)
                scores = obs_matrix.dot(self.W_attn_heads[h]) + self.b_attn_heads[h]
                attn_weights = self._softmax_masked(scores, mask)
                # Weighted sum: (16,) dot (16, 12) = (12,)
                pooled = np.sum(attn_weights[:, None] * obs_matrix, axis=0)
                pooled_list.append(pooled)
        
        if pooled_list:
            pooled_all = np.concatenate(pooled_list)  # (36,) for 3 heads
        else:
            pooled_all = np.zeros(self.obs_dim * self.num_heads)
        
        # Prior encoding using ReLU
        if self.W_prior_enc is not None:
            prior_enc = np.maximum(0, prior.dot(self.W_prior_enc.T) + self.b_prior_enc)
        else:
            prior_enc = prior
        
        # Meta features (3 dims)
        meta = np.array([
            test_case["prior_null_frac"],
            n_valid / self.max_obs,
            10.0 / 64.0,  # bucket count ratio
        ])
        
        # Combine all features
        context = np.concatenate([prior_enc, meta, pooled_all])
        
        # MLP forward pass
        x = context
        for i, layer in enumerate(self.mlp_layers):
            W = layer['W']  # (out_dim, in_dim)
            b = layer['b']  # (out_dim,)
            x = W.dot(x) + b
            if i < len(self.mlp_layers) - 1:  # ReLU for hidden layers
                x = np.maximum(0, x)
        
        # Residual prediction: delta is added to prior
        delta = x  # (9,)
        corrected = prior + delta
        
        # Post-processing: clip and enforce monotonicity
        corrected = np.clip(corrected, 0.0, 1.0)
        corrected = np.maximum.accumulate(corrected)
        
        return corrected.tolist()
        x = context
        for i, layer in enumerate(self.mlp_weights):
            x = x.dot(layer['W']) + layer['b']
            if i < len(self.mlp_weights) - 1:  # ReLU for hidden layers
                x = np.maximum(0, x)
        
        # Residual prediction
        delta = x
        corrected = prior + delta
        corrected = np.clip(corrected, 0.0, 1.0)
        corrected = np.maximum.accumulate(corrected)
        
        return corrected.tolist()


def run_experiment(dist_type: str, n_cases: int = 200, q_mods: int = 10, seed: int = 42) -> ExperimentResult:
    """Run experiment for a single initial distribution."""
    rng = random.Random(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Running {dist_type} distribution (q={q_mods}, n={n_cases})")
    print('='*60)
    
    # Load OASIS model
    oasis_model = None
    model_paths = [
        "/Users/qichutian/presto/presto-cdf-simulation/ablation_study/work_simplified/models/oasis_model.json",
    ]
    for path in model_paths:
        if Path(path).exists():
            try:
                oasis_model = MLPModelPredictor(path)
                print(f"Loaded MLP model: {path}")
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    if oasis_model is None:
        print("Warning: No OASIS model found")
    
    q_errors = {"stale": [], "stholes": [], "quicksel": [], "isomer": [], "oasis": []}
    
    for i in range(n_cases):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_cases}")
        
        test_case = build_test_case_for_distribution(dist_type, q_mods, rng)
        
        # Stale
        q_err = compute_q_error(test_case["prior_quantiles"], test_case["true_quantiles"])
        q_errors["stale"].append(q_err)
        
        # STHoles
        try:
            result = correct_stholes(
                test_case["min_val"], test_case["max_val"],
                test_case["prior_quantiles"], test_case["observations"]
            )
            q_err = compute_q_error(result, test_case["true_quantiles"])
        except Exception as e:
            q_err = q_errors["stale"][-1]
        q_errors["stholes"].append(q_err)
        
        # QuickSel-H
        try:
            result = correct_quicksel_h(
                test_case["min_val"], test_case["max_val"],
                test_case["prior_quantiles"], test_case["observations"]
            )
            q_err = compute_q_error(result, test_case["true_quantiles"])
        except Exception as e:
            q_err = q_errors["stale"][-1]
        q_errors["quicksel"].append(q_err)
        
        # ISOMER
        try:
            result = correct_isomer(
                test_case["min_val"], test_case["max_val"],
                test_case["prior_quantiles"], test_case["observations"]
            )
            q_err = compute_q_error(result, test_case["true_quantiles"])
        except Exception as e:
            q_err = q_errors["stale"][-1]
        q_errors["isomer"].append(q_err)
        
        # OASIS
        if oasis_model:
            try:
                result = oasis_model.predict_quantiles(test_case)
                q_err = compute_q_error(result, test_case["true_quantiles"])
            except Exception as e:
                print(f"  OASIS error: {e}")
                q_err = q_errors["stale"][-1]
        else:
            q_err = q_errors["stale"][-1]
        q_errors["oasis"].append(q_err)
    
    result = ExperimentResult(
        init_dist=dist_type,
        q_error_stale=np.mean(q_errors["stale"]),
        q_error_stholes=np.mean(q_errors["stholes"]),
        q_error_quicksel=np.mean(q_errors["quicksel"]),
        q_error_isomer=np.mean(q_errors["isomer"]),
        q_error_oasis=np.mean(q_errors["oasis"]),
    )
    
    print(f"\nResults for {dist_type}:")
    print(f"  Stale:    {result.q_error_stale:.3f}")
    print(f"  STHoles:  {result.q_error_stholes:.3f} ({(1-result.q_error_stholes/result.q_error_stale)*100:+.1f}%)")
    print(f"  QuickSel: {result.q_error_quicksel:.3f} ({(1-result.q_error_quicksel/result.q_error_stale)*100:+.1f}%)")
    print(f"  ISOMER:   {result.q_error_isomer:.3f} ({(1-result.q_error_isomer/result.q_error_stale)*100:+.1f}%)")
    print(f"  OASIS:    {result.q_error_oasis:.3f} ({(1-result.q_error_oasis/result.q_error_stale)*100:+.1f}%)")
    
    return result


def main():
    """Run experiments across different initial distributions."""
    distributions = [
        "gaussian_mixture",  # Baseline (same as training)
        "uniform",
        "skewed_powerlaw",
        "bimodal",
        "triangular",
        "exponential",
    ]
    
    results = []
    for dist in distributions:
        result = run_experiment(dist, n_cases=200, q_mods=10, seed=42)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: OASIS Generalization to Different Initial Distributions")
    print("="*80)
    print(f"{'Distribution':<20} {'Stale':>10} {'OASIS':>10} {'Improvement':>12}")
    print("-"*80)
    
    for r in results:
        improvement = (1 - r.q_error_oasis / r.q_error_stale) * 100
        print(f"{r.init_dist:<20} {r.q_error_stale:>10.3f} {r.q_error_oasis:>10.3f} {improvement:>11.1f}%")
    
    print("="*80)
    
    # Save results
    output = {
        "experiment": "distribution_generalization",
        "results": [
            {
                "distribution": r.init_dist,
                "q_error_stale": r.q_error_stale,
                "q_error_oasis": r.q_error_oasis,
                "improvement_pct": (1 - r.q_error_oasis / r.q_error_stale) * 100
            }
            for r in results
        ]
    }
    
    output_path = Path("distribution_generalization_results.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
