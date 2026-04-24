"""
attention_heatmap.py
====================
Generate attention heatmap visualization for OASIS model (ablation study model).

This script:
1. Loads a trained OASIS model (MlpHistogramModelV2)
2. Runs inference on sample data to extract attention weights
3. Creates heatmaps showing how each attention head focuses on different observations
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
from typing import List, Tuple, Optional

# Import the model
import sys
sys.path.insert(0, '/Users/qichutian/postgres/postgresql-14.17/postgres-cdf-simulation/cdf_kll_ml_pipeline')
from mlp_histogram_model_v2 import MlpHistogramModelV2


def load_model(model_path: str) -> MlpHistogramModelV2:
    """Load trained OASIS model."""
    return MlpHistogramModelV2.load(model_path)


def extract_attention_weights(model: MlpHistogramModelV2, features: np.ndarray) -> np.ndarray:
    """
    Extract attention weights from the model for given input features.
    
    Returns:
        attention_weights: shape (num_samples, num_heads, max_observations)
    """
    num_samples = features.shape[0]
    num_heads = model.num_heads
    max_obs = model.max_observations
    
    attention_weights = np.zeros((num_samples, num_heads, max_obs))
    
    W_attn = np.array(model.W_attn_heads)  # (num_heads, obs_dim)
    b_attn = np.array(model.b_attn_heads)  # (num_heads,)
    
    _, obs_start, mask_start = model._feature_dims()
    
    for i, xi in enumerate(features):
        # Split features
        obs_flat = xi[obs_start:mask_start]
        obs_slots = obs_flat.reshape(max_obs, model.obs_dim)  # (K, D_obs)
        mask = xi[mask_start:]  # (K,)
        
        # Compute attention for each head
        for h in range(num_heads):
            scores = obs_slots @ W_attn[h] + b_attn[h]  # (K,)
            scores = scores - scores.max()  # Numerical stability
            exp_s = np.exp(scores) * mask
            attn = exp_s / (exp_s.sum() + 1e-12)  # (K,)
            attention_weights[i, h, :] = attn
    
    return attention_weights


def generate_sample_data(num_samples: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sample data similar to the ablation study.
    Returns features and targets.
    """
    rng = np.random.RandomState(seed)
    
    # Feature dimensions matching the model
    prior_dim = 9
    meta_dim = 3
    max_obs = 16
    obs_dim = 12
    feature_dim = prior_dim + meta_dim + max_obs * obs_dim + max_obs  # = 220
    
    features = []
    targets = []
    
    for _ in range(num_samples):
        # Prior: normalized quantile values (0-1)
        prior = np.sort(rng.uniform(0, 1, prior_dim))
        
        # Meta: null_frac, obs_count_ratio, bucket_count
        meta = np.array([rng.uniform(0, 0.1), rng.uniform(0.5, 1.0), 10.0])
        
        # Observations: mix of valid and padded entries
        num_valid_obs = rng.randint(4, max_obs + 1)
        obs_slots = []
        for j in range(max_obs):
            if j < num_valid_obs:
                # Valid observation: (predicate_onehot_6, v, v_upper, s_hat, s_star, has_upper, span)
                pred_type = rng.randint(0, 6)
                pred_onehot = np.eye(6)[pred_type]
                v = rng.uniform(0, 1)
                v_upper = rng.uniform(v, 1) if pred_type == 5 else 0  # Between predicate
                s_hat = rng.uniform(0.1, 0.9)
                s_star = rng.uniform(0.1, 0.9)
                has_upper = 1.0 if pred_type == 5 else 0.0
                span = v_upper - v if pred_type == 5 else 0.0
                obs = np.concatenate([pred_onehot, [v, v_upper, s_hat, s_star, has_upper, span]])
            else:
                # Padded observation
                obs = np.zeros(obs_dim)
            obs_slots.extend(obs)
        
        # Mask: 1 for valid, 0 for padded
        mask = np.array([1.0 if j < num_valid_obs else 0.0 for j in range(max_obs)])
        
        # Combine all features
        feature = np.concatenate([prior, meta, obs_slots, mask])
        features.append(feature)
        
        # Target: corrected quantiles (prior + small delta)
        delta = rng.normal(0, 0.05, prior_dim)
        target = np.clip(prior + delta, 0, 1)
        targets.append(target)
    
    return np.array(features), np.array(targets)


def plot_attention_heatmap(attention_weights: np.ndarray, output_path: str, 
                           title: str = "OASIS Multi-Head Attention Pattern"):
    """
    Create attention heatmap visualization.
    
    Args:
        attention_weights: shape (num_samples, num_heads, max_observations)
    """
    num_samples, num_heads, max_obs = attention_weights.shape
    
    # Average across samples for overall pattern
    avg_attention = attention_weights.mean(axis=0)  # (num_heads, max_obs)
    
    # Also select a few individual samples for detailed view
    sample_indices = [0, num_samples // 4, num_samples // 2, 3 * num_samples // 4]
    
    fig = plt.figure(figsize=(16, 10))
    
    # Main heatmap: Average attention across all samples
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(avg_attention, cmap='YlOrRd', aspect='auto', vmin=0)
    ax1.set_xlabel('Observation Index (Time Step)', fontsize=11)
    ax1.set_ylabel('Attention Head', fontsize=11)
    ax1.set_title('Average Attention Weight Across All Samples', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(num_heads))
    ax1.set_yticklabels([f'Head {i+1}' for i in range(num_heads)])
    ax1.set_xticks(range(max_obs))
    plt.colorbar(im1, ax=ax1, label='Attention Weight')
    
    # Add text annotations
    for i in range(num_heads):
        for j in range(max_obs):
            text = ax1.text(j, i, f'{avg_attention[i, j]:.3f}',
                           ha="center", va="center", color="black" if avg_attention[i, j] < 0.5 else "white",
                           fontsize=7)
    
    # Individual sample heatmaps
    for idx, sample_idx in enumerate(sample_indices[:3]):
        ax = plt.subplot(2, 2, idx + 2)
        sample_attn = attention_weights[sample_idx]  # (num_heads, max_obs)
        im = ax.imshow(sample_attn, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xlabel('Observation Index (Time Step)', fontsize=10)
        ax.set_ylabel('Attention Head', fontsize=10)
        ax.set_title(f'Sample {sample_idx + 1} Attention Pattern', fontsize=11)
        ax.set_yticks(range(num_heads))
        ax.set_yticklabels([f'H{i+1}' for i in range(num_heads)])
        ax.set_xticks(range(0, max_obs, 2))
        plt.colorbar(im, ax=ax, label='Weight')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved attention heatmap to: {output_path}")


def plot_attention_distribution(attention_weights: np.ndarray, output_path: str):
    """
    Plot distribution of attention weights across heads and observations.
    """
    num_samples, num_heads, max_obs = attention_weights.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution per head (averaged across observations)
    ax = axes[0, 0]
    head_means = attention_weights.mean(axis=(0, 2))  # Average across samples and obs
    head_stds = attention_weights.std(axis=(0, 2))
    x = np.arange(num_heads)
    bars = ax.bar(x, head_means, yerr=head_stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_xlabel('Attention Head', fontsize=11)
    ax.set_ylabel('Average Attention Weight', fontsize=11)
    ax.set_title('Average Attention per Head', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Head {i+1}' for i in range(num_heads)])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, head_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Attention entropy per head (how focused is each head)
    ax = axes[0, 1]
    # Entropy: -sum(p * log(p)) - lower entropy means more focused attention
    epsilon = 1e-12
    entropy = -np.sum(attention_weights * np.log(attention_weights + epsilon), axis=2)  # (samples, heads)
    avg_entropy = entropy.mean(axis=0)
    max_entropy = np.log(max_obs)  # Maximum possible entropy (uniform distribution)
    normalized_entropy = avg_entropy / max_entropy  # 0 = perfectly focused, 1 = uniform
    
    bars = ax.bar(range(num_heads), normalized_entropy, color=['#d62728', '#9467bd', '#8c564b'])
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Uniform (max entropy)')
    ax.set_xlabel('Attention Head', fontsize=11)
    ax.set_ylabel('Normalized Entropy', fontsize=11)
    ax.set_title('Attention Focus per Head\n(lower = more focused)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f'Head {i+1}' for i in range(num_heads)])
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, ent) in enumerate(zip(bars, normalized_entropy)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ent:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Temporal attention pattern (how attention varies across observation index)
    ax = axes[1, 0]
    time_means = attention_weights.mean(axis=(0, 1))  # Average across samples and heads
    time_stds = attention_weights.std(axis=(0, 1))
    x = np.arange(max_obs)
    ax.fill_between(x, time_means - time_stds, time_means + time_stds, alpha=0.3, color='blue')
    ax.plot(x, time_means, marker='o', linewidth=2, markersize=4, color='blue')
    ax.set_xlabel('Observation Index (Time Step)', fontsize=11)
    ax.set_ylabel('Average Attention Weight', fontsize=11)
    ax.set_title('Temporal Attention Pattern\n(averaged across heads)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(x)
    
    # 4. Head specialization heatmap (correlation between heads)
    ax = axes[1, 1]
    # Reshape to (samples * max_obs, num_heads) and compute correlation
    flat_attn = attention_weights.transpose(1, 0, 2).reshape(num_heads, -1)  # (heads, samples*obs)
    corr_matrix = np.corrcoef(flat_attn)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xlabel('Attention Head', fontsize=11)
    ax.set_ylabel('Attention Head', fontsize=11)
    ax.set_title('Head Correlation Matrix', fontsize=12, fontweight='bold')
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_heads))
    ax.set_xticklabels([f'H{i+1}' for i in range(num_heads)])
    ax.set_yticklabels([f'H{i+1}' for i in range(num_heads)])
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add correlation values as text
    for i in range(num_heads):
        for j in range(num_heads):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                          fontsize=10)
    
    plt.suptitle('OASIS Attention Analysis (Ablation Study Model)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved attention distribution plot to: {output_path}")


def plot_combined_visualization(attention_weights: np.ndarray, output_path: str):
    """
    Create a comprehensive combined visualization.
    """
    num_samples, num_heads, max_obs = attention_weights.shape
    avg_attention = attention_weights.mean(axis=0)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main attention heatmap (large, top-left)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im1 = ax1.imshow(avg_attention, cmap='YlOrRd', aspect='auto', vmin=0)
    ax1.set_xlabel('Observation Index (Time Step)', fontsize=12)
    ax1.set_ylabel('Attention Head', fontsize=12)
    ax1.set_title('Average Attention Weights Across All Test Samples', fontsize=13, fontweight='bold')
    ax1.set_yticks(range(num_heads))
    ax1.set_yticklabels([f'Head {i+1}' for i in range(num_heads)])
    ax1.set_xticks(range(max_obs))
    cbar1 = plt.colorbar(im1, ax=ax1, label='Attention Weight')
    
    # Add annotations
    for i in range(num_heads):
        for j in range(max_obs):
            text = ax1.text(j, i, f'{avg_attention[i, j]:.3f}',
                           ha="center", va="center", 
                           color="black" if avg_attention[i, j] < 0.5 else "white",
                           fontsize=8)
    
    # 2. Head statistics (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    head_stats = []
    for h in range(num_heads):
        head_data = attention_weights[:, h, :].flatten()
        head_stats.append({
            'mean': head_data.mean(),
            'std': head_data.std(),
            'max': head_data.max(),
            'min': head_data.min()
        })
    
    metrics = ['Mean', 'Std', 'Max', 'Min']
    x_pos = np.arange(num_heads)
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [head_stats[h][metric.lower()] for h in range(num_heads)]
        ax2.bar(x_pos + i * width, values, width, label=metric)
    
    ax2.set_xlabel('Attention Head', fontsize=11)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('Head Statistics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels([f'H{i+1}' for i in range(num_heads)])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Temporal pattern (middle-right)
    ax3 = fig.add_subplot(gs[1, 2])
    for h in range(num_heads):
        time_pattern = attention_weights[:, h, :].mean(axis=0)
        ax3.plot(range(max_obs), time_pattern, marker='o', label=f'Head {h+1}', linewidth=2)
    ax3.set_xlabel('Observation Index', fontsize=11)
    ax3.set_ylabel('Avg Attention Weight', fontsize=11)
    ax3.set_title('Temporal Attention Pattern', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(range(0, max_obs, 2))
    
    # 4. Sample-specific patterns (bottom row)
    sample_indices = [0, num_samples // 3, 2 * num_samples // 3]
    titles = ['Sample 1\n(Low Drift)', f'Sample {sample_indices[1]+1}\n(Medium Drift)', 
              f'Sample {sample_indices[2]+1}\n(High Drift)']
    
    for idx, (sample_idx, title) in enumerate(zip(sample_indices, titles)):
        ax = fig.add_subplot(gs[2, idx])
        sample_attn = attention_weights[sample_idx]
        im = ax.imshow(sample_attn, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xlabel('Observation Index', fontsize=10)
        ax.set_ylabel('Head', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_yticks(range(num_heads))
        ax.set_yticklabels([f'H{i+1}' for i in range(num_heads)])
        ax.set_xticks(range(0, max_obs, 2))
        plt.colorbar(im, ax=ax, label='Weight')
    
    plt.suptitle('OASIS Multi-Head Attention Visualization\n(Ablation Study Model with 3 Heads × 16 Observations)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined visualization to: {output_path}")


def main():
    """Main function to generate attention heatmaps."""
    # Paths
    model_path = "/Users/qichutian/postgres/postgresql-14.17/postgres-cdf-simulation/ablation_study/work_final/models/oasis.json"
    output_dir = Path("/Users/qichutian/postgres/postgresql-14.17/postgres-cdf-simulation/attention_viz")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("OASIS Attention Heatmap Visualization")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path)
    print(f"Model loaded successfully!")
    print(f"  - Number of attention heads: {model.num_heads}")
    print(f"  - Max observations (K): {model.max_observations}")
    print(f"  - Observation dimension: {model.obs_dim}")
    print(f"  - Hidden dimensions: {model.hidden_dims}")
    
    # Generate sample data
    num_samples = 100
    print(f"\nGenerating {num_samples} synthetic test samples...")
    features, targets = generate_sample_data(num_samples=num_samples, seed=42)
    print(f"Generated features shape: {features.shape}")
    
    # Extract attention weights
    print("\nExtracting attention weights...")
    attention_weights = extract_attention_weights(model, features)
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  - Samples: {attention_weights.shape[0]}")
    print(f"  - Heads: {attention_weights.shape[1]}")
    print(f"  - Observations: {attention_weights.shape[2]}")
    
    # Statistics
    print(f"\nAttention Weight Statistics:")
    print(f"  - Mean: {attention_weights.mean():.6f}")
    print(f"  - Std: {attention_weights.std():.6f}")
    print(f"  - Min: {attention_weights.min():.6f}")
    print(f"  - Max: {attention_weights.max():.6f}")
    
    for h in range(model.num_heads):
        head_weights = attention_weights[:, h, :]
        print(f"  - Head {h+1} avg weight: {head_weights.mean():.6f} ± {head_weights.std():.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Basic heatmap
    plot_attention_heatmap(
        attention_weights, 
        str(output_dir / "attention_heatmap_basic.png"),
        title="OASIS Multi-Head Attention Pattern (Ablation Study)"
    )
    
    # 2. Distribution analysis
    plot_attention_distribution(
        attention_weights,
        str(output_dir / "attention_distribution.png")
    )
    
    # 3. Combined comprehensive visualization
    plot_combined_visualization(
        attention_weights,
        str(output_dir / "attention_heatmap_combined.png")
    )
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # List output files
    for f in output_dir.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
