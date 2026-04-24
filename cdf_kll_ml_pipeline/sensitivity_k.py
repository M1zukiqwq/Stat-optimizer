"""
Sensitivity Analysis on Observation Window Size (K)
"""
import argparse
import subprocess
import sys
from pathlib import Path
from ablation_experiment import run_ablation

def save_sensitivity_csv(results, output_path: Path):
    lines = ["K,QErr_Prior,QErr_Teacher,QErr_MLP"]
    for k, prior, teacher, mlp in results:
        lines.append(f"{k},{prior:.6f},{teacher:.6f},{mlp:.6f}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已保存到 {output_path}")

def plot_sensitivity(results, output_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ks = [r[0] for r in results]
    priors = [r[1] for r in results]
    teachers = [r[2] for r in results]
    mlps = [r[3] for r in results]

    plt.figure(figsize=(8, 5))
    plt.title("Sensitivity Analysis: K (Window Size) vs Q-Error (at q=10)")
    plt.plot(ks, priors, "o-", color="#e74c3c", label="Prior", linewidth=2.0)
    plt.plot(ks, teachers, "s--", color="#f39c12", label="Teacher", linewidth=2.0)
    plt.plot(ks, mlps, "^-", color="#8e44ad", label="MLP", linewidth=2.2)
    
    plt.xlabel("Window Size (K)", fontsize=12)
    plt.ylabel("Q-Error (↓ better)", fontsize=12)
    plt.xticks(ks)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    print(f"图表已保存到 {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("sensitivity_K_work"))
    args = parser.parse_args()
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # K values to test
    k_values = [4, 8, 16, 32]
    # We evaluate sensitivity at q=10
    test_q = 10
    
    results = [] # (K, QErr_Prior, QErr_Teacher, QErr_MLP)
    
    print(f"========== Sensitivity Analysis for K ==========")
    for k in k_values:
        print(f"\n--- Testing K = {k} ---")
        
        # We reuse run_ablation but overriding max_observations
        summaries = run_ablation(
            q_values=[test_q],
            k_train=1000,
            k_test=64,
            num_buckets=10,
            max_observations=k,
            work_dir=work_dir / f"k_{k}",
            seed=42,
            train_q_values=[test_q, 20],  # Same pre-train condition
            mlp_epochs=200
        )
        
        # summary has 1 element since q_values=[10]
        s = summaries[0]
        results.append((k, s.qerr_prior, s.qerr_teacher, s.qerr_mlp))
        
    print("\n" + "=" * 50)
    print(f"{'K':>4} {'QErr_Prior':>12} {'QErr_Teacher':>14} {'QErr_MLP':>12}")
    print("-" * 50)
    for k, p, t, m in results:
        print(f"{k:>4} {p:>12.4f} {t:>14.4f} {m:>12.4f}")
        
    save_sensitivity_csv(results, work_dir / "sensitivity_k_results.csv")
    plot_sensitivity(results, work_dir / "sensitivity_k_plot.png")

if __name__ == "__main__":
    main()
