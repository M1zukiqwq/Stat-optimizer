"""Run an existing numpy experiment with the v3 torch prior monkeypatched in.

Usage: V3_CKPT=ckpt.pt python run_v3.py <proj|comp|fj|ood> [experiment args...]
Patches both the source-module globals and the experiment-module bindings so
oasis_boundaries / correct_marginal_with_oasis use the trained v3 model.
"""
import importlib
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in [str(_HERE), str(_REPO / "cdf_kll_ml_pipeline"), str(_REPO / "experiments")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import v3_infer

MAP = {
    "proj": "projection_locality_experiment",
    "comp": "composition_family_experiment",
    "fj": "factorjoin_oasis_experiment",
    "ood": "ood_drift_realism_experiment",
    "stage1swap": "stage1_estimator_swap_experiment",
    "trace": "trace_grounded_drift_experiment",
    "public": "public_trace_workload_experiment",
    "odp": "optimizer_decision_proxy_experiment",
    "budget": "feedback_budget_sensitivity_experiment",
    "noise": "feedback_noise_robustness_experiment",
    "pg": "postgres_planner_stats_injection_experiment",
    "tpch": "postgres_runtime_tpch_experiment",
    "routerdiag": "router_diagnostics",
    "suite": "run_synthetic_paper_suite",
    "smoke": "oasis_accuracy_smoke",
}


def main():
    ckpt = os.environ["V3_CKPT"]
    v3_infer.load(ckpt)
    print(f"[run_v3] loaded {ckpt}")

    # Universal: make MlpHistogramModelV2.load return the v3 adapter everywhere
    # (covers any prior-injection site that goes through model.predict).
    v3_infer.patch_model_loader()

    import optimizer_decision_proxy_experiment as odp
    ob_v3 = v3_infer.make_oasis_boundaries_v3(odp.boundaries_from_quantiles, odp.observations_to_dicts)
    odp.oasis_boundaries = ob_v3  # affects build_method_boundaries (OOD)

    import copula_oasis_experiment as cox
    cox.correct_marginal_with_oasis = v3_infer.correct_marginal_with_oasis_v3

    # stage1_estimator_swap uses oasis_accuracy_smoke.oasis_boundaries
    try:
        import oasis_accuracy_smoke as smoke
        smoke.oasis_boundaries = v3_infer.make_oasis_boundaries_v3(
            smoke.boundaries_from_quantiles, smoke.observations_to_dicts)
    except Exception as e:
        print(f"[run_v3] smoke patch skipped: {e}")

    # run_synthetic_paper_suite uses its own predict_oasis_boundaries
    try:
        import run_synthetic_paper_suite as S
        S.predict_oasis_boundaries = v3_infer.make_oasis_boundaries_v3(odp.boundaries_from_quantiles)
    except Exception as e:
        print(f"[run_v3] suite patch skipped: {e}")

    name = sys.argv[1]
    rest = sys.argv[2:]
    mod = importlib.import_module(MAP[name])
    # rebind names imported into the experiment module namespace
    if hasattr(mod, "oasis_boundaries"):
        mod.oasis_boundaries = ob_v3
    if hasattr(mod, "correct_marginal_with_oasis"):
        mod.correct_marginal_with_oasis = v3_infer.correct_marginal_with_oasis_v3
    print(f"[run_v3] running {MAP[name]} with v3 prior")
    sys.argv = [MAP[name]] + rest
    if hasattr(mod, "main"):
        mod.main()
    elif hasattr(mod, "run") and hasattr(mod, "parse_args"):
        mod.run(mod.parse_args())
    else:
        raise RuntimeError(f"{MAP[name]}: no main()/run(parse_args()) entry point")


if __name__ == "__main__":
    main()
