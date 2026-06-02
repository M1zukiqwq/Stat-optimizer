"""Shared v3 inference + drop-in replacements for the numpy prior functions.

Lets the existing numpy experiment scripts use the torch v3 model by
monkeypatching `oasis_boundaries` / `correct_marginal_with_oasis` with versions
that run the trained set-transformer. Handles [min,max] normalization so it works
even when stale boundaries are not already on [0,1].
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from model import OasisTorchV3, boundaries_from_logits

PREDICATE_ORDER = ["<", "<=", ">", ">=", "BETWEEN", "="]
PIDX = {p: i for i, p in enumerate(PREDICATE_ORDER)}
N_PRED = len(PREDICATE_ORDER)
OBS_FEAT_DIM = N_PRED + 6

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_CFG = None


def load(ckpt_path):
    global _MODEL, _CFG
    ck = torch.load(ckpt_path, map_location=_DEVICE)
    _CFG = ck["config"]
    m = OasisTorchV3(num_buckets=_CFG["num_buckets"], max_obs=_CFG["max_obs"],
                     d_model=_CFG["d_model"], n_heads=_CFG["n_heads"],
                     n_layers=_CFG["n_layers"], residual_prior=_CFG["residual_prior"]).to(_DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval()
    _MODEL = m
    return m


def _mono01_boundaries(inner):
    out = [min(max(float(v), 0.0), 1.0) for v in inner]
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return [0.0] + out + [1.0]


@torch.no_grad()
def predict_boundaries(observations, stale_boundaries, max_obs):
    """Return v3 prior boundaries on the SAME scale as stale_boundaries."""
    K = _CFG["max_obs"]
    lo_v, hi_v = float(stale_boundaries[0]), float(stale_boundaries[-1])
    rng = max(hi_v - lo_v, 1e-12)
    norm_stale = [min(max((float(b) - lo_v) / rng, 0.0), 1.0) for b in stale_boundaries]
    feat = np.zeros((1, K, OBS_FEAT_DIM), np.float32)
    mask = np.zeros((1, K), np.float32)
    for i, o in enumerate(observations[:K]):
        pt = o.get("predicate_type", "<="); v = float(o.get("value", 0.5))
        vu = o.get("value_upper")
        vn = min(max((v - lo_v) / rng, 0.0), 1.0)
        vun = min(max((float(vu) - lo_v) / rng, 0.0), 1.0) if vu is not None else vn
        has_up = 1.0 if vu is not None else 0.0
        est = float(o.get("estimated_sel", o.get("estimated_selectivity", 0.0)))
        act = float(o.get("actual_sel", o.get("actual_selectivity", 0.0)))
        feat[0, i, PIDX.get(pt, 1)] = 1.0
        feat[0, i, N_PRED:] = [vn, vun, est, act, est - act, has_up]
        mask[0, i] = 1.0
    sb = torch.tensor([norm_stale], dtype=torch.float32, device=_DEVICE)
    logits = _MODEL(torch.tensor(feat, device=_DEVICE), torch.tensor(mask, device=_DEVICE), sb)
    nb = boundaries_from_logits(logits)[0].cpu().numpy().tolist()
    # denormalize back to original scale
    return [lo_v + b * rng for b in nb]


# ---- drop-in replacements (same signatures as the numpy originals) ----

def _obs_from_sample(sample):
    out = []
    for o in sample.observations:
        out.append({
            "predicate_type": getattr(o, "predicate_type", "<="),
            "value": getattr(o, "value", 0.5),
            "value_upper": getattr(o, "value_upper", None),
            "estimated_sel": getattr(o, "estimated_selectivity", 0.0),
            "actual_sel": getattr(o, "actual_selectivity", 0.0),
        })
    return out


def make_oasis_boundaries_v3(boundaries_from_quantiles, observations_to_dicts=None):
    def oasis_boundaries_v3(sample, model, max_observations):
        stale = boundaries_from_quantiles(sample.prior.quantile_values)
        obs = _obs_from_sample(sample)
        return predict_boundaries(obs, stale, max_observations)
    return oasis_boundaries_v3


def correct_marginal_with_oasis_v3(stale_boundaries, observations, model, num_buckets, max_obs):
    return predict_boundaries(list(observations), list(stale_boundaries), max_obs)


# ---- universal model adapter: parses the flat tensorizer feature and runs v3 ----
# tensorizer layout (use_time_decay=False): prior_norm[B-1] + meta[3] + K*obs_dim + mask[K]
# obs (obs_dim=12): 6-hot[TENS_ORDER] + value_norm, value_upper_norm, est, act, has_upper, span_norm
_TENS_ORDER = ["<", "<=", ">", ">=", "=", "BETWEEN"]


@torch.no_grad()
def _predict_from_tensor_feature(feat, K=16):
    Bm1 = 9  # B=10 in all paper experiments
    od = (len(feat) - Bm1 - 3 - K) // K
    prior_norm = list(feat[:Bm1])
    off = Bm1 + 3
    flat = feat[off:off + K * od]
    mask = feat[off + K * od: off + K * od + K]
    stale = [0.0] + prior_norm + [1.0]
    obs = []
    for i in range(K):
        if mask[i] < 0.5:
            continue
        row = flat[i * od:(i + 1) * od]
        onehot = row[:6]
        pt = _TENS_ORDER[int(max(range(6), key=lambda j: onehot[j]))]
        value_norm, vup_norm, est, act, has_upper = row[6], row[7], row[8], row[9], row[10]
        obs.append({"predicate_type": pt, "value": value_norm,
                    "value_upper": (vup_norm if has_upper > 0.5 else None),
                    "estimated_sel": est, "actual_sel": act})
    # prior is already normalized to [0,1] in the tensor feature
    return predict_boundaries(obs, stale, K)


class V3Model:
    """Guard adapter. The faithful v3 path rebuilds inputs from the RAW sample
    (predict_boundaries); the tensorizer reconstructs est_sel differently, which
    diverges from how v3 was trained. So model.predict must NEVER be reached: if
    it is, an entry point was missed and we fail loudly instead of producing
    subtly wrong numbers."""
    def predict(self, feature_list):
        raise RuntimeError(
            "v3 reached an UNPATCHED prior-injection site (model.predict). The "
            "tensor-feature path is not faithful to v3 training; add a per-function "
            "patch (rebuild obs from the raw sample) for this experiment.")


def patch_model_loader():
    """Make MlpHistogramModelV2.load return a guard so unpatched sites fail loudly."""
    import mlp_histogram_model_v2 as mm
    mm.MlpHistogramModelV2.load = staticmethod(lambda *a, **k: V3Model())
    return V3Model()
