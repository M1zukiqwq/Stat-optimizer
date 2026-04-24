"""
mlp_histogram_model.py
======================
Attention-pooled MLP for KLL histogram correction.

Architecture (CPU-efficient, pure-numpy):

    input feature vector shape:
        [prior_norm (B-1) | meta (3) | obs_flat (K×D_obs) | mask (K)]

    1. Split observation block into K slots of D_obs dims each.
    2. Per-slot attention score = w_attn · slot + b_attn  (scalar, learned)
       -- softmax over valid slots (mask=1), zero out padded slots
    3. Weighted sum → pooled observation vector (D_obs dims)
    4. Concat [prior_norm | meta | pooled_obs] → context vector (B-1+3+D_obs)
    5. MLP: context → hidden1 (ReLU) → hidden2 (ReLU) → output (B-1 dims)
       -- B-1 normalized quantile values

All weights stored as plain Python lists; numpy used only for matrix ops
during fit / predict so the runtime stays fully portable.

Save/load format: JSON (same convention as RidgeMultiOutputRegressor).
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Numpy import (required for efficient training)
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise ImportError("numpy is required for MlpHistogramModel. Install via: pip install numpy")


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _relu(x: List[float]) -> List[float]:
    return [max(0.0, v) for v in x]


def _softmax(scores: List[float], mask: List[float]) -> List[float]:
    """Masked softmax: padded positions (mask=0) get weight 0."""
    MAX = -1e18
    for i, (s, m) in enumerate(zip(scores, mask)):
        if m > 0.5 and s > MAX:
            MAX = s
    exps = [math.exp(s - MAX) * m for s, m in zip(scores, mask)]
    total = sum(exps) + 1e-12
    return [e / total for e in exps]


def _matmul_vec(weights: List[List[float]], x: List[float]) -> List[float]:
    """y = W x  where W is (out_dim × in_dim)."""
    return [sum(w * xi for w, xi in zip(row, x)) for row in weights]


def _add(a: List[float], b: List[float]) -> List[float]:
    return [ai + bi for ai, bi in zip(a, b)]


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class MlpHistogramModel:
    """
    Attention-pooled 2-hidden-layer MLP.

    Parameters
    ----------
    obs_dim        : feature dimension per observation slot (e.g. 12)
    prior_dim      : number of prior quantile levels (e.g. 9 for B=10)
    meta_dim       : meta block size (always 3)
    max_observations : observation window size K (e.g. 16)
    hidden_dims    : sizes of hidden layers (default [128, 64])
    alpha          : L2 regularisation coefficient
    lr             : learning rate for Adam
    epochs         : training epochs
    batch_size     : mini-batch size
    seed           : random seed
    """

    def __init__(
        self,
        obs_dim: int = 12,
        prior_dim: int = 9,
        meta_dim: int = 3,
        max_observations: int = 16,
        hidden_dims: Tuple[int, ...] = (128, 64),
        alpha: float = 1e-4,
        lr: float = 3e-3,
        epochs: int = 200,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.obs_dim = obs_dim
        self.prior_dim = prior_dim
        self.meta_dim = meta_dim
        self.max_observations = max_observations
        self.hidden_dims = list(hidden_dims)
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        # Trainable weights (initialised in fit())
        self.W_attn: Optional[List[float]] = None     # (obs_dim,)
        self.b_attn: float = 0.0
        self.layers: List[Dict] = []                  # list of {W, b}
        self._fitted = False

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _feature_dims(self) -> Tuple[int, int, int]:
        """Returns (prior_start, obs_start, mask_start) for slicing."""
        prior_start = 0
        obs_start = self.prior_dim + self.meta_dim
        mask_start = obs_start + self.max_observations * self.obs_dim
        return prior_start, obs_start, mask_start

    def _split(self, x: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Split flat vector into (prior, meta, obs_slots, mask)."""
        _, obs_start, mask_start = self._feature_dims()
        prior = x[:self.prior_dim]
        meta = x[self.prior_dim:obs_start]
        obs_flat = x[obs_start:mask_start]
        mask = x[mask_start:]
        obs_slots = obs_flat.reshape(self.max_observations, self.obs_dim)
        return prior, meta, obs_slots, mask

    # -----------------------------------------------------------------------
    # Forward pass (numpy)
    # -----------------------------------------------------------------------

    def _forward_np(self, X: "np.ndarray") -> "np.ndarray":
        """Batch forward pass.  X: (N, feature_dim). Returns (N, prior_dim)."""
        N = X.shape[0]
        outputs = []

        W_attn = np.array(self.W_attn)                         # (obs_dim,)
        b_attn = self.b_attn

        for xi in X:
            prior, meta, obs_slots, mask = self._split(xi)     # obs_slots: (K, D)

            # Attention scores and weights
            scores = obs_slots @ W_attn + b_attn                # (K,)
            scores = scores - scores.max()
            exp_s = np.exp(scores) * mask
            attn = exp_s / (exp_s.sum() + 1e-12)               # (K,)

            pooled = (attn[:, None] * obs_slots).sum(axis=0)   # (D_obs,)
            context = np.concatenate([prior, meta, pooled])     # (prior+meta+D_obs,)

            h = context
            for layer in self.layers[:-1]:
                h = np.maximum(0.0, layer["W"] @ h + layer["b"])
            out_layer = self.layers[-1]
            h = out_layer["W"] @ h + out_layer["b"]
            outputs.append(h)

        return np.vstack(outputs)                               # (N, prior_dim)

    # -----------------------------------------------------------------------
    # Fit (Adam + mini-batch gradient descent via numpy)
    # -----------------------------------------------------------------------

    def fit(self, features: List[List[float]], targets: List[List[float]]) -> None:
        _require_numpy()

        X = np.array(features, dtype=np.float32)
        Y = np.array(targets, dtype=np.float32)
        N, feat_dim = X.shape
        out_dim = Y.shape[1]

        # Derive dims from first sample; sanity check obs_dim
        context_dim = self.prior_dim + self.meta_dim + self.obs_dim

        rng = np.random.RandomState(self.seed)

        # ---- Initialise weights ----
        self.W_attn = (rng.randn(self.obs_dim) * 0.01).tolist()
        self.b_attn = 0.0

        layer_dims = [context_dim] + self.hidden_dims + [out_dim]
        self.layers = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            scale = math.sqrt(2.0 / fan_in)  # He init
            self.layers.append({
                "W": rng.randn(fan_out, fan_in).astype(np.float32) * scale,
                "b": np.zeros(fan_out, dtype=np.float32),
            })

        # Adam state
        W_attn_np = np.array(self.W_attn, dtype=np.float32)
        adam_params = [W_attn_np] + [p for layer in self.layers for p in (layer["W"], layer["b"])]
        m_state = [np.zeros_like(p) for p in adam_params]
        v_state = [np.zeros_like(p) for p in adam_params]
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        def _get_params():
            return [W_attn_np] + [p for layer in self.layers for p in (layer["W"], layer["b"])]

        def _set_params(params):
            nonlocal W_attn_np
            W_attn_np = params[0]
            idx = 1
            for layer in self.layers:
                layer["W"] = params[idx]; idx += 1
                layer["b"] = params[idx]; idx += 1

        step = 0
        idx_all = list(range(N))

        for epoch in range(self.epochs):
            rng.shuffle(idx_all)
            for start in range(0, N, self.batch_size):
                batch_idx = idx_all[start:start + self.batch_size]
                Xb = X[batch_idx]
                Yb = Y[batch_idx]
                Nb = len(batch_idx)

                # ---- Forward + backward (manual) ----
                # We use a numerical gradient via a lightweight pass
                # to avoid implementing full backprop.
                # For the batch size and model size here (~200 params) this is viable.
                # Actually let's do proper backprop for efficiency.

                # --- gather intermediates ---
                priors_b, metas_b, obs_slots_b, masks_b = [], [], [], []
                attns_b, pooleds_b, contexts_b = [], [], []
                hs_b = [[] for _ in self.layers]  # pre-activation per layer
                zs_b = [[] for _ in self.layers]  # post-activation

                for xi in Xb:
                    prior, meta, obs_slots, mask = self._split(xi)
                    priors_b.append(prior)
                    metas_b.append(meta)
                    obs_slots_b.append(obs_slots)
                    masks_b.append(mask)

                    scores = obs_slots @ W_attn_np + self.b_attn
                    scores -= scores.max()
                    exp_s = np.exp(scores) * mask
                    attn = exp_s / (exp_s.sum() + 1e-12)
                    pooled = (attn[:, None] * obs_slots).sum(axis=0)
                    context = np.concatenate([prior, meta, pooled])

                    attns_b.append(attn)
                    pooleds_b.append(pooled)
                    contexts_b.append(context)

                    h = context
                    for l_idx, layer in enumerate(self.layers):
                        z = layer["W"] @ h + layer["b"]
                        hs_b[l_idx].append(h)
                        zs_b[l_idx].append(z)
                        h = np.maximum(0.0, z) if l_idx < len(self.layers) - 1 else z

                preds = np.array(zs_b[-1])   # (Nb, out_dim) - last layer no relu
                delta = (preds - Yb) * 2.0 / Nb   # MSE gradient incorporates 1/Nb
                delta = np.clip(delta, -10.0, 10.0)

                # Backprop through layers (reverse)
                grads_W = [np.zeros_like(layer["W"]) for layer in self.layers]
                grads_b = [np.zeros_like(layer["b"]) for layer in self.layers]

                d = delta.copy()
                for l_idx in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[l_idx]
                    h_in = np.array(hs_b[l_idx])   # (Nb, fan_in)

                    if l_idx < len(self.layers) - 1:
                        relu_mask = (np.array(zs_b[l_idx]) > 0).astype(np.float32)
                        d = d * relu_mask

                    grads_W[l_idx] = d.T @ h_in + self.alpha * layer["W"]
                    grads_b[l_idx] = d.sum(axis=0)

                    d = d @ layer["W"]   # propagate: after loop, d has shape (Nb, context_dim)

                # d is now (Nb, context_dim) — gradient w.r.t. context
                grad_context = np.clip(d, -10.0, 10.0)

                # Split grad_context into grad_prior, grad_meta, grad_pooled
                grad_pooled = grad_context[:, self.prior_dim + self.meta_dim:]  # (Nb, D_obs)

                # Backprop through attention pooling
                grad_W_attn = np.zeros_like(W_attn_np)
                grad_b_attn = 0.0

                for i in range(Nb):
                    attn = attns_b[i]        # (K,)
                    obs_slots = obs_slots_b[i]  # (K, D_obs)
                    mask = masks_b[i]
                    gp = grad_pooled[i]      # (D_obs,)

                    # d_pooled/d_attn: obs_slots (K, D_obs) → gp·obs_slots^T → (K,)
                    d_attn_score = (obs_slots @ gp)   # (K,)  = d_pooled/d_attn_i

                    # d_attn/d_scores: jacobian of softmax
                    # d_scores_k = attn_k * (d_attn_score_k - attn·d_attn_score)
                    dot = float(attn @ d_attn_score)
                    d_scores = attn * (d_attn_score - dot)   # (K,)

                    # d_scores/d_W_attn = obs_slots   (K×D_obs)
                    grad_W_attn += (d_scores[:, None] * obs_slots).sum(axis=0)
                    grad_b_attn += d_scores.sum()

                grad_W_attn += self.alpha * W_attn_np

                # ---- Gradient clipping ----
                step += 1
                all_grads = [grad_W_attn] + [g for gW, gb in zip(grads_W, grads_b) for g in (gW, gb)]
                global_norm = np.sqrt(sum(float(np.sum(g * g)) for g in all_grads))
                max_grad_norm = 1.0
                if global_norm > max_grad_norm:
                    clip_coef = max_grad_norm / (global_norm + 1e-6)
                    all_grads = [g * clip_coef for g in all_grads]

                # ---- Adam update ----
                params = _get_params()

                new_params = []
                for p_idx, (p, g, m, v) in enumerate(zip(params, all_grads, m_state, v_state)):
                    m_state[p_idx] = beta1 * m + (1 - beta1) * g
                    v_state[p_idx] = beta2 * v + (1 - beta2) * g * g
                    m_hat = m_state[p_idx] / (1 - beta1 ** step)
                    v_hat = v_state[p_idx] / (1 - beta2 ** step)
                    new_params.append(p - self.lr * m_hat / (np.sqrt(v_hat) + eps_adam))

                _set_params(new_params)

        # Persist back to Python lists for JSON serialisation
        self.W_attn = W_attn_np.tolist()
        self.b_attn = float(self.b_attn)
        for layer in self.layers:
            layer["W"] = layer["W"].tolist()
            layer["b"] = layer["b"].tolist()

        self._fitted = True

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict(self, features: List[List[float]]) -> List[List[float]]:
        _require_numpy()
        if not self._fitted:
            raise ValueError("model is not fitted")

        X = np.array(features, dtype=np.float32)

        # Restore numpy arrays from lists for forward pass
        self.W_attn = list(self.W_attn)  # ensure it's a list
        W_attn_np = np.array(self.W_attn, dtype=np.float32)
        orig_W_attn = self.W_attn

        # Temporarily patch self.W_attn for _forward_np
        self.W_attn = orig_W_attn
        for layer in self.layers:
            layer["W"] = np.array(layer["W"], dtype=np.float32)
            layer["b"] = np.array(layer["b"], dtype=np.float32)

        result = self._forward_np(X).tolist()

        # Convert back to lists so JSON serialisation still works
        for layer in self.layers:
            layer["W"] = layer["W"].tolist()
            layer["b"] = layer["b"].tolist()

        return result

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, path: str, metadata: Optional[Dict[str, object]] = None) -> None:
        if not self._fitted:
            raise ValueError("model is not fitted")

        payload = {
            "model_type": "MlpHistogramModel",
            "obs_dim": self.obs_dim,
            "prior_dim": self.prior_dim,
            "meta_dim": self.meta_dim,
            "max_observations": self.max_observations,
            "hidden_dims": self.hidden_dims,
            "alpha": self.alpha,
            "W_attn": self.W_attn,
            "b_attn": self.b_attn,
            "layers": [{"W": layer["W"], "b": layer["b"]} for layer in self.layers],
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "MlpHistogramModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            obs_dim=payload["obs_dim"],
            prior_dim=payload["prior_dim"],
            meta_dim=payload.get("meta_dim", 3),
            max_observations=payload["max_observations"],
            hidden_dims=tuple(payload["hidden_dims"]),
            alpha=payload.get("alpha", 1e-4),
        )
        model.W_attn = payload["W_attn"]
        model.b_attn = payload["b_attn"]
        model.layers = payload["layers"]
        model._fitted = True
        return model
