"""
mlp_histogram_model_v2.py
==========================
Enhanced OASIS model with multi-head attention, residual connections, and deeper architecture.

Key improvements over v1:
1. Multi-head attention (3 heads) for diverse observation patterns
2. Residual prediction (predict delta instead of absolute values)
3. Prior encoder (dedicated MLP for prior distribution)
4. Deeper MLP with skip connections (128→128→64→64→9)

Architecture remains CPU-efficient with pure NumPy.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise ImportError("numpy is required. Install via: pip install numpy")


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _relu(x: List[float]) -> List[float]:
    return [max(0.0, v) for v in x]


def _layer_norm(x: "np.ndarray", eps: float = 1e-5) -> "np.ndarray":
    """Simple layer normalization."""
    mean = x.mean()
    std = x.std() + eps
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Enhanced Model with Multi-Head Attention
# ---------------------------------------------------------------------------

class MlpHistogramModelV2:
    """
    Enhanced OASIS model with:
    - Multi-head attention (num_heads=3)
    - Residual prediction (predict delta from prior)
    - Prior encoder (dedicated processing)
    - Deeper MLP with skip connections

    Parameters
    ----------
    obs_dim          : feature dimension per observation slot (e.g. 12)
    prior_dim        : number of prior quantile levels (e.g. 9 for B=10)
    meta_dim         : meta block size (always 3)
    max_observations : observation window size K (e.g. 16)
    num_heads        : number of attention heads (default 3)
    hidden_dims      : sizes of hidden layers (default [128, 128, 64, 64])
    prior_encoder_dim: prior encoder hidden size (default 32)
    alpha            : L2 regularisation coefficient
    lr               : learning rate for Adam
    epochs           : training epochs
    batch_size       : mini-batch size
    seed             : random seed
    """

    def __init__(
        self,
        obs_dim: int = 12,
        prior_dim: int = 9,
        meta_dim: int = 3,
        max_observations: int = 16,
        num_heads: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128, 64, 64),
        prior_encoder_dim: int = 32,
        alpha: float = 1e-4,
        lr: float = 3e-4,
        epochs: int = 150,
        batch_size: int = 32,
        seed: int = 42,
        activation_clip: float = 10.0,
        attention_score_clip: float = 20.0,
        parameter_clip: float = 2.0,
    ) -> None:
        self.obs_dim = obs_dim
        self.prior_dim = prior_dim
        self.meta_dim = meta_dim
        self.max_observations = max_observations
        self.num_heads = num_heads
        self.hidden_dims = list(hidden_dims)
        self.prior_encoder_dim = prior_encoder_dim
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.activation_clip = activation_clip
        self.attention_score_clip = attention_score_clip
        self.parameter_clip = parameter_clip

        # Trainable weights (initialised in fit())
        self.W_attn_heads: Optional[List[List[float]]] = None  # (num_heads, obs_dim)
        self.b_attn_heads: Optional[List[float]] = None        # (num_heads,)
        self.prior_encoder: List[Dict] = []                    # prior encoder layers
        self.layers: List[Dict] = []                           # main MLP layers
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
        """Batch forward pass. X: (N, feature_dim). Returns (N, prior_dim)."""
        N = X.shape[0]
        outputs = []

        W_attn_heads = np.array(self.W_attn_heads, dtype=np.float64)  # (num_heads, obs_dim)
        b_attn_heads = np.array(self.b_attn_heads, dtype=np.float64)  # (num_heads,)

        for xi in X:
            prior, meta, obs_slots, mask = self._split(xi)  # obs_slots: (K, D)

            # ---- Multi-head attention ----
            pooled_heads = []
            for h in range(self.num_heads):
                scores = obs_slots @ W_attn_heads[h] + b_attn_heads[h]  # (K,)
                scores = np.nan_to_num(scores, nan=0.0, posinf=self.attention_score_clip, neginf=-self.attention_score_clip)
                scores = np.clip(scores - scores.max(), -self.attention_score_clip, self.attention_score_clip)
                exp_s = np.exp(scores) * mask
                attn = exp_s / (exp_s.sum() + 1e-12)  # (K,)
                pooled = (attn[:, None] * obs_slots).sum(axis=0)  # (D_obs,)
                pooled_heads.append(pooled)

            # Concatenate all heads
            pooled_multi = np.concatenate(pooled_heads)  # (num_heads * D_obs,)

            # ---- Prior encoder ----
            prior_encoded = prior
            for layer in self.prior_encoder:
                z = layer["W"] @ prior_encoded + layer["b"]
                z = np.nan_to_num(z, nan=0.0, posinf=self.activation_clip, neginf=-self.activation_clip)
                z = np.clip(z, -self.activation_clip, self.activation_clip)
                prior_encoded = np.maximum(0.0, z)

            # ---- Context fusion ----
            context = np.concatenate([prior_encoded, meta, pooled_multi])

            # ---- Main MLP with skip connections ----
            h = context
            for l_idx, layer in enumerate(self.layers[:-1]):
                h_new = layer["W"] @ h + layer["b"]
                h_new = np.nan_to_num(h_new, nan=0.0, posinf=self.activation_clip, neginf=-self.activation_clip)
                h_new = np.clip(h_new, -self.activation_clip, self.activation_clip)
                h = np.maximum(0.0, h_new)

            # Output layer (no activation)
            out_layer = self.layers[-1]
            delta = out_layer["W"] @ h + out_layer["b"]  # (prior_dim,)
            delta = np.nan_to_num(delta, nan=0.0, posinf=self.activation_clip, neginf=-self.activation_clip)
            delta = np.clip(delta, -self.activation_clip, self.activation_clip)

            # ---- Residual prediction: corrected = prior + delta ----
            corrected = prior + delta
            outputs.append(corrected)

        return np.vstack(outputs)  # (N, prior_dim)

    # -----------------------------------------------------------------------
    # Fit (Adam + mini-batch gradient descent)
    # -----------------------------------------------------------------------

    def fit(self, features: List[List[float]], targets: List[List[float]]) -> None:
        _require_numpy()

        X = np.array(features, dtype=np.float64)
        Y = np.array(targets, dtype=np.float64)
        N, feat_dim = X.shape
        out_dim = Y.shape[1]

        rng = np.random.RandomState(self.seed)

        # ---- Initialise multi-head attention weights ----
        self.W_attn_heads = (rng.randn(self.num_heads, self.obs_dim) * 0.01).tolist()
        self.b_attn_heads = [0.0] * self.num_heads

        # ---- Initialise prior encoder ----
        self.prior_encoder = []
        prior_enc_dims = [self.prior_dim, self.prior_encoder_dim]
        for fan_in, fan_out in zip(prior_enc_dims[:-1], prior_enc_dims[1:]):
            scale = math.sqrt(2.0 / fan_in)
            self.prior_encoder.append({
                "W": rng.randn(fan_out, fan_in).astype(np.float64) * scale,
                "b": np.zeros(fan_out, dtype=np.float64),
            })

        # ---- Initialise main MLP ----
        context_dim = self.prior_encoder_dim + self.meta_dim + (self.num_heads * self.obs_dim)
        layer_dims = [context_dim] + self.hidden_dims + [out_dim]
        self.layers = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            scale = math.sqrt(2.0 / fan_in)  # He init
            self.layers.append({
                "W": rng.randn(fan_out, fan_in).astype(np.float64) * scale,
                "b": np.zeros(fan_out, dtype=np.float64),
            })

        # ---- Adam state ----
        W_attn_np = np.array(self.W_attn_heads, dtype=np.float64)
        b_attn_np = np.array(self.b_attn_heads, dtype=np.float64)

        adam_params = (
            [W_attn_np, b_attn_np] +
            [p for layer in self.prior_encoder for p in (layer["W"], layer["b"])] +
            [p for layer in self.layers for p in (layer["W"], layer["b"])]
        )
        m_state = [np.zeros_like(p) for p in adam_params]
        v_state = [np.zeros_like(p) for p in adam_params]
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        def _get_params():
            return (
                [W_attn_np, b_attn_np] +
                [p for layer in self.prior_encoder for p in (layer["W"], layer["b"])] +
                [p for layer in self.layers for p in (layer["W"], layer["b"])]
            )

        def _set_params(params):
            nonlocal W_attn_np, b_attn_np
            W_attn_np = params[0]
            b_attn_np = params[1]
            idx = 2
            for layer in self.prior_encoder:
                layer["W"] = params[idx]; idx += 1
                layer["b"] = params[idx]; idx += 1
            for layer in self.layers:
                layer["W"] = params[idx]; idx += 1
                layer["b"] = params[idx]; idx += 1

        step = 0
        idx_all = list(range(N))

        print(f"Training enhanced model: {self.num_heads} heads, {self.hidden_dims} hidden dims")
        print(f"Total parameters: ~{sum(p.size for p in adam_params):,}")

        for epoch in range(self.epochs):
            rng.shuffle(idx_all)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, N, self.batch_size):
                batch_idx = idx_all[start:start + self.batch_size]
                Xb = X[batch_idx]
                Yb = Y[batch_idx]
                Nb = len(batch_idx)

                # ---- Forward pass (collect intermediates) ----
                priors_b, metas_b, obs_slots_b, masks_b = [], [], [], []
                attns_heads_b = []  # (Nb, num_heads, K)
                pooled_heads_b = []  # (Nb, num_heads, D_obs)
                prior_encoded_b = []
                contexts_b = []
                hs_prior_enc_b = [[] for _ in self.prior_encoder]
                zs_prior_enc_b = [[] for _ in self.prior_encoder]
                hs_b = [[] for _ in self.layers]
                zs_b = [[] for _ in self.layers]

                for xi in Xb:
                    prior, meta, obs_slots, mask = self._split(xi)
                    priors_b.append(prior)
                    metas_b.append(meta)
                    obs_slots_b.append(obs_slots)
                    masks_b.append(mask)

                    # Multi-head attention
                    attns_heads = []
                    pooled_heads = []
                    for h in range(self.num_heads):
                        scores = obs_slots @ W_attn_np[h] + b_attn_np[h]
                        scores = np.nan_to_num(scores, nan=0.0, posinf=self.attention_score_clip, neginf=-self.attention_score_clip)
                        scores = np.clip(scores - scores.max(), -self.attention_score_clip, self.attention_score_clip)
                        exp_s = np.exp(scores) * mask
                        attn = exp_s / (exp_s.sum() + 1e-12)
                        pooled = (attn[:, None] * obs_slots).sum(axis=0)
                        attns_heads.append(attn)
                        pooled_heads.append(pooled)
                    attns_heads_b.append(attns_heads)
                    pooled_heads_b.append(pooled_heads)
                    pooled_multi = np.concatenate(pooled_heads)

                    # Prior encoder
                    h_prior = prior
                    for l_idx, layer in enumerate(self.prior_encoder):
                        z = layer["W"] @ h_prior + layer["b"]
                        z = np.nan_to_num(z, nan=0.0, posinf=self.activation_clip, neginf=-self.activation_clip)
                        z = np.clip(z, -self.activation_clip, self.activation_clip)
                        hs_prior_enc_b[l_idx].append(h_prior)
                        zs_prior_enc_b[l_idx].append(z)
                        h_prior = np.maximum(0.0, z)
                    prior_encoded_b.append(h_prior)

                    # Context fusion
                    context = np.concatenate([h_prior, meta, pooled_multi])
                    contexts_b.append(context)

                    # Main MLP
                    h = context
                    for l_idx, layer in enumerate(self.layers):
                        z = layer["W"] @ h + layer["b"]
                        z = np.nan_to_num(z, nan=0.0, posinf=self.activation_clip, neginf=-self.activation_clip)
                        z = np.clip(z, -self.activation_clip, self.activation_clip)

                        hs_b[l_idx].append(h)
                        zs_b[l_idx].append(z)
                        h = np.maximum(0.0, z) if l_idx < len(self.layers) - 1 else z

                # Predictions (residual: corrected = prior + delta)
                deltas = np.array(zs_b[-1])  # (Nb, out_dim)
                priors_batch = np.array(priors_b)  # (Nb, out_dim)
                preds = priors_batch + deltas

                # Loss
                loss = ((preds - Yb) ** 2).mean()
                epoch_loss += loss
                num_batches += 1

                # Gradient
                delta_grad = (preds - Yb) * 2.0 / Nb
                delta_grad = np.clip(delta_grad, -10.0, 10.0)
                delta_grad = np.nan_to_num(delta_grad, nan=0.0, posinf=10.0, neginf=-10.0)

                # Backprop through main MLP
                grads_W = [np.zeros_like(layer["W"]) for layer in self.layers]
                grads_b = [np.zeros_like(layer["b"]) for layer in self.layers]

                d = delta_grad.copy()
                for l_idx in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[l_idx]
                    h_in = np.array(hs_b[l_idx])  # (Nb, fan_in)

                    if l_idx < len(self.layers) - 1:
                        relu_mask = (np.array(zs_b[l_idx]) > 0).astype(np.float32)
                        d = d * relu_mask

                    grads_W[l_idx] = d.T @ h_in + self.alpha * layer["W"]
                    grads_b[l_idx] = d.sum(axis=0)
                    grads_W[l_idx] = np.nan_to_num(grads_W[l_idx], nan=0.0, posinf=10.0, neginf=-10.0)
                    grads_b[l_idx] = np.nan_to_num(grads_b[l_idx], nan=0.0, posinf=10.0, neginf=-10.0)

                    d = d @ layer["W"]
                    d = np.nan_to_num(d, nan=0.0, posinf=10.0, neginf=-10.0)

                grad_context = np.clip(d, -10.0, 10.0)  # (Nb, context_dim)

                # Split context gradient
                grad_prior_enc = grad_context[:, :self.prior_encoder_dim]
                grad_pooled_multi = grad_context[:, self.prior_encoder_dim + self.meta_dim:]

                # Backprop through prior encoder
                grads_prior_enc_W = [np.zeros_like(layer["W"]) for layer in self.prior_encoder]
                grads_prior_enc_b = [np.zeros_like(layer["b"]) for layer in self.prior_encoder]

                d_prior = grad_prior_enc.copy()
                for l_idx in range(len(self.prior_encoder) - 1, -1, -1):
                    layer = self.prior_encoder[l_idx]
                    h_in = np.array(hs_prior_enc_b[l_idx])
                    relu_mask = (np.array(zs_prior_enc_b[l_idx]) > 0).astype(np.float32)
                    d_prior = d_prior * relu_mask

                    grads_prior_enc_W[l_idx] = d_prior.T @ h_in + self.alpha * layer["W"]
                    grads_prior_enc_b[l_idx] = d_prior.sum(axis=0)
                    grads_prior_enc_W[l_idx] = np.nan_to_num(grads_prior_enc_W[l_idx], nan=0.0, posinf=10.0, neginf=-10.0)
                    grads_prior_enc_b[l_idx] = np.nan_to_num(grads_prior_enc_b[l_idx], nan=0.0, posinf=10.0, neginf=-10.0)

                    d_prior = d_prior @ layer["W"]
                    d_prior = np.nan_to_num(d_prior, nan=0.0, posinf=10.0, neginf=-10.0)

                # Backprop through multi-head attention
                grad_W_attn = np.zeros_like(W_attn_np)
                grad_b_attn = np.zeros_like(b_attn_np)

                for i in range(Nb):
                    for h in range(self.num_heads):
                        start_idx = h * self.obs_dim
                        end_idx = (h + 1) * self.obs_dim
                        gp = grad_pooled_multi[i, start_idx:end_idx]  # (D_obs,)

                        attn = attns_heads_b[i][h]  # (K,)
                        obs_slots = obs_slots_b[i]  # (K, D_obs)

                        d_attn_score = obs_slots @ gp  # (K,)
                        dot = float(attn @ d_attn_score)
                        d_scores = attn * (d_attn_score - dot)  # (K,)

                        grad_W_attn[h] += (d_scores[:, None] * obs_slots).sum(axis=0)
                        grad_b_attn[h] += d_scores.sum()

                grad_W_attn += self.alpha * W_attn_np
                grad_W_attn = np.nan_to_num(grad_W_attn, nan=0.0, posinf=10.0, neginf=-10.0)
                grad_b_attn = np.nan_to_num(grad_b_attn, nan=0.0, posinf=10.0, neginf=-10.0)

                # ---- Gradient clipping ----
                step += 1
                all_grads = (
                    [grad_W_attn, grad_b_attn] +
                    [g for gW, gb in zip(grads_prior_enc_W, grads_prior_enc_b) for g in (gW, gb)] +
                    [g for gW, gb in zip(grads_W, grads_b) for g in (gW, gb)]
                )
                all_grads = [np.nan_to_num(g, nan=0.0, posinf=10.0, neginf=-10.0) for g in all_grads]
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
                    updated = p - self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)
                    updated = np.nan_to_num(updated, nan=0.0, posinf=self.parameter_clip, neginf=-self.parameter_clip)
                    updated = np.clip(updated, -self.parameter_clip, self.parameter_clip)
                    new_params.append(updated)

                _set_params(new_params)

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/num_batches:.6f}")

        # Persist to Python lists
        self.W_attn_heads = W_attn_np.tolist()
        self.b_attn_heads = b_attn_np.tolist()
        for layer in self.prior_encoder:
            layer["W"] = layer["W"].tolist()
            layer["b"] = layer["b"].tolist()
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

        X = np.array(features, dtype=np.float64)

        # Convert to numpy for forward pass
        W_attn_np = np.array(self.W_attn_heads, dtype=np.float64)
        b_attn_np = np.array(self.b_attn_heads, dtype=np.float64)

        orig_W = self.W_attn_heads
        orig_b = self.b_attn_heads
        self.W_attn_heads = W_attn_np.tolist()
        self.b_attn_heads = b_attn_np.tolist()

        for layer in self.prior_encoder:
            layer["W"] = np.array(layer["W"], dtype=np.float64)
            layer["b"] = np.array(layer["b"], dtype=np.float64)
        for layer in self.layers:
            layer["W"] = np.array(layer["W"], dtype=np.float64)
            layer["b"] = np.array(layer["b"], dtype=np.float64)

        result = self._forward_np(X).tolist()

        # Convert back to lists
        self.W_attn_heads = orig_W
        self.b_attn_heads = orig_b
        for layer in self.prior_encoder:
            layer["W"] = layer["W"].tolist()
            layer["b"] = layer["b"].tolist()
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
            "model_type": "MlpHistogramModelV2",
            "obs_dim": self.obs_dim,
            "prior_dim": self.prior_dim,
            "meta_dim": self.meta_dim,
            "max_observations": self.max_observations,
            "num_heads": self.num_heads,
            "hidden_dims": self.hidden_dims,
            "prior_encoder_dim": self.prior_encoder_dim,
            "alpha": self.alpha,
            "activation_clip": self.activation_clip,
            "attention_score_clip": self.attention_score_clip,
            "parameter_clip": self.parameter_clip,
            "W_attn_heads": self.W_attn_heads,
            "b_attn_heads": self.b_attn_heads,
            "prior_encoder": [{"W": layer["W"], "b": layer["b"]} for layer in self.prior_encoder],
            "layers": [{"W": layer["W"], "b": layer["b"]} for layer in self.layers],
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "MlpHistogramModelV2":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            obs_dim=payload["obs_dim"],
            prior_dim=payload["prior_dim"],
            meta_dim=payload.get("meta_dim", 3),
            max_observations=payload["max_observations"],
            num_heads=payload.get("num_heads", 3),
            hidden_dims=tuple(payload["hidden_dims"]),
            prior_encoder_dim=payload.get("prior_encoder_dim", 32),
            alpha=payload.get("alpha", 1e-4),
            activation_clip=payload.get("activation_clip", 10.0),
            attention_score_clip=payload.get("attention_score_clip", 20.0),
            parameter_clip=payload.get("parameter_clip", 2.0),
        )
        model.W_attn_heads = payload["W_attn_heads"]
        model.b_attn_heads = payload["b_attn_heads"]
        model.prior_encoder = payload["prior_encoder"]
        model.layers = payload["layers"]
        model._fitted = True
        return model
