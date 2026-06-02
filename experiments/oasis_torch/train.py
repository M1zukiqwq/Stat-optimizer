"""Train OASIS Stage-1 v3 on the downstream/system-error objective.

Loss = w_fut * future-predicate log-Qerror (on projected marginal, held-out preds)
     + w_join * FactorJoin bilinear regret (projected vs fresh)
     + w_cons * feedback residual (prior stays consistent)
     + w_reg  * reconstruction MAE to fresh (valid/safe histogram)
Curriculum subsamples the feedback count K per batch.
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

from model import (OasisTorchV3, boundaries_from_logits, cell_masses, interval_coverage,
                   ipf_project, sel_from_masses)

K_CURRICULUM = [2, 4, 6, 8, 12, 16]


def make_grid(G, device):
    return torch.linspace(0, 1, G + 1, device=device)


def levels_of(B, device):
    return torch.linspace(0, 1, B + 1, device=device)


def cap_obs(batch, kcap):
    """Zero feedback mask beyond first kcap observations (curriculum)."""
    m = batch["obs_mask"].clone()
    if kcap < m.shape[1]:
        m[:, kcap:] = 0.0
    return m


def compute_loss(model, batch, grid, levels, B, w, kcap=None, ipf_iters=8, use_ipf=True):
    obs_mask = cap_obs(batch, kcap) if kcap is not None else batch["obs_mask"]
    logits = model(batch["obs_feat"], obs_mask, batch["stale"])
    bx = boundaries_from_logits(logits)                      # (Bsz,B+1)
    m = cell_masses(bx, levels, grid)                        # (Bsz,G)
    # ---- differentiable projection onto (curriculum-capped) feedback ----
    # use_ipf=False is the "raw-train + post-hoc projection" ablation: the loss
    # sees the RAW prior (projection only applied at eval), isolating the
    # contribution of calibration-IN-THE-LOOP vs post-hoc projection.
    if use_ipf:
        cover = interval_coverage(grid, batch["obs_lo"], batch["obs_hi"])   # (Bsz,K,G)
        m_proj = ipf_project(m, cover, batch["obs_tgt"], obs_mask, n_iter=ipf_iters)
    else:
        m_proj = m
    # ---- future-predicate log-Qerror on projected marginal ----
    fsel = sel_from_masses(m_proj, grid, None, batch["fut_lo"], batch["fut_hi"])  # (Bsz,P)
    fut = (torch.log(fsel) - torch.log(batch["fut_true"].clamp_min(1e-9))).abs().mean()
    # ---- FactorJoin bilinear regret (table A=proj, B=fresh; true=fresh,fresh) ----
    m_fresh = cell_masses(batch["fresh"], levels, grid)
    est_join = (m_proj * m_fresh).sum(-1).clamp_min(1e-12)
    true_join = (m_fresh * m_fresh).sum(-1).clamp_min(1e-12)
    join = (torch.log(est_join) - torch.log(true_join)).abs().mean()
    # ---- feedback consistency on the prior (not projected) ----
    psel = sel_from_masses(m, grid, None, batch["obs_lo"], batch["obs_hi"])       # (Bsz,K)
    resid = ((psel - batch["obs_tgt"]).abs() * obs_mask).sum() / obs_mask.sum().clamp_min(1)
    # ---- reconstruction reg (valid/safe histogram) ----
    recon = (bx - batch["fresh"]).abs().mean()
    total = w["fut"] * fut + w["join"] * join + w["cons"] * resid + w["reg"] * recon
    return total, {"fut": fut.item(), "join": join.item(), "cons": resid.item(),
                   "recon": recon.item(), "total": total.item()}


def iterate_batches(T, bs, device, shuffle=True):
    n = T["stale"].shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield {k: v[j].to(device) for k, v in T.items()}


@torch.no_grad()
def validate(model, T, grid, levels, B, w, device, bs=512):
    model.eval()
    tot = {"fut": 0.0, "join": 0.0, "cons": 0.0, "n": 0}
    for batch in iterate_batches(T, bs, device, shuffle=False):
        _, parts = compute_loss(model, batch, grid, levels, B, w)
        nb = batch["stale"].shape[0]
        for k in ("fut", "join", "cons"):
            tot[k] += parts[k] * nb
        tot["n"] += nb
    model.train()
    return {k: tot[k] / max(tot["n"], 1) for k in ("fut", "join", "cons")}


def run_smoke(device):
    """Tiny self-test on random tensors to catch shape bugs (no data needed)."""
    B, K, G, P, bsz = 10, 16, 40, 48, 8
    grid = make_grid(G, device); levels = levels_of(B, device)
    model = OasisTorchV3(num_buckets=B, max_obs=K).to(device)
    fake = {
        "stale": torch.sort(torch.rand(bsz, B + 1, device=device), -1).values,
        "fresh": torch.sort(torch.rand(bsz, B + 1, device=device), -1).values,
        "obs_feat": torch.rand(bsz, K, model.obs_proj.in_features, device=device),
        "obs_mask": (torch.rand(bsz, K, device=device) > 0.3).float(),
        "obs_lo": torch.rand(bsz, K, device=device) * 0.4,
        "obs_hi": 0.6 + torch.rand(bsz, K, device=device) * 0.4,
        "obs_tgt": torch.rand(bsz, K, device=device).clamp(0.05, 0.95),
        "fut_lo": torch.rand(bsz, P, device=device) * 0.4,
        "fut_hi": 0.6 + torch.rand(bsz, P, device=device) * 0.4,
        "fut_true": torch.rand(bsz, P, device=device).clamp(0.05, 0.95),
    }
    for b in ("stale", "fresh"):
        fake[b][:, 0] = 0.0; fake[b][:, -1] = 1.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    w = {"fut": 1.0, "join": 0.3, "cons": 0.3, "reg": 0.1}
    for step in range(3):
        opt.zero_grad()
        loss, parts = compute_loss(model, fake, grid, levels, B, w, kcap=K_CURRICULUM[step % len(K_CURRICULUM)])
        loss.backward(); opt.step()
        print(f"smoke step {step}: {parts}")
    print("SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="ckpt_v3.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--ipf-iters", type=int, default=8)
    ap.add_argument("--w-fut", type=float, default=1.0)
    ap.add_argument("--w-join", type=float, default=0.3)
    ap.add_argument("--w-cons", type=float, default=0.3)
    ap.add_argument("--w-reg", type=float, default=0.1)
    ap.add_argument("--no-residual-prior", action="store_true")
    ap.add_argument("--no-ipf", action="store_true",
                    help="Ablation: train loss on the RAW prior (no in-loop projection); "
                         "projection applied only at eval (post-hoc).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    if a.smoke:
        run_smoke(device); return

    blob = torch.load(a.data, map_location="cpu")
    Ttr, Tva, meta = blob["train"], blob["val"], blob["meta"]
    B = meta["num_buckets"]; K = meta["max_obs"]
    grid = make_grid(a.grid, device); levels = levels_of(B, device)
    model = OasisTorchV3(num_buckets=B, max_obs=K, d_model=a.d_model, n_heads=a.heads,
                         n_layers=a.layers, residual_prior=not a.no_residual_prior).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    w = {"fut": a.w_fut, "join": a.w_join, "cons": a.w_cons, "reg": a.w_reg}
    n = Ttr["stale"].shape[0]
    print(f"train n={n} val n={Tva['stale'].shape[0]} B={B} K={K} device={device}")
    best = math.inf
    rng = torch.Generator().manual_seed(a.seed)
    for ep in range(a.epochs):
        model.train()
        for batch in iterate_batches(Ttr, a.bs, device, shuffle=True):
            kcap = K_CURRICULUM[int(torch.randint(0, len(K_CURRICULUM), (1,), generator=rng))]
            opt.zero_grad()
            loss, _ = compute_loss(model, batch, grid, levels, B, w, kcap=kcap,
                                   ipf_iters=a.ipf_iters, use_ipf=not a.no_ipf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            v = validate(model, Tva, grid, levels, B, w, device)
            score = v["fut"]
            tag = ""
            if score < best:
                best = score
                torch.save({"state_dict": model.state_dict(),
                            "config": {"num_buckets": B, "max_obs": K, "d_model": a.d_model,
                                       "n_heads": a.heads, "n_layers": a.layers,
                                       "residual_prior": not a.no_residual_prior},
                            "val": v}, a.out)
                tag = " *best*"
            print(f"ep {ep:3d} val fut={v['fut']:.4f} join={v['join']:.4f} cons={v['cons']:.4f}{tag}")
    print(f"done. best val fut(log-qerr)={best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
