"""Encoding, PCA whitening, ranking and metrics shared by the retrieval and evaluation scripts.

Metrics follow the paper: Recall@K (share of queries with >= 1 relevant item in the top K), PRES@K
(Magdy & Jones, SIGIR 2010), mAP@K normalised by min(#relevant, K), and NAR (Hu et al. 2025).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------- encoding

class PathListDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path)
            return idx, self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc


def gem_pool(patches: torch.Tensor, p: float = 3.0) -> torch.Tensor:
    """(B, N, D) -> (B, D) GeM pooling of L2-normalised patch tokens (the frozen baselines)."""
    clamped = patches.clamp(min=1e-6)
    powered = clamped.pow(p).mean(dim=1)
    out = powered.pow(1.0 / p)
    return F.normalize(out, p=2, dim=-1)


@torch.no_grad()
def encode(paths, fn, tf, device, bs: int = 64, nw: int = 4) -> torch.Tensor:
    """fn: images (B, C, H, W) on device -> descriptors (B, D). Returns (len(paths), D) in path order."""
    out = []
    for idx, imgs in DataLoader(PathListDataset(paths, tf), batch_size=bs, num_workers=nw):
        # clone: a worker's idx tensor holds a shared-memory file descriptor until freed
        out.append((idx.clone(), fn(imgs.to(device)).float().cpu()))
    emb = torch.empty(len(paths), out[0][1].shape[1])
    for idx, d in out:
        emb[idx] = d
    return emb


# ---------------------------------------------------------------- whitening

def fit_whiten(X: torch.Tensor, eps: float = 1e-5):
    """PCA whitening (mean, W) fitted on the rows of X; apply as normalize((x - mean) @ W)."""
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)            # ascending
    evals = evals.flip(0); evecs = evecs.flip(1)     # descending
    W = evecs * (1.0 / torch.sqrt(evals.clamp(min=eps)))   # (D, D) whitening
    return mu, W


# ---------------------------------------------------------------- metrics

def nar_per_query(ranks: torch.Tensor, n_combined: int) -> float:
    n_rel = ranks.numel()
    if n_rel == 0:
        return float("nan")
    ideal = n_rel * (n_rel + 1) / 2
    return float((ranks.sum().item() - ideal) / (n_combined * n_rel))


def map_at_k_per_query(ranks: torch.Tensor, k: int) -> float:
    n_rel = ranks.numel()
    if n_rel == 0:
        return float("nan")
    sorted_ranks = np.sort(ranks.cpu().numpy())
    ap = 0.0
    hits = 0
    for rank in sorted_ranks:
        if rank > k:
            break
        hits += 1
        ap += hits / rank
    return ap / min(n_rel, k)


def recall_at_k_per_query(ranks: torch.Tensor, k: int) -> float:
    if ranks.numel() == 0:
        return float("nan")
    return float((ranks <= k).any().item())


MAP_KS = (10, 30, 50, 100)
RECALL_KS = (1, 5, 10, 50, 100, 1000)


def full_metric_record(ranks: torch.Tensor, n_combined: int) -> Dict[str, float]:
    """Per-query NAR + mAP@{10,30,50,100} + R@{1,5,10,50,100,1000} + MeanRank from 1-indexed ranks."""
    rec: Dict[str, float] = {"NAR": nar_per_query(ranks, n_combined)}
    for k in MAP_KS:
        rec[f"mAP@{k}"] = map_at_k_per_query(ranks, k)
    for k in RECALL_KS:
        rec[f"R@{k}"] = recall_at_k_per_query(ranks, k)
    rec["MeanRank"] = float(ranks.float().mean().item()) if ranks.numel() else float("nan")
    return rec


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    keys = records[0].keys()
    return {k: float(np.nanmean([r[k] for r in records])) for k in keys}


def pres_at(ranks: torch.Tensor, nmax: int) -> float:
    """PRES (Magdy & Jones, SIGIR 2010): recall-oriented and robust to incomplete qrels. Relevants ranked
    beyond the cutoff nmax count as not found (rank capped at nmax + 1). 1 when all n relevants sit at
    ranks 1..n; -> 0 as they fall past nmax."""
    n = ranks.numel()
    r = ranks.float().clamp(max=nmax + 1)
    return float(1.0 - (r.mean() - (n + 1) / 2.0) / nmax)


# ---------------------------------------------------------------- examiner-confusion benchmark

@torch.no_grad()
def score_confusion(q_emb, g_emb, rel_idx_per_q, device, qb: int = 64):
    """Whitened-cosine ranks of each query's relevant gallery items -> (aggregate, n_queries, per-query).

    Whitening is fitted on a random sample of up to 20,000 gallery descriptors (global torch RNG).
    rel_idx_per_q[i] is a LongTensor of gallery indices relevant to query i. Gallery items that are not
    cited for a query are unjudged: only the ranks of the cited marks are scored. Ranks are computed by a
    strict-greater count (no N-wide argsort).
    """
    N = g_emb.shape[0]
    samp = g_emb[torch.randperm(N)[:min(20000, N)]].to(device)
    mu, W = fit_whiten(samp)
    wg = F.normalize((g_emb.to(device) - mu) @ W, p=2, dim=1)          # (N, D')
    wq = F.normalize((q_emb.to(device) - mu) @ W, p=2, dim=1)          # (Q, D')
    recs = []
    for s in range(0, wq.shape[0], qb):
        S = wq[s:s + qb] @ wg.T                                         # (b, N)
        for j in range(S.shape[0]):
            rel = rel_idx_per_q[s + j]
            if rel.numel() == 0:
                continue
            srow = S[j]
            ranks = (srow.unsqueeze(0) > srow[rel].unsqueeze(1)).sum(1) + 1   # (R,) 1-indexed
            rec = full_metric_record(ranks.cpu(), N)
            rec["PRES@100"] = pres_at(ranks, 100)
            rec["PRES@1000"] = pres_at(ranks, 1000)
            recs.append(rec)
    return aggregate_metrics(recs), len(recs), recs


# ---------------------------------------------------------------- METU-v2 near-duplicate ruler

def eval_metu(emb_pool, emb_q, qgroups, qctypes, device):
    """METU-v2 protocol: queries are injected after the gallery pool; the relevant set of a query is the
    other queries of its group; the query itself is masked. Whitening is fitted on up to 20,000 pool
    descriptors (global torch RNG). Returns overall and per-content-type aggregates plus per-query ranks."""
    all_emb = torch.cat([emb_pool, emb_q]); P = len(emb_pool); qgi = list(range(P, P + len(emb_q)))
    samp = emb_pool[torch.randperm(P)[:min(20000, P)]]
    mu, W = fit_whiten(samp.to(device))
    w = F.normalize(((all_emb.to(device) - mu) @ W), p=2, dim=1)
    sim = (w[qgi] @ w.T).float().cpu()
    recs = {}; per_query = []
    for i, gi in enumerate(qgi):
        row = sim[i].clone(); row[gi] = -1e9
        rel = [qgi[j] for j in range(len(qgi)) if j != i and qgroups[j] == qgroups[i]]
        if not rel:
            continue
        order = row.argsort(descending=True)
        rp = torch.empty_like(order); rp[order] = torch.arange(len(order))
        ranks = (rp[torch.tensor(rel)] + 1)
        recs.setdefault(qctypes[i], []).append(full_metric_record(ranks, len(row)))
        per_query.append({"type": qctypes[i], "ranks": ranks.tolist(), "n": int(len(row))})
    allr = [r for v in recs.values() for r in v]
    agg = lambda rs: {k: float(sum(x[k] for x in rs) / len(rs)) for k in rs[0]}
    return {"overall": agg(allr), "per_query": per_query,
            **{t: {**agg(v), "n": len(v)} for t, v in recs.items()}}
