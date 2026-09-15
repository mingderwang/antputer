"""
antbrain.py — connectome access layer for the ant.

The Clonal Raider ANT connectome (CRANTb, *Ooceraea biroi*) is the first complete
ant brain wiring after the fruit fly. Its connectivity is exposed by `crantpy` as an
edge list  [pre, post, weight]  (same spirit as FlyWire's feather table that
flyputer uses). CRANTb is an in-progress research dataset: it requires a CAVE token
(browser auth) and is queried live through CAVE.

This module is the *port layer*. It mirrors `flysim.py`'s API so the rest of antputer
can consume connectivity the very same way flyputer does:

    antbrain.load_connectome()   -> DataFrame [pre, post, weight]
    antbrain.build_subcircuit(seed_ids) -> signed W[post, pre]

Two modes, chosen automatically:
  1. LIVE     — crantpy is installed + a CAVE token works   -> real CRANTb edges
  2. OFFLINE  — bundled *functional brain model*: a structurally-plausible ant
                olfactory circuit (antennal-lobe glomeruli -> Kenyon cells -> MBONs,
                plus central-complex output) with randomized sparse wiring, labelled
                clearly as a MODEL, not measured data.

Data files (never committed): data/crant_connectivity.feather (live snapshot),
data/crant_neuron_annotations.tsv (live neuron metadata).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EDGE_CACHE = os.path.join(DATA_DIR, "crant_connectivity.feather")
ANN_CACHE = os.path.join(DATA_DIR, "crant_neuron_annotations.tsv")

CONN: pd.DataFrame | None = None
ANN: pd.DataFrame | None = None

# --------------------------------------------------------------------------- #
# functional brain model (offline fallback) — clearly NOT measured data
# --------------------------------------------------------------------------- #

# Plausible ant-olfaction numbers (order-of-magnitude, from ant neuroanatomy):
# each antenna has a few hundred antennal-lobe glomeruli; Kenyon cells number in the
# thousands; a handful of MBON subtypes read the mushroom body out.
N_GLOMERULI = 96        # glomeruli per antenna (left+right pooled, reduced for speed)
N_KENYON = 2048         # Kenyon cells
N_MBON = 24             # mushroom-body output neurons (valence/flexible readouts)
N_COMMIT = 32           # "committed" parallel set for switching/decision output
MODEL_NOTE = "OFFLINE FUNCTIONAL MODEL (random sparse wiring; NOT measured CRANTb data)"


def _bincount_pairs(targets, n):
    """Deduplicate targets into (target, count) pairs."""
    if len(targets) == 0:
        return []
    vals, counts = np.unique(targets, return_counts=True)
    return list(zip(vals.tolist(), counts.tolist()))


def _load_or_build_model():
    """Build the offline model edges once (cached on disk so reconnectivity is stable
    across runs). Never committed to git — it lives in data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "model_connectivity.feather")
    if os.path.exists(cache):
        conn = pd.read_feather(cache)
        ann = pd.read_feather(os.path.join(DATA_DIR, "model_neurons.feather"))
        return conn, ann, MODEL_NOTE
    rng = np.random.default_rng(20240913)
    nodes = list(range(N_GLOMERULI + N_KENYON + N_MBON + 8 + N_COMMIT))
    pre, post, wgt = [], [], []
    for g in range(N_GLOMERULI):
        targets = rng.integers(0, N_KENYON, size=int(N_KENYON * 0.08))
        for k, c in _bincount_pairs(targets, N_KENYON):
            pre.append(g); post.append(N_GLOMERULI + k); wgt.append(float(c))
    for k in range(N_KENYON):
        for m in rng.choice(N_MBON, size=int(rng.integers(1, 4)), replace=False):
            pre.append(N_GLOMERULI + k); post.append(N_GLOMERULI + N_KENYON + int(m)); wgt.append(1.0)
    ring = N_GLOMERULI + N_KENYON + N_MBON
    for c in range(8):
        for nb in ((c - 1) % 8, c, (c + 1) % 8):
            pre.append(ring + c); post.append(ring + nb); wgt.append(0.5)
        for s in range(N_COMMIT):
            pre.append(ring + c); post.append(ring + 8 + s); wgt.append(0.25)
    ann = pd.DataFrame({"root_id": nodes,
                        "region": (["OL_antennal"] * N_GLOMERULI
                                   + ["MB_kenyon"] * N_KENYON
                                   + ["MBON"] * N_MBON
                                   + ["CX_ring"] * 8
                                   + ["CX_output"] * N_COMMIT),
                        "cell_class": (["antennal-lobe glomerulus"] * N_GLOMERULI
                                       + ["Kenyon cell"] * N_KENYON
                                       + ["MBON"] * N_MBON
                                       + ["compass"] * 8
                                       + ["steering"] * N_COMMIT)})
    conn = pd.DataFrame({"pre": pre, "post": post, "weight": wgt})
    conn.to_feather(cache)
    ann.to_feather(os.path.join(DATA_DIR, "model_neurons.feather"))
    return conn, ann, MODEL_NOTE


# --------------------------------------------------------------------------- #
# public API (mirrors flysim.py)
# --------------------------------------------------------------------------- #

def _try_live_connectome() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Attempt a live CRANTb pull. Returns (edges, annotations) or raises."""
    try:
        import crantpy
    except ImportError:
        raise RuntimeError("crantpy not installed")
    ann = None
    if os.path.exists(ANN_CACHE):
        ann = pd.read_csv(ANN_CACHE, sep="\t")
    if os.path.exists(EDGE_CACHE):
        return pd.read_feather(EDGE_CACHE), ann
    # The full-synapse pull is expensive + token-gated; for a first port we keep it
    # lazy: callers who actually want live data run ./scripts/port_crant.py
    raise RuntimeError("live CRANTb snapshot not cached; run scripts/port_crant.py")


def _ensure() -> None:
    global CONN, ANN
    if CONN is not None:
        return
    # 1) prefer a live/cached CRANTb snapshot
    try:
        conn, ann = _try_live_connectome()
    except Exception:
        conn, ann, note = _load_or_build_model()
        CONN = conn
        ANN = ann
        _MODE.set_model(note)
        return
    CONN = conn
    ANN = ann
    _MODE.set_model("LIVE CRANTb connectome")


class _Mode:
    note = MODEL_NOTE

    def set_model(self, n):
        self.note = n


_MODE = _Mode()


def mode() -> str:
    """Which brain is backing the colony: live CRANTb or the offline functional model."""
    _ensure()
    return _MODE.note


def load_connectome():
    """Edge list  [pre, post, weight]  (pre = presynaptic, post = postsynaptic)."""
    _ensure()
    return CONN


def annotations():
    """Neuron metadata if available (region / cell_class), else an empty frame."""
    _ensure()
    return ANN if ANN is not None else pd.DataFrame()


def build_subcircuit(seed_ids, hops=2, max_neurons=1500, min_syn=1):
    """BFS the downstream neighbourhood of the seeds; return signed W[post, pre]."""
    _ensure()
    seeds = list(dict.fromkeys(int(s) for s in seed_ids))
    edges = CONN[CONN.weight >= min_syn]
    order = list(seeds)
    seen = set(seeds)
    frontier = list(seeds)
    for _ in range(hops):
        if not frontier or len(order) >= max_neurons:
            break
        targets = edges[edges.pre.isin(frontier)].post.values
        new = []
        for x in targets:
            xi = int(x)
            if xi not in seen:
                seen.add(xi)
                order.append(xi)
                new.append(xi)
                if len(order) >= max_neurons:
                    break
        frontier = new
    nodes = order[:max_neurons]
    idx = {n: i for i, n in enumerate(nodes)}
    sub = edges[edges.pre.isin(set(nodes)) & edges.post.isin(set(nodes))]
    N = len(nodes)
    W = np.zeros((N, N), dtype=np.float32)
    for pre, post, w in zip(sub.pre.values, sub.post.values, sub.weight.values):
        W[idx[int(post)], idx[int(pre)]] += float(w)
    return nodes, idx, W


def neuron_ids(query: str | None = None, limit: int = 50):
    """Neuron lookup by cell class/region (stub of flysim.find_neurons). Offline model
    only has the olfaction + compass classes; live mode gets real annotions."""
    _ensure()
    ann = ANN
    if ann is None or len(ann) == 0:
        return []
    df = ann.copy()
    if query:
        q = query.lower()
        mask = (df["cell_class"].astype(str).str.lower().str.contains(q, na=False)
                | df["region"].astype(str).str.lower().str.contains(q, na=False))
        df = df[mask]
    return [int(r) for r in df["root_id"].head(limit).tolist()]


if __name__ == "__main__":
    print("mode:", mode())
    c = load_connectome()
    ann = annotations()
    print("edges:", c.shape)
    print("neurons:", len(ann) if len(ann) else "n/a")
    print("classes:", ", ".join(sorted(set(ann["cell_class"]))) if len(ann) else "n/a")