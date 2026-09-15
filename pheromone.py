"""
pheromone.py — the ant learns which smells mean WATER / FOOD / ALARM without
forgetting the others.

A direct port of flyputer's `sniff.py`: the ant brain does olfactory learning the
same architectural way the fly does it — a sparse flickering of Kenyon cells per
smell, so teaching smell B barely touches smell A. Here the inputs are realistic ant
pheromones instead of generic odors:

  - water trail pheromone   (recruitment to a found water/food source)
  - alarm pheromone         (danger -> avoid + recruit defenders)
  - home (nest) scent       (the path back to the colony)

The glomeruli -> Kenyon cell -> MBON wiring comes from `antbrain` (live CRANTb when
available, otherwise the bundled functional model).

CLI:
    .venv/bin/python pheromone.py
"""
from __future__ import annotations

import numpy as np

import antbrain

GLOMERULI = "antennal-lobe glomerulus"
KENYON = "Kenyon cell"
MBON = "MBON"

_CIRC = None


def _cls_ids(val):
    ann = antbrain.annotations()
    if len(ann) == 0:
        return []
    m = ann["cell_class"].astype(str).str.fullmatch(val, case=False, na=False).to_numpy()
    return [int(r) for r in ann.root_id[m].tolist()]


def circuit(min_syn=0):
    """Real wiring slice: glomeruli -> Kenyon cells -> MBONs. Cached."""
    global _CIRC
    if _CIRC is not None:
        return _CIRC
    conn = antbrain.load_connectome()
    gl = sorted(_cls_ids(GLOMERULI))
    kc = sorted(_cls_ids(KENYON))
    mb = sorted(_cls_ids(MBON))
    if not (gl and kc and mb):
        raise RuntimeError("brain has no olfactory circuit (wrong mode?)")
    gi = {g: i for i, g in enumerate(gl)}
    ki = {k: i for i, k in enumerate(kc)}
    mi = {m: i for i, m in enumerate(mb)}
    e1 = conn[conn.pre.isin(set(gl)) & conn.post.isin(set(kc))]
    e2 = conn[conn.pre.isin(set(kc)) & conn.post.isin(set(mb))]
    Wgk = np.zeros((len(kc), len(gl)), dtype=np.float32)   # KC x glomerulus (fan)
    for pre, post in zip(e1.pre.values, e1.post.values):
        Wgk[ki[int(post)], gi[int(pre)]] += 1.0
    Wkm = np.zeros((len(mb), len(kc)), dtype=np.float32)   # MBON x KC
    for pre, post, w in zip(e2.pre.values, e2.post.values, e2.weight.values):
        Wkm[mi[int(post)], ki[int(pre)]] += float(w)
    _CIRC = {"gl": gl, "kc": kc, "mb": mb, "Wgk": Wgk, "Wkm": Wkm}
    return _CIRC


def smell(seed, active_frac=0.15):
    """A pheromone blend = a random subset of glomeruli firing (one input pattern).
    Seed convention: 0=water, 1=alarm, 2=home. Any other seed = a 'new' smell."""
    C = circuit()
    n = len(C["gl"])
    r = np.random.default_rng(seed)
    return (r.random(n) < active_frac).astype(np.float32)


def kc_code(blend, coincidence=3):
    """Sparse Kenyon-cell code: a KC fires if >= `coincidence` of its glomeruli are
    active (coincidence detection -> each smell lights a tiny near-disjoint subset)."""
    C = circuit()
    active_partners = (C["Wgk"] > 0) @ blend
    return active_partners >= coincidence


class AntMemory:
    """Mushroom-body learning: a dopamine-gated DEPRESSION of KC->MBON output for
    whichever KCs were active during an associated (good or bad) smell. New memories
    touch their own near-disjoint KCs, so they don't overwrite old ones."""

    def __init__(self):
        C = circuit()
        self.n_kc = len(C["kc"])
        self.good = np.zeros(self.n_kc, dtype=bool)   # KCs whose output means GOOD
        self.bad = np.zeros(self.n_kc, dtype=bool)    # KCs whose output means BAD

    def teach_good(self, code):
        self.good |= code

    def teach_bad(self, code):
        self.bad |= code

    def valence(self, code):
        """Learned evaluation of a smell in [-1, 1]: + attract (water/food/home),
        - avoid (alarm)."""
        n = int(code.sum())
        if n == 0:
            return 0.0
        return float((code & self.good).sum() / n - (code & self.bad).sum() / n)

    def forget(self, frac=0.02):
        """A little noise so learned memories aren't literally permanent."""
        r = np.random.default_rng()
        good_noise = r.random(self.n_kc) < frac
        bad_noise = r.random(self.n_kc) < frac
        self.good &= ~good_noise
        self.bad &= ~bad_noise


def train_ant(mem: AntMemory, n_repeats: int = 3, coincidence: int = 3):
    """Teach the ant: water & home smell GOOD, alarm smell BAD. Returns a summary."""
    taught = 0
    for rep in range(n_repeats):
        for seed, good in ((0, True), (2, True), (1, False)):
            mem.teach_good(kc_code(smell(seed), coincidence)) if good \
                else mem.teach_bad(kc_code(smell(seed), coincidence))
            taught += 1
    return {"taught": taught, "n_taught_good_kc": int(mem.good.sum()),
            "n_taught_bad_kc": int(mem.bad.sum())}


def run_experiment(seedA=0, seedB=1, coincidence=3):
    """Water-smell vs alarm-smell: two near-disjoint Kenyon-cell codes, and the ant
    reliably evaluates the allowed smells right without confusing new ones."""
    mem = AntMemory()
    train_ant(mem, coincidence=coincidence)
    sm = {s: smell(s) for s in (0, 1, 2)}
    codes = {s: kc_code(sm[s], coincidence) for s in sm}
    overlap = (codes[0] & codes[1]).sum() / max(1, (codes[0] | codes[1]).sum())
    vals = {s: mem.valence(codes[s]) for s in codes}
    novel = mem.valence(kc_code(smell(99), coincidence))
    return {
        "kc_overlap_water_alarm": float(overlap),
        "valence_water": vals[0], "valence_alarm": vals[1], "valence_home": vals[2],
        "valence_novel": float(novel),
        "kc_active_water": float(codes[0].mean()), "kc_active_alarm": float(codes[1].mean()),
    }


if __name__ == "__main__":
    print("brain mode:", antbrain.mode())
    C = circuit()
    print("Ant olfactory circuit: %d glomeruli -> %d Kenyon cells -> %d MBONs\n"
          % (len(C["gl"]), len(C["kc"]), len(C["mb"])))
    r = run_experiment()
    print("Water & alarm smells light two near-disjoint sparse KC codes (%.1f%% overlap)."
          % (100 * r["kc_overlap_water_alarm"]))
    print("Learned valence  ->  water %+.2f   home %+.2f   alarm %+.2f   novel %+.2f"
          % (r["valence_water"], r["valence_home"], r["valence_alarm"], r["valence_novel"]))
    if r["valence_water"] > 0.2 and r["valence_alarm"] < -0.2 and abs(r["valence_novel"]) < 0.3:
        print("\nThe ant knows water=good, alarm=bad, and doesn't panic about new smells.")
    else:
        print("\n(check tuning: valences should be water>0, home>0, alarm<0, novel~0)")