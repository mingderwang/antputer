"""
scripts/port_crant.py — pull the real Clonal Raider ANT connectome (CRANTb) into
antputer, in the exact edge-list format the rest of the code consumes.

CRANTb is a live, token-gated dataset (Connectome Annotation Versioning Engine,
CAVE). It is NOT downloadable as a static file yet (v1.0 is planned for 2026).

Prereqs:
    .venv/bin/pip install crantpy          # from the crantpy GitHub repo if not on PyPI
    # one-time browser auth:
    .venv/bin/python -c "import crantpy as cp; cp.generate_cave_token(save=True)"

Run:
    .venv/bin/python scripts/port_crant.py

It writes (git-ignored):
    data/crant_connectivity.feather     # columns: pre, post, weight
    data/crant_neuron_annotations.tsv   # columns: root_id, cell_class, region

After that, antbrain.py auto-detects the snapshot and the whole app runs on real
ant wiring instead of the offline functional model.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# Cell classes we care about: the ant olfactory learning circuit + the central
# complex (compass / steering), the parts the colony simulation leans on.
WANTED_CLASSES = [
    "antennal-lobe glomerulus", "antennal lobe projection neuron", "ALPN",
    "Kenyon cell", "spiny_kenyon_cell", "MBON",
    "compass", "ring neuron", "PFN", "PFL3", "DNa02", "steering",
]


def main():
    try:
        import crantpy as cp
    except ImportError:
        sys.exit("crantpy not installed.\n"
                 "  .venv/bin/pip install git+https://github.com/"
                 "Social-Evolution-and-Behavior/crantpy.git\n"
                 "(or pip install crantpy once it is on PyPI)")

    os.makedirs(DATA, exist_ok=True)
    try:
        client = cp.get_cave_client()
        print("connected to datastack:", client.datastack_name)
    except Exception as e:
        sys.exit(f"CAVE auth failed ({e}).\n"
                 "  .venv/bin/python -c \"import crantpy as cp; cp.generate_cave_token(save=True)\"")

    ann_rows, edge_frames = [], []
    for cls in WANTED_CLASSES:
        try:
            crit = cp.NeuronCriteria(cell_class=cls)
            ids = list(crit.get_roots())
        except Exception as e:
            print(f"  skip {cls!r}: {e}")
            continue
        if not ids:
            continue
        print(f"  {cls}: {len(ids)} neurons")
        ann_rows += [{"root_id": int(i), "cell_class": cls, "region": ""} for i in ids]
        try:
            conn = cp.get_connectivity(ids, threshold=1)   # [pre, post, weight]
            edge_frames.append(conn)
        except Exception as e:
            print(f"    connectivity failed: {e}")

    if not edge_frames:
        sys.exit("no connectivity returned — nothing to port")

    edges = pd.concat(edge_frames, ignore_index=True)
    edges = edges.rename(columns={"pre": "pre", "post": "post", "weight": "weight"})
    edges.to_feather(os.path.join(DATA, "crant_connectivity.feather"))
    pd.DataFrame(ann_rows).drop_duplicates("root_id").to_csv(
        os.path.join(DATA, "crant_neuron_annotations.tsv"), sep="\t", index=False)
    print(f"\nwrote {len(edges):,} edges and {len(ann_rows):,} neuron rows -> data/")
    print("antbrain.py will now use REAL CRANTb wiring.")


if __name__ == "__main__":
    main()