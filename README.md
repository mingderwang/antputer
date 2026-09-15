# antputer

A terminal ant colony: many ants crawl a 2D map, communicate by **pheromone**, and
**decide collectively** whether to find water or relocate the nest — the classic
emergent-swarm intelligence, forked conceptually from the fruit-fly brain simulation
[`flyputer`](../googlebrain/flyputer) and adapted to an **ant brain**.

## What it does

* **Swarm behaviour** — each ant noses the pheromone field, follows trails, recruits
  nestmates by teaching them where the water is (tandem-running), and the colony
  converges on water sources through positive feedback.
* **Two scenarios** — `find_water` (foraging + recruitment) and `relocate` (a source
  dries up, alarm spreads, the colony re-anchors its nest on a new source).
* **Sensory learning** — the ant's olfactory circuit (glomeruli → Kenyon cells →
  MBONs) learns *water smells good / alarm smells bad* with sparse, near-disjoint
  codes that don't interfere — a port of `flyputer`'s `sniff.py`.
* **Live terminal animation** — the 180×180 world is drawn with braille dots; ants are
  braille spinner glyphs `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` coloured by state (walking / foraging /
  carrying / drinking / alarmed / dead).
* **Brain-backed or offline** — runs on the real **CRANTb** ant connectome when data
  is available, otherwise on a clearly-labelled offline functional model.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python screen.py            # animated colony: find water
.venv/bin/python screen.py relocate   # animated colony: nest relocation
.venv/bin/python simulate.py --plot   # headless run + ant_colony.png
.venv/bin/python pheromone.py         # olfactory-learning demo
```

## Architecture

| File | Role |
|---|---|
| `antbrain.py` | connectome access layer — live CRANTb edge list `[pre, post, weight]` or offline functional model |
| `pheromone.py` | ant olfactory learning: glomeruli → Kenyon cells → MBONs (sparse codes) |
| `colony.py` | stepable 2D swarm simulation: pheromone fields, recruitment, relocation, metabolism |
| `screen.py` | terminal animation: braille-dot map + spinner ants, coloured by state |
| `simulate.py` | headless scenario runner + optional matplotlib plot |
| `scripts/port_crant.py` | pull the real CRANTb connectome into `data/` |

## Real ant brain data (CRANTb)

The app targets **CRANTb** (Clonal Raider ANT, *Ooceraea biroi*) — the first complete
ant brain connectome after the fruit fly. It is an in-progress, token-gated dataset,
so the repo ships an offline functional model by default:

```bash
.venv/bin/pip install crantpy
.venv/bin/python -c "import crantpy as cp; cp.generate_cave_token(save=True)"
.venv/bin/python scripts/port_crant.py   # writes data/crant_connectivity.feather
```

Once present, `antbrain.py` switches automatically to real CRANTb wiring.

## Notes

* The behavioural model is a credible *mechanism*, not biological fidelity for any
  one species; numbers are order-of-magnitude.
* DNA of the design: [FlyWire](https://flywire.ai) fruit-fly connectome, `crantpy`,
  `fafbseg-py` (Philipp Schlegel).
* Data files, `.venv/`, and generated images are git-ignored and never committed.