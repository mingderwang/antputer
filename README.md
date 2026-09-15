# antputer 🐜

A colony of ants crawling on a 2D map, **talking by pheromone**, **learning which
smells mean water**, and **deciding as a group** where to go — to find water, or to
pick up the nest and relocate (搬家).

It is a port of the approach in
[`flyputer`](../googlebrain/flyputer/) (a fruit-fly **FlyWire** brain simulation) to
the ant: same idea — drive behaviour from a real connectome when you can, fall back to
a labelled functional model when you can't — plus a live **terminal animation**.

> The ant connectome it targets is **CRANTb** (Clonal Raider ANT, *Ooceraea biroi*),
> the first complete ant brain wiring after the fruit fly. It is an in-progress,
> token-gated research dataset, so this repo ships an **offline functional model** by
> default and a one-command path to the real data.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# animated 2D ant colony in your terminal (braille ants 🐜)
.venv/bin/python screen.py                 # water foraging
.venv/bin/python screen.py relocate        # nest-relocation scenario
.venv/bin/python screen.py --simple        # ASCII fallback, no braille

# headless metrics
.venv/bin/python simulate.py               # find_water
.venv/bin/python simulate.py relocate --plot   # saves ant_colony.png

# the ant's sense of smell (sparse Kenyon-cell learning)
.venv/bin/python pheromone.py
```

## What you'll see

`screen.py` draws two layers:

* **Background** — braille dots (2×4 sub-dots per cell) for the 180×180 world:
  green dotted **pheromone trails**, blue **water**, gold **nest**, red **alarm**.
* **Ants** — the braille spinner glyphs `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`, one frame per tick so they look
  like they are walking, coloured by state:

  | glyph/colour | state | meaning |
  |---|---|---|
  | grey ⠋ | walking | exploring, naive |
  | green ⠋ | foraging | knows a source, commuting to it |
  | cyan ⠋ | carrying | hauling water home, laying trail |
  | blue ⠋ | drinking | at the water, refilling |
  | red ⠋ | alarmed | fleeing a dried-out source |
  | ✕ | dead | starved (energy hit zero) |

The status bar shows live counts and total water deliveries.

## How the colony decides

1. **Smell.** Each ant noses the pheromone + water-scent field at its feet (short
   range — ants can't smell water across the map).
2. **Recruit.** An ant that finds water drinks, picks up a load, runs home laying a
   strong trail, and on arrival **teaches nearby naive nestmates where the water is**
   (modelled on antennation / tandem running).
3. **Positive feedback.** Learners set off for the source; their own trail reinforces
   the route → a visible **ant column** forms and the whole colony streams to water.
4. **Relocate.** If a source **dries out**, it releases **alarm pheromone**; committed
   foragers forget it, scouts spread, and once enough deliveries point at the surviving
   source the colony **re-anchors its nest** there.
5. **Metabolism.** Ants spend energy; drinking restores it; starving ants die — so the
   map has live / drinking / carrying / dead states.

## Real ant brain data (CRANTb)

```bash
.venv/bin/pip install crantpy     # or from the crantpy GitHub repo
.venv/bin/python -c "import crantpy as cp; cp.generate_cave_token(save=True)"
.venv/bin/python scripts/port_crant.py
```

This writes `data/crant_connectivity.feather` (`pre, post, weight`) and
`data/crant_neuron_annotations.tsv`; `antbrain.py` then uses **real CRANTb wiring**
instead of the offline model. Without the snapshot, the app runs the offline
functional model — clearly labelled everywhere as *not measured data*.

## Files

| file | role |
|---|---|
| `antbrain.py` | connectome access layer (live CRANTb ↔ offline functional model) |
| `pheromone.py` | ant olfactory learning: glomeruli → Kenyon cells → MBONs (sparse codes) |
| `colony.py` | the stepable 2D swarm simulation (pheromone, recruitment, relocation) |
| `screen.py` | live terminal animation (braille map + spinner ants) |
| `simulate.py` | headless scenario runner + optional matplotlib plot |
| `scripts/port_crant.py` | pull the real CRANTb connectome into `data/` |

## Data & credits

* CRANTb / CRANTpy — Clonal Raider Ant Brain project (Lee, Kronauer, Haberkern labs
  and the CRANTb community). `crantpy` is derived from `fafbseg-py` (Philipp Schlegel)
  and the FlyWire project.
* The behavioural model is a **credible mechanism, not biological fidelity** for any
  one species. Numbers are order-of-magnitude, chosen for legibility.
* No large data files are committed; fetch them with the scripts above.
