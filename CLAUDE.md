# antputer — notes for Claude Code

A local app ported from `flyputer` (fruit-fly FlyWire brain simulation) over to the
**ant**: a colony of many ants crawling on a 2D map, communicating via pheromones,
**learning** which smells mean water, and collectively deciding to **find water** or
**relocate the nest** (搬家). Optionally backed by the **CRANTb** ant connectome
(*Ooceraea biroi*).

## Rules
- **Git commits: NEVER add a "Co-Authored-By" trailer or any AI attribution to
  commit messages.** Commit as the user only.
- Never commit large data files (`data/*.feather`, `data/*.tsv`, `.venv/`, caches,
  `*.png`, `*.gif`), or generated artifacts — they're in `.gitignore`. Data is fetched
  via `scripts/port_crant.py`.
- CRANTb is an in-progress research dataset (needs a CAVE token). When unavailable,
  the code transparently falls back to a **bundled functional brain model** clearly
  flagged, NOT measured data.
- Keep the framing factual: we simulate a credible *mechanism*, we don't claim
  biological fidelity for any specific species.

## Layout
- `antbrain.py` — connectome access layer: live CRANTb edge list `[pre, post, weight]`
  when available, else the bundled functional model. Mirrors `flysim.py`'s API.
- `pheromone.py` — ant olfactory learning: antennal-lobe glomeruli -> Kenyon cells ->
  MBONs (sparse, near-disjoint codes; a port of `flyputer/sniff.py`). Ants learn
  water=good, alarm=bad, without forgetting old smells.
- `colony.py` — the stepable 2D multi-ant swarm simulation: crawling, pheromone
  fields, tandem-recruitment, quorum-based nest relocation, metabolism (energy/live/dead).
- `screen.py` — live terminal animation: braille-dot background (2×4 sub-dots/cell)
  for trails/water/nest/alarm, plus braille spinner ants `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` coloured by
  state (walking / foraging / carrying / drinking / alarmed / dead).
- `simulate.py` — headless batch runner, prints colony metrics, optional matplotlib plot.
- `scripts/port_crant.py` — pull real CRANTb connectivity into `data/`.

## Run
- `.venv/bin/python screen.py`            # animated terminal colony (default find_water)
- `.venv/bin/python screen.py relocate    # nest-relocation animation
- `.venv/bin/python screen.py --simple    # ASCII fallback, no braille sub-dots
- `.venv/bin/python simulate.py`           # batch metrics (find_water)
- `.venv/bin/python simulate.py relocate --plot   # saves ant_colony.png
- `.venv/bin/python pheromone.py`          # olfactory-learning demo

## Data modes
`antbrain.py` runs in one of two modes, chosen automatically:
1. **LIVE** — `data/crant_connectivity.feather` exists (ported via `scripts/port_crant.py`)
   -> real CRANTb edge list
2. **OFFLINE** — bundled functional brain model (random sparse wiring, structurally
   plausible sizes; clearly flagged, not measured data)