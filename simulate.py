"""
simulate.py — run an ant colony scenario and print the colony-level result.

    .venv/bin/python simulate.py                 # water foraging (default)
    .venv/bin/python simulate.py relocate        # nest relocation
    .venv/bin/python simulate.py find_water --plot   # also save ant_colony.png

Metrics reported:
  find_water : when the colony first finds water, how many ants converge, and how
               many water deliveries the recruitment loop sustains.
  relocate   : when the source dried, when the colony re-anchored its nest, and
               where it moved to.
"""
from __future__ import annotations

import argparse

import antbrain
import colony as C


def report(mode: str, ticks: int, seed: int, plot: bool = False):
    col = C.Colony(mode, seed=seed, ticks=ticks)
    r = col.run()
    print(f"brain: {antbrain.mode()}\n")
    print(f"scenario          : {r['mode']}")
    print(f"ticks simulated   : {ticks}")
    print(f"first water found : t={r['first_find_tick']} "
          f"({'never' if r['first_find_tick'] < 0 else ''})")
    print(f"peak ants at water: {r['peak_at_source']}/{C.N_ANTS}")
    print(f"ants at water end : {r['at_source_final']}/{C.N_ANTS} ({r['at_source_frac']:.0%})")
    print(f"water deliveries  : {r['deliveries']}  (per source: {r['visited']})")
    print(f"colony alive      : {r['alive']}/{C.N_ANTS} ({r['alive_frac']:.0%})")
    if mode == "relocate":
        print(f"nest re-anchored  : t={r['reanchor_tick']}  ->  {tuple(r['moved_nest'])}")
    if plot:
        _plot(col, r)
    return r


def _plot(col: C.Colony, r: dict, out: str = "ant_colony.png"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    ax1.plot(r["waveform"], color="#2a9d8f")
    ax1.set_title("ants at water over time")
    ax1.set_xlabel("tick")
    ax1.set_ylabel("ants at a source")
    ax1.axhline(C.N_ANTS, ls=":", c="grey", label="whole colony")
    if r["first_find_tick"] >= 0:
        ax1.axvline(r["first_find_tick"], ls="--", c="tab:blue", label="first find")
    if r["reanchor_tick"] >= 0:
        ax1.axvline(r["reanchor_tick"], ls="--", c="tab:red", label="nest moved")
    ax1.legend(fontsize=8)

    ax2.imshow(np.log1p(col.trail), origin="lower", cmap="Greens", alpha=0.7)
    yy, xx = np.mgrid[0:C.H, 0:C.W]
    sm = np.zeros((C.H, C.W), bool)
    for s in col.sources:
        if not s.anhydrous:
            sm |= (xx - s.x) ** 2 + (yy - s.y) ** 2 <= s.r ** 2
    ax2.contour(sm, colors="tab:blue", linewidths=1)
    nx, ny = col.moved_nest if r["mode"] == "relocate" and r["reanchor_tick"] >= 0 else C.NEST
    ax2.plot(nx, ny, marker="s", ms=10, c="gold", mec="k", label="nest")
    color = {"walking": "grey", "foraging": "tab:green", "carrying": "tab:cyan",
             "drinking": "tab:blue", "alarmed": "tab:red", "dead": "black"}
    for (px, py, st) in col.ant_status():
        ax2.plot(px, py, ".", ms=5, color=color.get(st, "k"))
    ax2.set_title("final map (green=trail, blue=water)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nsaved plot -> {out}")


def main():
    ap = argparse.ArgumentParser(description="ant colony simulator")
    ap.add_argument("mode", nargs="?", default="find_water",
                    choices=["find_water", "relocate"])
    ap.add_argument("--ticks", type=int, default=2200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    report(a.mode, a.ticks, a.seed, a.plot)


if __name__ == "__main__":
    main()