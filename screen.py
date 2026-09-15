"""
screen.py — live terminal animation of the ant colony on a 2D screen.

Two layers:
  * BACKGROUND is drawn with braille dots (2x4 sub-dots per terminal cell) so the
    180x180 world renders at a usable resolution: pheromone trails as green dotted
    density, water sources as blue dots, the nest as gold, alarm marks as red.
  * ANTS are drawn on top with the braille "spinner" glyphs (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏). The glyph
    cycles every tick so the ants look like they are walking, and its colour tells
    you their state: walking / foraging / carrying water / drinking / alarmed / dead.

Run:
    .venv/bin/python screen.py                 # water foraging, animated
    .venv/bin/python screen.py relocate        # nest-relocation scenario
    .venv/bin/python screen.py --plain         # no alt-screen (log-friendly)
    .venv/bin/python screen.py --frames 5 --plain --simple   # ASCII-ish fallback
Keys:  Ctrl-C to quit.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter

import numpy as np

import antbrain
import colony as C

SPIN = C.SPIN
DEAD = C.DEAD_GLYPH

RESET = "\033[0m"
ANT_COLOR = {
    "walking": "\033[37m",     # light grey
    "foraging": "\033[92m",    # bright green — committed forager
    "carrying": "\033[96m",    # bright cyan  — hauling water home
    "drinking": "\033[94m",    # bright blue  — at the water
    "alarmed": "\033[91m",     # bright red   — fleeing
    "dead": "\033[90m",        # dim grey
}
TRAIL_COLOR = "\033[32m"
WATER_COLOR = "\033[94m"
NEST_COLOR = "\033[93m"
ALARM_COLOR = "\033[91m"

HIDE, SHOW = "\033[?25l", "\033[?25h"
ALT_ON, ALT_OFF = "\033[?1049h", "\033[?1049l"
HOME, CLEAR = "\033[H", "\033[2J"

# braille dot bit weights for a cell's 2x4 sub-grid
BW = np.array([[0x01, 0x08],
               [0x02, 0x10],
               [0x04, 0x20],
               [0x40, 0x80]], dtype=np.uint8)


def _source_mask(sources):
    yy, xx = np.mgrid[0:C.H, 0:C.W]
    m = np.zeros((C.H, C.W), dtype=bool)
    for s in sources:
        if not s.anhydrous:
            m |= (xx - s.x) ** 2 + (yy - s.y) ** 2 <= s.r ** 2
    return m


def render_braille(colony: C.Colony, cols: int, rows: int, color: bool = True) -> list[str]:
    body = rows - 1
    dot_cols, dot_rows = cols * 2, body * 4
    scale = min(dot_cols / C.W, dot_rows / C.H)
    dw, dh = C.W * scale, C.H * scale
    ox, oy = (dot_cols - dw) / 2.0, (dot_rows - dh) / 2.0

    # world coordinate of every sub-dot
    dc = np.arange(dot_cols)
    dr = np.arange(dot_rows)
    wx = np.clip(((dc - ox) / scale).astype(int), 0, C.W - 1)
    wy = np.clip(((dr - oy) / scale).astype(int), 0, C.H - 1)
    inside = ((((dc - ox) / scale) >= 0) & (((dc - ox) / scale) < C.W))[None, :] & \
             ((((dr - oy) / scale) >= 0) & (((dr - oy) / scale) < C.H))[:, None]

    srcmask = _source_mask(colony.sources)
    wtr = srcmask[wy][:, wx] & inside
    alarmv = colony.alarm[wy][:, wx] * inside
    nestv = colony.nest[wy][:, wx] * inside
    trailv = colony.trail[wy][:, wx] * inside

    # trail is drawn only where it is strong (the well-trodden path), with a
    # checkerboard dither for the medium halo so it reads as a dotted trail.
    tmax = float(colony.trail.max()) or 1.0
    strong = trailv > max(0.8, 0.25 * tmax)
    medium = trailv > max(0.3, 0.10 * tmax)
    dither = ((dr[:, None] + dc[None, :]) % 2 == 0)
    trailfill = strong | (medium & dither)

    filled = wtr | trailfill | (nestv > 0.5) | (alarmv > 0.08)

    # fold sub-dots into (body, 4, cols, 2) blocks -> 8-bit braille mask + feature flags
    blk = filled.reshape(body, 4, cols, 2)
    mask = (blk * BW[None, :, None, :]).sum(axis=(1, 3)).astype(np.uint8)
    wtr_c = wtr.reshape(body, 4, cols, 2).any(axis=(1, 3))
    nest_c = (nestv > 0.5).reshape(body, 4, cols, 2).any(axis=(1, 3))
    alarm_c = (alarmv > 0.08).reshape(body, 4, cols, 2).any(axis=(1, 3))
    trail_c = trailfill.reshape(body, 4, cols, 2).any(axis=(1, 3))

    # colour per cell by priority: water > nest > alarm > trail
    cellcol = np.full((body, cols), RESET, dtype=object)
    cellcol[trail_c] = TRAIL_COLOR
    cellcol[alarm_c] = ALARM_COLOR
    cellcol[nest_c] = NEST_COLOR
    cellcol[wtr_c] = WATER_COLOR
    cellch = np.vectorize(lambda m: chr(0x2800 + int(m)) if m else " ")(mask)

    # overlay ants (spinner glyphs), counting clusters per cell
    acount = np.zeros((body, cols), dtype=int)
    antst = {}
    for (ax, ay, status) in colony.ant_status():
        c = int(ox / 2 + ax * (scale / 2))
        r = int(oy / 4 + ay * (scale / 4))
        if 0 <= r < body and 0 <= c < cols:
            acount[r, c] += 1
            antst[(r, c)] = status

    lines = []
    frame = SPIN[colony.t % len(SPIN)]
    for r in range(body):
        parts, cur = [], None
        for c in range(cols):
            if acount[r, c]:
                ch = DEAD if antst[(r, c)] == "dead" else (frame if acount[r, c] == 1 else "⠿")
                cc = ANT_COLOR.get(antst[(r, c)], RESET)
            else:
                ch, cc = cellch[r, c], cellcol[r, c]
            if color:
                if cc != cur:
                    parts.append(cc)
                    cur = cc
                parts.append(ch)
            else:
                parts.append(ch)
        if color:
            parts.append(RESET)
        lines.append("".join(parts))

    cnt = Counter(s for _, _, s in colony.ant_status())
    alive = sum(v for k, v in cnt.items() if k != "dead")
    bar = (f"{colony.mode} t={colony.t:5d} {alive:2d}/{C.N_ANTS}  "
           f"w{cnt.get('walking',0):2d} f{cnt.get('foraging',0):2d} "
           f"c{cnt.get('carrying',0):2d} d{cnt.get('drinking',0):2d} "
           f"a{cnt.get('alarmed',0):2d} x{cnt.get('dead',0):2d}  "
           f"deliveries {sum(colony.visits):4d}  [{antbrain.mode().split('(')[0].strip()}]")
    lines.append(bar[:cols].ljust(cols))
    return lines


# --- simple (non-braille) fallback ------------------------------------------ #
def _trail_glyph(v):
    if v > 4.0:
        return "▓"
    if v > 1.5:
        return "▒"
    if v > 0.3:
        return "░"
    if v > 0.02:
        return "·"
    return " "


def render_simple(colony: C.Colony, cols: int, rows: int, color: bool = True) -> list[str]:
    W, H = C.W, C.H
    body = rows - 1
    scale = min((cols - 1) / W, (body - 1) / H)
    ox, oy = int((cols - W * scale) / 2), int((body - H * scale) / 2)
    ch = [[" "] * cols for _ in range(body)]
    col = [[RESET] * cols for _ in range(body)]
    srcmask = _source_mask(colony.sources)
    for r in range(body):
        for c in range(cols):
            wx, wy = int((c - ox) / scale), int((r - oy) / scale)
            if not (0 <= wx < W and 0 <= wy < H):
                continue
            if srcmask[wy, wx]:
                ch[r][c], col[r][c] = "≈", WATER_COLOR
            elif colony.nest[wy, wx] > 0.5:
                ch[r][c], col[r][c] = "⌂", NEST_COLOR
            elif colony.alarm[wy, wx] > 0.08:
                ch[r][c], col[r][c] = "✕", ALARM_COLOR
            else:
                g = _trail_glyph(float(colony.trail[wy, wx]))
                if g != " ":
                    ch[r][c], col[r][c] = g, TRAIL_COLOR
    frame = SPIN[colony.t % len(SPIN)]
    for (ax, ay, status) in colony.ant_status():
        c, r = ox + int(ax * scale), oy + int(ay * scale)
        if 0 <= r < body and 0 <= c < cols:
            ch[r][c] = DEAD if status == "dead" else frame
            col[r][c] = ANT_COLOR.get(status, RESET)
    lines = []
    for r in range(body):
        if color:
            parts, cur = [], None
            for c in range(cols):
                if col[r][c] != cur:
                    parts.append(col[r][c]); cur = col[r][c]
                parts.append(ch[r][c])
            parts.append(RESET)
            lines.append("".join(parts))
        else:
            lines.append("".join(ch[r]))
    cnt = Counter(s for _, _, s in colony.ant_status())
    alive = sum(v for k, v in cnt.items() if k != "dead")
    bar = (f"{colony.mode} t={colony.t:5d} {alive:2d}/{C.N_ANTS}  "
           f"deliveries {sum(colony.visits):4d}")
    lines.append(bar[:cols].ljust(cols))
    return lines


def animate(mode, fps, steps, plain, color, max_frames, seed, simple):
    colony = C.Colony(mode, seed=seed)
    render = render_simple if simple else render_braille
    out = sys.stdout
    if not plain:
        out.write(ALT_ON + HIDE + CLEAR)
    try:
        fi = 0
        while True:
            for _ in range(steps):
                colony.step()
            size = shutil.get_terminal_size((100, 34))
            lines = render(colony, size.columns, size.lines, color=color)
            out.write(HOME + "\n".join(lines))
            out.flush()
            fi += 1
            if max_frames is not None and fi >= max_frames:
                break
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        if not plain:
            out.write(SHOW + ALT_OFF)
        out.flush()


def main():
    ap = argparse.ArgumentParser(description="terminal ant colony")
    ap.add_argument("mode", nargs="?", default="find_water",
                    choices=["find_water", "relocate"])
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--simple", action="store_true", help="ASCII fallback, no braille")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    animate(a.mode, a.fps, a.steps, a.plain, not a.no_color, a.frames, a.seed, a.simple)


if __name__ == "__main__":
    main()