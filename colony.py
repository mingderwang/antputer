"""
colony.py — many ants crawling on a 2D map, talking by pheromone, and deciding as a
colony where to go (find water) or whether to move the nest (relocate).

The behavioural model is deliberately simple and mechanistic, in the same spirit as
`flyputer`'s toy LIF sim:

  * each ant noses the pheromone field at its feet + a compass bearing home;
  * pheromones evaporate/diffuse like a real ant trail (recruitment) pheromone;
  * a water source = a patch of "water" scent on the map — the smell the ant's
    mushroom body has learned to like (see pheromone.py);
  * when an ant finds water it marches home laying a heavy trail, then recruits the
    neighbours it passes at the nest;
  * recruitment is the COLONY DECISION LOOP: more visits -> stronger trail ->
    more ants follow -> the group converges on the water;
  * if a water source dries out, alarm pheromone pushes ants away and the nest
    marker re-anchors on the richest surviving source (relocation / 搬家);
  * ants spend energy; drinking restores it, starving kills them — so the display
    can show who is live / drinking / carrying / fleeing / dead.

The `Colony` class is stepable so a renderer (screen.py) or an LLM agent (agent.py)
can drive it one tick at a time.

Scenario modes:  "find_water"  (default) |  "relocate"
"""
from __future__ import annotations

import dataclasses
import math
import random
from typing import Callable

import numpy as np

import antbrain
import pheromone
from pheromone import AntMemory

# --- tunables --------------------------------------------------------------- #
W = H = 180                 # world size (cells)
N_ANTS = 60                 # colony size
NEST = (W // 2, H // 2)     # nest coordinate
ANT_SPEED = 1.3             # cells / tick
EVAP = 0.990                # pheromone persistence per tick
DIFFUSE = 0.02              # diffusion toward neighbours (keeps the trail a path)
TRAIL_STRENGTH = 3.0        # trail laid per step while returning home
FOLLOWER_TRAIL = 0.25       # trail laid by an ant that is following a trail (feedback)
TRAIL_CAP = 50.0            # soft cap — high enough that the path becomes a ridge
RECRUIT_STRENGTH = 8.0      # deposit when an ant announces the find at the nest
RECRUIT_PROB = 0.5          # chance a naive nestmate is taught the source location
NEST_RADIUS = 6             # how close counts as "home"
FOLLOW_THRESHOLD = 0.01     # trail strength at which an ant starts following
SCENT_THRESHOLD = 0.12      # water-scent strength an ant can smell (short range)
DRINK_UNTIL = 0.85          # ants top up to this energy before carrying water home
MAX_TURN = 0.5              # rad/tick — ants can't turn on a dime (keeps them on-trail)
ENERGY_DECAY = 0.00025      # energy lost per tick while alive
ENERGY_PER_DRINK = 0.06     # energy restored per tick spent on water

# water sources: (x, y, radius). capacity is set per scenario.
WATER_SOURCES = [
    (W // 2 + 58, H // 2 - 40, 8),
    (W // 2 - 60, H // 2 + 46, 7),
]

# braille "walking legs" spinner — cycling this each tick makes an ant look alive
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
DEAD_GLYPH = "✕"


@dataclasses.dataclass
class Water:
    x: int
    y: int
    r: int
    cap: float
    anhydrous: bool = False


class Ant:
    """One worker. Senses the world, decides a heading, moves, drops scent, ages."""

    __slots__ = ("x", "y", "heading", "memory", "has_water", "found", "known",
                 "alarm_timer", "energy", "dead", "age", "rng")

    def __init__(self, rng: random.Random):
        self.x, self.y = float(NEST[0]), float(NEST[1])
        self.heading = rng.uniform(0, 2 * math.pi)
        self.memory = AntMemory()
        pheromone.train_ant(self.memory)     # learn: water good, home good, alarm bad
        self.has_water = False
        self.found = None                    # source it is currently hauling from
        self.known = None                    # source it has LEARNED (path integration)
        self.alarm_timer = 0
        self.energy = 1.0
        self.dead = False
        self.age = 0
        self.rng = rng

    # -- status (for the renderer / LLM) ------------------------------------ #
    def status(self, on_water: bool) -> str:
        if self.dead:
            return "dead"
        if on_water:
            return "drinking"
        if self.has_water:
            return "carrying"
        if self.known is not None:
            return "foraging"
        if self.alarm_timer > 0:
            return "alarmed"
        return "walking"

    # -- controller: smell -> action ---------------------------------------- #
    def decide(self, trail, alarm, water, nest, sources) -> str:
        rng = self.rng
        wx, wy = int(self.x), int(self.y)

        # 1) standing on a live water source? drink, learn it, then load up.
        on_i = None
        for i, s in enumerate(sources):
            if not s.anhydrous and (wx - s.x) ** 2 + (wy - s.y) ** 2 <= (s.r + 3) ** 2:
                on_i = i
                break
        if on_i is not None:
            self.known = on_i                              # path-integration memory
            self.found = on_i
            self.energy = min(1.0, self.energy + ENERGY_PER_DRINK)
            if not self.has_water and self.energy >= DRINK_UNTIL:
                self.has_water = True
                sources[on_i].cap -= 1.0
            if not self.has_water:                         # still topping up: linger
                self.heading = self._wander(rng)
                return "drinking"

        if self.dead:
            return "dead"

        # 2) carrying water -> march home, laying a heavy trail
        if self.has_water:
            if (wx - NEST[0]) ** 2 + (wy - NEST[1]) ** 2 <= NEST_RADIUS ** 2:
                self.heading = self._wander(rng)
                return "deliver"
            self.heading = self._turn_toward(self._steer_to(NEST[0], NEST[1], rng, wander=0.2))
            return "return_home"

        # 2b) forget a source that has dried up
        if self.known is not None and sources[self.known].anhydrous:
            self.known = None

        # 3) strong alarm -> flee (naive ants scatter; committed foragers keep going)
        if float(alarm[wy, wx]) > 0.05 and self.known is None:
            self.heading = self._turn_toward(self._best_away(alarm))
            self.alarm_timer = 20
            return "flee_alarm"

        # 4) committed forager -> commute straight to the source it knows
        if self.known is not None:
            s = sources[self.known]
            self.heading = self._turn_toward(self._steer_to(s.x, s.y, rng, wander=0.15))
            return "commute"

        # 5) a trail at my feet -> follow it (recruitment), may lead to water
        if trail[wy, wx] > FOLLOW_THRESHOLD:
            self.heading = self._turn_toward(self._follow_grade(trail))
            return "follow_trail"

        # 6) nose the water scent directly (short range)
        if water[wy, wx] > SCENT_THRESHOLD:
            self.heading = self._turn_toward(self._follow_grade(water))
            return "scent_water"

        # 7) default: explore
        self.heading = self._wander(rng)
        return "explore"

    # -- steering ----------------------------------------------------------- #
    def _turn_toward(self, want, max_turn=MAX_TURN):
        """Limited turn rate: an ant can't spin in place, which keeps it locked on
        a thin pheromone trail instead of jittering off it."""
        diff = (want - self.heading + math.pi) % (2 * math.pi) - math.pi
        return self.heading + max(-max_turn, min(max_turn, diff))

    def _steer_to(self, tx, ty, rng, wander=0.0):
        want = math.atan2(ty - self.y, tx - self.x)
        return want + rng.uniform(-wander, wander) if wander else want

    def _follow_grade(self, f):
        x, y = int(self.x), int(self.y)
        o = 2
        gx = f[y, min(W - 1, x + o)] - f[y, max(0, x - o)]
        gy = f[min(H - 1, y + o), x] - f[max(0, y - o), x]
        if gx * gx + gy * gy < 1e-9:
            return self.heading
        return math.atan2(gy, gx)

    def _best_away(self, f):
        x, y = int(self.x), int(self.y)
        best, besta = 1e9, self.heading
        for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            sx = min(W - 1, max(0, x + int(math.cos(ang) * 3)))
            sy = min(H - 1, max(0, y + int(math.sin(ang) * 3)))
            if f[sy, sx] < best:
                best, besta = f[sy, sx], ang
        return besta

    def _wander(self, rng):
        if self.alarm_timer > 0:
            self.alarm_timer -= 1
            return self.heading + rng.uniform(-0.3, 0.3)
        return self.heading + rng.uniform(-0.8, 0.8)

    # -- movement, metabolism & parade -------------------------------------- #
    def move(self, speed: float):
        if self.dead:
            return
        self.x = min(W - 1, max(0.0, self.x + math.cos(self.heading) * speed))
        self.y = min(H - 1, max(0.0, self.y + math.sin(self.heading) * speed))

    def metabolize(self):
        if self.dead:
            return
        self.age += 1
        self.energy -= ENERGY_DECAY
        if self.energy <= 0:
            self.energy = 0.0
            self.dead = True
            self.has_water = False

    def drop_scent(self, trail: np.ndarray, action: str):
        if self.dead:
            return
        x, y = int(self.x), int(self.y)
        if self.has_water:                    # a laden ant lays a strong trail
            trail[y, x] = min(TRAIL_CAP, trail[y, x] + TRAIL_STRENGTH)
        elif action == "follow_trail":        # followers reinforce the column
            trail[y, x] = min(TRAIL_CAP, trail[y, x] + FOLLOWER_TRAIL)


def spawn_ants(rng: random.Random) -> list[Ant]:
    return [Ant(rng) for _ in range(N_ANTS)]


def build_worldfields():
    """trail, alarm, water-scent, home-scent grids."""
    trail = np.zeros((H, W))
    alarm = np.zeros((H, W))
    water = np.zeros((H, W))
    nest = np.zeros((H, W))
    for (sx, sy, r) in WATER_SOURCES:
        for dy in range(-W, W + 1):
            for dx in range(-W, W + 1):
                y, x = sy + dy, sx + dx
                if not (0 <= x < W and 0 <= y < H):
                    continue
                d2 = dx * dx + dy * dy
                d = math.sqrt(d2)
                if d <= r:
                    water[y, x] = 1.0                             # the source itself
                else:
                    # moisture smell fades fast -> only nearby ants smell water
                    water[y, x] = max(water[y, x], math.exp(-(d - r) / 8.0))
    _paint_home(nest, NEST)
    return trail, alarm, water, nest


def _paint_home(nest: np.ndarray, center):
    cx, cy = center
    for dy in range(-14, 15):
        for dx in range(-14, 15):
            d2 = dx * dx + dy * dy
            if d2 <= 196:
                nest[cy + dy, cx + dx] = math.exp(-d2 / 40.0)


def diffuse_evap(f: np.ndarray, evap: float, diffus: float):
    f[:] = (0.25 * np.roll(f, 1, 0) + 0.25 * np.roll(f, -1, 0)
            + 0.25 * np.roll(f, 1, 1) + 0.25 * np.roll(f, -1, 1)) * (1 - diffus) + f * diffus
    f *= evap


class Colony:
    """Stepable colony state. Call `step()` repeatedly; inspect `frame()` to draw."""

    def __init__(self, mode: str = "find_water", seed: int = 7, ticks: int = 2200):
        self.mode = mode
        self.ticks = ticks
        self.t = 0
        self.rng = random.Random(seed)
        self.ants = spawn_ants(self.rng)
        self.trail, self.alarm, self.water, self.nest = build_worldfields()
        caps = [10 ** 9, 10 ** 9] if mode != "relocate" else [40, 10 ** 9]
        self.sources = [Water(x, y, r, c) for (x, y, r), c in zip(WATER_SOURCES, caps)]
        self.visits = [0, 0]
        self.first_find_tick = -1
        self.quarter_tick = -1
        self.reanchor_tick = -1
        self.moved_nest = list(NEST)
        self.at_source: list[int] = []

    # -- simulation --------------------------------------------------------- #
    def step(self):
        trail, alarm, water, nest, ants = self.trail, self.alarm, self.water, self.nest, self.ants
        diffuse_evap(trail, EVAP, DIFFUSE)
        diffuse_evap(alarm, EVAP, DIFFUSE)

        for a in ants:
            action = a.decide(trail, alarm, water, nest, self.sources)

            if action == "deliver":
                src = a.found if a.found is not None else a.known
                if src is not None:
                    self.visits[src] += 1
                if self.first_find_tick < 0:
                    self.first_find_tick = self.t
                # announce at the nest -> recruit naive nestmates (tandem-running cue):
                # they learn WHERE the water is (path-integration vector) and set off.
                for da in ants:
                    if da is not a and da.known is None and not da.dead and \
                       (da.x - a.x) ** 2 + (da.y - a.y) ** 2 < 14 ** 2:
                        if self.rng.random() < RECRUIT_PROB:
                            da.known = src
                            da.heading = math.atan2(a.y - da.y, a.x - da.x)
                trail[int(a.y), int(a.x)] = min(TRAIL_CAP, trail[int(a.y), int(a.x)] + RECRUIT_STRENGTH)
                a.has_water = False
                a.found = None

            a.move(ANT_SPEED)
            a.metabolize()
            a.drop_scent(trail, action)

        for i, s in enumerate(self.sources):          # a source ran dry
            if not s.anhydrous and s.cap <= 0:
                s.anhydrous = True
                alarm[max(0, s.y - 4):s.y + 5, max(0, s.x - 4):s.x + 5] = 0.9
                for a2 in ants:
                    if a2.found == i:
                        a2.found = None
                        a2.has_water = False

        if self.mode == "relocate":                   # nest drifts to richest source
            alive = [(i, s) for i, s in enumerate(self.sources) if not s.anhydrous]
            if alive:
                bi, best = max(alive, key=lambda pr: self.visits[pr[0]])
                if self.visits[bi] > 0 and \
                   (best.x - self.moved_nest[0]) ** 2 + (best.y - self.moved_nest[1]) ** 2 > 20 ** 2:
                    self.moved_nest = [best.x, best.y]
                    self.reanchor_tick = self.t
                    nest[:] = 0.0
                    _paint_home(nest, (best.x, best.y))

        here = sum(1 for a in ants
                   if any(not s.anhydrous and (a.x - s.x) ** 2 + (a.y - s.y) ** 2 <= (s.r + 6) ** 2
                          for s in self.sources))
        self.at_source.append(here)
        if self.quarter_tick < 0 and here >= N_ANTS * 0.25:
            self.quarter_tick = self.t
        self.t += 1
        return self

    def ant_status(self):
        """Per-ant (x, y, status) for rendering/inspection."""
        out = []
        for a in self.ants:
            on_water = any(not s.anhydrous and
                           (int(a.x) - s.x) ** 2 + (int(a.y) - s.y) ** 2 <= (s.r + 3) ** 2
                           for s in self.sources)
            out.append((a.x, a.y, a.status(on_water)))
        return out

    def frame(self) -> dict:
        return {"t": self.t, "mode": self.mode,
                "trail": self.trail, "alarm": self.alarm, "water": self.water,
                "nest": self.nest, "ants": self.ants, "sources": self.sources,
                "statuses": self.ant_status(),
                "deliveries": sum(self.visits)}

    # -- batch run ---------------------------------------------------------- #
    def run(self, report: Callable[[dict], None] | None = None) -> dict:
        for _ in range(self.ticks):
            self.step()
            if report is not None and self.t % 40 == 0:
                report(self.frame())
        alive = sum(1 for a in self.ants if not a.dead)
        n_reached = self.at_source[-1] if self.at_source else 0
        return {
            "mode": self.mode,
            "first_find_tick": self.first_find_tick,
            "quarter_tick": self.quarter_tick,
            "at_source_final": n_reached,
            "at_source_frac": n_reached / N_ANTS,
            "peak_at_source": max(self.at_source) if self.at_source else 0,
            "deliveries": sum(self.visits),
            "visited": self.visits,
            "alive": alive,
            "alive_frac": alive / N_ANTS,
            "reanchor_tick": self.reanchor_tick,
            "moved_nest": self.moved_nest,
            "brain_mode": antbrain.mode(),
            "waveform": self.at_source,
        }


def run_scenario(mode: str = "find_water", ticks: int = 2200, seed: int = 7,
                 report: Callable[[dict], None] | None = None):
    """Convenience wrapper: build a colony and run it. See Colony.run()."""
    return Colony(mode=mode, seed=seed, ticks=ticks).run(report)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "find_water"
    print("brain:", antbrain.mode())
    r = run_scenario(mode)
    for k, v in r.items():
        if k != "waveform":
            print(f"  {k}: {v}")
    print("  peak at_source:", r["peak_at_source"])