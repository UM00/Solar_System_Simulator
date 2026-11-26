"""
experiments.py - the "what if?" laboratory.

WHAT IT DOES
------------
Lets you change the real Solar System and watch the consequences:
    - make any body heavier or lighter (or switch its gravity off)
    - make gravity itself stronger or weaker (change G)
    - stop a planet, halve or double its speed, or reverse it
    - one-click PRESETS for classic questions ("what if Earth stopped?")
"Back to the real Solar System" restores everything exactly as it was when
the experiment started.

While an experiment is on, the simulation no longer matches NASA's data.
That's the point. The HUD shows a badge so you never confuse the two.

WHY EXPERIMENTS NEED DIFFERENT PHYSICS MACHINERY
------------------------------------------------
The normal simulation (simulation.py) uses one fixed time step, chosen for
the real Solar System (Io's 1.77-day orbit sets ~25 minutes). Experiments
break that assumption:

1. ADAPTIVE TIME STEPS. Make Earth 318x heavier and the Moon dives at it at
   high speed; stop Earth and it falls into the Sun. A fixed 25-minute step
   would leap right past such close passes and give nonsense. So before
   every step we look at every pair (body, puller) and estimate two times:
       crossing time   r / v           (how soon they could meet)
       orbit time      sqrt(r^3 / GM)  (Kepler's 3rd law: period / 2 pi)
   and take ETA = 0.06 of the shortest. That's about 100 steps per orbit,
   the same rule of thumb as before, but now applied every single step
   (Chapter 6 of "Moving Planets Around": variable step-size integration).
   Near a close pass the steps shrink automatically. The animation slows
   down there: that's the computer taking care.

2. COLLISIONS. The physics treats bodies as points, so two bodies that
   should hit would pass through each other's centres, where 1/r^2 blows up
   and flings them out at absurd speeds. Instead, when two bodies touch
   (using their REAL radii) they MERGE: the heavier one absorbs the lighter.
   Total momentum is kept exactly (Newton's third law):
       v_new = (m1 v1 + m2 v2) / (m1 + m2)
   A massless asteroid that touches a body is simply destroyed.

The asteroid belt still uses its once-a-day slow lane, but each day is cut
into smaller pieces if any asteroid has a close pass. The pullers' positions
during the day are estimated in a straight line between start and end.
"""

import math

import numpy as np

from physics import G, accelerations, accelerations_from
from simulation import SLOW_LANE_STEP

ETA = 0.06                 # step = ETA x shortest crossing/orbit time (~100 steps per orbit)
MAX_SUBSTEPS = 600         # per call, so one frame can't freeze the program
MAX_BELT_SUBSTEPS = 50     # per belt day

# (label, what to expect, what to do). Every preset starts from the real Solar System.
PRESETS = [
    ("Halve the Sun's mass",
     "Earth's orbital speed sqrt(GM/r) is now EXACTLY the escape speed sqrt(2 G (M/2) / r): "
     "Earth and the others are barely bound and drift off into space.",
     lambda lab: lab.set_mass_factor("Sun", 0.5)),
    ("Double the Sun's mass",
     "Every planet is now too slow for a circle. It falls inward on an ellipse whose closest point "
     "is 1/3 of its old distance (vis-viva: a = 2r/3).",
     lambda lab: lab.set_mass_factor("Sun", 2.0)),
    ("Switch off the Sun's gravity",
     "Newton's first law: with no force, every planet flies off in a straight line at its current speed.",
     lambda lab: lab.set_mass_factor("Sun", 0.0)),
    ("Stop Earth in its orbit",
     "With no sideways speed, Earth falls straight into the Sun, in about 65 days (free-fall time = year / (4 sqrt 2)).",
     lambda lab: lab.change_speed("Earth", "stop")),
    ("Make Earth as heavy as Jupiter (x318)",
     "The Moon is far too slow for the new pull. It plunges toward Earth on a narrow ellipse and crashes.",
     lambda lab: lab.set_mass_factor("Earth", 317.8)),
    ("Turn Jupiter into a small star (x80)",
     "80 Jupiter masses is about the lightest a star can be. Its moons, suddenly far too slow for the new pull, "
     "fall into it within hours; the asteroid belt and the inner planets feel a much stronger tug.",
     lambda lab: lab.set_mass_factor("Jupiter", 80.0)),
    ("Double the strength of gravity (G x2)",
     "Same as doubling EVERY mass at once: all orbits fall inward, like doubling the Sun's mass.",
     lambda lab: lab.set_g_scale(2.0)),
]

SPEED_CHANGES = {"stop": 0.0, "half": 0.5, "double": 2.0, "reverse": -1.0}


class ExperimentLab:
    def __init__(self, sim):
        self.sim = sim
        self.active = False
        self.g_scale = 1.0
        self.mass_factor = {}       # name -> factor (only the changed ones)
        self.speed_notes = []       # e.g. "Earth stopped"
        self.events = []            # e.g. "Day 65: Earth hit the Sun and merged"
        self._snapshot = None
        self._slow_elapsed = 0.0
        self._belt_start_positions = None

        bodies = sim.bodies
        self.radii = np.array([b.radius for b in bodies])
        self.real_masses = sim.masses.copy()
        self.is_test = np.array([b.is_test_particle for b in bodies])
        self.in_swarm = np.array([bool(b.swarm) for b in bodies])
        self.sun = int(np.argmax(sim.masses))
        self.reference = [sim.names.index(b.parent) if b.parent in sim.names else self.sun for b in bodies]

    # ---------------------------------------------------------- start/stop
    def begin(self):
        """First change: remember the real state so it can be restored exactly."""
        if self.active:
            return
        s = self.sim
        self._snapshot = {k: getattr(s, k).copy() for k in
                          ("positions", "velocities", "masses", "gravity_masses", "acc", "alive")}
        self._snapshot.update(time=s.time, steps=s._fast_steps_done, lag=s.slow_lag)
        self._slow_elapsed = s.slow_lag
        self._belt_start_positions = s.positions.copy()
        self.active = True

    def reset(self):
        """Back to the real Solar System, exactly as it was when the experiment began."""
        if not self.active:
            return
        s, snap = self.sim, self._snapshot
        for k in ("positions", "velocities", "masses", "gravity_masses", "acc", "alive"):
            setattr(s, k, snap[k].copy())
        s.time, s._fast_steps_done, s.slow_lag = snap["time"], snap["steps"], snap["lag"]
        self.active = False
        self.g_scale = 1.0
        self.mass_factor.clear()
        self.speed_notes.clear()
        self.events.clear()

    # ------------------------------------------------------------- changes
    def index(self, name):
        return self.sim.names.index(name) if name in self.sim.names else None

    def set_mass_factor(self, name, factor):
        """Mass = real mass x factor. 0 switches the body's gravity off."""
        i = self.index(name)
        if i is None:
            return False
        self.begin()
        s = self.sim
        s.masses[i] = self.real_masses[i] * factor
        s.gravity_masses[i] = 0.0 if (self.is_test[i] or not s.alive[i]) else s.masses[i]
        if factor == 1.0:
            self.mass_factor.pop(name, None)
        else:
            self.mass_factor[name] = factor
        self._refresh_accelerations()
        return True

    def set_g_scale(self, factor):
        """Make gravity itself stronger or weaker: G -> G x factor."""
        self.begin()
        self.g_scale = factor
        self._refresh_accelerations()

    def change_speed(self, name, how):
        """
        Change a body's speed RELATIVE TO WHAT IT ORBITS (its planet for moons,
        otherwise the Sun): "stop", "half", "double" or "reverse".
        """
        i = self.index(name)
        if i is None:
            return False
        self.begin()
        s, j = self.sim, self.reference[i]
        if j == i:                                       # the Sun itself: relative to the centre of mass
            s.velocities[i] *= SPEED_CHANGES[how]
        else:
            s.velocities[i] = s.velocities[j] + SPEED_CHANGES[how] * (s.velocities[i] - s.velocities[j])
        self.speed_notes.append({"stop": f"{name} stopped", "half": f"{name} half speed",
                                 "double": f"{name} double speed", "reverse": f"{name} reversed"}[how])
        return True

    def apply_preset(self, k):
        """Start from the real Solar System, then apply preset number k."""
        self.reset()
        label, _note, action = PRESETS[k]
        return action(self) is not False

    def days(self):
        """Simulated days since the experiment started."""
        return (self.sim.time - self._snapshot["time"]) / 86400 if self.active else 0.0

    def describe(self):
        parts = [f"{n} mass x{f:g}" for n, f in self.mass_factor.items()]
        if self.g_scale != 1.0:
            parts.append(f"G x{self.g_scale:g}")
        parts += self.speed_notes
        return ", ".join(parts) if parts else "no changes yet"

    def _refresh_accelerations(self):
        s = self.sim
        s.acc = self.g_scale * accelerations(s.positions, s.gravity_masses * s.alive)

    # ------------------------------------------------------------- physics
    def advance(self, total):
        """Move the experiment forward by `total` seconds with adaptive steps. Returns the time covered."""
        s = self.sim
        done, n = 0.0, 0
        while total - done > 1e-6 and n < MAX_SUBSTEPS:
            f = np.flatnonzero(s.alive & ~self.in_swarm)
            h = min(total - done, s.dt, self._safe_step(f))
            gm = s.gravity_masses[f]
            accel = lambda pos: self.g_scale * accelerations(pos, gm)
            pos, vel, acc = s.integrator(s.positions[f], s.velocities[f], s.acc[f], accel, h)
            s.positions[f], s.velocities[f], s.acc[f] = pos, vel, acc
            s.time += h
            done += h
            n += 1
            self._handle_collisions(f)
            if s.slow.size:
                self._slow_elapsed += h
                if self._slow_elapsed >= SLOW_LANE_STEP:
                    self._step_belt()
        s.slow_lag = self._slow_elapsed
        return done

    def _pair_times(self, targets, sources):
        """ETA x the shortest crossing / orbit time between targets and sources."""
        s = self.sim
        if targets.size == 0 or sources.size == 0:
            return math.inf
        dp = s.positions[targets][:, None, :] - s.positions[sources][None, :, :]
        dv = s.velocities[targets][:, None, :] - s.velocities[sources][None, :, :]
        r = np.linalg.norm(dp, axis=2)
        v = np.linalg.norm(dv, axis=2)
        mu = G * self.g_scale * (s.gravity_masses[targets][:, None] + s.gravity_masses[sources][None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cross = np.where((r > 0) & (v > 0), r / v, np.inf)
            t_orbit = np.where((r > 0) & (mu > 0), np.sqrt(r**3 / mu), np.inf)
        return ETA * min(t_cross.min(), t_orbit.min())

    def _safe_step(self, f):
        sources = np.flatnonzero((self.sim.gravity_masses > 0) & self.sim.alive)
        return self._pair_times(f, sources)

    def _handle_collisions(self, f):
        s = self.sim
        p = s.positions[f]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        touching = np.argwhere(np.triu(d < (self.radii[f][:, None] + self.radii[f][None, :])))
        merged = False
        for a, b in touching:
            i, j = f[a], f[b]
            if not (s.alive[i] and s.alive[j]) or (s.gravity_masses[i] == 0 and s.gravity_masses[j] == 0):
                continue
            big, small = (i, j) if s.masses[i] >= s.masses[j] else (j, i)
            m_big, m_small = s.gravity_masses[big], s.gravity_masses[small]
            if m_small > 0:   # momentum is conserved: (m1 v1 + m2 v2) / (m1 + m2)
                s.velocities[big] = (m_big * s.velocities[big] + m_small * s.velocities[small]) / (m_big + m_small)
                s.masses[big] += s.masses[small]
                s.gravity_masses[big] += m_small
            s.alive[small] = False
            s.gravity_masses[small] = 0.0
            s.velocities[small] = 0.0
            how = "and merged" if m_small > 0 else "and was destroyed"
            self.events.append(f"after {self.days():.1f} days: {s.names[small]} hit {s.names[big]} {how}")
            merged = True
        if merged:
            self._refresh_accelerations()

    def _step_belt(self):
        """
        One belt "day" (Section 16 of the guide), cut into smaller pieces if
        any asteroid has a close pass. Pullers' positions during the day are
        interpolated in a straight line between the start and the end.
        """
        s = self.sim
        h, self._slow_elapsed = self._slow_elapsed, 0.0
        belt = s.slow[s.alive[s.slow]]
        sources = np.flatnonzero((s.gravity_masses > 0) & s.alive)
        if belt.size:
            safe = self._pair_times(belt, sources)
            k = int(min(MAX_BELT_SUBSTEPS, max(1, math.ceil(h / safe))))
            hk = h / k
            start, end = self._belt_start_positions[sources], s.positions[sources]
            gm = s.gravity_masses[sources]
            pos, vel, acc = s.positions[belt], s.velocities[belt], s.acc[belt]
            for q in range(k):
                half = vel + 0.5 * hk * acc
                pos = pos + hk * half
                src_pos = start + (end - start) * ((q + 1) / k)
                acc = self.g_scale * accelerations_from(pos, src_pos, gm)
                vel = half + 0.5 * hk * acc
            s.positions[belt], s.velocities[belt], s.acc[belt] = pos, vel, acc
            # Asteroids that fell into a body are destroyed.
            d = np.linalg.norm(pos[:, None, :] - end[None, :, :], axis=2)
            hit = (d < self.radii[sources][None, :]).any(axis=1)
            if hit.any():
                s.alive[belt[hit]] = False
                self.events.append(f"after {self.days():.1f} days: {hit.sum()} asteroid(s) fell into a planet or the Sun")
        self._belt_start_positions = s.positions.copy()
