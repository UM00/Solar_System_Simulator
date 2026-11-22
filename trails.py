"""
trails.py - orbit trails that show about the last TWO orbits of each body,
or (optionally) the full history.

WHY NOT VPYTHON'S BUILT-IN TRAILS?
----------------------------------
VPython can keep "the last N trail points", but N points cover very different
amounts of TIME depending on the animation speed. At 10 days per second Io
circles Jupiter in about 10 frames, so 360 points would show ~36 orbits: a
messy scribble. We want "2 orbits" whatever the speed, so each trail point is
stored with the SIMULATED TIME it was recorded. Points older than
2 x (that body's orbital period) are removed.

WHERE THE PERIOD COMES FROM: KEPLER'S THIRD LAW
-----------------------------------------------
physics.orbital_period() turns a body's position and velocity RELATIVE TO
WHAT IT ORBITS into its period (vis-viva gives the orbit size a, then
T = 2*pi*sqrt(a^3 / GM)). That is:
    moons            -> relative to their parent planet
    everything else  -> relative to the Sun (the heaviest body)
It's worked out once, at the start. Orbits barely change over a session.

TWO SAMPLING RULES
------------------
- A point is added about every 1/180 of an orbit, so a circle looks smooth.
  When the animation is fast, that's at most one point per frame.
- Bodies that aren't on a closed orbit (period unknown) get a point every 2
  simulated days and keep 2 years of trail.

Trails follow the DRAWN position (moons are drawn exaggerated, see
visualize.py), so a moon's trail spirals around its planet as both move on.

KEEPING LINES THIN AT ANY DISTANCE (found by testing)
---------------------------------------------------
A VPython `curve` is a 3D tube whose thickness is in WORLD units. Perspective
makes a fixed thickness look fat up close and invisible far away. Two
dead ends: radius=0 means "pick a thickness from the view size" (thick
tubes), and `points` measure their "pixel" size at the distance of the
camera's look point, not at each dot.
The fix is PERSPECTIVE IN REVERSE. Something at distance d looks
(size / d) big on screen, so each trail's radius is set to
WIDTH_PER_DISTANCE x (distance from the camera to that body). Each trail is
then about 1.5 pixels wide around its body, whether the body is next to you
or across the Solar System. update_widths() redoes this a few times a second.
Trails are emissive (they glow), so the Sun's lighting doesn't darken them.
"""

from collections import deque

from vpython import curve, mag, vector

WIDTH_PER_DISTANCE = 0.0018   # radius / distance: about 1.5 px wide on a ~850 px tall view
WIDTH_UPDATE_EVERY = 6        # frames between thickness updates

from data_sources import DAY
from physics import G, orbital_period

POINTS_PER_ORBIT = 180
ORBITS_SHOWN = 2.0
HISTORY_CAP = 20000          # max points per body in "full history" mode
FALLBACK_PERIOD = 365.25 * DAY


class OrbitTrails:
    def __init__(self, sim, spheres):
        self.spheres = spheres
        self.full_history = False
        self._visible = True
        self._frame = 0
        self.trails = {}

        star_index = int(sim.gravity_masses.argmax())   # the Sun
        index = {name: i for i, name in enumerate(sim.names)}
        for i, body in enumerate(sim.bodies):
            if i == star_index:
                continue  # the Sun doesn't get a trail
            j = index.get(body.parent, star_index)
            mu = G * (sim.masses[j] + sim.masses[i])
            period = orbital_period(sim.positions[i] - sim.positions[j],
                                    sim.velocities[i] - sim.velocities[j], mu)
            if period is None or period > 1000 * 365.25 * DAY:
                period = FALLBACK_PERIOD
            self.trails[body.name] = {
                "curve": curve(color=vector(*body.color) * 0.5, radius=0.01, emissive=True),
                "radius": 0.01,
                "times": deque(),
                "period": period,
                "every": period / POINTS_PER_ORBIT,
                "next": 0.0,
            }

    def update(self, sim_time):
        """Call each frame after the simulation has moved."""
        for name, t in self.trails.items():
            if sim_time < t["next"]:
                continue
            t["curve"].append(pos=self.spheres[name].pos)
            t["times"].append(sim_time)
            t["next"] = sim_time + t["every"]
            self._trim(t, sim_time)

    def update_widths(self, camera_pos):
        """Call every frame. Rescales each trail so it stays ~1.5 px wide (see the top)."""
        self._frame += 1
        if self._frame % WIDTH_UPDATE_EVERY:
            return
        for name, t in self.trails.items():
            wanted = WIDTH_PER_DISTANCE * max(mag(self.spheres[name].pos - camera_pos), 1e-6)
            if abs(wanted - t["radius"]) > 0.1 * t["radius"]:   # skip tiny changes: fewer messages
                t["radius"] = wanted
                t["curve"].radius = wanted

    def _trim(self, t, sim_time):
        if self.full_history:
            while len(t["times"]) > HISTORY_CAP:
                t["times"].popleft()
                t["curve"].shift()  # remove the oldest point
        else:
            oldest_allowed = sim_time - ORBITS_SHOWN * t["period"]
            while t["times"] and t["times"][0] < oldest_allowed:
                t["times"].popleft()
                t["curve"].shift()

    # ---------------------------------------------------------- switches
    def set_full_history(self, on, sim_time):
        """Full history = keep everything (up to HISTORY_CAP). Off = last 2 orbits."""
        self.full_history = on
        if not on:
            for t in self.trails.values():
                self._trim(t, sim_time)

    def set_visible(self, on):
        self._visible = on
        for t in self.trails.values():
            t["curve"].visible = on

    def clear(self):
        for t in self.trails.values():
            t["curve"].clear()
            t["times"].clear()
