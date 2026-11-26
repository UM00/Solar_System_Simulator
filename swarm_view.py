"""
swarm_view.py - draws crowds of thousands of asteroids (the asteroid belt,
Jupiter's trojans) as clouds of dots.

WHY NOT A SPHERE EACH?
----------------------
Every VPython object that moves costs one message from Python to the browser
per frame, plus its own label and trail. For 2,000+ asteroids that's far too
slow and an unreadable mess. So each swarm is ONE VPython `points` object,
and only its dots' positions are updated.

The asteroids themselves are full members of the simulation: test particles
pulled by the Sun, the planets and the big moons. Only their DRAWING is
simplified.

HOW THE DOTS ARE UPDATED (all found by testing)
-----------------------------------------------
- In the browser, a `points` object is really one small sphere per point.
  Clearing and re-adding 3,000 points 15 times a second made the browser
  build and throw away 45,000 objects a second, and it froze. So the points
  are created ONCE and afterwards only MOVED (VPython's "modify" command).
- Moving only a fixed number of dots per frame (DOTS_PER_FRAME, cycling
  round-robin through the whole crowd) keeps the browser's work per frame
  small and even. Moving ALL dots every few frames meant ~37,000 commands a
  second; the browser was so busy that it stopped reporting key presses in
  time, and the camera ignored the keyboard. With 150 per frame, a
  1,800-asteroid crowd is fully refreshed 5 times a second.
- The messages are built straight from NumPy arrays, skipping VPython's
  per-point vector objects. That cut the Python-side cost from about the
  cost of all the physics to ~1 ms per frame.
- Dot size: in WORLD units, set in proportion to the camera's distance from
  the Sun (perspective in reverse, as for the trails), so the belt looks
  equally fine from close up or from the edge of the Solar System. VPython's
  "pixel" dots are measured at the camera's look-point distance instead, and
  came out as giant blobs.

WHAT TO LOOK FOR
----------------
- The main belt between Mars and Jupiter, about 2.1-3.3 AU from the Sun.
  It's thin and mostly flat, but nothing like the crowded rock-fields of
  films: real asteroids are typically ~1 million km apart.
- The KIRKWOOD GAPS (see the guide's histogram): rings where hardly any
  asteroid orbits, because there it would circle the Sun exactly 3, 5/2 or
  7/3 times per Jupiter orbit. Jupiter's tug then always lands at the same
  point in the orbit, and the repeated pushes add up (RESONANCE, like pushing
  a swing at the right moment) until the orbit is stretched out of the gap.
- The JUPITER TROJANS: two clouds sharing Jupiter's orbit, about 60 degrees
  ahead of and behind it, at the Lagrange points L4 and L5. There the Sun's
  and Jupiter's pulls combine so an asteroid stays in step with Jupiter.
  Watch them travel around the Sun together with Jupiter.
"""

import numpy as np
from vpython import mag, points, vector

DOTS_PER_FRAME = 150        # dots moved per frame, cycling through all swarms
SIZE_PER_DISTANCE = 0.0016  # dot radius / (camera's distance from the Sun)


def _send(cloud, method, val):
    """Queue one raw command for this cloud (the same thing VPython's own methods send)."""
    cloud.appendcmd({"val": val, "method": method, "idx": cloud.idx})


class SwarmClouds:
    def __init__(self, sim, to_scene_array):
        """
        sim            : the Simulation
        to_scene_array : function turning an (N, 3) array of simulation
                         positions into (N, 3) screen coordinates
        """
        self.sim = sim
        self.to_scene_array = to_scene_array
        self._cursor = 0          # where the round-robin refresh continues next frame
        self._radius = None
        self._resize_left = 0     # dots still waiting for the latest size
        self.groups = {}
        for i, body in enumerate(sim.bodies):
            if body.swarm:
                group = self.groups.setdefault(body.swarm, {"indices": [], "color": body.color})
                group["indices"].append(i)

        for group in self.groups.values():
            idx = np.array(group["indices"])
            group["indices"] = idx
            cloud = points(color=vector(*group["color"]), radius=0.02, size_units="world", emissive=True)
            # Create every dot ONCE, in a single message.
            xyz = self._screen_positions(idx)
            _send(cloud, "append", [{"pos": p} for p in xyz.tolist()])
            group["cloud"] = cloud
            group["visible"] = True

        # One flat list of (group, dot number) to cycle through, all swarms together.
        self._queue = [(g, n) for g in self.groups.values() for n in range(len(g["indices"]))]

    def _screen_positions(self, idx):
        # The belt moves on its own slower clock (simulation.py) and can be up
        # to a day behind. A straight-line nudge (position + velocity x lag)
        # draws it where it is NOW; over less than a day of a 4-year orbit a
        # straight line is an excellent approximation of the curve.
        lag = self.sim.slow_lane_lag()
        return self.to_scene_array(self.sim.positions[idx] + self.sim.velocities[idx] * lag)

    def update(self, moved=True):
        """
        Call EVERY frame (also while paused). Sends the next DOTS_PER_FRAME
        dots' positions and sizes. moved=False (paused) skips the work unless
        a size change still has to reach every dot. Found by testing: when
        updates only ran while the simulation moved, dots sized for a
        far-away view stayed huge after flying in while paused.
        """
        if not self._queue or (not moved and self._resize_left <= 0):
            return
        self._resize_left -= DOTS_PER_FRAME
        batch = [self._queue[(self._cursor + k) % len(self._queue)]
                 for k in range(min(DOTS_PER_FRAME, len(self._queue)))]
        self._cursor = (self._cursor + len(batch)) % len(self._queue)
        for group in self.groups.values():
            numbers = [n for g, n in batch if g is group]
            if not numbers or not group["visible"]:
                continue   # hidden swarms cost nothing: no updates sent
            xyz = self._screen_positions(group["indices"][numbers])
            cloud = group["cloud"]
            r = self._radius or 0.02
            alive = self.sim.alive[group["indices"][numbers]].tolist()   # experiments can destroy asteroids
            for n, p, a in zip(numbers, xyz.tolist(), alive):
                _send(cloud, "modify", [n, [{"pos": p, "radius": r, "visible": a}]])

    def update_size(self, camera_pos, sun_pos):
        """
        Keep dots looking the same size wherever the camera is (see the top).
        Found by testing: changing the whole cloud's radius does NOT resize
        dots that already exist. So the size travels with each dot's position
        update instead, and every dot catches up within one refresh cycle.
        """
        wanted = SIZE_PER_DISTANCE * max(mag(camera_pos - sun_pos), 1e-6)
        if self._radius is None or abs(wanted - self._radius) > 0.05 * self._radius:
            self._radius = wanted
            self._resize_left = len(self._queue)   # every dot needs the new size once

    def labels(self):
        """The swarm names, e.g. ["Main belt", "Jupiter trojans"]."""
        return list(self.groups)

    def set_visible(self, label, on):
        self.groups[label]["visible"] = on
        self.groups[label]["cloud"].visible = on

    def count(self, label):
        return len(self.groups[label]["indices"])
