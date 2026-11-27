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
- Only dots whose picture has visibly CHANGED are sent (moved about a third
  of a pixel, size out of date, or destroyed), at most DOTS_PER_FRAME per
  frame, most-changed first. Moving ALL dots every few frames meant ~37,000
  commands a second; the browser was so busy that it stopped reporting key
  presses in time, and the camera ignored the keyboard. When paused, or
  zoomed far out where the belt barely moves on screen, almost nothing is
  sent at all.
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

DOTS_PER_FRAME = 150        # at most this many dots sent per frame
SIZE_PER_DISTANCE = 0.0016  # dot radius / (camera's distance from the Sun)
POSITION_TOLERANCE = 0.0005 # a dot is re-sent once it has moved this x its distance from the camera


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
        self._radius = None
        self.groups = {}
        for i, body in enumerate(sim.bodies):
            if body.swarm:
                group = self.groups.setdefault(body.swarm, {"indices": [], "color": body.color})
                group["indices"].append(i)

        for group in self.groups.values():
            group["indices"] = np.array(group["indices"])
            group["visible"] = True
            group["cloud"] = None       # created lazily, on the first update() (see _create)

    def _create(self, group):
        """
        LAZY LOADING: the dots are only created the first time they're needed
        (visualize.py starts updating them a few frames after the Solar System
        appears), so the first picture shows up sooner.
        """
        idx = group["indices"]
        cloud = points(color=vector(*group["color"]), radius=0.02, size_units="world", emissive=True)
        # Create every dot ONCE, in a single message.
        xyz = self._screen_positions(idx)
        _send(cloud, "append", [{"pos": p} for p in xyz.tolist()])
        cloud.visible = group["visible"]
        group["cloud"] = cloud
        # What the browser is currently showing for each dot (see update()).
        group["sent_pos"] = xyz
        group["sent_radius"] = np.full(len(idx), 0.02)
        group["sent_alive"] = np.ones(len(idx), dtype=bool)

    def _screen_positions(self, idx):
        # The belt moves on its own slower clock (simulation.py) and can be up
        # to a day behind. A straight-line nudge (position + velocity x lag)
        # draws it where it is NOW; over less than a day of a 4-year orbit a
        # straight line is an excellent approximation of the curve.
        lag = self.sim.slow_lane_lag()
        return self.to_scene_array(self.sim.positions[idx] + self.sim.velocities[idx] * lag)

    def update(self, camera_pos):
        """
        Call EVERY frame (also while paused). Sends only the dots whose
        picture has really changed, up to DOTS_PER_FRAME, most-changed first.

        For ALL dots at once (NumPy, cheap) it compares where each dot is now
        with what the browser is showing, and picks a dot if
          - it has moved more than POSITION_TOLERANCE x its distance from the
            camera (about a third of a pixel), or
          - its size is out of date by more than 5% (the camera moved), or
          - it was destroyed / revived in an experiment.
        When paused, or zoomed far out where the belt barely moves on screen,
        almost nothing is sent. Found by testing: before this, 150 dots were
        sent every frame no matter what; and when updates only ran while the
        simulation moved, dots sized for a far-away view stayed huge after
        flying in while paused.
        """
        cam = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
        r = self._radius or 0.02
        budget = DOTS_PER_FRAME
        for group in self.groups.values():
            if not group["visible"] or budget <= 0:
                continue   # hidden swarms cost nothing: no updates sent
            if group["cloud"] is None:
                self._create(group)
                continue
            idx = group["indices"]
            xyz = self._screen_positions(idx)
            alive = self.sim.alive[idx]                       # experiments can destroy asteroids
            tolerance = POSITION_TOLERANCE * np.linalg.norm(xyz - cam, axis=1)
            moved = np.linalg.norm(xyz - group["sent_pos"], axis=1) / np.maximum(tolerance, 1e-12)
            stale = (moved > 1) | (np.abs(group["sent_radius"] - r) > 0.05 * r) | (alive != group["sent_alive"])
            todo = np.flatnonzero(stale)
            if todo.size > budget:                            # most-changed first
                todo = todo[np.argsort(-moved[todo])[:budget]]
            budget -= todo.size
            cloud = group["cloud"]
            for n, p, a in zip(todo.tolist(), xyz[todo].tolist(), alive[todo].tolist()):
                _send(cloud, "modify", [n, [{"pos": p, "radius": r, "visible": a}]])
            group["sent_pos"][todo] = xyz[todo]
            group["sent_radius"][todo] = r
            group["sent_alive"][todo] = alive[todo]

    def update_size(self, camera_pos, sun_pos):
        """
        Keep dots looking the same size wherever the camera is (see the top).
        Found by testing: changing the whole cloud's radius does NOT resize
        dots that already exist. So the size travels with each dot's position
        update instead, and every dot catches up within one refresh cycle.
        """
        self._radius = SIZE_PER_DISTANCE * max(mag(camera_pos - sun_pos), 1e-6)

    def labels(self):
        """The swarm names, e.g. ["Main belt", "Jupiter trojans"]."""
        return list(self.groups)

    def set_visible(self, label, on):
        self.groups[label]["visible"] = on
        if self.groups[label]["cloud"] is not None:
            self.groups[label]["cloud"].visible = on

    def count(self, label):
        return len(self.groups[label]["indices"])
