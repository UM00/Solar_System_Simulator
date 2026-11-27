"""
rings.py - Saturn's rings, drawn from NASA's measured ring data.

WHAT THE RINGS ARE
------------------
Not an asteroid belt: billions of chunks of almost pure WATER ICE, from dust
to house-sized, each on its own circular orbit around SATURN (not the Sun).
The main rings are ~280,000 km across but only ~10-100 m thick, thinner
relative to their width than a sheet of paper. They lie exactly in Saturn's
equatorial plane, so they share Saturn's 26.7 deg tilt. (The planet's bulging
equator pulls the particles' orbits into that plane.)

WHY RINGS AND NOT A MOON: THE ROCHE LIMIT
-----------------------------------------
Close to a planet, its pull on the near side of a lump is noticeably
stronger than on the far side. Inside the "Roche limit" that difference,
the TIDAL force, beats the lump's own gravity, so loose material can't clump
into a moon. It stays a ring. For icy lumps (density ~0.92 g/cm3) around
Saturn (0.687 g/cm3) the classic estimate is
    2.44 x Saturn's radius x (0.687 / 0.92)^(1/3)  =  about 2.2 Saturn radii.
Saturn's main rings end at 2.27 radii, right around this limit; its big
moons all orbit well outside it.

THE DATA (NASA Saturnian Rings Fact Sheet)
------------------------------------------
For each ring: inner and outer edge (in units of Saturn's equatorial radius,
60,268 km), OPTICAL DEPTH tau, and ALBEDO (how much light it reflects).
  - Optical depth: light passing straight through a ring is dimmed by a
    factor e^(-tau). So the fraction it BLOCKS (how solid it looks) is
    opacity = 1 - e^(-tau). B ring (tau ~1.5): 78% solid. C ring (tau ~0.1):
    10%, a faint veil you can see Saturn through.
  - Albedo sets brightness: B and A reflect 40-60%, the C ring only 12-30%.
  - The CASSINI DIVISION (1.95-2.03 radii) looks like a gap but has thin
    material. It's a RESONANCE, exactly like the Kirkwood gaps (guide
    Section 9): a particle there would orbit twice for every orbit of the
    moon Mimas, so Mimas's tugs keep adding up and clear it out.
Not drawn: the D, G and E rings (optical depth 0.000001-0.00001: far too
faint to see) and the Encke and Keeler gaps (325 km and 42 km wide: thinner
than a pixel at this scale).

DRAWING
-------
Each ring is a flat, very thin annulus (a disc with a hole), sized in
Saturn's DRAWN radius so it fits the enlarged planet exactly as the real
rings fit the real one, and turned so its axis is Saturn's real spin axis
(from NASA's pole data). Then it rides along with Saturn.
"""

import math

from vpython import cross, extrusion, mag, norm, shapes, vector

# (name, inner edge, outer edge [Saturn equatorial radii], optical depth, albedo)
# Middle of NASA's quoted ranges where a range is given.
SATURN_RINGS = [
    ("C ring",           1.239, 1.526, 0.10, 0.20),
    ("B ring",           1.526, 1.950, 1.50, 0.50),
    ("Cassini Division", 1.950, 2.030, 0.05, 0.30),
    ("A ring",           2.030, 2.270, 0.70, 0.50),
    ("F ring",           2.318, 2.322, 0.10, 0.60),
]
SATURN_EQUATORIAL_RADIUS_KM = 60268.0
ICE_COLOR = vector(1.0, 0.93, 0.80)      # pale, slightly warm ice
THICKNESS = 0.003                        # in drawn Saturn radii (visibly thin, still drawable)


def orient(obj, from_axis, to_axis, origin):
    """Rotate obj so that its from_axis points along to_axis (shortest turn)."""
    a, b = norm(from_axis), norm(to_axis)
    angle = math.acos(max(-1.0, min(1.0, a.dot(b))))
    if angle < 1e-9:
        return
    axis = cross(a, b)
    if mag(axis) < 1e-9:                 # exactly opposite: any perpendicular axis will do
        axis = vector(1, 0, 0) if abs(a.x) < 0.9 else vector(0, 0, 1)
    obj.rotate(angle=angle, axis=norm(axis), origin=origin)


class SaturnRings:
    def __init__(self, planet_sphere, spin_axis_scene, planet_radius_km):
        """
        planet_sphere    : Saturn's drawn sphere
        spin_axis_scene  : Saturn's spin axis in screen coordinates
        planet_radius_km : the radius Saturn's sphere is drawn for (mean radius)
        """
        self.planet = planet_sphere
        # One drawn "Saturn equatorial radius", in screen units.
        r_eq = planet_sphere.radius * SATURN_EQUATORIAL_RADIUS_KM / planet_radius_km
        self.parts = []
        for name, inner, outer, tau, albedo in SATURN_RINGS:
            h = THICKNESS * r_eq
            ring = extrusion(
                path=[vector(0, -h / 2, 0), vector(0, h / 2, 0)],   # a very short extrusion along y...
                shape=[shapes.circle(radius=outer * r_eq, np=256),  # ...of a circle
                       shapes.circle(radius=inner * r_eq, np=256)], # with a circular hole
                color=ICE_COLOR * (0.45 + albedo),
                opacity=1 - math.exp(-tau),                         # optical depth -> how solid
                shininess=0,
            )
            ring.pos = planet_sphere.pos
            orient(ring, vector(0, 1, 0), spin_axis_scene, ring.pos)
            self.parts.append(ring)

    def update(self):
        """Ride along with Saturn (the ring plane's direction stays fixed in space).
        Only sends updates when Saturn has actually been moved (see visualize._redraw)."""
        if self.parts and self.parts[0].pos.equals(self.planet.pos):
            return
        for ring in self.parts:
            ring.pos = self.planet.pos

    def set_visible(self, on):
        for ring in self.parts:
            ring.visible = on
