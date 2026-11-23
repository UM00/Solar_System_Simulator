"""
body.py - the one data type every object in the simulator shares.

KEY DESIGN IDEA
---------------
The simulator never asks "is this a planet?" or "is this an asteroid?".
It only sees Bodies: things that have a mass, a position and a velocity.
The Sun, Earth, the Moon, Ceres, Halley's Comet, or a star you invent are
all just Body objects. Adding a new kind of object later means creating
another Body. The physics code does not change.

UNITS (used everywhere in this project)
---------------------------------------
    distance  -> metres (m)
    time      -> seconds (s)
    mass      -> kilograms (kg)
    angles    -> degrees (only for axial tilt, converted when used)

COORDINATE FRAME
----------------
"Ecliptic" coordinates, centred on the Solar System's balance point:
    x-y plane = the plane of Earth's orbit (the ecliptic)
    +x        = direction of the March equinox
    +z        = "ecliptic north", perpendicular to Earth's orbit
Seen from +z looking down, every planet orbits counter-clockwise.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Body:
    # ---- Used by the PHYSICS ---------------------------------------------
    name: str
    mass: float           # kg
    position: np.ndarray  # m,   [x, y, z]
    velocity: np.ndarray  # m/s, [vx, vy, vz]

    # A "test particle" FEELS gravity but is too light to PULL on anything.
    # True for asteroids, comets, spacecraft. Ceres, the biggest asteroid,
    # has about 1/6000 of Earth's mass, so its pull really is negligible.
    # Skipping those pulls saves a lot of computation (see physics.py).
    is_test_particle: bool = False

    # ---- Used only for DRAWING and SPINNING --------------------------------
    # These never affect the orbit. In this model gravity acts as if all of
    # a body's mass sat at its centre (a "point mass"), so size and spin
    # don't change where the body goes.
    radius: float = 1.0e6                  # m
    rotation_period: float | None = None   # s, SIDEREAL (one spin relative to the stars).
                                           # None = unknown, so it won't spin
    axial_tilt: float = 0.0                # degrees between spin axis and orbit-north.
                                           # Over 90 means it spins "backwards" (Venus, Uranus)
    color: tuple = (1.0, 1.0, 1.0)         # (red, green, blue), each 0..1
    texture: str | None = None             # VPython texture name, e.g. "earth"
    parent: str | None = None              # what it orbits closely (Moon -> "Earth").
                                           # Only used to exaggerate moon distances on screen.
    swarm: str | None = None               # e.g. "Main belt": part of a crowd of thousands,
                                           # drawn as one dot in a shared cloud instead of
                                           # its own sphere, label and trail.

    def __post_init__(self):
        # Always store vectors as floating-point numpy arrays.
        # (np.array([0, 0, 0]) would be an INTEGER array. Adding decimals
        #  to it can silently go wrong, so we force float here.)
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.position.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError(f"{self.name}: position and velocity need exactly 3 numbers each")
        if self.mass < 0:
            raise ValueError(f"{self.name}: mass cannot be negative")
