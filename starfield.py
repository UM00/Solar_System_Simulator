"""
starfield.py - a background of distant stars (and a faint Milky Way band).

THE SKYBOX TRICK
----------------
Real stars are so far away that flying across the whole Solar System doesn't
shift them at all: no parallax. Even the nearest star is ~270,000 AU away.
We fake that with a "skybox": a sphere of star dots that is always CENTRED
ON THE CAMERA. When you fly, the sphere flies with you, so the stars never
get closer. When you turn, you see different stars, just like reality.

The stars are tiny emissive boxes (they glow and aren't lit by the Sun), all
merged into ONE compound object. Moving one object per frame is cheap.
Moving 1,500 separate objects would not be.

THE CLIPPING CATCH (see camera.py)
----------------------------------
VPython only draws things between d/100 and 10*d from the camera, where d is
the camera's current look distance. So the star sphere's radius is kept at
4*d, safely inside that window, and rescaled whenever d changes. Each dot is
scaled with it, so a star always looks the same size on screen.

THE MILKY WAY
-------------
Our galaxy is a flat disc and we sit inside it, so many of the stars we see
lie near one great circle across the sky: the Milky Way band. Its plane is
tilted about 60 degrees to the ecliptic (the plane of Earth's orbit), so 40%
of our dots are pulled toward a plane tilted 60 degrees.
"""

import math
import random

from vpython import box, compound, vector

BASE_RADIUS = 1000.0       # the sphere is built at this radius, then rescaled
RADIUS_FACTOR = 4.0        # star sphere radius = 4 x camera look distance
GALACTIC_TILT_DEG = 60.0


class StarField:
    def __init__(self, count=1500, seed=7):
        rng = random.Random(seed)  # fixed seed = the same sky every run
        tilt = math.radians(GALACTIC_TILT_DEG)
        plane_normal = vector(0, math.cos(tilt), math.sin(tilt))

        parts = []
        for i in range(count):
            # A random direction, evenly spread over the sphere:
            # pick a random height z in [-1, 1] and a random angle around.
            z = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2 * math.pi)
            r_xy = math.sqrt(1 - z * z)
            d = vector(r_xy * math.cos(phi), z, r_xy * math.sin(phi))
            if i < 0.4 * count:
                # Milky Way: squash the part pointing out of the galactic plane
                d = (d - plane_normal * (d.dot(plane_normal) * 0.88)).norm()

            # Most stars are faint, a few are bright (like the real sky).
            brightness = 0.25 + 0.75 * rng.random() ** 3
            tint = rng.random()
            if tint < 0.15:
                c = vector(0.75, 0.85, 1.0)      # hot blue-white stars
            elif tint < 0.30:
                c = vector(1.0, 0.88, 0.70)      # cooler yellow-orange stars
            else:
                c = vector(1.0, 1.0, 1.0)
            size = BASE_RADIUS * 0.0016 * (0.7 + 1.3 * brightness)
            parts.append(box(pos=d * BASE_RADIUS, size=vector(size, size, size),
                             color=c * brightness, emissive=True, shininess=0))

        # One object, measured from the sphere's centre.
        self.obj = compound(parts, origin=vector(0, 0, 0))
        self.obj.emissive = True
        self.base_size = vector(self.obj.size)
        self._scale = 1.0

    def follow(self, camera_pos, look_distance):
        """Keep the sky centred on the camera and inside VPython's drawing range.
        Sends nothing when the camera hasn't moved (no message = no work)."""
        if not self.obj.pos.equals(camera_pos):
            self.obj.pos = camera_pos
        scale = RADIUS_FACTOR * look_distance / BASE_RADIUS
        if abs(scale - self._scale) > 0.03 * self._scale:
            self._scale = scale
            self.obj.size = self.base_size * scale

    @property
    def visible(self):
        return self.obj.visible

    @visible.setter
    def visible(self, value):
        self.obj.visible = value
