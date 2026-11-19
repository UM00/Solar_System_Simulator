"""
camera.py - a free-flying camera, like in a 3D game.

WHY THIS FILE EXISTS
--------------------
VPython's built-in camera always ORBITS around one point (scene.center):
dragging swings you around that point, and scrolling moves you toward it.
You can never just *fly somewhere*. This class replaces that with a camera
that has its own position and its own viewing direction.

CONTROLS (click the 3D view first, so it receives your key presses)
-------------------------------------------------------------------
    W / S              fly forward / backward
    A / D              slide left / right
    E or Space / Q     rise / sink
    Arrow keys         turn left/right, look up/down
    Hold LEFT mouse    steer: point where you want to turn. The further from
                       the centre of the screen, the faster you turn
    Mouse wheel        fly forward / backward in steps
    Shift (hold)       5x faster

HOW A CAMERA DIRECTION IS STORED: YAW AND PITCH
-----------------------------------------------
Instead of storing an arrow directly, we store two angles, like a person:
    yaw   = how far you've turned left/right (turning your whole body)
    pitch = how far you're looking up/down (tilting your head)
From these two angles we build three arrows each frame:
    forward = where you're looking
    right   = to your right-hand side
    up      = above your head
Moving with W/A/S/D just means "move along one of these arrows".
Pitch is limited to just under 90 degrees. Looking EXACTLY straight up
makes "right" undefined (any direction is perpendicular), and the view
would flip. Every 3D game does the same.

AUTOMATIC SPEED
---------------
The Solar System has sizes from 0.1 units (the Moon, on screen) to 300
units (Neptune's orbit). One fixed flying speed would be too slow out
there or uncontrollable near a planet. So speed scales with the distance to
the NEAREST object: near Earth you creep, in deep space you race.
Space-exploration programs like Celestia and SpaceEngine use this trick.

VPYTHON DETAIL: THE CLIPPING TRAP
---------------------------------
VPython only draws things between distance/100 and distance*10 from the
camera, where "distance" is how far away scene.center (the look point) is.
(Found in VPython's own code: zNear = distance/100, zFar = distance*10.)
So we place the look point 10x the distance of the nearest object ahead:
    nearest object at 1 unit -> drawn from 0.1 to 100 units away.
That keeps the nearby planet un-clipped while distant planets stay visible.
"""

import math

from vpython import cross, keysdown, mag, norm, vector

WORLD_UP = vector(0, 1, 0)        # VPython's screen "up" = ecliptic north
MAX_PITCH = math.radians(89.0)    # just short of straight up/down
TURN_SPEED = math.radians(70.0)   # arrow keys: degrees per second
STEER_GAIN = 2.5                  # mouse steering: turn rate per radian of pointer offset
STEER_DEADZONE = 0.02             # radians; ignore tiny offsets so you can hold still
SHIFT_BOOST = 5.0
MIN_SPEED_DISTANCE = 0.02         # never slower than this, even touching a planet
LOOK_DISTANCE_FACTOR = 10.0       # look point = 10 x nearest-object distance (see above)


class FlyCamera:
    def __init__(self, scene):
        self.scene = scene

        # Switch off VPython's built-in orbit camera. We control it ourselves.
        scene.autoscale = False
        scene.userspin = False
        scene.userpan = False
        # The wheel stays ON. VPython turns it into a zoom, and we turn that
        # zoom into flying forward/backward (see _absorb_wheel).
        scene.userzoom = True
        scene.up = WORLD_UP

        self.pos = vector(0, 0, 10)
        self.yaw = -math.pi / 2       # facing -z
        self.pitch = 0.0
        self.speed_factor = 1.0       # set by the flight-speed slider
        self.steering = False         # True while the left mouse button is held

        self.target = None            # sphere we're following, or None
        self._target_last_pos = None
        self._home = (vector(0, 0, 10), vector(0, 0, 0))

        self._look_distance = 1.0
        self._sent_ranges = []        # recent ranges WE sent (to tell them apart from wheel zooms)
        self._force_resend = False
        self._last_forward = None

        scene.bind("mousedown", self._mouse_down)
        scene.bind("mouseup", self._mouse_up)
        scene.bind("mouseleave", self._mouse_up)  # releasing outside the view must stop steering too

    # ------------------------------------------------------------ directions
    def forward(self):
        """Unit arrow pointing where the camera looks, built from yaw and pitch."""
        cp = math.cos(self.pitch)
        return vector(cp * math.cos(self.yaw), math.sin(self.pitch), cp * math.sin(self.yaw))

    def right(self):
        # forward x up = the arrow pointing to your right (right-hand rule)
        return norm(cross(self.forward(), WORLD_UP))

    def up(self):
        return cross(self.right(), self.forward())

    # ------------------------------------------------------------ placement
    def look_at(self, point):
        """Turn (without moving) to face a point."""
        d = point - self.pos
        if mag(d) == 0:
            return
        d = norm(d)
        self.pitch = max(-MAX_PITCH, min(MAX_PITCH, math.asin(d.y)))
        self.yaw = math.atan2(d.z, d.x)

    def set_home(self, pos, look_at):
        self._home = (vector(pos), vector(look_at))

    def go_home(self):
        """Back to the overview position, in free flight."""
        self.target = None
        self.pos = vector(self._home[0])
        self.look_at(self._home[1])

    def follow(self, sphere):
        """
        Fly to a body and ride along with it. You can still fly around it
        with the keys: following only adds the body's own motion to yours.
        """
        self.target = sphere
        self._target_last_pos = vector(sphere.pos)
        # Park at 8 radii away, slightly above, looking at it.
        offset = (-self.forward() * 8 + WORLD_UP * 2) * sphere.radius
        self.pos = sphere.pos + offset
        self.look_at(sphere.pos)

    # ------------------------------------------------------------ mouse
    def _mouse_down(self, _evt):
        self.steering = True

    def _mouse_up(self, _evt):
        self.steering = False

    def _pointer_offset(self):
        """
        How far the mouse pointer is from the centre of the screen, as two
        angles (radians): (to the right, upward).
        VPython gives the mouse as a RAY: an arrow from the camera through
        the pointer. Comparing it with our forward/right/up arrows gives
        the angles.
        """
        ray = self.scene.mouse.ray
        f, r, u = self.forward(), self.right(), self.up()
        ahead = ray.dot(f)
        if ahead <= 0:
            return 0.0, 0.0
        return math.atan2(ray.dot(r), ahead), math.atan2(ray.dot(u), ahead)

    # ------------------------------------------------------------ per frame
    def update(self, dt, spheres):
        """Call once per animation frame. dt = real seconds since last frame."""
        nearest = self._nearest_surface_distance(spheres)

        # 1. FOLLOWING: move by however much the target moved since last frame.
        if self.target is not None:
            self.pos += self.target.pos - self._target_last_pos
            self._target_last_pos = vector(self.target.pos)

        keys = {k.lower() for k in keysdown()}

        # 2. TURNING with the arrow keys
        self.yaw += TURN_SPEED * dt * (("right" in keys) - ("left" in keys))
        self.pitch += TURN_SPEED * dt * (("up" in keys) - ("down" in keys))

        # 3. STEERING with the mouse: the turn RATE depends on how far the
        #    pointer is from the centre. (Turning by "how far the mouse moved"
        #    would be nicer, but VPython reports the mouse a frame or two late,
        #    which makes that jittery. Rate steering is immune to the delay.)
        if self.steering:
            sx, sy = self._pointer_offset()
            if abs(sx) > STEER_DEADZONE:
                self.yaw += STEER_GAIN * sx * dt
            if abs(sy) > STEER_DEADZONE:
                self.pitch += STEER_GAIN * sy * dt

        self.pitch = max(-MAX_PITCH, min(MAX_PITCH, self.pitch))

        # 4. MOVING with W/A/S/D/E/Q, speed scaled by the nearest distance
        f, r = self.forward(), self.right()
        direction = (
            f * (("w" in keys) - ("s" in keys))
            + r * (("d" in keys) - ("a" in keys))
            + WORLD_UP * ((("e" in keys) or (" " in keys)) - ("q" in keys))
        )
        if mag(direction) > 0:
            speed = self.speed_factor * max(nearest, MIN_SPEED_DISTANCE)
            if "shift" in keys:
                speed *= SHIFT_BOOST
            self.pos += norm(direction) * speed * dt

        # 5. MOUSE WHEEL -> fly forward/back
        self._absorb_wheel(nearest)

        # 6. Tell VPython where the camera is
        self._apply(nearest)

    # ------------------------------------------------------------ helpers
    def _nearest_surface_distance(self, spheres):
        best = math.inf
        for s in spheres:
            best = min(best, mag(s.pos - self.pos) - s.radius)
        return max(best, 1e-3)

    def _absorb_wheel(self, nearest):
        """
        The wheel makes VPython shrink or grow scene.range, which moves its
        camera toward or away from the look point. We spot a range that we
        did NOT send ourselves, copy the same move into our own position,
        and the next _apply() puts the range back to normal.
        """
        current = self.scene.range
        if not self._sent_ranges:
            return
        if any(abs(current - r) <= 1e-9 * r for r in self._sent_ranges):
            return  # it's one of ours, so no wheel movement
        ratio = current / self._sent_ranges[-1]
        step = self._look_distance * (1.0 - ratio)  # >0 = wheel toward you = fly forward
        step = min(step, 0.5 * nearest)            # never jump through a planet
        self.pos += self.forward() * step
        self._sent_ranges.append(current)         # don't count this same zoom twice
        self._force_resend = True                 # make _apply put the range back

    def _apply(self, nearest):
        # Pick the look-point distance from the nearest object (the clipping
        # trap above), but only change it when it's off by more than 5%.
        # Every change resends scene.range, and fewer resends make wheel
        # zooms easier to recognise.
        wanted = max(LOOK_DISTANCE_FACTOR * nearest, 0.05)
        resend = (self._force_resend or not self._sent_ranges
                  or abs(wanted - self._look_distance) > 0.05 * self._look_distance)
        self._force_resend = False
        if resend:
            self._look_distance = wanted

        f = self.forward()
        if self._last_forward is None or mag(f - self._last_forward) > 1e-12:
            self.scene.forward = f
            self._last_forward = f
        # VPython puts its camera at:  center - forward * range / tan(fov/2).
        # Setting center and range like this puts it exactly at self.pos.
        self.scene.center = self.pos + f * self._look_distance
        if resend:
            new_range = self._look_distance * math.tan(self.scene.fov / 2)
            self.scene.range = new_range
            self._sent_ranges = (self._sent_ranges + [new_range])[-8:]
