"""
camera.py - a free-flying "spaceship" camera with inertia.

WHY THIS FILE EXISTS
--------------------
VPython's built-in camera always ORBITS around one point (scene.center).
This class replaces it with a camera that has its own position, its own
viewing direction, and its own VELOCITY, so it moves like a craft in space.

CONTROLS (the 3D view must be under the mouse for the keys to register)
-----------------------------------------------------------------------
    W / S              thrust forward / backward
    A / D              thrust left / right
    E or Space / Q     thrust up / down
    Arrow keys         turn left/right, look up/down
    Hold LEFT mouse    steer: point where you want to turn. The further from
                       the centre of the screen, the faster you turn
    Mouse wheel        a short burst of forward/backward thrust
                       (ui_theme.py turns each wheel click into a brief
                       PageUp/PageDown press, and those keys also work)
    Shift (hold)       5x faster

INERTIA: WHY IT FEELS LIKE SPACE
--------------------------------
A real spacecraft doesn't start or stop instantly. Here the keys don't set
the camera's position directly. They set a TARGET velocity, and the actual
velocity eases toward it a bit every frame:

    velocity += (target_velocity - velocity) * (1 - exp(-dt / glide))

`glide` is a time in seconds (the "Camera glide" slider). After one glide
time the velocity has covered 63% of the gap; after three, 95%. That's
EXPONENTIAL SMOOTHING, a "damped" motion. Let go of the keys and the camera
coasts to a stop instead of halting dead. Turning works the same way with an
angular velocity. Using exp(-dt / glide) instead of a fixed fraction keeps
the feel the same even if the frame rate changes.

(Strictly, a frictionless spacecraft would coast forever. Pure Newtonian
flight is very hard to steer, so the gentle damping is a deliberate
compromise that most space games make.)

HOW A DIRECTION IS STORED: YAW AND PITCH
----------------------------------------
    yaw   = how far you've turned left/right (turning your whole body)
    pitch = how far you're looking up/down (tilting your head)
From these two angles we build three arrows each frame: forward, right
(= forward x up, right-hand rule) and up. Thrust moves along those arrows.
Pitch stops just short of 90 degrees: looking EXACTLY straight up makes
"right" undefined and the view would flip.

AUTOMATIC SPEED
---------------
Sizes on screen range from 0.1 units (the Moon) to 300 units (Neptune's
orbit), so top speed scales with the distance to the NEAREST object: you
creep near a planet and race through empty space (as in Celestia and
SpaceEngine).

SMOOTH FLIGHTS ("Follow" and "Overview")
----------------------------------------
Picking a body to follow, or pressing Overview, doesn't teleport you. It
starts a FLIGHT: each frame the camera covers a fraction of the remaining
distance (the same exponential smoothing), so it speeds away, then slows
gently into place while turning to face the target. Touch any flying
control and you take over immediately.

VPYTHON DETAIL: THE CLIPPING TRAP
---------------------------------
VPython only draws things between distance/100 and distance*10 from the
camera, where "distance" is how far away scene.center (the look point) is
(from VPython's code: zNear = distance/100, zFar = distance*10). So the look
point sits 10x the nearest object's distance ahead of the camera.
"""

import math

from vpython import cross, keysdown, mag, norm, vector

WORLD_UP = vector(0, 1, 0)        # VPython's screen "up" = ecliptic north
MAX_PITCH = math.radians(89.0)
TURN_SPEED = math.radians(55.0)   # arrow keys: degrees per second at full turn
TURN_SMOOTHING = 0.20             # seconds; how quickly turning starts/stops
STEER_GAIN = 2.0                  # mouse steering: turn rate per radian of pointer offset
STEER_DEADZONE = 0.03             # radians; ignore tiny offsets so you can hold still
SHIFT_BOOST = 5.0
# PageUp/PageDown (mouse wheel) thrust. Keep it modest: speed is proportional
# to the distance from the nearest object, so flying AWAY speeds you up, which
# takes you further away, which speeds you up more. That's EXPONENTIAL growth
# (like compound interest). In testing, a boost of 6 turned one second of
# scrolling back into a trip thousands of times further out, and 2.5 still
# moved ~1.85x per click. Each wheel click moves you roughly a fixed RATIO
# closer or further, like the steps of a zoom; at 1.5 it measured ~1.55x per click.
WHEEL_BOOST = 1.5
MIN_SPEED_DISTANCE = 0.02         # never slower than this, even touching a planet
LOOK_DISTANCE_FACTOR = 10.0       # look point = 10 x nearest-object distance
FLIGHT_TIME = 0.55                # seconds; smoothing time of Follow/Overview flights

MOVE_KEYS = {"w", "s", "a", "d", "e", "q", " ", "pageup", "pagedown"}
TURN_KEYS = {"left", "right", "up", "down"}


def _ease(dt, tau):
    """Fraction of the remaining gap to close this frame (exponential smoothing)."""
    return 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0


def _wrap(angle):
    """Wrap an angle into [-pi, pi), so we always turn the short way round."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class FlyCamera:
    def __init__(self, scene):
        self.scene = scene

        # Switch off VPython's built-in orbit camera; we control it ourselves.
        scene.autoscale = False
        scene.userspin = False
        scene.userpan = False
        scene.userzoom = False   # the wheel becomes thrust instead (see ui_theme.py)
        scene.up = WORLD_UP

        self.pos = vector(0, 0, 10)
        self.yaw = -math.pi / 2
        self.pitch = 0.0
        self.vel = vector(0, 0, 0)   # screen units per second
        self.yaw_rate = 0.0          # radians per second
        self.pitch_rate = 0.0

        self.speed_factor = 1.0      # flight-speed slider
        self.glide = 0.45            # seconds; camera-glide slider
        self.steering = False        # left mouse button held

        self.target = None           # sphere we ride along with
        self._target_last_pos = None
        self._flight = None          # an in-progress smooth flight, or None
        self._home = (vector(0, 0, 10), vector(0, 0, 0))

        self.look_distance = 1.0
        self._range_sent = False
        self._last_forward = None

        scene.bind("mousedown", self._mouse_down)
        scene.bind("mouseup", self._mouse_up)
        scene.bind("mouseleave", self._mouse_up)  # releasing outside the view stops steering too

    # ------------------------------------------------------------ directions
    def forward(self):
        cp = math.cos(self.pitch)
        return vector(cp * math.cos(self.yaw), math.sin(self.pitch), cp * math.sin(self.yaw))

    def right(self):
        return norm(cross(self.forward(), WORLD_UP))

    def up(self):
        return cross(self.right(), self.forward())

    # ------------------------------------------------------------ placement
    def _angles_towards(self, point):
        d = point - self.pos
        if mag(d) == 0:
            return self.yaw, self.pitch
        d = norm(d)
        return math.atan2(d.z, d.x), max(-MAX_PITCH, min(MAX_PITCH, math.asin(d.y)))

    def look_at(self, point):
        """Turn instantly to face a point."""
        self.yaw, self.pitch = self._angles_towards(point)

    def set_home(self, pos, look_at):
        self._home = (vector(pos), vector(look_at))

    def jump_home(self):
        """Instantly to the overview (used once at start-up)."""
        self.target = None
        self._flight = None
        self.pos = vector(self._home[0])
        self.look_at(self._home[1])

    def go_home(self):
        """Smooth flight back to the overview, in free flight."""
        self.target = None
        self._start_flight(goal=self._home[0], look=self._home[1])

    def follow(self, sphere):
        """Smooth flight to a body, then ride along with it."""
        self.target = sphere
        self._target_last_pos = vector(sphere.pos)
        # Park 8 radii away (from the current side), slightly above, looking at it.
        approach = norm(self.pos - sphere.pos) if mag(self.pos - sphere.pos) > 0 else -self.forward()
        offset = (approach * 8 + WORLD_UP * 2.5) * sphere.radius
        self._start_flight(offset=offset)

    def stop_following(self):
        self.target = None
        self._flight = None

    def _start_flight(self, goal=None, look=None, offset=None):
        self.vel = vector(0, 0, 0)
        self.yaw_rate = self.pitch_rate = 0.0
        self._flight = {"goal": goal, "look": look, "offset": offset}

    # ------------------------------------------------------------ mouse
    def _mouse_down(self, _evt):
        self.steering = True

    def _mouse_up(self, _evt):
        self.steering = False

    def _pointer_offset(self):
        """Pointer's angle from the screen centre (radians): (to the right, upward)."""
        ray = self.scene.mouse.ray
        f, r, u = self.forward(), self.right(), self.up()
        ahead = ray.dot(f)
        if ahead <= 0:
            return 0.0, 0.0
        return math.atan2(ray.dot(r), ahead), math.atan2(ray.dot(u), ahead)

    # ------------------------------------------------------------ per frame
    def update(self, dt, spheres):
        """Call once per animation frame. dt = real seconds since the last frame."""
        nearest = self._nearest_surface_distance(spheres)
        keys = {k.lower() for k in keysdown()}

        # 1. Ride along with the followed body.
        if self.target is not None:
            self.pos += self.target.pos - self._target_last_pos
            self._target_last_pos = vector(self.target.pos)

        # Any flying input cancels an automatic flight: you take the controls.
        if self._flight and (keys & (MOVE_KEYS | TURN_KEYS) or self.steering):
            self._flight = None

        # 2. Turning: a target turn rate, smoothed into the actual turn rate.
        want_yaw = TURN_SPEED * (("right" in keys) - ("left" in keys))
        want_pitch = TURN_SPEED * (("up" in keys) - ("down" in keys))
        if self.steering:
            sx, sy = self._pointer_offset()
            if abs(sx) > STEER_DEADZONE:
                want_yaw += STEER_GAIN * sx
            if abs(sy) > STEER_DEADZONE:
                want_pitch += STEER_GAIN * sy
        k = _ease(dt, TURN_SMOOTHING)
        self.yaw_rate += (want_yaw - self.yaw_rate) * k
        self.pitch_rate += (want_pitch - self.pitch_rate) * k
        self.yaw += self.yaw_rate * dt
        self.pitch += self.pitch_rate * dt

        # 3. Thrust: a target velocity, smoothed into the actual velocity (inertia).
        f, r = self.forward(), self.right()
        direction = (
            f * (("w" in keys) - ("s" in keys) + WHEEL_BOOST * (("pageup" in keys) - ("pagedown" in keys)))
            + r * (("d" in keys) - ("a" in keys))
            + WORLD_UP * ((("e" in keys) or (" " in keys)) - ("q" in keys))
        )
        top_speed = self.speed_factor * max(nearest, MIN_SPEED_DISTANCE)
        if "shift" in keys:
            top_speed *= SHIFT_BOOST
        want_vel = direction * top_speed  # (length > 1 only for wheel bursts)
        self.vel += (want_vel - self.vel) * _ease(dt, self.glide)
        # Never let a coast carry us through a planet in one frame.
        step = self.vel * dt
        if mag(step) > 0.5 * nearest:
            step = norm(step) * 0.5 * nearest
        self.pos += step

        # 4. Automatic flight (Follow / Overview), if one is running.
        if self._flight:
            self._fly(dt)

        self.pitch = max(-MAX_PITCH, min(MAX_PITCH, self.pitch))
        self._apply(nearest)

    def _fly(self, dt):
        fl = self._flight
        if fl["offset"] is not None and self.target is not None:
            goal = self.target.pos + fl["offset"]
            look = self.target.pos
        else:
            goal, look = fl["goal"], fl["look"]
        remaining = goal - self.pos
        self.pos += remaining * _ease(dt, FLIGHT_TIME)
        want_yaw, want_pitch = self._angles_towards(look)
        a = _ease(dt, FLIGHT_TIME * 0.6)
        self.yaw += _wrap(want_yaw - self.yaw) * a
        self.pitch += (want_pitch - self.pitch) * a
        # Done when within 0.5% of the way, so tiny leftovers don't linger forever.
        scale = mag(fl["offset"]) if fl["offset"] is not None else max(mag(goal), 1e-9)
        if mag(remaining) < 0.005 * scale and abs(_wrap(want_yaw - self.yaw)) < 0.01:
            self._flight = None

    # ------------------------------------------------------------ helpers
    def _nearest_surface_distance(self, spheres):
        best = math.inf
        for s in spheres:
            best = min(best, mag(s.pos - self.pos) - s.radius)
        return max(best, 1e-3)

    def _apply(self, nearest):
        """Tell VPython where the camera is (see the clipping trap above)."""
        wanted = max(LOOK_DISTANCE_FACTOR * nearest, 0.05)
        # Only change the look distance when it's off by more than 5%:
        # each change resends scene.range, so fewer changes = less traffic.
        resend = not self._range_sent or abs(wanted - self.look_distance) > 0.05 * self.look_distance
        if resend:
            self.look_distance = wanted

        f = self.forward()
        if self._last_forward is None or mag(f - self._last_forward) > 1e-12:
            self.scene.forward = f
            self._last_forward = f
        # VPython puts its camera at: center - forward * range / tan(fov/2).
        # Setting center and range like this puts it exactly at self.pos.
        self.scene.center = self.pos + f * self.look_distance
        if resend:
            self.scene.range = self.look_distance * math.tan(self.scene.fov / 2)
            self._range_sent = True
