"""
visualize.py - the 3D view, drawn with VPython (opens in a browser tab).

THE GOLDEN RULE: physics in TRUE units, drawing in FAKE units
--------------------------------------------------------------
At true scale the Solar System is almost entirely empty space. If Earth's
orbit were a 10 m wide circle, Earth would be a 0.4 mm speck and the Sun
a 9 cm ball. You'd see nothing. Every planetarium app cheats, and so do we.
But we cheat ONLY when drawing:

    - distances : shrunk by one fixed factor (1 AU = 10 screen units)
    - sizes     : enlarged A LOT, and compressed with a square root so
                  Jupiter (11x Earth's width) doesn't swallow everything
    - moons     : pushed further from their planet, or they'd sit inside
                  the (enlarged) planet sphere

The simulation itself never sees any of this.

TWO KINDS OF MOTION
-------------------
    REVOLUTION : moving around the Sun. Comes from the gravity simulation.
    ROTATION   : spinning around its own axis. Doesn't affect the orbit in
                 this model (gravity acts on the centre of mass), so it's
                 simple geometry: angle turned = 360 deg * elapsed / period.

AXIAL TILT AND THE SEASONS
--------------------------
A spinning planet acts like a gyroscope: its axis keeps pointing the SAME
way in space all year. Earth's leans 23.44 deg. In June the northern
half leans toward the Sun (long days, summer). Six months later Earth is
on the other side of the Sun with the axis still leaning the same way,
so the north leans AWAY (winter). Watch Earth's lit side to see it.

COORDINATES
-----------
The simulation uses z = ecliptic north (body.py). VPython uses y = up on
screen. So simulation (x, y, z) is drawn at screen (x, z, -y). That's a
plain rotation, so left/right handedness is kept and orbits still run
counter-clockwise when seen from above.

THE PAGE LAYOUT
---------------
ui_theme.py turns VPython's page into a full-window view with:
    - a HUD (top-left): date, speed, what you're following
    - a sidebar (button top-right, or the H key) with all the controls
The other helpers: camera.py (flying), trails.py (orbit trails),
starfield.py (background stars).
"""

import math
from datetime import datetime, timedelta

import numpy as np
from vpython import (
    button,
    canvas,
    checkbox,
    color,
    label,
    local_light,
    menu,
    rate,
    slider,
    sphere,
    textures,
    vector,
    wtext,
)

import ui_theme
from camera import FlyCamera
from data_sources import AU, DAY
from starfield import StarField
from trails import OrbitTrails

# ---- drawing scales (change these freely; they don't touch the physics) ---
DISTANCE_SCALE = 10.0 / AU   # 1 AU -> 10 screen units
EARTH_RADIUS_M = 6.371e6
PLANET_SIZE = 0.20           # Earth's drawn radius, in screen units
STAR_SIZE = 1.2              # the Sun's drawn radius (capped, or it would swallow Mercury)
MIN_SIZE = 0.012             # smallest drawn radius, so small moons/asteroids stay visible
MOON_DISTANCE_EXAGGERATION = 30.0
FPS = 60
SLIDER_LENGTH = 290          # pixels, to fit the sidebar


def to_scene(p):
    """Simulation coordinates (metres, z-up) -> VPython screen vector (y-up)."""
    return vector(p[0], p[2], -p[1]) * DISTANCE_SCALE


def display_radius(body):
    if body.radius > 1e8:  # bigger than any planet -> it's a star
        return STAR_SIZE
    # Square root keeps the size ORDER right (Jupiter > Earth > Moon)
    # without the giants being absurdly large. A minimum size keeps tiny
    # moons and asteroids (Phobos is 22 km wide) visible as dots.
    return max(PLANET_SIZE * math.sqrt(body.radius / EARTH_RADIUS_M), MIN_SIZE)


class SolarSystemView:
    def __init__(self, sim, start_date, days_per_second=10.0):
        self.sim = sim
        self.start = datetime.fromisoformat(start_date)
        self.days_per_second = days_per_second
        self.paused = False
        self.time_debt = 0.0  # simulated seconds owed but not yet stepped
        self.following = None

        # The size is only a starting value: ui_theme stretches it to the window.
        self.scene = canvas(title="", width=1280, height=800, background=color.black)
        # The Sun is the ONLY light source, so planets get a real
        # day side and night side. A little ambient light keeps the
        # night side faintly visible.
        self.scene.lights = []
        self.scene.ambient = color.gray(0.12)
        self.sun_light = None

        # HUD (top-left): one block of HTML, rewritten every frame.
        self.hud = wtext(pos=self.scene.title_anchor, text="")

        self.spheres = {}
        self.labels = {}
        self.spin_axes = {}  # screen-space spin axis of each body
        for body in sim.bodies:
            self._create_sphere(body)

        self.stars = StarField()
        self.trails = OrbitTrails(sim, self.spheres)
        self.camera = FlyCamera(self.scene)
        self._fit_camera()
        self._build_controls()
        ui_theme.install(self.scene)   # last: restyles the page into HUD + sidebar

    # --------------------------------------------------------------- setup
    def _create_sphere(self, body):
        is_star = body.radius > 1e8
        textured = body.texture is not None
        extra = {"texture": getattr(textures, body.texture)} if textured else {}
        s = sphere(
            pos=to_scene(body.position),
            radius=display_radius(body),
            color=color.white if textured else vector(*body.color),
            **extra,
            emissive=is_star,  # stars glow; they aren't lit by something else
            shininess=0.0,
        )
        if is_star and self.sun_light is None:
            self.sun_light = local_light(pos=s.pos, color=color.white)

        # ---- axial tilt -------------------------------------------------
        # Untilted, the axis points straight "up" (ecliptic north).
        # We lean it by the tilt angle toward the simulation's +y direction.
        # For EARTH that's the real direction: the +x axis is DEFINED as the
        # line where Earth's equator crosses its orbit (the equinox
        # direction), so the axis leans exactly sideways from it. Result:
        # northern summer arrives in June, as it should.
        # For other planets the lean direction is a simplification. Their
        # real poles point in other directions; only the tilt ANGLE is real.
        tilt = math.radians(body.axial_tilt)
        s.rotate(angle=-tilt, axis=vector(1, 0, 0), origin=s.pos)
        self.spin_axes[body.name] = vector(0, math.cos(tilt), -math.sin(tilt))

        self.spheres[body.name] = s
        self.labels[body.name] = label(
            pos=s.pos, text=body.name, xoffset=12, yoffset=12, height=11,
            color=color.white, box=False, opacity=0, line=False,
        )

    def _fit_camera(self):
        """Overview: back off along a downward-slanted line until every orbit fits."""
        farthest = max(np.linalg.norm(p) for p in self.sim.positions) * DISTANCE_SCALE
        # Distance needed so a circle of that radius fits the view:
        # half-width = distance * tan(half the field of view)
        distance = 1.15 * farthest / math.tan(self.scene.fov / 2)
        direction = vector(0, -0.5, -1).norm()      # looking down at an angle
        self.camera.set_home(-direction * distance, look_at=vector(0, 0, 0))
        self.camera.jump_home()

    def _build_controls(self):
        """The sidebar. Everything appended to the caption ends up in it."""
        sc = self.scene
        h = lambda title: sc.append_to_caption(f"<div class='ss-h'>{title}</div>")

        h("Simulation")
        self.pause_button = button(text="Pause", bind=self._toggle_pause)
        sc.append_to_caption("\nTime speed ")
        self.speed_text = wtext(text="")
        sc.append_to_caption("\n")
        # Logarithmic slider: -2..2.5 -> 0.01 .. ~316 days per second
        slider(min=-2, max=2.5, value=math.log10(self.days_per_second),
               length=SLIDER_LENGTH, bind=self._set_speed)
        self._update_speed_text()

        h("Camera")
        sc.append_to_caption("Follow ")
        self.follow_menu = menu(choices=["(free flight)"] + list(self.spheres),
                                bind=self._choose_follow)
        sc.append_to_caption(" ")
        button(text="Overview", bind=self._go_home)
        sc.append_to_caption("\nFlight speed ")
        self.flight_text = wtext(text="1.0x")
        sc.append_to_caption("\n")
        # -1..1 -> 0.1x .. 10x the automatic flying speed
        slider(min=-1, max=1, value=0, length=SLIDER_LENGTH, bind=self._set_flight_speed)
        sc.append_to_caption("\nCamera glide ")
        self.glide_text = wtext(text=f"{self.camera.glide:.2f} s")
        sc.append_to_caption("\n")
        # How long the camera takes to speed up / coast to a stop
        slider(min=0.05, max=1.5, value=self.camera.glide, length=SLIDER_LENGTH, bind=self._set_glide)

        h("Trails")
        checkbox(text="Show trails", checked=True, bind=self._toggle_trails)
        sc.append_to_caption("\n")
        checkbox(text="Full trail history", checked=False, bind=self._toggle_history)
        sc.append_to_caption("\n<span class='ss-note'>Off: each body keeps its last 2 orbits.</span>\n")
        button(text="Clear trails", bind=lambda _b: self.trails.clear())

        h("Display")
        checkbox(text="Labels", checked=True, bind=self._toggle_labels)
        sc.append_to_caption("   ")
        checkbox(text="Stars", checked=True, bind=self._toggle_stars)
        sc.append_to_caption("\n<span id='ss-fullscreen-slot'></span>")

        h("Flying")
        sc.append_to_caption(
            "<kbd>W</kbd>/<kbd>S</kbd> forward/back &nbsp; <kbd>A</kbd>/<kbd>D</kbd> left/right"
            "\n<kbd>E</kbd>/<kbd>Q</kbd> up/down &nbsp; <kbd>Shift</kbd> 5x faster"
            "\n<kbd>Arrows</kbd> turn &nbsp; <kbd>Wheel</kbd> glide forward/back"
            "\n<b>Hold left mouse</b> to steer toward the pointer"
            "\n<kbd>H</kbd> show/hide this panel &nbsp; <kbd>Esc</kbd> close it"
            "\n<span class='ss-note'>Speed adapts by itself: slow near planets, fast in open space. "
            "To watch Earth spin, follow Earth and set time speed below ~1 day/s.</span>"
        )

    # ------------------------------------------------------------ controls
    def _toggle_pause(self, b):
        self.paused = not self.paused
        b.text = "Play" if self.paused else "Pause"

    def _choose_follow(self, m):
        if m.index == 0:
            self.camera.stop_following()  # stay where you are, stop riding along
            self.following = None
        else:
            self.camera.follow(self.spheres[m.selected])
            self.following = m.selected

    def _go_home(self, _btn):
        self.follow_menu.index = 0
        self.following = None
        self.camera.go_home()

    def _set_flight_speed(self, s):
        self.camera.speed_factor = 10 ** s.value
        self.flight_text.text = f"{self.camera.speed_factor:.1f}x"

    def _set_glide(self, s):
        self.camera.glide = s.value
        self.glide_text.text = f"{s.value:.2f} s"

    def _set_speed(self, s):
        self.days_per_second = 10 ** s.value
        self._update_speed_text()

    def _update_speed_text(self):
        self.speed_text.text = f"{self.days_per_second:.2f} days / second"

    def _toggle_trails(self, cb):
        self.trails.set_visible(cb.checked)

    def _toggle_history(self, cb):
        self.trails.set_full_history(cb.checked, self.sim.time)

    def _toggle_labels(self, cb):
        for lab in self.labels.values():
            lab.visible = cb.checked

    def _toggle_stars(self, cb):
        self.stars.visible = cb.checked

    # ------------------------------------------------------------ drawing
    def _scene_position(self, i, body, positions_by_name):
        """Where to DRAW a body (moons get their distance exaggerated)."""
        pos = self.sim.positions[i]
        if body.parent and body.parent in positions_by_name:
            parent_pos = positions_by_name[body.parent]
            offset = (pos - parent_pos) * MOON_DISTANCE_EXAGGERATION
            return to_scene(parent_pos + offset)
        return to_scene(pos)

    def _redraw(self, seconds_advanced):
        positions_by_name = dict(zip(self.sim.names, self.sim.positions))
        for i, body in enumerate(self.sim.bodies):
            s = self.spheres[body.name]
            s.pos = self._scene_position(i, body, positions_by_name)
            self.labels[body.name].pos = s.pos

            # ROTATION: turn by (time passed / spin period) of a full circle.
            # rotate() uses the right-hand rule, so a positive angle is a
            # counter-clockwise spin seen from the north, which is how
            # Earth spins (the Sun rises in the east).
            if body.rotation_period and seconds_advanced > 0:
                angle = 2 * math.pi * seconds_advanced / body.rotation_period
                s.rotate(angle=angle, axis=self.spin_axes[body.name], origin=s.pos)

            if self.sun_light is not None and body.radius > 1e8:
                self.sun_light.pos = s.pos

    def _update_hud(self):
        now = self.start + timedelta(seconds=self.sim.time)
        state = "&#10074;&#10074; Paused" if self.paused else f"&#9654; {self.days_per_second:.2f} days/s"
        follow = f" &nbsp;·&nbsp; following <b>{self.following}</b>" if self.following else ""
        text = (f"<div class='ss-date'>{now:%Y-%m-%d %H:%M}</div>"
                f"<div class='ss-info'>{state}{follow}</div>"
                f"<div class='ss-hint'>H or &#9776; for controls &nbsp;·&nbsp; W/A/S/D fly &nbsp;·&nbsp; "
                f"hold left mouse to steer</div>")
        if text != self.hud.text:
            self.hud.text = text

    # --------------------------------------------------------------- loop
    def run(self):
        """Animate forever (close the browser tab / Ctrl+C to stop)."""
        dt = self.sim.dt
        while True:
            rate(FPS)  # at most FPS frames per second
            if not self.paused:
                # How much simulated time this frame should cover...
                self.time_debt += self.days_per_second * DAY / FPS
                # ...in whole physics steps. Any leftover carries to the next
                # frame, so even very slow speeds (less than one step per frame)
                # run at the correct average rate.
                n_steps = int(self.time_debt // dt)
                self.time_debt -= n_steps * dt
                if n_steps:
                    self.sim.step(n_steps)
                    self._redraw(n_steps * dt)
                    self.trails.update(self.sim.time)
            # The camera keeps working while paused, so you can fly around a
            # frozen Solar System. Its clock is REAL time (1/FPS s per frame),
            # not simulated time.
            self.camera.update(1.0 / FPS, self.spheres.values())
            self.stars.follow(self.camera.pos, self.camera.look_distance)
            self.trails.update_widths(self.camera.pos)
            self._update_hud()
