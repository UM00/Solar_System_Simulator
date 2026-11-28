"""
events.py - predict REAL astronomical events with our simulation and check
them against NASA's published answers.

THE IDEA
--------
A simulation is only worth trusting if it can predict something that can be
checked. So this module starts from NASA's positions on START_DATE, runs our
own physics (velocity Verlet, Newtonian gravity) forward by months or years,
and asks: WHEN does the event happen in our simulation, and HOW CLOSE is it?
Then it puts our answer next to NASA's.

    Predict  : a separate, behind-the-scenes simulation (no drawing). Only
               the Sun, the planets, the Moon and (for a flyby) the asteroid.
               This is the honest test: everything after START_DATE is ours.
    Show me  : the 3D view "time-travels" to the event (NASA's positions on
               that day) and plays it in slow motion with a live readout.

Nothing here runs until you click, so it costs nothing at start-up.

THE EVENTS AND WHAT "CLOSE" MEANS
---------------------------------
1. ASTEROID FLYBY (Apophis, 13 April 2029). The 340 m asteroid passes
   closer to Earth than our geostationary satellites. We look for the moment
   the Earth-Apophis distance is smallest.

2. SOLAR ECLIPSE. The Moon's shadow points along the line from the Sun
   through the Moon (the SHADOW AXIS). An eclipse happens when that line
   passes close to Earth. Astronomers measure how close with GAMMA:
       gamma = (shortest distance from Earth's centre to the shadow axis)
               / (Earth's equatorial radius, 6,378 km)
   gamma 0 = the shadow hits the middle of Earth's disc; around 1 = it
   grazes the edge (near the poles); much above 1 = it misses Earth (no
   eclipse). Positive = the axis passes NORTH of Earth's centre.
   The moment gamma is smallest is called GREATEST ECLIPSE.

3. LUNAR ECLIPSE. The same with the roles swapped: Earth's shadow points
   along the Sun->Earth line, and gamma is the Moon's distance from it.

Whether a solar eclipse is TOTAL or ANNULAR depends on whether the tip of
the Moon's dark shadow cone (the UMBRA) reaches Earth's surface. Similar
triangles give the cone's length:
    umbra length = (Sun-Moon distance) x R_moon / (R_sun - R_moon)
which is ~374,000 km, while the Moon is 357,000-406,000 km away. That's
why some eclipses are total and others annular (a "ring of fire").

FINDING THE EXACT MOMENT
------------------------
Stepping with 25-minute or 1-hour steps would put the answer on a coarse
grid. So we run normally until SEARCH_WINDOW before NASA's date, then switch
to FINE_STEP (1 minute) steps and record (distance)^2 at every step. Near its
minimum, a straight-line pass gives (distance)^2 = d0^2 + v^2 (t - t0)^2,
an exact parabola, so a parabola through the 3 smallest samples pins the
moment down to well under a second. The window only says WHERE to look;
the simulation has to find the minute itself.

WHAT WE IGNORE (and why the small differences are expected)
-----------------------------------------------------------
- General relativity, the shape (bulge) of Earth and the Moon, and tides.
  NASA's ephemeris includes them; ours is pure Newton with point masses.
- Light travel time (~8 minutes from the Sun): it shifts the shadow by only
  a few km here.
"""

import math
from datetime import datetime, timedelta

import numpy as np

from data_sources import AU, DAY, KM, build_system, pole_vector, suggest_time_step
from simulation import Simulation

R_EARTH_EQ = 6378.137 * KM   # Earth's EQUATORIAL radius: what NASA's gamma is measured in
R_MOON = 1737.4 * KM
R_SUN = 695700.0 * KM
SPEED_OF_LIGHT = 299792458.0  # m/s

# The behind-the-scenes prediction uses only the bodies that matter for these
# events. Jupiter..Neptune are their barycenters (planet + moons as one point).
PREDICTION_BODIES = ["Sun", "Mercury", "Venus", "Earth", "Moon", "Mars",
                     "Jupiter", "Saturn", "Uranus", "Neptune"]
PREDICTION_DT = 300.0        # 5-minute steps for the long run (measured: 15 min put eclipses ~4x further off)
FINE_STEP = 60.0             # 1-minute steps near the event
SEARCH_WINDOW = 2 * DAY      # fine steps from 2 days before NASA's time to 2 days after
WATCH_LEAD = 3 * 3600.0      # "Show me" starts this long before the event
WATCH_DAYS_PER_SECOND = 0.01 # ...and plays at this speed (1 hour of sky = 4.2 s)

# ---------------------------------------------------------------------------
# NASA's published answers (TD/TDB time, the same steady clock our data uses)
# ---------------------------------------------------------------------------
CAD_API = "https://ssd-api.jpl.nasa.gov/cad.api?des=99942&date-min=2029-01-01&date-max=2029-12-31&body=Earth"
SOLAR_CANON = "https://eclipse.gsfc.nasa.gov/SEcat5/SE2001-2100.html"
LUNAR_CANON = "https://eclipse.gsfc.nasa.gov/LEcat5/LE2001-2100.html"

EVENTS = [
    {"key": "lunar-2026-03", "kind": "lunar", "title": "Total lunar eclipse, 3 Mar 2026",
     "time": "2026-03-03 11:34:52", "gamma": -0.3765, "type": "total", "source": LUNAR_CANON},
    {"key": "solar-2026-08", "kind": "solar", "title": "Total solar eclipse, 12 Aug 2026 (Iceland, Spain)",
     "time": "2026-08-12 17:47:06", "gamma": 0.8977, "type": "total", "source": SOLAR_CANON},
    {"key": "lunar-2026-08", "kind": "lunar", "title": "Partial lunar eclipse, 28 Aug 2026",
     "time": "2026-08-28 04:14:04", "gamma": 0.4964, "type": "partial", "source": LUNAR_CANON},
    {"key": "solar-2027-08", "kind": "solar", "title": "Total solar eclipse, 2 Aug 2027 (Egypt, 6m23s)",
     "time": "2027-08-02 10:07:50", "gamma": 0.1421, "type": "total", "source": SOLAR_CANON},
    {"key": "apophis-2029", "kind": "flyby", "title": "Apophis passes Earth, 13 Apr 2029",
     "body": "Apophis", "designation": "99942",
     # CNEOS close-approach data: jd 2462240.407091969 = 2029-04-13 21:46:13 TDB
     "time": "2029-04-13 21:46:13", "distance_km": 0.000254090910419299 * AU / KM,
     "speed_kms": 7.42253895678452, "source": CAD_API},
]


def titles():
    """The event names for the sidebar menu."""
    return [e["title"] for e in EVENTS]


def seconds_between(start_date, when):
    return (datetime.fromisoformat(when) - datetime.fromisoformat(start_date)).total_seconds()


# ---------------------------------------------------------------------------
# The geometry: one number per event that is smallest at the event
# ---------------------------------------------------------------------------
def _axis_offset(source, through, target):
    """
    The shortest vector from `target` to the straight line that starts at
    `source` and goes through `through` (e.g. Sun -> Moon = the Moon's
    shadow axis). Projection: drop the target onto the line, keep the rest.
    """
    axis = (through - source) / np.linalg.norm(through - source)
    rel = target - through
    return -(rel - np.dot(rel, axis) * axis), axis


def shadow_positions(event, pos, vel):
    """
    LIGHT TRAVEL TIME. A shadow is made of light that is not there. The
    shadow touching Earth NOW was cast by the Moon 1.3 seconds AGO (the time
    light takes from the Moon to Earth), by light that left the Sun 8 minutes
    before that. In those 1.3 s the Moon, racing around the Sun with Earth at
    ~30 km/s, moved about 38 km, and the shadow crosses Earth at ~1 km/s. So
    using everyone's position at the same instant would put the eclipse
    ~40 seconds late. (Measured: without this step every eclipse came out
    40-50 s late.) Astronomers call the same effect "aberration".

    So we use the positions where the light really was: position minus
    velocity x travel time (a straight line is perfect over 1 second).
    Lunar eclipse: the same, with Earth casting the shadow onto the Moon.
    """
    if vel is None:
        return pos
    pos = dict(pos)
    caster = "Moon" if event["kind"] == "solar" else "Earth"
    receiver = "Earth" if event["kind"] == "solar" else "Moon"
    tau = np.linalg.norm(pos[receiver] - pos[caster]) / SPEED_OF_LIGHT
    tau_sun = tau + np.linalg.norm(pos[caster] - pos["Sun"]) / SPEED_OF_LIGHT
    pos[caster] = pos[caster] - vel[caster] * tau
    pos["Sun"] = pos["Sun"] - vel["Sun"] * tau_sun
    return pos


def measure(event, pos, vel=None):
    """
    pos: {"Sun": xyz, "Earth": xyz, ...} in metres (vel: the same in m/s,
    used for the light-travel-time correction, see shadow_positions).
    Returns the (signed) quantity that is smallest at the event:
      flyby : distance between the body and Earth (m)
      eclipse : gamma (Earth radii, + = north)
    and the unsigned distance in metres used to find the minimum.
    """
    if event["kind"] == "flyby":
        d = np.linalg.norm(pos[event["body"]] - pos["Earth"])
        return d, d
    pos = shadow_positions(event, pos, vel)
    if event["kind"] == "solar":     # Moon's shadow, measured at Earth
        offset, _ = _axis_offset(pos["Sun"], pos["Moon"], pos["Earth"])
    else:                            # Earth's shadow, measured at the Moon
        offset, _ = _axis_offset(pos["Sun"], pos["Earth"], pos["Moon"])
    # The sign: is the axis NORTH or SOUTH of Earth's centre (lunar: is the
    # Moon north or south of the axis)? "North" = along Earth's spin axis.
    north = pole_vector("Earth", event["time"][:10])
    toward_axis = offset if event["kind"] == "solar" else -offset
    sign = 1.0 if np.dot(toward_axis, north) >= 0 else -1.0
    d = np.linalg.norm(offset)
    return sign * d / R_EARTH_EQ, d


def eclipse_type(event, pos, gamma):
    """
    total / annular / partial, from cone geometry (see the top of the file).
    """
    if event["kind"] == "solar":
        sun, moon, earth = pos["Sun"], pos["Moon"], pos["Earth"]
        p = abs(gamma) * R_EARTH_EQ
        if p > R_EARTH_EQ:
            return "partial"     # the shadow axis misses Earth (only the outer shadow touches)
        umbra_length = np.linalg.norm(moon - sun) * R_MOON / (R_SUN - R_MOON)
        # From the Moon along the axis to where it pierces Earth's surface:
        _, axis = _axis_offset(sun, moon, earth)
        to_surface = np.dot(earth - moon, axis) - math.sqrt(R_EARTH_EQ**2 - p**2)
        return "total" if umbra_length > to_surface else "annular"
    # Lunar: the radius of Earth's umbra where the Moon is. Earth's
    # atmosphere makes the real shadow ~2% bigger (a standard correction).
    sun, earth, moon = pos["Sun"], pos["Earth"], pos["Moon"]
    d_moon = np.linalg.norm(moon - earth)
    umbra = 1.02 * (R_EARTH_EQ - (R_SUN - R_EARTH_EQ) * d_moon / np.linalg.norm(earth - sun))
    p = abs(gamma) * R_EARTH_EQ
    if p + R_MOON <= umbra:
        return "total"
    if p - R_MOON < umbra:
        return "partial"
    return "penumbral"


# ---------------------------------------------------------------------------
# Predict: our own simulation, from START_DATE to the event
# ---------------------------------------------------------------------------
def predict(event, start_date, progress=None):
    """
    Run a behind-the-scenes simulation from start_date to the event and
    measure it. progress(fraction) is called now and then (for the sidebar).
    Returns a dict with our numbers next to NASA's.
    """
    t_nasa = seconds_between(start_date, event["time"])
    if t_nasa <= SEARCH_WINDOW:
        raise ValueError(f"This event is before (or too close to) START_DATE {start_date}.")
    extra = [(event["designation"], event["body"])] if event["kind"] == "flyby" else []
    bodies = build_system(start_date, PREDICTION_BODIES, extra_small_bodies=extra)
    safe, _ = suggest_time_step(bodies)
    sim = Simulation(bodies, dt=min(PREDICTION_DT, safe or PREDICTION_DT))
    sim.move_to_center_of_mass()
    names = sim.names

    # 1. The long run, in chunks so we can report progress.
    t_fine = t_nasa - SEARCH_WINDOW
    total_steps = int(t_fine // sim.dt)
    done = 0
    while done < total_steps:
        n = min(2000, total_steps - done)
        sim.step(n)
        done += n
        if progress:
            progress(0.9 * done / total_steps)

    # 2. Fine steps through the search window, recording the distance.
    sim.set_dt(FINE_STEP)
    times, dist2 = [], []
    best = None
    for k in range(int(2 * SEARCH_WINDOW / FINE_STEP) + 1):
        pos = dict(zip(names, sim.positions))
        vel = dict(zip(names, sim.velocities))
        signed, d = measure(event, pos, vel)
        times.append(sim.time)
        dist2.append(d * d)
        if best is None or d < best[0]:
            best = (d, k, signed, shadow_positions(event, {n: p.copy() for n, p in pos.items()}, vel)
                    if event["kind"] != "flyby" else pos, {n: v.copy() for n, v in vel.items()})
        sim.step(1)
        if progress and k % 500 == 0:
            progress(0.9 + 0.1 * k / (2 * SEARCH_WINDOW / FINE_STEP))

    # 3. A parabola through the 3 samples around the smallest one
    #    (exact for a straight-line pass, see the top of the file).
    d_min, k, signed, pos_min, vel_min = best
    if k in (0, len(times) - 1):
        raise RuntimeError("The closest moment fell outside the search window.")
    y0, y1, y2 = dist2[k - 1], dist2[k], dist2[k + 1]
    curvature = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / curvature           # in steps, between -1 and 1
    t_ours = times[k] + shift * FINE_STEP
    d2_ours = y1 - 0.25 * (y0 - y2) * shift
    d_ours = math.sqrt(max(d2_ours, 0.0))

    when_ours = datetime.fromisoformat(start_date) + timedelta(seconds=t_ours)
    result = {
        "event": event,
        "days_simulated": t_ours / DAY,
        "time_ours": when_ours,
        "time_nasa": datetime.fromisoformat(event["time"]),
        "time_error_s": t_ours - t_nasa,
    }
    if event["kind"] == "flyby":
        rel_v = vel_min[event["body"]] - vel_min["Earth"]
        result.update(distance_km=d_ours / KM,
                      distance_error_km=d_ours / KM - event["distance_km"],
                      speed_kms=np.linalg.norm(rel_v) / KM)
    else:
        gamma = math.copysign(d_ours / R_EARTH_EQ, signed)
        result.update(gamma=gamma, gamma_error=gamma - event["gamma"],
                      gamma_error_km=(gamma - event["gamma"]) * R_EARTH_EQ / KM,
                      type=eclipse_type(event, pos_min, gamma))
    if progress:
        progress(1.0)
    return result


def describe(result):
    """The prediction-vs-NASA table, as HTML for the sidebar."""
    e = result["event"]
    dt = result["time_error_s"]
    rows = [("When", f"{result['time_ours']:%Y-%m-%d %H:%M:%S}", f"{result['time_nasa']:%Y-%m-%d %H:%M:%S}",
             f"{dt:+.0f} s")]
    if e["kind"] == "flyby":
        rows.append(("Distance from Earth's centre", f"{result['distance_km']:,.0f} km",
                     f"{e['distance_km']:,.0f} km", f"{result['distance_error_km']:+,.0f} km"))
        rows.append(("Speed relative to Earth", f"{result['speed_kms']:.3f} km/s",
                     f"{e['speed_kms']:.3f} km/s", f"{result['speed_kms'] - e['speed_kms']:+.3f}"))
    else:
        rows.append(("Gamma", f"{result['gamma']:+.4f}", f"{e['gamma']:+.4f}",
                     f"{result['gamma_error']:+.4f} ({result['gamma_error_km']:+,.0f} km)"))
        rows.append(("Type", result["type"], e["type"], "&#10003;" if result["type"] == e["type"] else "&#10007;"))
    head = "<tr><th></th><th>Ours</th><th>NASA</th><th>Diff</th></tr>"
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)
    return (f"<table class='ss-evt-table'>{head}{body}</table>"
            f"<span class='ss-note'>Predicted from NASA's {result['days_simulated']:.0f}-day-old starting "
            f"positions with our own physics. <a href='{e['source']}' target='_blank'>NASA source</a></span>")


# ---------------------------------------------------------------------------
# Show me: where to time-travel the 3D view, and the live readout
# ---------------------------------------------------------------------------
def watch_plan(event, start_date):
    """
    Returns (load_date, load_time, play_from_time) in seconds since start_date:
    load NASA's positions at midnight of the day 3+ hours before the event,
    then fast-forward to WATCH_LEAD before it.
    """
    moment = datetime.fromisoformat(event["time"])
    load = (moment - timedelta(seconds=WATCH_LEAD)).replace(hour=0, minute=0, second=0)
    t_start = (load - datetime.fromisoformat(start_date)).total_seconds()
    t_play = seconds_between(start_date, event["time"]) - WATCH_LEAD
    return load.strftime("%Y-%m-%d"), t_start, t_play


def live_readout(event, names, positions, velocities, time_to_event):
    """One line for the HUD while watching, from the LIVE simulation."""
    signed, _ = measure(event, dict(zip(names, positions)), dict(zip(names, velocities)))
    clock = f"{'T-' if time_to_event > 0 else 'T+'}{abs(time_to_event) / 60:.0f} min"
    if event["kind"] == "flyby":
        return f"{event['body']} is {signed / KM:,.0f} km from Earth's centre &nbsp;·&nbsp; {clock}"
    what = "Moon's shadow axis passes" if event["kind"] == "solar" else "Moon is"
    where = "north" if signed >= 0 else "south"
    return (f"{what} {abs(signed):.3f} Earth radii {where} of "
            f"{'Earth' if event['kind'] == 'solar' else 'the shadow axis'} &nbsp;·&nbsp; gamma now {signed:+.4f}"
            f" &nbsp;·&nbsp; {clock}")
