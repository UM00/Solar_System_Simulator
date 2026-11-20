"""
data_sources.py - real data from NASA, turned into Body objects.

TWO KINDS OF DATA
-----------------
1. STATE: where a body is and how fast it moves, on a given date.
   -> fetched live from NASA JPL's Horizons system (free, no API key).
   These are the INITIAL CONDITIONS. Gravity tells us HOW things move;
   the starting positions and velocities decide WHAT the orbits are.

2. PHYSICAL FACTS: mass, size, spin period, axial tilt.
   -> typed in below from NASA's Planetary Fact Sheet
      (nssdc.gsfc.nasa.gov/planetary/factsheet). These never change,
      so there's no need to download them.

Downloaded states are saved in data/horizons_cache.json. Running again
with the same date is instant, works offline, and doesn't hammer NASA.
"""

import json
import re
from pathlib import Path

import numpy as np

import nasa_api
from body import Body
from physics import G, orbital_period

AU = 1.495978707e11  # metres in one astronomical unit (exact, by definition)
DAY = 86400.0        # seconds in one day
HOUR = 3600.0
KM = 1000.0

CACHE_FILE = Path(__file__).parent / "data" / "horizons_cache.json"

# ---------------------------------------------------------------------------
# Physical facts
#   horizons_id    : the number Horizons uses. 10 = Sun, X99 = planet X's body
#                    (399 = Earth). Plain 3 would be the Earth-Moon BARYCENTER.
#                    For Jupiter..Neptune we deliberately use the BARYCENTER
#                    (5, 6, 7, 8): the planet plus all its moons, treated as one
#                    point at their shared balance point. We don't simulate their
#                    dozens of moons, so this is the most accurate stand-in.
#   GM (km^3/s^2)  : G times mass, from NASA JPL's DE440 planetary model.
#                    WHY GM AND NOT MASS? Gravity only ever uses the product
#                    G*M. Astronomers measure that product directly from
#                    spacecraft orbits, to ~11 digits. G on its own is only
#                    known to ~5 digits, so "mass in kg" is the LESS precise
#                    number. Using kg masses here made Earth end up ~60,000 km
#                    off after one year. Using GM fixes that.
#                    For the barycenters, GM includes the moons ("system" GM).
#   radius, rotation, tilt: NASA Planetary Fact Sheet.
#   rotation_hours : SIDEREAL period, one spin relative to the stars.
#                    Earth's is 23.93 h, not 24 h: while it spins, Earth also moves
#                    along its orbit, so it has to turn ~1 degree extra to face
#                    the Sun again. 24 h is the SOLAR day.
#   tilt_deg       : angle between the spin axis and the orbit's "north".
#                    Over 90 = spins backwards (retrograde). Venus is 177 deg,
#                    nearly upside-down, so it spins slowly backwards.
#                    Uranus at 98 deg rolls along its orbit on its side.
# ---------------------------------------------------------------------------
PHYSICAL_DATA = {
    #           id     GM (km^3/s^2)         radius(km) rotation(h) tilt(deg) color (r,g,b)
    "Sun":     ("10",  132712440041.279419, 695700.0,  609.12,     7.25,  (1.00, 0.85, 0.30)),
    "Mercury": ("199",        22031.868551,   2439.7, 1407.6,      0.034, (0.60, 0.58, 0.55)),
    "Venus":   ("299",       324858.592000,   6051.8, 5832.5,    177.4,   (0.95, 0.80, 0.50)),
    "Earth":   ("399",       398600.435507,   6371.0,   23.9345,  23.44,  (0.20, 0.45, 1.00)),
    "Moon":    ("301",         4902.800118,   1737.4,  655.72,     6.68,  (0.80, 0.80, 0.80)),
    "Mars":    ("499",        42828.375816,   3389.5,   24.6229,  25.19,  (0.85, 0.35, 0.20)),
    "Jupiter": ("5",      126712764.100000,  69911.0,    9.9250,   3.13,  (0.85, 0.70, 0.55)),
    "Saturn":  ("6",       37940584.841800,  58232.0,   10.656,   26.73,  (0.90, 0.80, 0.55)),
    "Uranus":  ("7",        5794556.400000,  25362.0,   17.24,    97.77,  (0.55, 0.85, 0.90)),
    "Neptune": ("8",        6836527.100580,  24622.0,   16.11,    28.32,  (0.30, 0.45, 0.95)),
}

# Bodies that orbit something other than the Sun (used only for drawing).
PARENTS = {"Moon": "Earth"}

# VPython has a built-in Earth texture, so the spin and tilt are visible.
TEXTURES = {"Earth": "earth"}


# ---------------------------------------------------------------------------
# Talking to NASA
# ---------------------------------------------------------------------------
def _load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def fetch_state_vector(horizons_id, date, id_type=None, **vector_options):
    """
    Ask JPL Horizons where a body is on `date` (e.g. "2026-01-01").

    Returns (position [m], velocity [m/s]) relative to the
    SOLAR SYSTEM BARYCENTER (balance point), in ecliptic coordinates.

    Settings sent to Horizons, and why:
      location='500@0'    -> measure from the Solar System Barycenter (id 0).
                             The Sun itself wobbles around this point,
                             mostly because of Jupiter.
      refplane='ecliptic' -> x-y plane = Earth's orbital plane, matching body.py.
      epochs (TDB scale)  -> "Barycentric Dynamical Time", the steady clock
                             used for planetary motion. It differs from
                             everyday UTC by about 69 seconds.
    """
    key = f"{horizons_id}|{id_type}|{date}"
    if vector_options:  # e.g. comets: closest_apparition=True
        key += "|" + ",".join(f"{k}={v}" for k, v in sorted(vector_options.items()))
    cache = _load_cache()
    if key in cache:
        entry = cache[key]
        return np.array(entry["position"]), np.array(entry["velocity"])

    # Imported here so the rest of the project works without astroquery installed.
    from astropy.time import Time
    from astroquery.jplhorizons import Horizons

    epoch = Time(date, scale="tdb")
    query = Horizons(id=horizons_id, id_type=id_type, location="500@0", epochs=epoch.jd)
    table = query.vectors(refplane="ecliptic", **vector_options)
    row = table[0]

    # Horizons answers in AU and AU/day. Convert to SI (metres, m/s).
    position = np.array([float(row["x"]), float(row["y"]), float(row["z"])]) * AU
    velocity = np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])]) * AU / DAY

    cache[key] = {
        "name": str(row["targetname"]),  # what Horizons actually matched, worth checking
        "position": position.tolist(),
        "velocity": velocity.tolist(),
    }
    _save_cache(cache)
    return position, velocity


# ---------------------------------------------------------------------------
# Building Bodies
# ---------------------------------------------------------------------------
def make_body(name, date, without_moons_gm=None):
    """
    Create a Sun/planet/Moon Body with real NASA starting conditions.

    without_moons_gm: leave as None normally. When this planet's moons are
        ALSO being simulated, pass the planet-only GM here. The planet then
        switches from "planet + moons as one point" (barycenter, e.g. id 5)
        to "the planet's own body" (id 599), so no mass is counted twice.
    """
    if name not in PHYSICAL_DATA:
        raise KeyError(f"No physical data for {name!r}. Known: {list(PHYSICAL_DATA)}")
    horizons_id, gm_km3, radius_km, rotation_h, tilt, color = PHYSICAL_DATA[name]
    if without_moons_gm is not None:
        horizons_id = f"{nasa_api.PLANET_NUMBERS[name]}99"
        gm_km3 = without_moons_gm
    position, velocity = fetch_state_vector(horizons_id, date)
    # km^3 -> m^3 is a factor of 1000^3 = 1e9. Dividing by G gives kg, and the
    # physics multiplies by G again, so G*mass == NASA's GM exactly.
    mass = gm_km3 * 1e9 / G
    return Body(
        name=name,
        mass=mass,
        position=position,
        velocity=velocity,
        radius=radius_km * KM,
        rotation_period=rotation_h * HOUR,
        axial_tilt=tilt,
        color=color,
        texture=TEXTURES.get(name),
        parent=PARENTS.get(name),
    )


def make_small_body(designation, date, name=None, radius_km=50.0, color=(0.7, 0.7, 0.6)):
    """
    Create an asteroid or comet as a TEST PARTICLE (pulled, but doesn't pull).

    designation examples:
        "1"    -> asteroid number 1, Ceres
        "4"    -> Vesta
        "99942"-> Apophis
    Horizons treats a plain number as asteroid number N when
    id_type="smallbody".

    The radius is just for drawing. Asteroids are tiny, so the default is
    a placeholder. You can get real sizes later from JPL's Small-Body
    Database (SBDB) API.
    """
    position, velocity = fetch_state_vector(designation, date, id_type="smallbody")
    return Body(
        name=name or f"Asteroid {designation}",
        mass=0.0,
        position=position,
        velocity=velocity,
        is_test_particle=True,
        radius=radius_km * KM,
        color=color,
    )


# ---------------------------------------------------------------------------
# Moons, asteroids and comets from nasa_api.py
# ---------------------------------------------------------------------------
MOON_COLOR = (0.75, 0.72, 0.68)
ASTEROID_COLOR = (0.70, 0.66, 0.55)
COMET_COLOR = (0.60, 0.85, 1.00)


def make_moon(moon, date, parent_body, physical=None):
    """
    Build a moon Body from a list_moons() entry, e.g.
        {"horizons_id": "501", "name": "Io", "planet": "Jupiter"}

    physical    : the moon_physical_data() entry for it, if JPL has one.
                  Moons with a measured GM pull on things. Tiny moons
                  without one become test particles (pulled, not pulling).
    parent_body : the planet's Body. Needed to work out the moon's orbit.

    SPIN: nearly every large moon is TIDALLY LOCKED. The planet's gravity
    stretched it slightly into an egg shape, and over millions of years the
    friction of that stretching slowed its spin until one rotation took
    exactly one orbit. That's why we always see the same face of our Moon.
    So spin period = orbital period, which Kepler's laws give us from the
    moon's position and velocity relative to its planet.
    """
    position, velocity = fetch_state_vector(moon["horizons_id"], date)
    physical = physical or {}
    gm_km3 = physical.get("gm")
    mass = gm_km3 * 1e9 / G if gm_km3 else 0.0

    mu = G * (parent_body.mass + mass)
    period = orbital_period(position - parent_body.position,
                            velocity - parent_body.velocity, mu)
    return Body(
        name=moon["name"],
        mass=mass,
        position=position,
        velocity=velocity,
        is_test_particle=not gm_km3,
        radius=(physical.get("radius_km") or 5.0) * KM,
        rotation_period=period,  # tidally locked (see above)
        color=MOON_COLOR,
        parent=parent_body.name,
    )


def short_name(full_name):
    """'1 Ceres (A801 AA)' -> 'Ceres' ;  '15789 (1993 SC)' -> '1993 SC' ;  '1P/Halley' -> '1P/Halley'"""
    m = re.match(r"^\d+\s+([^(]+?)\s*(\(.*\))?$", full_name)
    if m and m.group(1):
        return m.group(1)
    m = re.match(r"^\d+\s+\((.+)\)$", full_name)
    return m.group(1) if m else full_name


def make_small_body_from_record(record, date):
    """
    Build an asteroid/comet Body from a nasa_api.query_small_bodies() result.

    If JPL has measured its GM (only the biggest ones, like Ceres and Vesta),
    it gets real gravity. Otherwise it's a test particle.

    How we ask Horizons for its position (both found by testing):
      asteroid : "DES=<spkid>;"  (the plain number alone isn't accepted)
      comet    : its designation + closest_apparition. A comet like Halley
                 has a separate orbit solution for each return to the Sun,
                 so we pick the one closest to our date.
    """
    if record["kind"] == "comet":
        position, velocity = fetch_state_vector(
            record["designation"], date, id_type="designation",
            closest_apparition=True, no_fragments=True)
        color = COMET_COLOR
    else:
        position, velocity = fetch_state_vector(f"DES={record['spkid']};", date)
        color = ASTEROID_COLOR

    gm_km3 = record.get("gm")
    return Body(
        name=short_name(record["name"]),
        mass=gm_km3 * 1e9 / G if gm_km3 else 0.0,
        position=position,
        velocity=velocity,
        is_test_particle=not gm_km3,
        radius=(record.get("diameter_km") or 10.0) / 2 * KM,
        rotation_period=record["rotation_hours"] * HOUR if record.get("rotation_hours") else None,
        color=color,
    )


def build_system(date, planet_names, moons=None, small_body_queries=(), extra_small_bodies=()):
    """
    Assemble a whole simulation in one call.

    planet_names       : ["Sun", "Earth", "Moon", "Jupiter", ...]  (PHYSICAL_DATA names)
    moons              : {"Jupiter": ["Io", "Europa"], "Saturn": "measured", "Mars": "all"}
                           a list of names -> just those moons
                           "measured"      -> every moon JPL has a mass for
                           "all"           -> every moon NASA tracks (Saturn: ~290!)
    small_body_queries : list of dicts for nasa_api.query_small_bodies,
                         e.g. [{"kind": "asteroid", "limit": 10}]
    extra_small_bodies : [("1", "Ceres"), ...] the older hand-picked style

    THE DOUBLE-COUNTING RULE: if a planet's moons are included, that planet
    switches to its body-only position and GM (see make_body). Otherwise its
    moons' mass would be counted twice.
    """
    moons = moons or {}
    bodies = []
    planet_gm = nasa_api.planet_gm_without_moons() if moons else {}
    for name in planet_names:
        has_moons_added = name in moons and name in nasa_api.PLANET_NUMBERS
        bodies.append(make_body(name, date, planet_gm.get(name) if has_moons_added else None))

    names_taken = {b.name for b in bodies}
    if moons:
        physical = nasa_api.moon_physical_data()
        for planet, choice in moons.items():
            parent = next((b for b in bodies if b.name == planet), None)
            if parent is None:
                raise ValueError(f"Moons of {planet} requested, but {planet} is not in planet_names")
            available = nasa_api.list_moons(planet)
            if choice == "all":
                chosen = available
            elif choice == "measured":
                chosen = [m for m in available if (physical.get(m["horizons_id"]) or {}).get("gm")]
            else:
                by_name = {m["name"]: m for m in available}
                missing = [n for n in choice if n not in by_name]
                if missing:
                    raise ValueError(f"{planet} has no moon called {missing}. Try nasa_api.list_moons({planet!r})")
                chosen = [by_name[n] for n in choice]
            for moon in chosen:
                if moon["name"] in names_taken:  # e.g. Earth's "Moon" already listed
                    continue
                bodies.append(make_moon(moon, date, parent, physical.get(moon["horizons_id"])))
                names_taken.add(moon["name"])

    seen_ids = set()
    for query in small_body_queries:
        for record in nasa_api.query_small_bodies(**query):
            if record["spkid"] in seen_ids:  # same object from two queries
                continue
            seen_ids.add(record["spkid"])
            body = make_small_body_from_record(record, date)
            # Names can clash across kinds: asteroid "52 Europa" vs Jupiter's
            # moon Europa. Then use the full catalogue name instead.
            if body.name in names_taken:
                body.name = record["name"]
            bodies.append(body)
            names_taken.add(body.name)

    for designation, label in extra_small_bodies:
        if label not in names_taken:
            bodies.append(make_small_body(designation, date, name=label))
            names_taken.add(label)
    return bodies


def suggest_time_step(bodies, steps_per_orbit=100):
    """
    The largest SAFE time step (seconds) for these bodies.

    Rule of thumb from the integrator: you want ~100 steps per orbit of the
    FASTEST thing in the system. Fewer, and the integrator cuts corners
    on the tight curve, and the moon spirals off or crashes.
        Moon around Earth : 27.3 days -> dt up to ~6.5 hours
        Io around Jupiter : 1.77 days -> dt up to ~25 minutes
        Phobos around Mars: 7.65 hours -> dt up to ~4.6 minutes (slow to run!)
    Returns (dt_seconds, name_of_fastest_body), or (None, None) if no body has a parent.
    """
    by_name = {b.name: b for b in bodies}
    fastest, fastest_name = None, None
    for b in bodies:
        if not b.parent or b.parent not in by_name:
            continue
        p = by_name[b.parent]
        period = orbital_period(b.position - p.position, b.velocity - p.velocity,
                                G * (p.mass + b.mass))
        if period and (fastest is None or period < fastest):
            fastest, fastest_name = period, b.name
    if fastest is None:
        return None, None
    return fastest / steps_per_orbit, fastest_name
