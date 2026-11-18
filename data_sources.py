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
from pathlib import Path

import numpy as np

from body import Body
from physics import G

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


def fetch_state_vector(horizons_id, date, id_type=None):
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
    cache = _load_cache()
    if key in cache:
        entry = cache[key]
        return np.array(entry["position"]), np.array(entry["velocity"])

    # Imported here so the rest of the project works without astroquery installed.
    from astropy.time import Time
    from astroquery.jplhorizons import Horizons

    epoch = Time(date, scale="tdb")
    query = Horizons(id=horizons_id, id_type=id_type, location="500@0", epochs=epoch.jd)
    table = query.vectors(refplane="ecliptic")
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
def make_body(name, date):
    """Create a Sun/planet/Moon Body with real NASA starting conditions."""
    if name not in PHYSICAL_DATA:
        raise KeyError(f"No physical data for {name!r}. Known: {list(PHYSICAL_DATA)}")
    horizons_id, gm_km3, radius_km, rotation_h, tilt, color = PHYSICAL_DATA[name]
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
