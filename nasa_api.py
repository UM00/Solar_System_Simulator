"""
nasa_api.py - one simple doorway to NASA/JPL data on moons, asteroids and comets.

WHY THIS FILE EXISTS
--------------------
NASA has no single "give me every moon and asteroid" API. The data is spread
over several official JPL services, each with its own format:

  WHAT                                   JPL SOURCE                       FORMAT
  ------------------------------------   ------------------------------   ---------
  List of every moon of every planet     Horizons "major body" index      plain text
  Mass (GM) and size of moons            Planetary Satellite Physical     HTML table
                                         Parameters page
  Planet mass WITHOUT its moons          same page, second table          HTML table
  Search the 1.5 million+ asteroids      SBDB Query API                   JSON
    and ~4,000 comets
  Orbits of thousands of asteroids       SBDB Query API (elements)        JSON
    at once (asteroid belt, trojans)
  Find any object by name                Horizons Lookup API              JSON

This file hides those differences behind plain Python functions:

    list_moons("Jupiter")             -> every moon NASA tracks for Jupiter
    moon_physical_data()              -> GM / radius / density for each moon
    planet_gm_without_moons()         -> {"Jupiter": 126686531.9, ...}
    query_small_bodies(...)           -> asteroids or comets matching a filter
    query_orbital_elements("MBA")     -> orbits of the N biggest main-belt asteroids
    lookup("Halley")                  -> what NASA calls this object, and its IDs

The rest of the project never needs to know which website an answer came
from. This is called an ADAPTER: if NASA changes a format one day, you fix
it here and nothing else changes.

CACHING
-------
Every answer is saved in data/nasa_cache/. The second time you ask the same
question it's instant and works offline. Pass refresh=True to any function to
force a fresh download (for example, to catch newly discovered moons).

All functions use only Python's standard library (urllib, json, html.parser),
so there's nothing extra to install.
"""

import hashlib
import json
import re
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data" / "nasa_cache"

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
LOOKUP_API = "https://ssd.jpl.nasa.gov/api/horizons_lookup.api"
SBDB_QUERY_API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
SAT_PHYS_PAGE = "https://ssd.jpl.nasa.gov/sats/phys_par/"

# Horizons numbers each planet 1-9 (Mercury = 1 ... Pluto = 9).
# Its moons are then numbered X01, X02, ... (Jupiter's: 501 Io, 502 Europa...).
PLANET_NUMBERS = {
    "Mercury": 1, "Venus": 2, "Earth": 3, "Mars": 4, "Jupiter": 5,
    "Saturn": 6, "Uranus": 7, "Neptune": 8, "Pluto": 9,
}


# ---------------------------------------------------------------------------
# Downloading (with cache)
# ---------------------------------------------------------------------------
def _fetch_text(url, params=None, refresh=False):
    """Download a URL (GET) and return its text, using the on-disk cache."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    # The cache file name is a fingerprint (hash) of the full URL, so every
    # distinct question gets its own file.
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.txt"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    # LAZY LOADING: the networking library is only imported when something
    # really has to be downloaded. With a warm cache it's never loaded at all.
    # (Imported by name: writing "import urllib.request" inside this function
    # would make "urllib" a LOCAL name for the whole function, and the
    # urllib.parse line above would then fail. A classic Python scoping trap.)
    from urllib.request import Request, urlopen

    request = Request(url, headers={"User-Agent": "solar-system-simulator"})
    with urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _fetch_json(url, params=None, refresh=False):
    return json.loads(_fetch_text(url, params, refresh))


def _to_float(text):
    """'5959.9155' -> 5959.9155 ;  '' or None or '-' -> None"""
    if text is None:
        return None
    text = str(text).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 1. MOONS: the list (Horizons major-body index)
# ---------------------------------------------------------------------------
def list_moons(planet, refresh=False):
    """
    Every moon of `planet` that NASA tracks, as a list of dicts:
        [{"horizons_id": "501", "name": "Io", "planet": "Jupiter"}, ...]

    Source: Horizons' "MB" (major body) listing, a plain-text table of every
    object with a pre-computed trajectory. Moon IDs follow a pattern:
        X01..X98  -> moons of planet X  (e.g. 501..5xx for Jupiter)
        X5nnn     -> newer moons that ran out of 3-digit numbers (e.g. 55501)
    Some recently found moons have no name yet, only a designation like
    "S2003_J2". We use that as the name.
    """
    if planet not in PLANET_NUMBERS:
        raise KeyError(f"Unknown planet {planet!r}. Choose from {list(PLANET_NUMBERS)}")
    p = str(PLANET_NUMBERS[planet])

    text = _fetch_text(HORIZONS_API, {"format": "text", "COMMAND": "'MB'"}, refresh)
    moons = []
    for line in text.splitlines():
        # Lines look like:  "      501  Io                                  JI"
        match = re.match(r"^\s+(\d+)\s{2}(.*)$", line)
        if not match:
            continue
        hid, rest = match.group(1), match.group(2)
        is_classic = len(hid) == 3 and hid[0] == p and hid[1:] not in ("99", "00")
        is_newer = len(hid) == 5 and hid[0] == p and hid[1] == "5"
        if not (is_classic or is_newer):
            continue
        # Columns: name (35 chars), designation, aliases
        name = rest[:35].strip()
        designation = rest[35:48].strip()
        moons.append({"horizons_id": hid, "name": name or designation, "planet": planet})
    return moons


# ---------------------------------------------------------------------------
# 2. MOONS: mass and size (JPL Planetary Satellite Physical Parameters page)
# ---------------------------------------------------------------------------
class _TableReader(HTMLParser):
    """
    Collects the text of every table on a web page as lists of rows.
    (That JPL page doesn't always close its <td> tags, so we start a new
    cell whenever a new <td> or <th> opens instead of waiting for </td>.)
    """

    def __init__(self):
        super().__init__()
        self.tables = []   # list of tables; each table = list of rows; row = list of cells
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._finish_row()
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._finish_cell()
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "tr":
            self._finish_row()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def _finish_cell(self):
        if self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
        self._cell = None

    def _finish_row(self):
        self._finish_cell()
        if self._row:
            self.tables[-1].append(self._row)
        self._row = None


def _satellite_page_tables(refresh=False):
    reader = _TableReader()
    reader.feed(_fetch_text(SAT_PHYS_PAGE, refresh=refresh))
    return reader.tables


def moon_physical_data(refresh=False):
    """
    Measured GM, mean radius and density for every moon JPL has values for,
    keyed by Horizons ID:
        {"501": {"name": "Io", "planet": "Jupiter", "gm": 5959.9155,
                 "radius_km": 1821.49, "density": 3.528}, ...}
    Values that haven't been measured are None.

    WHY GM AGAIN? Same reason as for planets (see data_sources.py): JPL
    measures G*M directly from how spacecraft and other moons are pulled.
    """
    result = {}
    for table in _satellite_page_tables(refresh):
        for row in table:
            # Data rows: planet, satellite, code, GM, sigma, ref, radius, sigma, ref, density, ...
            if len(row) >= 10 and row[2].isdigit():
                result[row[2]] = {
                    "name": row[1],
                    "planet": row[0],
                    "gm": _to_float(row[3]),
                    "radius_km": _to_float(row[6]),
                    "density": _to_float(row[9]),
                }
    return result


def planet_gm_without_moons(refresh=False):
    """
    GM of each planet's body ALONE (moons not included), km^3/s^2:
        {"Earth": 398600.436, "Mars": 42828.37362, "Jupiter": 126686531.9, ...}

    WHY THIS MATTERS: data_sources.py normally treats Jupiter as one point
    holding Jupiter + all its moons (the "system" GM). If you ALSO simulate Io
    separately, Io's mass would be counted twice. So when a planet's moons are
    added, the planet must switch to this planet-only GM.
    """
    result = {}
    for table in _satellite_page_tables(refresh):
        for row in table:
            if len(row) >= 2 and row[0] in PLANET_NUMBERS:
                # cell looks like "126686531.9 ±0.42" -> keep the first number
                value = _to_float(row[1].split()[0]) if row[1] else None
                if value is not None:
                    result[row[0]] = value
    return result


# ---------------------------------------------------------------------------
# 3. ASTEROIDS AND COMETS (SBDB Query API)
# ---------------------------------------------------------------------------
# Orbit classes you can filter by (sb_class=...). The common ones:
ORBIT_CLASSES = {
    "MBA": "Main-belt asteroid (between Mars and Jupiter)",
    "IMB": "Inner main-belt asteroid",
    "OMB": "Outer main-belt asteroid",
    "APO": "Apollo: near-Earth, crosses Earth's orbit from outside",
    "ATE": "Aten: near-Earth, crosses Earth's orbit from inside",
    "AMO": "Amor: near-Earth, approaches but doesn't cross Earth's orbit",
    "IEO": "Atira: orbit entirely inside Earth's",
    "MCA": "Mars-crosser",
    "TJN": "Jupiter trojan (shares Jupiter's orbit, 60 deg ahead/behind)",
    "CEN": "Centaur (between Jupiter and Neptune)",
    "TNO": "Trans-Neptunian object (beyond Neptune, Kuiper belt)",
    "JFc": "Jupiter-family comet",
    "HTC": "Halley-type comet",
    "COM": "Comet (generic)",
}

_SBDB_FIELDS = ["spkid", "full_name", "pdes", "kind", "class", "GM",
                "diameter", "rot_per", "a", "e", "i", "per_y"]


def query_small_bodies(kind="asteroid", sb_class=None, near_earth=False,
                       limit=20, sort="-diameter", refresh=False):
    """
    Search JPL's Small-Body Database. Returns a list of dicts like:
        {"spkid": "20000001", "name": "1 Ceres (A801 AA)", "designation": "1",
         "kind": "asteroid", "orbit_class": "MBA", "gm": 62.6284,
         "diameter_km": 939.4, "rotation_hours": 9.07, "a_au": 2.77,
         "e": 0.0797, "i_deg": 10.59, "period_years": 4.6}

    kind       : "asteroid" or "comet"
    sb_class   : orbit class code (see ORBIT_CLASSES), e.g. "TJN" for trojans
    near_earth : True = only near-Earth objects
    limit      : how many to return (the database has 1.5 million+ asteroids!)
    sort       : field to sort by; "-" in front = largest first.
                 "-diameter" = biggest first. Objects with UNKNOWN diameter
                 are sorted last, so huge but unmeasured objects can be missing.

    WHAT THE ORBIT NUMBERS MEAN
      a (semimajor axis) : the orbit's "average radius", in AU
      e (eccentricity)   : 0 = circle, 0.9 = very stretched, >= 1 = not bound
      i (inclination)    : tilt of the orbit vs Earth's orbit, degrees
      period             : from Kepler's 3rd law, T^2 = a^3 (T in years, a in AU)
    """
    params = {
        "fields": ",".join(_SBDB_FIELDS),
        "sb-kind": "c" if kind.startswith("c") else "a",
        "limit": int(limit),
        "sort": sort,
        "full-prec": "1",
    }
    if sb_class:
        params["sb-class"] = sb_class
    if near_earth:
        params["sb-group"] = "neo"

    reply = _fetch_json(SBDB_QUERY_API, params, refresh)
    fields = reply.get("fields", [])
    bodies = []
    for row in reply.get("data", []):
        r = dict(zip(fields, row))
        bodies.append({
            "spkid": str(r["spkid"]),
            "name": r["full_name"].strip(),
            "designation": r["pdes"],
            "kind": "comet" if str(r["kind"]).startswith("c") else "asteroid",
            "orbit_class": r["class"],
            "gm": _to_float(r["GM"]),
            "diameter_km": _to_float(r["diameter"]),
            "rotation_hours": _to_float(r["rot_per"]),
            "a_au": _to_float(r["a"]),
            "e": _to_float(r["e"]),
            "i_deg": _to_float(r["i"]),
            "period_years": _to_float(r["per_y"]),
        })
    return bodies


def query_orbital_elements(sb_class="MBA", limit=2000, refresh=False):
    """
    ORBITAL ELEMENTS (not positions) for many asteroids in ONE request, for
    drawing whole populations like the main belt. Returns a dict of lists:
        {"spkid", "name", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg", "epoch_jd"}
    orbital_elements.py turns these into positions (see there for what each
    element means).

    Choices made here (found by testing the API):
      - numbered asteroids only (sb-ns=n): their orbits are well determined
        and share one recent reference date. Unnumbered ones can have
        orbits last fitted in 1999.
      - sorted by H, the "absolute magnitude": a brightness, where SMALLER
        means BRIGHTER, which for asteroids mostly means BIGGER. So the
        first N are roughly the N largest.
    """
    params = {
        "fields": "spkid,full_name,a,e,i,om,w,ma,epoch",
        "sb-class": sb_class,
        "sb-ns": "n",
        "sort": "H",
        "limit": int(limit),
        "full-prec": "1",
    }
    reply = _fetch_json(SBDB_QUERY_API, params, refresh)
    out = {k: [] for k in ("spkid", "name", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg", "epoch_jd")}
    for spkid, name, a, e, i, om, w, ma, epoch in reply.get("data", []):
        values = [_to_float(v) for v in (a, e, i, om, w, ma, epoch)]
        if None in values or values[1] >= 1:   # skip incomplete or unbound orbits
            continue
        out["spkid"].append(str(spkid))
        out["name"].append(name.strip())
        for key, v in zip(("a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg", "epoch_jd"), values):
            out[key].append(v)
    return out


def count_small_bodies(kind="asteroid", refresh=False):
    """How many asteroids (or comets) does JPL know about right now?"""
    params = {"sb-kind": "c" if kind.startswith("c") else "a"}
    return int(_fetch_json(SBDB_QUERY_API, params, refresh)["count"])


# ---------------------------------------------------------------------------
# 4. LOOKUP BY NAME (Horizons Lookup API)
# ---------------------------------------------------------------------------
def lookup(name, group=None, refresh=False):
    """
    Ask NASA what it calls an object and which ID numbers it uses:
        lookup("Io")       -> [{"name": "Io", "spkid": "501", "type": "natural satellite", ...}]
        lookup("Halley")   -> the comet AND an asteroid also named Halley
    group: None, or one of "ast", "com", "pln", "sat" to narrow the search.
    """
    params = {"sstr": name}
    if group:
        params["group"] = group
    return _fetch_json(LOOKUP_API, params, refresh).get("result", [])


# ---------------------------------------------------------------------------
# Try it:  python nasa_api.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Asteroids known to JPL:", f"{count_small_bodies('asteroid'):,}")
    print("Comets known to JPL   :", f"{count_small_bodies('comet'):,}")

    phys = moon_physical_data()
    for planet in ("Mars", "Jupiter", "Saturn"):
        moons = list_moons(planet)
        print(f"\n{planet}: {len(moons)} moons tracked. First five:")
        for m in moons[:5]:
            p = phys.get(m["horizons_id"], {})
            print(f"   {m['horizons_id']:>6}  {m['name']:<12} GM={p.get('gm')}  R={p.get('radius_km')} km")

    print("\nPlanet GM without moons:", planet_gm_without_moons())

    print("\nFive biggest asteroids:")
    for b in query_small_bodies("asteroid", limit=5):
        print(f"   {b['name']:<28} {b['diameter_km']} km  class {b['orbit_class']}")
