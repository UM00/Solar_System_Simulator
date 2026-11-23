"""
main.py - start the 3D Solar System.

Run from a terminal with your .venv activated:

    python main.py

A browser tab opens with the 3D view.

GROWING THE SYSTEM STEP BY STEP
-------------------------------
Everything you might want to change is in the SETTINGS block below.
    Phase 1: Sun + Earth
    Phase 2: add "Moon"
    Phase 3: add the other planets
    Phase 4: add moons of other planets       (MOONS)
    Phase 5: add asteroids and comets         (SMALL_BODY_QUERIES / SMALL_BODIES)
    Phase 6: the asteroid belt and trojans    (SWARMS)
No physics code needs to change for any of these. That's the point of the
Body / Simulation design. All moon and asteroid data comes from NASA/JPL
through nasa_api.py.
"""

from data_sources import build_system, suggest_time_step
from simulation import Simulation
from visualize import SolarSystemView

# ============================== SETTINGS ===================================
START_DATE = "2026-01-01"   # real NASA positions on this date

# BODY_NAMES = ["Sun", "Earth"]
# BODY_NAMES = ["Sun", "Earth", "Moon"]
BODY_NAMES = ["Sun", "Mercury", "Venus", "Earth", "Moon", "Mars",
              "Jupiter", "Saturn", "Uranus", "Neptune"]

# Moons of other planets, fetched from NASA. For each planet, give:
#   a list of names  -> just those moons
#   "measured"       -> every moon whose mass JPL has measured (the big ones)
#   "all"            -> every moon NASA tracks (Saturn alone has ~290!)
# Try:  python -c "import nasa_api; print(nasa_api.list_moons('Saturn'))"
MOONS = {
    "Jupiter": ["Io", "Europa", "Ganymede", "Callisto"],   # the 4 Galilean moons
    "Saturn": ["Titan", "Rhea", "Iapetus"],
    # "Mars": ["Phobos", "Deimos"],  # WARNING: Phobos orbits in 7.7 h -> tiny dt -> slow
    # "Uranus": "measured",
    # "Neptune": ["Triton"],
}

# Asteroid/comet searches in JPL's Small-Body Database (see nasa_api.py).
SMALL_BODY_QUERIES = [
    {"kind": "asteroid", "limit": 10},                        # the 10 biggest asteroids
    # {"kind": "asteroid", "sb_class": "TJN", "limit": 20},    # Jupiter trojans
    # {"kind": "asteroid", "near_earth": True, "limit": 20},   # biggest near-Earth asteroids
    # {"kind": "comet", "limit": 5},                           # 5 biggest comets
]

# Hand-picked extras as (Horizons designation, display name).
SMALL_BODIES = [("99942", "Apophis")]

# Crowds of real asteroids, drawn as clouds of dots (see swarm_view.py).
# Orbits come from NASA in one request each; "count" = how many of the
# largest. Classes: IMB/MBA/OMB = inner/main/outer belt, TJN = Jupiter trojans.
# More dots = each dot is refreshed less often (swarm_view.py moves 150 dots
# per frame, so 1,800 dots are all refreshed 5 times a second).
SWARMS = [
    {"label": "Main belt", "sb_class": "IMB,MBA,OMB", "count": 1500},
    {"label": "Jupiter trojans", "sb_class": "TJN", "count": 300},
]
# SWARMS = []   # no belt

DT_HOURS = 1.0              # largest time step allowed. Made smaller automatically for fast moons
DAYS_PER_SECOND = 10.0      # starting animation speed (adjustable with the slider)
# ===========================================================================


def main():
    print(f"Loading bodies for {START_DATE} from NASA/JPL (cached after the first run)...")
    bodies = build_system(START_DATE, BODY_NAMES, MOONS, SMALL_BODY_QUERIES, SMALL_BODIES, SWARMS)
    named = [b.name for b in bodies if not b.swarm]
    print(f"  {len(named)} bodies: {', '.join(named)}")
    for spec in SWARMS:
        print(f"  + {sum(b.swarm == spec['label'] for b in bodies):,} asteroids in '{spec['label']}'")

    # A fast moon needs small steps (see suggest_time_step in data_sources.py).
    dt = DT_HOURS * 3600.0
    safe_dt, fastest = suggest_time_step(bodies)
    if safe_dt is not None and safe_dt < dt:
        dt = safe_dt
        print(f"  Time step reduced to {dt / 60:.1f} min so {fastest}'s fast orbit stays accurate.")

    sim = Simulation(bodies, dt=dt)
    # Keep our mini-universe from drifting off screen (see simulation.py).
    sim.move_to_center_of_mass()

    print("Opening the 3D view in your browser...")
    view = SolarSystemView(sim, START_DATE, days_per_second=DAYS_PER_SECOND)
    view.run()


if __name__ == "__main__":
    main()
