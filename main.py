"""
main.py - start the 3D Solar System.

Run from a terminal with your .venv activated:

    python main.py

A browser tab opens with the 3D view.

GROWING THE SYSTEM STEP BY STEP
-------------------------------
Everything you might want to change is in the SETTINGS block below.
    Phase 1: Sun + Earth                          (the default)
    Phase 2: add "Moon"
    Phase 3: add the other planets
    Phase 4: add asteroids to SMALL_BODIES
No physics code needs to change for any of these. That's the point of the
Body / Simulation design.
"""

from data_sources import make_body, make_small_body
from simulation import Simulation
from visualize import SolarSystemView

# ============================== SETTINGS ===================================
START_DATE = "2026-01-01"   # real NASA positions on this date

#BODY_NAMES = ["Sun", "Earth"]
# BODY_NAMES = ["Sun", "Earth", "Moon"]
BODY_NAMES = ["Sun", "Mercury", "Venus", "Earth", "Moon", "Mars",
               "Jupiter", "Saturn", "Uranus", "Neptune"]

# Asteroids / comets as (Horizons designation, display name).
# A plain number = asteroid number N in the official catalogue.
#SMALL_BODIES = []
SMALL_BODIES = [("1", "Ceres"), ("4", "Vesta"), ("99942", "Apophis")]

DT_HOURS = 1.0              # physics time step. 1 h handles even the Moon's fast orbit
DAYS_PER_SECOND = 10.0      # starting animation speed (adjustable with the slider)
# ===========================================================================


def main():
    print(f"Loading {len(BODY_NAMES) + len(SMALL_BODIES)} bodies for {START_DATE}...")
    bodies = [make_body(name, START_DATE) for name in BODY_NAMES]
    bodies += [make_small_body(des, START_DATE, name=label) for des, label in SMALL_BODIES]

    sim = Simulation(bodies, dt=DT_HOURS * 3600.0)
    # Keep our mini-universe from drifting off screen (see simulation.py).
    sim.move_to_center_of_mass()

    print("Opening the 3D view in your browser...")
    view = SolarSystemView(sim, START_DATE, days_per_second=DAYS_PER_SECOND)
    view.run()


if __name__ == "__main__":
    main()
