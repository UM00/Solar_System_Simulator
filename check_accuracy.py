"""
check_accuracy.py - is the simulator right? Measure it, don't eyeball it.

Two tests, no 3D window needed:

  TEST 1 - CONSERVATION: total energy should stay constant.
           A growing energy error means the integrator is the problem.

  TEST 2 - REALITY CHECK: start from NASA's real positions, simulate one
           year, then ask NASA where Earth REALLY was a year later.
           The gap is your accuracy score.

Run it from a terminal (with the .venv activated):

    python check_accuracy.py                                  # Sun + Earth
    python check_accuracy.py --integrator euler               # compare with Chapter 3's method
    python check_accuracy.py --bodies Sun Mercury Venus Earth Moon Mars Jupiter Saturn Uranus Neptune

WHAT TO EXPECT (measured, start 2026-01-01, 1 year, dt = 1 h)
--------------------------------------------------------------
    Sun + Earth, euler         : ~5,300,000 km off, energy error ~1e-2
    Sun + Earth, verlet        : ~1,160,000 km off, energy error ~1e-8
    Sun + all planets + Moon   :        ~110 km off
    ...same with --dt-hours 0.25:         ~50 km off

Sun + Earth alone is "wrong" by about a million km, and that ISN'T a bug.
The real Earth is tugged by the Moon, Venus, Jupiter and the rest, and your
mini-universe doesn't contain them. Add them and the error drops ~10,000x.
That shows, with numbers, that the planets really do pull on each other.

    ...plus --asteroids 10      :         ~50 km off (no measurable change!)

The last ~50 km comes from physics this model leaves out, mainly Einstein's
general relativity (a tiny correction to Newton near the Sun; it shifts
Earth by roughly 30 km per year). Measured: adding the 10 biggest asteroids'
gravity changes nothing at this level. Their combined mass is about 1/2000
of Earth's.
"""

import argparse
from datetime import date, timedelta

import numpy as np

from data_sources import AU, DAY, PHYSICAL_DATA, build_system, fetch_state_vector
from integrator import INTEGRATORS
from simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description="Measure simulator accuracy against NASA data.")
    parser.add_argument("--bodies", nargs="+", default=["Sun", "Earth"])
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--days", type=float, default=365.0)
    parser.add_argument("--dt-hours", type=float, default=1.0)
    parser.add_argument("--integrator", choices=list(INTEGRATORS), default="verlet")
    parser.add_argument("--asteroids", type=int, default=0,
                        help="also include the N biggest asteroids from NASA (their gravity counts if JPL knows their mass)")
    args = parser.parse_args()

    for required in ("Sun", "Earth"):
        if required not in args.bodies:
            parser.error("--bodies must include Sun and Earth (the comparison uses them)")

    # ---- build the system from real NASA data --------------------------
    print(f"Fetching starting positions for {args.start} from JPL Horizons...")
    queries = [{"kind": "asteroid", "limit": args.asteroids}] if args.asteroids else []
    bodies = build_system(args.start, args.bodies, small_body_queries=queries)
    dt = args.dt_hours * 3600.0
    sim = Simulation(bodies, dt=dt, integrator=INTEGRATORS[args.integrator])
    # NOTE: no move_to_center_of_mass() here. We compare against NASA's
    # frame, and we only compare Earth RELATIVE TO the Sun, which doesn't
    # care where the origin is.

    energy_start = sim.energy()
    n_steps = round(args.days * DAY / dt)

    # ---- run, watching the energy along the way -------------------------
    print(f"Simulating {args.days:g} days in {n_steps} steps of {args.dt_hours:g} h "
          f"using '{args.integrator}'...")
    worst_energy_error = 0.0
    for _ in range(100):  # 100 chunks so we can track the energy as we go
        sim.step(n_steps // 100)
        err = abs((sim.energy() - energy_start) / energy_start)
        worst_energy_error = max(worst_energy_error, err)
    sim.step(n_steps - 100 * (n_steps // 100))  # leftover steps

    final_energy_error = abs((sim.energy() - energy_start) / energy_start)

    # ---- compare with where NASA says Earth really was -------------------
    end_date = (date.fromisoformat(args.start) + timedelta(days=args.days)).isoformat()
    print(f"Fetching NASA's real positions for {end_date}...")
    real_sun, _ = fetch_state_vector(PHYSICAL_DATA["Sun"][0], end_date)
    real_earth, _ = fetch_state_vector(PHYSICAL_DATA["Earth"][0], end_date)

    # Earth's position RELATIVE TO THE SUN, simulated vs real
    sim_rel = sim.positions[sim.index("Earth")] - sim.positions[sim.index("Sun")]
    real_rel = real_earth - real_sun
    error_km = np.linalg.norm(sim_rel - real_rel) / 1000.0

    # ---- report ------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Bodies      : {', '.join(args.bodies)}"
          + (f" + {args.asteroids} biggest asteroids" if args.asteroids else ""))
    print(f"Integrator  : {args.integrator},  dt = {args.dt_hours:g} h")
    print("-" * 60)
    print("TEST 1 - energy conservation (relative error, smaller = better)")
    print(f"   worst during run : {worst_energy_error:.2e}")
    print(f"   at the end       : {final_energy_error:.2e}")
    print("TEST 2 - Earth's position vs NASA after the run")
    print(f"   off by           : {error_km:,.0f} km")
    print(f"                    = {error_km * 1000 / AU * 100:.4f} % of the Earth-Sun distance")
    print(f"                    = {error_km / 384400:.2f} x the Earth-Moon distance")
    print("=" * 60)


if __name__ == "__main__":
    main()
