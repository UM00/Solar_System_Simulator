"""
simulation.py - holds all the bodies and runs time forward.

WHY ARRAYS INSTEAD OF A LIST OF Body OBJECTS?
---------------------------------------------
Body objects are nice to read and to create. But numpy is fastest when every
position sits in ONE (N, 3) array, so gravity for all pairs is computed in a
single operation. So:

    - The Simulation keeps the "live" state in arrays (positions, velocities).
    - Body objects are updated from those arrays only when you call
      sync_bodies() (for example before reading body.position).
"""

import numpy as np

from body import Body
from integrator import velocity_verlet_step
from physics import (
    accelerations,
    total_angular_momentum,
    total_energy,
    total_momentum,
)


class Simulation:
    def __init__(self, bodies, dt, integrator=velocity_verlet_step):
        """
        bodies     : list of Body
        dt         : time step in seconds. It must be MUCH smaller than the
                     fastest orbit in the system (rule of thumb: about 1/100
                     of it or less). Earth's year = 3.2e7 s, the Moon's month
                     = 2.4e6 s, so 1 hour (3600 s) is safe for both.
        integrator : any function from integrator.py
        """
        self.bodies = list(bodies)
        self.dt = float(dt)
        self.integrator = integrator
        self.time = 0.0  # seconds since the start
        self._pack()

    # ---------------------------------------------------------------- setup
    def _pack(self):
        """Copy the Body objects into numpy arrays."""
        self.names = [b.name for b in self.bodies]
        self.positions = np.array([b.position for b in self.bodies], dtype=float)
        self.velocities = np.array([b.velocity for b in self.bodies], dtype=float)
        self.masses = np.array([b.mass for b in self.bodies], dtype=float)
        # Test particles are pulled by gravity but don't pull (see body.py).
        self.gravity_masses = np.where(
            [b.is_test_particle for b in self.bodies], 0.0, self.masses
        )
        self.acc = accelerations(self.positions, self.gravity_masses)

    def add_body(self, body: Body):
        """
        The way to bring new objects in: asteroids, comets, a rogue star...
        Can be called before running or in the middle of a run.
        """
        if body.name in self.names:
            raise ValueError(f"A body called {body.name!r} already exists")
        self.sync_bodies()  # save the current state before rebuilding the arrays
        self.bodies.append(body)
        self._pack()

    def index(self, name):
        return self.names.index(name)

    def move_to_center_of_mass(self):
        """
        Shift positions and velocities so the balance point of OUR bodies
        sits still at the origin.

        PHYSICS: NASA gives coordinates relative to the balance point of the
        WHOLE Solar System. If we only simulate Sun + Earth, Jupiter is
        missing. NASA's Sun is moving a little (Jupiter tugs it at about
        13 m/s), and nothing in our mini-universe explains that motion, so
        the whole pair would slowly drift off. Removing the centre-of-mass
        motion fixes it: total momentum becomes exactly zero.
        """
        m = self.gravity_masses[:, None]
        total = self.gravity_masses.sum()
        com_pos = (m * self.positions).sum(axis=0) / total
        com_vel = (m * self.velocities).sum(axis=0) / total
        self.positions -= com_pos
        self.velocities -= com_vel
        self.acc = accelerations(self.positions, self.gravity_masses)

    # ----------------------------------------------------------------- run
    def _accel(self, pos):
        return accelerations(pos, self.gravity_masses)

    def step(self, n_steps=1):
        """Advance the simulation by n_steps * dt seconds."""
        for _ in range(n_steps):
            self.positions, self.velocities, self.acc = self.integrator(
                self.positions, self.velocities, self.acc, self._accel, self.dt
            )
            self.time += self.dt

    def sync_bodies(self):
        """Copy the live arrays back into the Body objects."""
        for i, b in enumerate(self.bodies):
            b.position = self.positions[i].copy()
            b.velocity = self.velocities[i].copy()

    # ------------------------------------------------------------- checks
    def energy(self):
        return total_energy(self.positions, self.velocities, self.gravity_masses)

    def momentum(self):
        return total_momentum(self.velocities, self.gravity_masses)

    def angular_momentum(self):
        return total_angular_momentum(self.positions, self.velocities, self.gravity_masses)
