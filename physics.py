"""
physics.py - Newtonian gravity for any number of bodies.

THE PHYSICS IN PLAIN WORDS
--------------------------
Newton's law of gravity: every mass pulls on every other mass.
The pull of body j on body i:

    - points FROM i TOWARD j        (gravity always attracts)
    - is proportional to m_j        (heavier things pull harder)
    - weakens as 1 / r^2            (twice as far -> a quarter of the pull)

Combined with F = m*a, the ACCELERATION of body i is

    a_i = G * sum over j of  m_j * (r_j - r_i) / |r_j - r_i|^3

The "/ |r|^3" looks wrong for an inverse-SQUARE law, but (r_j - r_i) carries
one power of distance on top. Dividing by r^3 turns it into
"a direction of length 1" times "1/r^2".

Body i's own mass does not appear: it cancels out (F = G*m_i*m_j/r^2 and
a = F/m_i). That's why a feather and a planet fall the same way in a vacuum,
and why asteroids can be "test particles": their motion doesn't depend on
their own mass at all.

Everything here works on whole numpy arrays at once (all bodies, all pairs).
That is much faster than Python for-loops.
"""

import numpy as np

G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2 (CODATA 2018)


def accelerations(positions, gravity_masses):
    """
    Gravitational acceleration of every body due to every other body.

    positions      : array (N, 3), metres
    gravity_masses : array (N,), kg. Test particles have 0 here, so they
                     are pulled but don't pull anything themselves.
    returns        : array (N, 3), m/s^2
    """
    # diff[i, j] = vector pointing FROM body i TO body j, for every pair at once.
    # [np.newaxis] adds a dimension so numpy "broadcasts" (N,1,3) against (1,N,3),
    # giving shape (N, N, 3). That's all N*N separation vectors in one line.
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # Distance between every pair: shape (N, N)
    dist = np.linalg.norm(diff, axis=2)

    # A body doesn't pull on itself. Its distance to itself is 0, which
    # would mean dividing by zero. Setting it to infinity makes 1/inf = 0.
    np.fill_diagonal(dist, np.inf)

    inv_r3 = 1.0 / dist**3

    # a_i = G * sum_j  m_j * diff_ij / r_ij^3
    # The [None, :, None] and [:, :, None] reshape the arrays so every term lines up as (N, N, 3).
    # No minus sign is needed: diff already points TOWARD the attracting body.
    # (Your Chapter 3 code used -(x - X), which is the same vector flipped.)
    pull = gravity_masses[None, :, None] * inv_r3[:, :, None] * diff
    return G * pull.sum(axis=1)


# ---------------------------------------------------------------------------
# CONSERVED QUANTITIES - your accuracy checks
# ---------------------------------------------------------------------------
# In a closed system with only gravity, three things NEVER change:
#   energy, momentum, angular momentum.
# The simulation doesn't enforce this. If the numbers drift, the numerical
# method is making errors. Watching them is how you know your code is good.
#
# Note: only bodies that pull (gravity_masses > 0) are counted. Test particles
# get pushed around without pushing back, so they aren't part of the
# conserved system.

def kinetic_energy(velocities, gravity_masses):
    """Energy of motion: sum of 1/2 m v^2 (joules)."""
    speed_sq = np.sum(velocities**2, axis=1)
    return 0.5 * np.sum(gravity_masses * speed_sq)


def potential_energy(positions, gravity_masses):
    """
    Gravitational binding energy: sum over each PAIR of -G m_i m_j / r_ij.

    It's negative because bodies are "stuck in a well": you'd have to ADD
    energy to pull them apart to infinity (where potential energy is 0).
    """
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(len(positions), k=1)  # each pair counted once (i < j)
    return -G * np.sum(gravity_masses[i] * gravity_masses[j] / dist[i, j])


def total_energy(positions, velocities, gravity_masses):
    """
    Kinetic + potential. For a bound orbit this total is negative.
    A planet trades between the two: it speeds up (more kinetic) as it falls
    closer (more negative potential), but the SUM stays fixed.
    """
    return kinetic_energy(velocities, gravity_masses) + potential_energy(positions, gravity_masses)


def total_momentum(velocities, gravity_masses):
    """
    sum of m * v (kg m/s). Newton's 3rd law: every pull has an equal and
    opposite pull back, so the total never changes.
    """
    return np.sum(gravity_masses[:, None] * velocities, axis=0)


def orbital_period(rel_position, rel_velocity, mu):
    """
    How long one orbit takes (seconds), for a body moving around a central
    body. Uses the position and velocity RELATIVE to that central body.
    mu = G * (M_central + m_orbiting), m^3/s^2.

    Two classic laws, in two steps:

    1. VIS-VIVA (energy conservation) gives the orbit's size, the
       semimajor axis a, from the current distance r and speed v:
            v^2 = mu * (2/r - 1/a)   ->   a = 1 / (2/r - v^2/mu)
       Fast for its distance = big orbit. Slow = small orbit.

    2. KEPLER'S THIRD LAW gives the period from the size:
            T = 2 * pi * sqrt(a^3 / mu)
       Twice the orbit size -> 2.8x longer to go round.

    Returns None if the body isn't on a closed orbit (it's escaping).
    """
    r = np.linalg.norm(rel_position)
    v2 = np.dot(rel_velocity, rel_velocity)
    inv_a = 2.0 / r - v2 / mu
    if inv_a <= 0:
        return None  # parabola/hyperbola: never comes back
    a = 1.0 / inv_a
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def total_angular_momentum(positions, velocities, gravity_masses):
    """
    sum of m * (r x v). Roughly "how much the system is spinning".
    Conserving it is why planets move faster when close to the Sun
    (Kepler's 2nd law: equal areas in equal times).
    """
    return np.sum(gravity_masses[:, None] * np.cross(positions, velocities), axis=0)
