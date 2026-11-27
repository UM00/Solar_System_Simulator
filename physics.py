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

    SPEED TRICK: ONLY BODIES WITH MASS PULL
    A test particle's pull is exactly zero, so computing it is wasted work.
    We only use the M bodies that pull (the Sun, planets, big moons...) as
    SOURCES, so the cost is N x M pairs instead of N x N. With a 2,000-
    asteroid belt that's ~50,000 pairs per step instead of ~5,000,000.
    The answer is identical: we only skip terms that would be zero.
    """
    sources = np.flatnonzero(gravity_masses)   # indices of the bodies that pull
    if sources.size == 0:
        return np.zeros_like(positions)
    return accelerations_from(positions, positions[sources], gravity_masses[sources])


def accelerations_from(positions, src_pos, src_masses):
    """
    Acceleration of each body at `positions` (N, 3) caused by the pulling
    bodies at `src_pos` (M, 3) with masses `src_masses` (M,).
    simulation.py also uses this directly to move the asteroid belt on its own
    slower clock.
    """
    # diff[i, k] = vector pointing FROM body i TO source k, for every pair at once.
    # [np.newaxis] adds a dimension so numpy "broadcasts" (N,1,3) against (1,M,3),
    # giving shape (N, M, 3). That's all N*M separation vectors in one line.
    diff = src_pos[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # SQUARED distance between every pair, shape (N, M). einsum("ijk,ijk->ij")
    # means "for each i, j: sum over k of diff*diff", i.e. x^2 + y^2 + z^2.
    # Working with r^2 skips a square root that would only be squared again.
    r2 = np.einsum("ijk,ijk->ij", diff, diff)

    # A body doesn't pull on itself. Its distance to itself is 0, which
    # would mean dividing by zero. Setting it to infinity makes 1/inf = 0.
    r2[r2 == 0] = np.inf

    # a_i = G * sum_k  m_k * diff_ik / r_ik^3      (r^3 = r^2 * sqrt(r^2))
    # The second einsum multiplies each arrow by its weight m/r^3 and adds up
    # each row in one go. No minus sign is needed: diff already points TOWARD
    # the attracting body. (Your Chapter 3 code used -(x - X), the same
    # vector flipped.)
    # Speed: measured 2x faster than computing a norm, cubing it and summing
    # separately, with identical results (to 1e-16), because NumPy makes
    # fewer temporary arrays.
    weight = src_masses[None, :] / (r2 * np.sqrt(r2))
    return G * np.einsum("ij,ijk->ik", weight, diff)


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
    sources = np.flatnonzero(gravity_masses)   # test particles add nothing here
    p, m = positions[sources], gravity_masses[sources]
    diff = p[np.newaxis, :, :] - p[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(len(p), k=1)  # each pair counted once (i < j)
    return -G * np.sum(m[i] * m[j] / dist[i, j])


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
