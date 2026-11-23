"""
orbital_elements.py - from the six numbers that describe an orbit to a
position and velocity, by solving KEPLER'S EQUATION.

WHY THIS FILE EXISTS
--------------------
NASA's Small-Body Database can send the orbits of thousands of asteroids in
one request, but as ORBITAL ELEMENTS, not positions. Asking Horizons for
positions one asteroid at a time takes about a second each (half an hour for
a 2,000-asteroid belt). So we do the conversion ourselves. It's the same
maths astronomers have used since Kepler, and it runs for all asteroids at
once with NumPy.

THE SIX ORBITAL ELEMENTS
------------------------
Any two-body orbit (Section 7 of the guide) is an ellipse, which takes six
numbers to pin down:

  SIZE AND SHAPE
    a   semimajor axis      the ellipse's size (half its longest width)
    e   eccentricity        0 = circle ... 0.99 = very stretched

  ORIENTATION IN SPACE (three angles)
    i   inclination         tilt of the orbit relative to the ecliptic
    Om  longitude of the    where the orbit crosses the ecliptic going
        ascending node      "upward" (northward), measured from +x
    w   argument of         angle from that crossing to the perihelion,
        perihelion          measured within the orbit's own plane

  WHERE THE BODY IS ON THE ELLIPSE
    M   mean anomaly        an angle that grows at a perfectly steady rate,
                            M = M0 + n * (t - t0). It is NOT the real angle
                            around the Sun: a body moves faster near
                            perihelion (Kepler's 2nd law), so the real angle
                            runs ahead of, or behind, this steady clock.

KEPLER'S EQUATION
-----------------
The link between the steady clock M and the real position is the
"eccentric anomaly" E, defined by Kepler's equation:

        M = E - e * sin(E)

Easy to go from E to M, but we need the reverse. There's no formula for
that, so we solve it NUMERICALLY with Newton's method: guess E, see how far
off M is, correct the guess by (error / slope), repeat. Each round roughly
doubles the number of correct digits, so a few rounds reach machine
precision. (Newton invented the method partly for exactly this problem.)

Once E is known, everything else is geometry (see elements_to_state).
"""

import numpy as np


def solve_kepler(M, e, tol=1e-13, max_iter=50):
    """
    Solve M = E - e sin E for E (radians), for whole arrays at once.

    Newton's method: the "function" is f(E) = E - e sin E - M, its slope is
    f'(E) = 1 - e cos E, and each step does E <- E - f / f'.
    Starting guess: E = M works well for small e; for very stretched orbits
    (e > 0.8) starting at pi is safer.
    """
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi)
    for _ in range(max_iter):
        step = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - step
        if np.max(np.abs(step)) < tol:
            break
    return E


def elements_to_state(a, e, i, Om, w, M, mu):
    """
    Orbital elements -> position (m) and velocity (m/s), RELATIVE TO THE
    CENTRAL BODY, in the ecliptic frame. All inputs are NumPy arrays.

    a in metres; i, Om, w, M in radians; mu = G * M_central in m^3/s^2.
    Only for closed orbits (e < 1).

    Two steps:
    1. Position and velocity in the orbit's own plane, with x pointing at
       perihelion:
           x = a (cos E - e)              y = a sqrt(1 - e^2) sin E
           vx = -k sin E                  vy =  k sqrt(1 - e^2) cos E
       where r = a (1 - e cos E) is the distance and k = sqrt(mu a) / r.
    2. Turn that flat picture into 3D with three rotations: by w within the
       plane, tilt by i, then swing by Om around the ecliptic's axis. Written
       out, that gives two unit vectors P (toward perihelion) and Q (90 deg
       ahead of it, in the direction of motion). The 3D position is x*P + y*Q.
    """
    E = solve_kepler(M, e)
    cos_E, sin_E = np.cos(E), np.sin(E)
    b = np.sqrt(1 - e**2)
    r = a * (1 - e * cos_E)

    x = a * (cos_E - e)
    y = a * b * sin_E
    k = np.sqrt(mu * a) / r
    vx = -k * sin_E
    vy = k * b * cos_E

    cO, sO = np.cos(Om), np.sin(Om)
    cw, sw = np.cos(w), np.sin(w)
    ci, si = np.cos(i), np.sin(i)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1)

    pos = x[:, None] * P + y[:, None] * Q
    vel = vx[:, None] * P + vy[:, None] * Q
    return pos, vel


def propagate_mean_anomaly(M0, a, mu, dt):
    """
    Move each body along its ellipse by dt seconds (two-body motion only):
        n = sqrt(mu / a^3)    (the mean motion: Kepler's 3rd law, n = 2 pi / T)
        M = M0 + n * dt
    """
    return M0 + np.sqrt(mu / a**3) * dt
