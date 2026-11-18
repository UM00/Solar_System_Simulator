"""
integrator.py - moving time forward by one small step `dt`.

WHAT AN INTEGRATOR DOES
-----------------------
Gravity tells us the ACCELERATION right now. To find where everything will be
later, we take many small steps:

    acceleration --(changes)--> velocity --(changes)--> position

How you take that step matters a lot. Both methods below cost about the same
per step (one gravity calculation), but they behave very differently over
thousands of orbits.

Every integrator here has the same signature, so you can swap them freely:

    new_pos, new_vel, new_acc = step(pos, vel, acc, accel_func, dt)

`acc` is the acceleration at the CURRENT positions. It is passed in and
returned so we never compute gravity twice for the same positions.
"""


def forward_euler_step(pos, vel, acc, accel_func, dt):
    """
    The method from your Chapter 3 notebook. Kept here so you can compare.

        position += velocity     * dt   (uses the OLD velocity)
        velocity += acceleration * dt

    WHY IT FAILS FOR ORBITS: each step moves the body in a straight line along
    its current direction. A straight line tangent to a circle always ends up
    slightly OUTSIDE the circle. So every step adds a sliver of energy, and the
    orbit spirals outward forever. Smaller dt slows this down but never stops it.
    """
    new_pos = pos + vel * dt
    new_vel = vel + acc * dt
    new_acc = accel_func(new_pos)
    return new_pos, new_vel, new_acc


def velocity_verlet_step(pos, vel, acc, accel_func, dt):
    """
    Velocity Verlet, also called "leapfrog" or "kick-drift-kick" (Chapter 5 of the book).

        1. KICK  : half a step of velocity change, using the current force
        2. DRIFT : a full step of motion, at that halfway velocity
        3.         recompute gravity at the new positions
        4. KICK  : the other half of the velocity change, using the NEW force

    WHY IT WORKS: the force is "shared" between the start and the end of the
    step, so the method is time-symmetric. Run it backwards and you retrace
    your path exactly. That makes it SYMPLECTIC: its energy error does not
    pile up, it just wobbles slightly up and down forever. Orbits stay closed
    for millions of steps. That's why real planetary simulators use this
    family of methods.
    """
    half_vel = vel + 0.5 * dt * acc       # 1. kick
    new_pos = pos + dt * half_vel         # 2. drift
    new_acc = accel_func(new_pos)         # 3. new forces
    new_vel = half_vel + 0.5 * dt * new_acc  # 4. kick
    return new_pos, new_vel, new_acc


# Lets check_accuracy.py pick an integrator by name from the command line.
INTEGRATORS = {
    "verlet": velocity_verlet_step,
    "euler": forward_euler_step,
}
