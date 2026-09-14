"""Tiny dependency-free 3D vector helpers.

Deliberately stdlib-only: the whole simulation core must run on a laptop with
no numpy/torch installed so that the connectome-side of the project can be
developed and unit-tested independently of the big GPU brain backend.

World convention (right-handed, z-up, ENU):
    x = east, y = north, z = up   [metres]
Heading psi is measured from +x toward +y.
"""
from __future__ import annotations

import math

Vec3 = tuple


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def lerp(a, b, t):
    return a + (b - a) * t


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def angle_between(a, b):
    """Unoriented angle in [0, pi] between two vectors."""
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = clamp(dot(a, b) / (na * nb), -1.0, 1.0)
    return math.acos(c)


def vel_from_angles(v, psi, gamma):
    """Velocity vector from speed, heading, flight-path angle."""
    cg = math.cos(gamma)
    return (v * cg * math.cos(psi), v * cg * math.sin(psi), v * math.sin(gamma))


def right_of(forward, up=(0.0, 0.0, 1.0)):
    """Unit vector pointing to the right of `forward` for a z-up frame."""
    r = cross(up, forward)
    n = norm(r)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (r[0] / n, r[1] / n, r[2] / n)


def up_of(forward, right):
    return cross(forward, right)
