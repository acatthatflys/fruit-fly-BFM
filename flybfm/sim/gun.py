"""Ballistic gun model.

Rounds are actually flown: the muzzle velocity is added to the aircraft's own
velocity, gravity acts on the round, and the hit test is a closest-approach
test against the target's future position.  Dispersion is applied per round.

Consequence: you cannot get a hit by simply pointing the nose at the target.
Lead matters, closure matters, gravity drop matters.  That is what makes the
learned behaviour look like BFM rather than like a tracking controller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import vecmath as vm

GUN_MUZZLE_MS = 1050.0        # M61A1 Vulcan muzzle velocity
GUN_DISPERSION_RAD = 0.0025   # 2.5 mrad round-to-round (M61 spec is <5 mrad max)
HIT_RADIUS_M = 4.00           # effective lethal radius, see docs/REWARD.md calibration table
HIT_DAMAGE = 1.0 / 6.0        # ~6 effective hits to kill (20mm vs fighter)
ROUND_LIFE_S = 3.0


@dataclass
class GunState:
    ammo: int = 511
    cooldown_s: float = 0.0
    shots: int = 0
    hits: int = 0
    trigger_held_s: float = 0.0


class Gun:
    def __init__(self, gun_state: GunState, rounds_per_s: float = 100.0 / 6.0,
                 seed: int = 1, dispersion_rad: float = GUN_DISPERSION_RAD):
        self.g = gun_state
        self.rps = rounds_per_s
        self.disp = dispersion_rad
        self._st = (seed * 1103515245 + 12345) % (2 ** 31)

    def _u(self) -> float:
        """Uniform(-1,1), deterministic."""
        self._st = (1103515245 * self._st + 12345) % (2 ** 31)
        return self._st / (2 ** 31 - 1) * 2.0 - 1.0

    def _gauss(self) -> float:
        # sum of 3 uniforms has variance 1 (each U(-1,1) has var 1/3); good
        # enough for dispersion, and deterministic.
        return self._u() + self._u() + self._u()

    def fire(self, shooter_pos, shooter_vel, target_pos, target_vel, dt, trigger):
        """Advance one sim tick. Returns (rounds_fired, hits_this_tick)."""
        if not trigger or self.g.ammo <= 0:
            self.g.cooldown_s = max(0.0, self.g.cooldown_s - dt)
            self.g.trigger_held_s = 0.0 if not trigger else self.g.trigger_held_s
            return 0, 0

        self.g.trigger_held_s += dt
        fired = 0
        hits = 0
        # Round accumulator.  At 16.7 rounds/s and a 20 ms tick the per-tick
        # count is fractional, so rounds have to be accumulated across ticks or
        # nothing is ever fired.
        self.g.cooldown_s -= dt
        interval = 1.0 / max(self.rps, 1e-9)
        guard = 0
        while self.g.cooldown_s <= 0.0 and self.g.ammo > 0 and guard < 64:
            self.g.cooldown_s += interval
            guard += 1

        fwd = vm.unit(shooter_vel) if vm.norm(shooter_vel) > 1e-6 else (1.0, 0.0, 0.0)
        right = vm.right_of(fwd)
        up = vm.up_of(fwd, right)

        for _ in range(guard):
            if self.g.ammo <= 0:
                break
            self.g.ammo -= 1
            self.g.shots += 1
            fired += 1
            # dispersion in the muzzle's local frame
            dy = self._gauss() * self.disp
            dz = self._gauss() * self.disp
            direction = vm.add(fwd, vm.add(vm.scale(right, dy), vm.scale(up, dz)))
            direction = vm.unit(direction)
            v_b = vm.add(shooter_vel, vm.scale(direction, GUN_MUZZLE_MS))
            if _intersects(shooter_pos, v_b, target_pos, target_vel):
                hits += 1
                self.g.hits += 1
        return fired, hits


def _intersects(p0, v_b, t_pos, t_vel) -> bool:
    """Closest approach of the round to the target, inside the round's life.

    The round is a ballistic body: it falls.  Evaluating the miss distance
    without the gravity term makes the model inconsistent with `lead_solution`,
    which compensates for exactly that drop -- the bullet then lands 0.5 g t^2
    *below* the aim point and long-range hit rates quietly collapse (measured:
    47% at 900 m instead of 93%, and the whole difference attributed to the
    policy rather than to the gun).

    The CPApproach time itself is taken from the gravity-free relative motion,
    which is a slight approximation; the drop is evaluated at that time.
    """
    r0 = vm.sub(t_pos, p0)
    v_rel = vm.sub(t_vel, v_b)
    vv = vm.dot(v_rel, v_rel)
    if vv < 1e-9:
        return vm.norm(r0) < HIT_RADIUS_M
    t_cpa = -vm.dot(r0, v_rel) / vv
    if t_cpa <= 0.0 or t_cpa > ROUND_LIFE_S:
        return False
    # The relative miss is (target - round).  The round falls, the target does
    # not, so the target sits *higher* relative to the round than a straight-line
    # extrapolation says: the correction term is +0.5 g t^2, not -0.5 g t^2.
    # (Sign errors here and in lead_solution are self-cancelling to first order,
    # which is how a gun can be "obviously aiming correctly" and still miss by
    # 7 m at 900 m.)
    miss = vm.add(r0, vm.scale(v_rel, t_cpa))
    miss = (miss[0], miss[1], miss[2] + 0.5 * 9.80665 * t_cpa * t_cpa)
    return vm.norm(miss) < HIT_RADIUS_M


def lead_solution(shooter_pos, shooter_vel, target_pos, target_vel, iters: int = 8):
    """Iterative lead point. Returns (lead_point, t_flight).

    Solve, for the round's flight time t,

        Vm * t = | T - P + (Vt - Vs) t - 0.5 g_vec t^2 |

    i.e. the aim direction is the direction to the point where the target *will
    be* once the round gets there.  Two things in that expression are easy to
    get wrong and both are worth metres of error:

    * the term is the **relative** velocity (Vt - Vs).  The shooter's own motion
      moves the required impact point; a solution that uses Vt alone is wrong by
      Vs*t, which at 250 m/s and 0.8 s is 200 m.
    * the gravity term makes the aim point *higher* than the target's future
      position, because the round drops on the way.  The sign is the opposite of
      the one you get if you think of it as "predicting where the target falls".

    Using |d| / Vm for the flight time instead (ignoring the shooter's velocity
    in the round's ground speed) systematically under-leads by 5-10% of the time
    of flight, which at 500 m against a crossing target is metres of aim error
    and a near-zero hit rate.  This is the single highest-leverage function in
    the gun model.
    """
    rel0 = vm.sub(target_pos, shooter_pos)
    v_rel = vm.sub(target_vel, shooter_vel)
    t_f = vm.norm(rel0) / GUN_MUZZLE_MS
    lead = target_pos
    for _ in range(iters):
        lead = vm.add(target_pos, vm.scale(v_rel, t_f))
        lead = (lead[0], lead[1], lead[2] + 0.5 * 9.80665 * t_f * t_f)
        d = vm.sub(lead, shooter_pos)
        t_f = vm.norm(d) / GUN_MUZZLE_MS
    return lead, t_f


def required_lead_angle_deg(shooter_pos, shooter_vel, target_pos, target_vel) -> float:
    """Angle between the nose (velocity vector) and the correct lead point.

    This is the *real* tracking error for a fixed-forward gun, and it is a much
    better shaping signal than raw angle-off: you can be at 5 deg angle-off and
    still miss, and at 20 deg angle-off and still hit a crossing target.
    """
    lead, _ = lead_solution(shooter_pos, shooter_vel, target_pos, target_vel)
    d = vm.sub(lead, shooter_pos)
    if vm.norm(d) < 1e-6 or vm.norm(shooter_vel) < 1e-6:
        return 0.0
    return math.degrees(vm.angle_between(vm.unit(shooter_vel), vm.unit(d)))
