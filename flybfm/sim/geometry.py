"""Air-combat geometry: the quantities every BFM decision is actually about.

Terminology used throughout the project (all angles in degrees at the API
boundary, radians internally):

  LOS      line of sight, from the shooter to the target
  AA       "antenna train angle" / angle-off.  Angle between the shooter's
           velocity vector and the LOS.  0 deg = nose pointed at the target.
           Signed: positive = target is to the shooter's right.
  TAA      target aspect angle.  Angle between the target's TAIL and the LOS
           from the target to the shooter.
                0 deg  -> shooter is dead astern  (classic offensive)
              180 deg  -> head-on                  (neutral)
           TAA < 90  -> shooter is in the target's rear hemisphere
  HCA      heading crossing angle between the two velocity vectors.
                0 deg  -> co-speed, co-heading
              180 deg  -> head-on
  WEZ      weapon engagement zone (here: the gun envelope)
  Es       specific energy  h + V^2/2g   [m].  "Energy state".
  Ps       specific excess power [m/s]; >0 = gaining energy.

Getting these definitions right matters: a reward built on a mis-signed aspect
angle trains a fighter that turns the wrong way, and it is the single most
common bug in hobby air-combat RL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import vecmath as vm
from .aircraft import G


@dataclass
class Geometry:
    range_m: float
    los_el_deg: float          # LOS elevation relative to shooter's horizontal
    aa_deg: float              # signed, + = target right of shooter's nose
    aa_vert_deg: float         # signed, + = target above shooter's nose plane
    taa_deg: float             # 0 = shooter on target's tail
    hca_deg: float             # 0 = co-heading, 180 = head-on
    closure_ms: float          # + = range decreasing
    aspect_sign: int           # +1 shooter is in target's rear hemisphere
    azimuth_el: tuple          # unit LOS in world frame (for sensor encoding)

    @property
    def in_rear_hemisphere(self) -> bool:
        return self.taa_deg < 90.0


def compute(shooter, target) -> Geometry:
    """Geometry from `shooter`'s frame of reference.

    `shooter` and `target` are sim.aircraft.State objects (or anything with
    .pos, .vel).
    """
    los = vm.sub(target.pos, shooter.pos)
    r = vm.norm(los)
    if r < 1e-6:
        return Geometry(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, (0.0, 0.0, 0.0))
    los_u = vm.scale(los, 1.0 / r)

    v_sh = shooter.vel
    v_tg = target.vel
    f_sh = vm.unit(v_sh) if vm.norm(v_sh) > 1e-6 else (1.0, 0.0, 0.0)

    right = vm.right_of(f_sh)
    up = vm.up_of(f_sh, right)

    aa = angle_from_axis(f_sh, los_u)
    sign_h = 1.0 if vm.dot(los_u, right) >= 0 else -1.0
    sign_v = 1.0 if vm.dot(los_u, up) >= 0 else -1.0

    # Aspect is measured *at the target*: the angle between the target's tail
    # direction and the line of sight from the target back to the shooter.
    #   0 deg   -> the shooter is dead astern of the target (offensive)
    #   180 deg -> head-on (defensive)
    # The extra vm.scale(..., -1.0) that used to be here flipped the sign, so
    # every consumer of TAA -- the aspect term of the reward, the timeout points
    # decision, the manoeuvre classifier and the scripted pilots' offensive /
    # defensive branch -- was inverted.  That is worth remembering: a single
    # sign error in a geometry convention does not announce itself as "wrong",
    # it announces itself as "the learner keeps doing something stupid".
    back = vm.sub(shooter.pos, target.pos)
    tail_dir = vm.scale(vm.unit(v_tg), -1.0) if vm.norm(v_tg) > 1e-6 else (0.0, 0.0, 0.0)
    if vm.norm(tail_dir) < 1e-6:
        taa = 0.0
    else:
        taa = vm.angle_between(tail_dir, vm.unit(back))

    hca = vm.angle_between(v_sh, v_tg) if (vm.norm(v_sh) > 1e-6 and vm.norm(v_tg) > 1e-6) else 0.0

    # LOS rate (rad/s) over one step is derived by the caller from history
    los_el = math.asin(vm.clamp(los_u[2], -1.0, 1.0))

    v_sh_u = vm.unit(v_sh) if vm.norm(v_sh) > 1e-6 else (1.0, 0.0, 0.0)
    v_tg_u = vm.unit(v_tg) if vm.norm(v_tg) > 1e-6 else (1.0, 0.0, 0.0)
    closure = -vm.dot(vm.sub(v_tg, v_sh), los_u)  # + when closing

    return Geometry(
        range_m=r,
        los_el_deg=math.degrees(los_el),
        aa_deg=math.degrees(aa) * sign_h,
        aa_vert_deg=math.degrees(aa) * sign_v,
        taa_deg=math.degrees(taa),
        hca_deg=math.degrees(hca),
        closure_ms=closure,
        aspect_sign=1 if taa < math.pi / 2 else -1,
        azimuth_el=los_u,
    )


def angle_from_axis(axis_u, v_u) -> float:
    c = vm.clamp(vm.dot(axis_u, v_u), -1.0, 1.0)
    return math.acos(c)


def angle_off(shooter, target) -> float:
    """Convenience: |AA| in degrees."""
    return abs(compute(shooter, target).aa_deg)


def specific_energy_adv(me, them) -> float:
    """Es difference in metres (+ = I have more energy)."""
    return me.specific_energy() - them.specific_energy()


def energy_height(v_ms: float) -> float:
    return v_ms * v_ms / (2.0 * G)


# --------------------------------------------------------------------- WEZ
@dataclass
class GunEnvelope:
    """Gun weapon-engagement-zone parameters (metres, degrees)."""
    r_min_m: float = 150.0
    r_max_m: float = 900.0        # ~3000 ft, realistic max tracking range
    aa_track_deg: float = 12.0    # inside this angle-off you are "tracking"
    aa_max_deg: float = 25.0      # snapping / fleeting shot


DEFAULT_GUN_ENVELOPE = GunEnvelope()


def in_wez(geom: Geometry, env: GunEnvelope = DEFAULT_GUN_ENVELOPE) -> bool:
    return (env.r_min_m <= geom.range_m <= env.r_max_m
            and abs(geom.aa_deg) <= env.aa_max_deg
            and geom.closure_ms > -50.0)


def tracking_quality(geom: Geometry, env: GunEnvelope = DEFAULT_GUN_ENVELOPE) -> float:
    """0..1 score for how good the current gun solution is. Used for shaping.

    This is *shaping only*; the actual hit/miss decision is done ballistically
    in gun.py so that the reward cannot diverge from physics.
    """
    if not (env.r_min_m * 0.5 <= geom.range_m <= env.r_max_m * 1.5):
        return 0.0
    a = max(0.0, 1.0 - abs(geom.aa_deg) / env.aa_max_deg)
    r_mid = 0.5 * (env.r_min_m + env.r_max_m)
    r_half = 0.5 * (env.r_max_m - env.r_min_m)
    r_score = max(0.0, 1.0 - abs(geom.range_m - r_mid) / (r_half * 1.5))
    return a * a * r_score


def overshoot_flag(geom: Geometry, closing: bool = True) -> bool:
    """Flight-path overshoot: too close, too fast, still trying to pull.

    Overshoot is *the* classic BFM failure mode, so it gets its own signal.
    """
    return geom.range_m < DEFAULT_GUN_ENVELOPE.r_min_m * 1.6 and geom.closure_ms > 60.0
