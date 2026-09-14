"""Scripted opponents -- the opponent pool.

The point of this file is *strategy diversity*, not difficulty.  A learner that
only ever fights "nose-on-the-target" will learn to beat nose-on-the-target.
Each class here encodes a different solution to the 1v1 problem, drawn from
real BFM teaching, so the pool spans the behaviour space a fighter must
generalise over:

    LevelFlight      - the "does it even engage" floor
    NoseOn           - pure pursuit: predictable, exploitable, always pressing
    LeadPursuit      - correct gun tracking; the thing you actually want to learn
    LagPursuit       - correct when overshooting; teaches the agent about closure
    GunSolution      - lead pursuit + trigger discipline
    BreakTurn        - defensive max-G turn; teaches the agent to stay in-phase
    VerticalFighter  - high/low yo-yo; energy tactics
    BoomAndZoom      - dive, shoot, extend, re-attack; disengage cycles
    Jinker           - random reversals; anti-overfitting noise
    Instructor       - adaptive: switches among the above by geometry

Controller core: a velocity-vector steering law.  We compute the perpendicular
component of the direction we want to fly, convert it to a commanded bank angle
in the velocity frame, and set load factor proportional to the pointing error.
That is a faithful controller for the point-mass model in aircraft.py.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .. import vecmath as vm
from .aircraft import Command, G
from .geometry import DEFAULT_GUN_ENVELOPE, compute, angle_off
from .gun import lead_solution, required_lead_angle_deg


def other_of(env, me):
    return env.red if me is env.blue else env.blue


# --------------------------------------------------------------- steering law
def n_sustained(me, margin: float = 0.98) -> float:
    """Load factor the engine can *hold* at the current q and altitude.

    Solving  q*S*(Cd0 + k*Cl^2) = T  for n.  Using this cap is what stops the
    scripted bots from doing what the first version of this file did: pulling
    9G until they are at 145 m/s and cannot turn at all.  Energy discipline is
    not a stylistic choice, it is what makes the bot a usable opponent.
    """
    ac, s = me.ac, me.s
    p = ac.p
    q = ac.q_dyn()
    if q < 1e-6:
        return 1.0
    T = ac.thrust_available()
    c = T / (q * p.wing_area_m2)
    if c <= p.cd0:
        return 1.0
    cl = math.sqrt(max(c - p.cd0, 0.0) / p.oswald_k)
    n = cl * q * p.wing_area_m2 / (p.mass_kg * G)
    return max(1.0, min(n * margin, p.n_max))


def n_for_ps(me, ps_min: float = -20.0) -> float:
    """Max load factor that keeps specific excess power above `ps_min` [m/s].

    Ps = (T - D) * V / W, and D grows with n^2 through induced drag.  Inverting
    that gives the load factor you can *afford* rather than the load factor the
    airframe can survive.  A fighter with a -200 m/s Ps budget will out-turn you
    for about fifteen seconds and then be a glider; this function is how the
    scripted bots avoid being that fighter.
    """
    p, s = me.ac.p, me.s
    q = me.ac.q_dyn()
    if q < 1e-6:
        return 1.0
    W = p.mass_kg * G
    T = me.ac.thrust_available() * max(0.05, s.throttle)
    d_allow = T - ps_min * W / max(s.v, 1.0)
    num = (d_allow - q * p.wing_area_m2 * p.cd0) * q * p.wing_area_m2
    if num <= 0.0:
        return 1.0
    n2 = num / (p.oswald_k * W * W)
    if n2 <= 0.0:
        return 1.0
    return max(1.0, min(math.sqrt(n2), p.n_max))


def steer_to(me, aim_point, pull_gain: float = 2.0, n_floor: float = 1.0,
             energy_aware: bool = True, damping: float = 0.0, aim_vel=None,
             roll_gain: float = 3.0, ps_budget: float = -25.0,
             n_min_maneuver: float = 2.5):
    """Return (roll_cmd, pull_cmd) to rotate the velocity vector toward a point.

    Load factor is commanded as a *turn rate* rather than as a raw error gain:
    for the point-mass model the achievable turn rate is approximately
    w = g*n/V, so n_des = w_des*V/g with w_des = k*err.  That is a first-order
    response in the pointing error and, unlike a pure P-gain on err, it does not
    wind up into a two-aircraft orbiting stall.

    KNOWN LIMITATION (measured, not suspected).  The bank command is derived from
    the *direction* of the required turn, mu_des = atan2(lateral, vertical).  That
    direction is ill-conditioned exactly where it matters most: with the pipper
    nearly on the target the required turn is almost purely lateral, so mu_des
    swings between +-90 deg on a 0.5 deg pointing error and the aircraft rolls
    through 180 deg of bank while |AA| still looks small in a trace.  The
    consequence is a limit cycle in bank angle that costs gun accuracy.

    A bank-to-turn law fixes this in principle -- derive the bank from the
    *rate* the lift has to produce (A = n sin mu = psi_dot V cos(gamma)/g,
    B = n cos mu = gamma_dot V/g + cos(gamma), mu = atan2(A, B), which gives a
    small bank for a small error) and decompose the feed-forward in the
    (lateral, vertical) frame rather than projecting it on unit(f x d).  Both
    were implemented, both were *worse in the sim* (perch: 449 shots -> 2),
    because a pull-only vertical channel plus a large feed-forward term leaves
    the load factor oscillating between 0.4 and 4 g while the geometry walks
    away.  The tuned law below is what the current baselines were measured
    with; replacing it is a real task with its own tuning, not a one-line fix.
    """
    s = me.s
    to_aim = vm.sub(aim_point, s.pos)
    if vm.norm(to_aim) < 1e-6 or s.v < 1.0:
        return 0.0, _pull_from_n(me, 1.0)
    d = vm.unit(to_aim)
    f = vm.unit(s.vel)
    right = vm.right_of(f)
    up = vm.up_of(f, right)

    proj = vm.dot(d, f)
    d_perp = vm.sub(d, vm.scale(f, proj))
    if vm.norm(d_perp) < 1e-6:
        # The aim point is directly behind us, so the turn plane is undefined.
        # Returning "no turn" here is how the previous version flew straight
        # out of the arena whenever it had to reverse: fall back to continuing
        # the turn in the direction we are already banked, which is what a real
        # pilot does (pick a side and *commit*).
        d_perp = vm.add(vm.scale(right, math.sin(s.mu)), vm.scale(up, math.cos(s.mu)))
        if vm.norm(d_perp) < 1e-6:
            d_perp = right
    d_perp = vm.unit(d_perp)

    mu_des = math.atan2(vm.dot(d_perp, right), vm.dot(d_perp, up))
    err = vm.angle_between(f, d)                    # 0..pi

    # Turn-rate command, in three parts:
    #   1. trim      - the load factor that simply *holds* the current flight
    #                  path (n = cos(gamma)/cos(mu)).  Without this the aircraft
    #                  sinks under gravity whenever it is banked, which at a
    #                  10 Hz decision rate is ~0.2 deg of aim jitter per tick
    #                  and is why the first version could not hold a pipper.
    #   2. feedback  - k * pointing error
    #   3. feed-fwd  - the rotation rate of the aim direction itself, so we
    #                  track a moving lead point instead of chasing it.
    n_trim = math.cos(s.gamma) / max(math.cos(s.mu), 0.2)
    w_des = (pull_gain * err
             - damping * _los_of(me)
             + _feed_forward(me, to_aim, aim_vel))
    n_cmd = n_trim + w_des * s.v / G

    n_cap = me.ac.n_available()
    if energy_aware:
        # affordability (Ps budget), never below a usable maneuvering floor
        n_cap = min(n_cap, max(n_for_ps(me, ps_budget), min(n_min_maneuver, n_cap)))
    n_des = vm.clamp(n_cmd, max(me.ac.p.n_min, -0.5), n_cap)

    # Roll is a *rate* command, not a bang-bang "roll at max rate until the
    # wings are where I want them".  Commanding max roll rate for any error
    # produces a limit cycle in bank angle, which jitters the lift vector and
    # wrecks the gun solution; a proportional rate command gives a clean
    # first-order bank response.
    roll_err = vm.wrap_pi(mu_des - s.mu)
    roll_rate_des = roll_err * roll_gain
    roll_cmd = vm.clamp(roll_rate_des / math.radians(me.ac.p.max_roll_rate_dps),
                        -1.0, 1.0)
    return roll_cmd, _pull_from_n(me, n_des)


def _feed_forward(me, to_aim, aim_vel) -> float:
    """Component of the aim direction's angular velocity along the turn axis.

    Rigorous version of "lead the lead": if d is the unit aim direction from us
    and v_rel the relative velocity, then d_dot = v_perp/r, so the aim
    direction's angular velocity vector is omega_d = d x v_perp / r, and the
    part of it we have to match is its component along the axis we rotate
    about, axis = unit(f x d).  Adding that to the feedback term makes the
    controller a tracker instead of a chaser, which is the entire difference
    between a 2% and a 15% gun hit rate.
    """
    if aim_vel is None:
        return 0.0
    r = vm.norm(to_aim)
    if r < 1e-6 or vm.norm(me.s.vel) < 1e-6:
        return 0.0
    d = vm.unit(to_aim)
    f = vm.unit(me.s.vel)
    v_rel = vm.sub(aim_vel, me.s.vel)
    v_perp = vm.sub(v_rel, vm.scale(d, vm.dot(v_rel, d)))
    omega_d = vm.scale(vm.cross(d, v_perp), 1.0 / r)
    axis = vm.cross(f, d)
    if vm.norm(axis) < 1e-9:
        return 0.0
    return vm.dot(omega_d, vm.unit(axis))


def _los_of(me) -> float:
    """Signed LOS rotation rate [rad/s], used to damp the turn-rate command."""
    return getattr(me, "los_rate_rad", 0.0)


def _pull_from_n(me, n):
    p = me.ac.p
    return vm.clamp(2.0 * (n - p.n_min) / (p.n_max - p.n_min) - 1.0, -1.0, 1.0)


def throttle_for(me, other, g, engage_range=4000.0, saddle_range=900.0):
    """Energy management: A/B to close, chop throttle in the saddle.

    The middle term is the one that matters: closing at 150 m/s when you are
    already inside gun range is how you fly through the target and lose the
    fight.  Real pilots pull the throttle back before they get there.
    """
    s = me.s
    if s.v < me.ac.p.corner_speed() * 0.95:
        return 1.0
    if g.range_m < saddle_range and abs(g.aa_deg) < 25.0:
        return 0.15 if g.closure_ms > 40.0 else 0.45
    if g.range_m > engage_range:
        return 1.0
    if g.closure_ms > 90.0:
        return 0.4
    return 0.85


def safety(me, cmd: Command, floor_m: float = 800.0) -> Command:
    """Ground-avoidance layer every scripted bot runs. Also used by baselines."""
    s = me.s
    def _level_roll(me):
        err = -vm.wrap_pi(me.s.mu)
        return vm.clamp(err * 3.0 / math.radians(me.ac.p.max_roll_rate_dps), -1.0, 1.0)

    # The floor has to be *dive-rate* dependent: at 250 m/s and -60 deg you need
    # a turn radius of ~700 m to level off, so a fixed 800 m floor is a
    # guaranteed crash.  h_required = V^2 / (2 g n) for the pull-out.
    v = max(s.v, 60.0)
    n_pull = max(4.0, me.ac.p.n_max * 0.7)
    h_needed = floor_m + 2.0 * (v * v / (2.0 * G * n_pull)) * max(0.0, -math.sin(s.gamma))
    if s.pos[2] < h_needed:
        return Command(roll=_level_roll(me), pull=1.0, throttle=1.0, trigger=False)
    if s.pos[2] > 11500.0 and s.gamma > 0.0:
        return Command(roll=_level_roll(me), pull=1.0, throttle=0.4, trigger=False)
    return cmd


# ------------------------------------------------------------------ policies
class ScriptedPolicy:
    """Base class. Subclasses implement act(me, other, env) -> Command."""
    name = "scripted"

    def __init__(self, seed: int = 0, trigger: bool = True, **kw):
        self.rng = random.Random(seed)
        self.use_trigger = trigger
        self.kw = kw

    def __call__(self, me, env) -> Command:
        other = other_of(env, me)
        cmd = self.act(me, other, env)
        cmd = self._bounds(me, env, cmd)
        return safety(me, cmd)

    def _bounds(self, me, env, cmd: Command) -> Command:
        """Keep the fight inside the arena.

        Without this, ~half of randomised episodes end with somebody flying out
        of the box, which teaches the learner nothing except "the map ends".
        Pilots bug out toward the centre instead.
        """
        r = math.hypot(me.s.pos[0], me.s.pos[1])
        limit = 0.55 * env.cfg.arena_radius_m
        if r < limit:
            return cmd
        centre_dir = vm.unit((-me.s.pos[0], -me.s.pos[1], 0.0))
        aim = vm.add(me.s.pos, vm.scale(centre_dir, 4000.0))
        roll, pull = steer_to(me, aim, pull_gain=2.0)
        return Command(roll=roll, pull=pull, throttle=1.0, trigger=False)

    def act(self, me, other, env) -> Command:  # pragma: no cover - abstract
        raise NotImplementedError

    def can_shoot(self, me, other, lead_tol_deg: float = 4.0) -> bool:
        """Trigger discipline.

        Firing the moment the nose is in the general direction of the target
        wastes the ammo budget and (worse) trains a learner on noise.  A human
        waits for the pipper to settle: here that is an explicit tolerance on
        the *lead angle*, the actual pointing error for a fixed-forward gun.
        """
        if not self.use_trigger:
            return False
        g = compute(me.s, other.s)
        return (g.range_m <= DEFAULT_GUN_ENVELOPE.r_max_m * 1.3
                and abs(g.aa_deg) <= DEFAULT_GUN_ENVELOPE.aa_track_deg
                and g.closure_ms > -30.0
                and abs(getattr(me, "lead_angle", 99.0)) <= lead_tol_deg)


class LevelFlight(ScriptedPolicy):
    """Flies straight and level (with a lazy weave). The floor baseline."""
    name = "level"

    def act(self, me, other, env):
        s = me.s
        roll = vm.clamp(-vm.wrap_pi(s.mu) * 2.0 / math.radians(me.ac.p.max_roll_rate_dps),
                        -1.0, 1.0)
        return Command(roll=roll, pull=_pull_from_n(me, 1.0),
                       throttle=0.7, trigger=False)


class NoseOn(ScriptedPolicy):
    """Pure pursuit: always point at the target. Dangerous, but exploitable."""
    name = "nose-on"

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        roll, pull = steer_to(me, other.s.pos, pull_gain=3.0)
        return Command(roll, pull, throttle_for(me, other, g), self.can_shoot(me, other))


class LagPursuit(ScriptedPolicy):
    name = "lag"

    def __init__(self, lag_s: float = 1.4, **kw):
        super().__init__(**kw)
        self.lag_s = lag_s

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        aim = vm.sub(other.s.pos, vm.scale(vm.unit(other.s.vel), other.s.v * self.lag_s))
        roll, pull = steer_to(me, aim, pull_gain=2.4, aim_vel=other.s.vel)
        return Command(roll, pull, throttle_for(me, other, g), self.can_shoot(me, other, lead_tol_deg=2.0))


class LeadPursuit(ScriptedPolicy):
    """Aim at the ballistic lead point -- a real gun solution."""
    name = "lead"

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        lead, _ = lead_solution(me.s.pos, me.s.vel, other.s.pos, other.s.vel)
        roll, pull = steer_to(me, lead, pull_gain=2.2, aim_vel=other.s.vel)
        thr = throttle_for(me, other, g)
        return Command(roll, pull, thr, self.can_shoot(me, other, lead_tol_deg=2.5))


class GunSolution(LeadPursuit):
    """Lead pursuit with trigger discipline, closure control and a yo-yo brake.

    This is the bot the whole project is ultimately trying to beat, so it is
    tuned to fly the way the BFM manual actually tells you to fly:
      * nose on the *ballistic lead point*, not on the target,
      * blend to LAG pursuit as closure builds (the classic anti-overshoot fix),
      * pull only as hard as the energy state supports,
      * short, disciplined bursts rather than holding the trigger down.
    """
    name = "guns"

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        lead, t_f = lead_solution(me.s.pos, me.s.vel, other.s.pos, other.s.vel)
        # anti-overshoot: bias the aim backwards along the target's velocity as
        # closure grows.  lag_bias is in seconds of target travel.
        lag_bias = vm.clamp((g.closure_ms - 25.0) / 60.0, 0.0, 1.6)
        aim = vm.sub(lead, vm.scale(vm.unit(other.s.vel), other.s.v * lag_bias))
        tracking = g.range_m < DEFAULT_GUN_ENVELOPE.r_max_m and abs(g.aa_deg) < 25.0
        if tracking:
            # in the saddle: spend energy for angles, but keep it bounded
            roll, pull = steer_to(me, aim, pull_gain=1.6, aim_vel=other.s.vel,
                                  ps_budget=-200.0, n_min_maneuver=4.0)
        else:
            # not yet in the saddle: protect the energy state so we can close
            roll, pull = steer_to(me, aim, pull_gain=1.2, aim_vel=other.s.vel,
                                  ps_budget=-15.0, n_min_maneuver=3.0)
        thr = throttle_for(me, other, g)
        if lag_bias > 0.8:
            thr = min(thr, 0.35)                      # pull it back, stay behind
        return Command(roll, pull, thr, self.can_shoot(me, other))


class BreakTurn(ScriptedPolicy):
    """Defensive: max-G turn into the attacker, then reverse. Classic break."""
    name = "break"

    def __init__(self, break_period_s: float = 6.0, **kw):
        super().__init__(**kw)
        self.break_period_s = break_period_s

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        # turn perpendicular to the attacker's LOS, alternating direction
        phase = int(env.t / self.break_period_s) % 2
        los = g.azimuth_el
        f = vm.unit(me.s.vel)
        right = vm.right_of(f)
        perp = right if phase == 0 else vm.scale(right, -1.0)
        # blend toward "into the attacker" so it is a real break, not a circle
        d = vm.unit(vm.add(vm.scale(perp, 1.0), vm.scale(los, 0.6)))
        roll, pull = steer_to(me, vm.add(me.s.pos, vm.scale(d, 1000.0)), pull_gain=3.4)
        # jink the throttle to stay out of phase
        thr = 1.0 if (env.t % 8.0) < 5.0 else 0.3
        return Command(roll, pull, thr, self.can_shoot(me, other))


class VerticalFighter(ScriptedPolicy):
    """Uses the vertical: pulls up when offensive, dives when defensive.

    A crude but real high/low yo-yo: convert energy to angles when it helps,
    buy it back when it does not.
    """
    name = "vertical"

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        lead, _ = lead_solution(me.s.pos, me.s.vel, other.s.pos, other.s.vel)
        offensive = g.taa_deg < 120.0
        if offensive:
            # high yo-yo: aim above the lead point to avoid overshoot
            aim = vm.add(lead, (0.0, 0.0, 350.0))
            roll, pull = steer_to(me, aim, pull_gain=2.2)
            thr = 0.3
        else:
            # defensive: split-S / dive to buy energy and separation
            d = vm.add(vm.scale(g.azimuth_el, -1.0), (0.0, 0.0, -0.8))
            roll, pull = steer_to(me, vm.add(me.s.pos, vm.scale(vm.unit(d), 1500.0)),
                                  pull_gain=2.6)
            thr = 1.0
        return Command(roll, pull, thr, self.can_shoot(me, other))


class BoomAndZoom(ScriptedPolicy):
    """Dive in, shoot, extend, re-attack. Punishes agents that lose energy."""
    name = "bnz"

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        energy_adv = me.s.specific_energy() - other.s.specific_energy()
        if g.range_m > 3500.0 and energy_adv > -200.0:
            # attack: point at intercept
            lead, _ = lead_solution(me.s.pos, me.s.vel, other.s.pos, other.s.vel)
            roll, pull = steer_to(me, lead, pull_gain=2.8)
            return Command(roll, pull, 1.0, self.can_shoot(me, other))
        if g.range_m < 900.0:
            roll, pull = steer_to(me, other.s.pos, pull_gain=3.0)
            return Command(roll, pull, 0.5, self.can_shoot(me, other))
        # extend: fly the reciprocal away, climbing to rebuild energy
        away = vm.scale(g.azimuth_el, -1.0)
        d = vm.unit(vm.add(away, (0.0, 0.0, 0.25)))
        roll, pull = steer_to(me, vm.add(me.s.pos, vm.scale(d, 2000.0)), pull_gain=2.0)
        return Command(roll, pull, 1.0, False)


class Jinker(ScriptedPolicy):
    """Random reversals and G jitter -- the anti-overfit noise opponent."""
    name = "jinker"

    def __init__(self, jink_period_s: float = 2.5, **kw):
        super().__init__(**kw)
        self.jink_period_s = jink_period_s

    def act(self, me, other, env):
        g = compute(me.s, other.s)
        lead, _ = lead_solution(me.s.pos, me.s.vel, other.s.pos, other.s.vel)
        phase = int(env.t / self.jink_period_s)
        r = random.Random(phase * 7919 + 13)
        offset = (r.uniform(-600, 600), r.uniform(-600, 600), r.uniform(-400, 400))
        aim = vm.add(lead, offset)
        roll, pull = steer_to(me, aim, pull_gain=2.6 + r.uniform(-0.6, 0.8), aim_vel=other.s.vel)
        thr = 0.4 + 0.6 * r.random()
        return Command(roll, pull, thr, self.can_shoot(me, other))


class Instructor(ScriptedPolicy):
    """Adaptive BFM bot: chooses a mode from geometry.

    This is the 'competent human' stand-in: offensive -> tracking gun solution
    with lag/lead discipline; neutral -> vertical merge; defensive -> break and
    jink. It is what an evaluation suite should be measured against, because it
    is the opponent whose *strategies* (plural) the rest of the file supplies.
    """
    name = "instructor"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.modes = {"lead": LeadPursuit(seed=1), "lag": LagPursuit(seed=2),
                      "guns": GunSolution(seed=3), "break": BreakTurn(seed=4),
                      "vertical": VerticalFighter(seed=5), "bnz": BoomAndZoom(seed=6)}

    def act(self, me, other, env) -> Command:
        g = compute(me.s, other.s)
        vc = me.s.v - other.s.v
        if g.taa_deg < 60.0 and g.range_m < 1500.0:
            if g.range_m < 500.0 and g.closure_ms > 50.0:
                return self.modes["lag"].act(me, other, env)       # avoid overshoot
            return self.modes["guns"].act(me, other, env)
        if g.taa_deg < 110.0:
            return self.modes["vertical"].act(me, other, env)
        if g.taa_deg < 150.0:
            return self.modes["bnz"].act(me, other, env)
        return self.modes["break"].act(me, other, env)             # defensive


# --------------------------------------------------------------- opponent pool
def default_pool(seed: int = 0):
    """The starter league. Deliberately heterogeneous."""
    return [
        LevelFlight(seed=seed + 1),
        NoseOn(seed=seed + 2),
        LeadPursuit(seed=seed + 3),
        LagPursuit(seed=seed + 4),
        GunSolution(seed=seed + 5),
        BreakTurn(seed=seed + 6),
        VerticalFighter(seed=seed + 7),
        BoomAndZoom(seed=seed + 8),
        Jinker(seed=seed + 9),
        Instructor(seed=seed + 10),
    ]


POOL_BY_NAME = {
    "level": LevelFlight,
    "nose-on": NoseOn,
    "lead": LeadPursuit,
    "lag": LagPursuit,
    "guns": GunSolution,
    "break": BreakTurn,
    "vertical": VerticalFighter,
    "bnz": BoomAndZoom,
    "jinker": Jinker,
    "instructor": Instructor,
}
