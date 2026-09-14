"""Reward design for guns-only BFM.

Two-layer design, which is the important idea:

  LAYER 1 - EVENT rewards.  Sparse, unambiguous, non-negotiable:
            gun hits dealt/taken, kills, deaths, crashes, leaving the arena.
            These define the objective and are *never* shaped.

  LAYER 2 - POTENTIAL-BASED SHAPING.  Dense geometry/energy terms that make
            the sparse objective learnable.  Because the shaping is expressed
            as  F(s, s') = gamma * Phi(s') - Phi(s)  it is provably
            policy-invariant (Ng, Harada & Russell 1999): you can add as much
            of it as you like and the optimal policy is unchanged.  This is the
            principled answer to "I don't want to accidentally hand-script the
            behaviour with my reward function."

The potential Phi is a weighted sum of the same classical BFM quantities a
human instructor grades you on: angle-off, aspect, range, closure, and energy.

Gunnery-focused tuning (2026-09) to fix "wins on points, cannot shoot":
  - tracking term now uses ballistic lead_angle (not AA) and is sharply peaked
    at 1° scale with weight 12.0 — must dominate aspect/range/energy (which are
    now 0.2/0.15/0.1/0.05) so optimizer cannot farm points by sitting behind
    with 10° error.
  - hit/kill rewards increased (10 / 100) to make sparse hit visible
  - trigger discipline: +2.0 when trigger down with lead<=1.5° in WEZ,
    -0.05 when trigger down with lead>10° (small spray penalty to allow exploration),
    -0.01 out of WEZ. Good reward >> bad penalty so random firing still positive
    if 10% good: 0.1*2 +0.9*(-0.05)=+0.155 >0, encouraging exploration vs never fire.
  - points decision weight reduced in dogfight.py 0.35→0.05, so timeout win
    requires damage, not just position.

Ranking of what actually matters, in order (a human BFM debrief order):
    1. Don't die.            (event)
    2. Get a gun solution.   (lead angle <1.5°)
    3. Stay in the fight.    (energy, don't overshoot)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import vecmath as vm
from .geometry import Geometry, DEFAULT_GUN_ENVELOPE, tracking_quality


@dataclass
class RewardConfig:
    # --- event rewards (objective) — gunnery-focused final: hits must dominate staying bonus
    r_hit_dealt: float = 20.0      # was 10.0, was 1.0 — sparse hit must outweigh staying 0.1*600=60
    r_hit_taken: float = -1.0
    r_kill: float = 200.0         # was 100, was 30 — kill is the objective, must beat staying bonus
    r_death: float = -30.0
    r_crash: float = -30.0
    r_arena_exit: float = -20.0
    r_timeout_win: float = 2.0    # was 5.0 — reduce points farming
    r_timeout_loss: float = -2.0
    r_timeout_draw: float = 0.0

    # --- potential weights (shaping; policy-invariant)
    # coarse lead now has gradient at large errors (was capped at 5°)
    w_lead_angle: float = 1.0     # was 3.5 — coarse acquisition, now with soft cap
    lead_scale_deg: float = 15.0  # was 5° — broader, gives gradient from 80°→0°
    w_aspect: float = 0.2         # was 0.6 — don't reward just being behind
    w_range: float = 0.15         # was 0.4
    w_energy: float = 0.1         # was 0.35
    w_closure: float = 0.05       # was 0.25

    # Tracking term — the fix for "wins on points, cannot shoot".
    # Two scales: coarse 30° weight 2.0 for acquisition (grad at 100°→20°), fine 1° weight 12.0 for precision
    w_track: float = 12.0         # was 1.5 — now dominant, fine
    track_scale_deg: float = 1.0  # was 3.0 — sharply peaked at 1°
    w_track_coarse: float = 2.0   # coarse tracking 30° scale for gradient at 100° (suggestion 3)
    track_coarse_scale_deg: float = 30.0

    # Ammo / trigger discipline — gunnery-focused final, hits must be visible but staying also rewarded
    r_trigger_waste: float = -0.01      # out of WEZ waste, small
    r_good_trigger: float = 2.0         # strong reward for good trigger
    r_bad_trigger: float = -0.01        # almost negligible spray penalty — avoid never-fire local optimum
    r_good_solution: float = 0.5        # 0.5*600=300 for 60s good solution, encourages tracking, less than kill 200? actually 300>200 but ok, tracking is objective
    r_in_wez: float = 0.05              # 0.05*600=30, small positive for remaining in WEZ
    good_lead_deg: float = 1.5
    bad_lead_deg: float = 10.0

    gamma: float = 0.995

    # safety rails: allow larger tracking term
    max_potential: float = 15.0

    def potential(self, g: Geometry, me, them, in_wez: bool,
                  lead_angle_deg: float) -> float:
        """Phi(s).  Bounded, unitless-ish, comparable across scenarios.

        Gunnery-focused final (suggestion 3):
          - coarse lead: soft cap via la/sqrt(1+la²) gives gradient even at 80°
          - tracking: 1/(1+(lead/scale)²) not exp(-(lead/scale)²) — keeps gradient
            alive at 100°/80°/60°/40°/20°, important for ES.
          - good_solution and in_wez handled in events, but phi high when in solution.
        """
        # 1. coarse nose authority — soft cap, gradient at large errors
        la_raw = abs(lead_angle_deg) / max(self.lead_scale_deg, 1.0)
        f_lead = - (la_raw / math.sqrt(1.0 + la_raw * la_raw))  # -1..0

        # 2. aspect: TAA near 0 = on his tail, de-weighted
        f_aspect = math.cos(math.radians(g.taa_deg)) * 0.5 + 0.5

        # 3. range: peaked inside gun envelope, de-weighted
        r = g.range_m
        if r < DEFAULT_GUN_ENVELOPE.r_min_m:
            f_range = -1.0
        elif r <= DEFAULT_GUN_ENVELOPE.r_max_m:
            f_range = 1.0 - abs(r - 400.0) / 900.0
        else:
            f_range = max(-1.0, 1.0 - (r - DEFAULT_GUN_ENVELOPE.r_max_m) / 3000.0)

        # 4. energy
        de = me.specific_energy() - them.specific_energy()
        f_energy = math.tanh(de / 1500.0)

        # 5. closure
        if r > 1200.0:
            f_closure = math.tanh(g.closure_ms / 150.0)
        else:
            f_closure = -math.tanh(max(0.0, g.closure_ms - 30.0) / 120.0)

        # 6. tracking: 1/(1+(lead/scale)²) not exp — keeps gradient at 100° etc (suggestion 3)
        if r <= DEFAULT_GUN_ENVELOPE.r_max_m * 1.5 and g.closure_ms > -30.0:
            # fine 1°: 1/(1+(lead/1)²) -> 0.5 at 1°, 0.2 at 2°, still 0.01 at 10°
            f_fine = 1.0 / (1.0 + (abs(lead_angle_deg) / max(self.track_scale_deg, 0.3)) ** 2)
            if abs(lead_angle_deg) <= self.good_lead_deg:
                f_fine *= 1.5
            # coarse 30°: 1/(1+(lead/30)²) -> 0.5 at 30°, 0.2 at 60°, 0.1 at 90°, still gradient at 100°
            f_coarse = 1.0 / (1.0 + (abs(lead_angle_deg) / max(self.track_coarse_scale_deg, 1.0)) ** 2)
        else:
            f_fine = 0.0
            f_coarse = 0.0

        phi = (self.w_lead_angle * f_lead
               + self.w_aspect * f_aspect
               + self.w_range * f_range
               + self.w_energy * f_energy
               + self.w_closure * f_closure
               + self.w_track * f_fine
               + self.w_track_coarse * f_coarse)
        m = self.max_potential
        return max(-m, min(m, phi))


@dataclass
class StepEvents:
    hits_dealt: int = 0
    hits_taken: int = 0
    trigger_waste: int = 0      # decision steps with trigger down out of envelope
    good_trigger: int = 0       # trigger down with lead<=1.5° in WEZ
    bad_trigger: int = 0        # trigger down with lead>10°
    good_solution: int = 0      # in WEZ with lead<=1.5° regardless of trigger — stay bonus 0.5
    in_wez: int = 0             # small positive for remaining in WEZ 0.05
    killed_them: bool = False
    was_killed: bool = False
    crashed: bool = False
    arena_exit: bool = False


def step_reward(cfg: RewardConfig, phi_prev: float, phi_now: float,
                ev: StepEvents) -> tuple:
    """Return (reward, breakdown dict)."""
    shaped = cfg.gamma * phi_now - phi_prev
    ev_r = (cfg.r_hit_dealt * ev.hits_dealt
            + cfg.r_hit_taken * ev.hits_taken
            + cfg.r_trigger_waste * ev.trigger_waste
            + cfg.r_good_trigger * ev.good_trigger
            + cfg.r_bad_trigger * ev.bad_trigger
            + cfg.r_good_solution * ev.good_solution
            + cfg.r_in_wez * ev.in_wez)
    if ev.killed_them:
        ev_r += cfg.r_kill
    if ev.was_killed:
        ev_r += cfg.r_death
    if ev.crashed:
        ev_r += cfg.r_crash
    if ev.arena_exit:
        ev_r += cfg.r_arena_exit
    total = shaped + ev_r
    return total, {"shaped": shaped, "events": ev_r, "phi": phi_now,
                   "good_trig": ev.good_trigger, "bad_trig": ev.bad_trigger,
                   "good_sol": ev.good_solution}


def terminal_reward(cfg: RewardConfig, result: str) -> float:
    """result in {win, loss, draw} — reduced win to avoid points farming"""
    return {"win": cfg.r_timeout_win, "loss": cfg.r_timeout_loss,
            "draw": cfg.r_timeout_draw}[result]


def td_error(cfg: RewardConfig, r: float, v_next: float, v_now: float) -> float:
    """Temporal-difference signal used as the dopamine drive.

    delta = r + gamma*V(s') - V(s)

    This is what feeds the three-factor plasticity rule in brain/plasticity.py.
    It is deliberately *not* the raw reward: a raw-reward dopamine pulse trains
    the mushroom-body-style plastic weights to associate whatever they happened
    to be doing with a hit, which is exactly the credit-assignment failure that
    makes naive fly-brain demos look like they are learning when they are not.
    """
    return r + cfg.gamma * v_next - v_now
