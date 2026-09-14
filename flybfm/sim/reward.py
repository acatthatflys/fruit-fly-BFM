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

Ranking of what actually matters, in order (a human BFM debrief order):
    1. Don't die.            (event)
    2. Get a gun solution.   (angle-off / lead angle)
    3. Stay in the fight.    (energy, don't overshoot)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import vecmath as vm
from .geometry import Geometry, DEFAULT_GUN_ENVELOPE, tracking_quality


@dataclass
class RewardConfig:
    # --- event rewards (objective)
    r_hit_dealt: float = 1.0
    r_hit_taken: float = -1.0
    r_kill: float = 30.0
    r_death: float = -30.0
    r_crash: float = -30.0
    r_arena_exit: float = -20.0
    r_timeout_win: float = 5.0
    r_timeout_loss: float = -5.0
    r_timeout_draw: float = 0.0

    # --- potential weights (shaping; policy-invariant)
    w_lead_angle: float = 3.5     # the single most BFM-relevant term
    lead_scale_deg: float = 12.0  # saturation scale for the lead-angle term
    w_aspect: float = 0.6         # offensive vs defensive geometry
    w_range: float = 0.4
    w_energy: float = 0.35
    w_closure: float = 0.25

    # Tracking term.  This is the one that decides whether the agent ever learns
    # to *shoot* rather than merely to arrive behind somebody.  The gun needs the
    # nose inside about one degree at 600 m, and none of the terms above is
    # sensitive on that scale: an agent can sit in the gun envelope, pointed
    # within 10 degrees, collect most of the shaping reward and hit nothing.
    # Measured: without this term the learner fired 21,000 rounds for 0 hits.
    w_track: float = 1.5
    track_scale_deg: float = 3.0

    # Ammo discipline.  Trigger down outside the gun envelope is waste, and a
    # real BFM instructor will tell you so.  Without a cost, a learner discovers
    # that holding the trigger is free.
    r_trigger_waste: float = -0.01

    gamma: float = 0.995

    # safety rails: never let a shaped term exceed these (keeps Phi bounded)
    max_potential: float = 4.0

    def potential(self, g: Geometry, me, them, in_wez: bool,
                  lead_angle_deg: float) -> float:
        """Phi(s).  Bounded, unitless-ish, comparable across scenarios."""
        # 1. nose authority: we would rather be pointed correctly than close.
        la = min(abs(lead_angle_deg) / max(self.lead_scale_deg, 1.0), 1.0)
        f_lead = -(la ** 2)                       # -1 best .. 0 worst-ish

        # 2. aspect: TAA near 0 = we are on his tail (offensive)
        f_aspect = math.cos(math.radians(g.taa_deg)) * 0.5 + 0.5   # 1 .. 0

        # 3. range: peaked inside the gun envelope
        r = g.range_m
        if r < DEFAULT_GUN_ENVELOPE.r_min_m:
            f_range = -1.0
        elif r <= DEFAULT_GUN_ENVELOPE.r_max_m:
            f_range = 1.0 - abs(r - 400.0) / 900.0
        else:
            f_range = max(-1.0, 1.0 - (r - DEFAULT_GUN_ENVELOPE.r_max_m) / 3000.0)

        # 4. energy: specific-energy advantage, saturating (so "run away and
        #    climb" cannot farm reward indefinitely)
        de = me.specific_energy() - them.specific_energy()
        f_energy = math.tanh(de / 1500.0)

        # 5. closure: want to close when far, want lower closure when close
        if r > 1200.0:
            f_closure = math.tanh(g.closure_ms / 150.0)
        else:
            # near the band, being *too* fast into the band is an overshoot risk
            f_closure = -math.tanh(max(0.0, g.closure_ms - 30.0) / 120.0)

        # 6. tracking: a sharply peaked function of the *nose* error, active
        #    only where a round could actually connect
        aa = abs(g.aa_deg)
        if r <= DEFAULT_GUN_ENVELOPE.r_max_m * 1.3 and g.closure_ms > -30.0:
            f_track = math.exp(-(aa / max(self.track_scale_deg, 0.5)) ** 2)
        else:
            f_track = 0.0

        phi = (self.w_lead_angle * f_lead
               + self.w_aspect * f_aspect
               + self.w_range * f_range
               + self.w_energy * f_energy
               + self.w_closure * f_closure
               + self.w_track * f_track)
        m = self.max_potential
        return max(-m, min(m, phi))


@dataclass
class StepEvents:
    hits_dealt: int = 0
    hits_taken: int = 0
    trigger_waste: int = 0      # decision steps with the trigger down out of the envelope
    killed_them: bool = False
    was_killed: bool = False
    crashed: bool = False
    arena_exit: bool = False


def step_reward(cfg: RewardConfig, phi_prev: float, phi_now: float,
                ev: StepEvents) -> tuple:
    """Return (reward, breakdown dict).

    Shaping first, then events.  The shaped part telescopes, so over an episode
    the total shaping is bounded by gamma*Phi(s_T) - Phi(s_0): the agent cannot
    get rich by loitering in a high-potential region.
    """
    shaped = cfg.gamma * phi_now - phi_prev
    ev_r = (cfg.r_hit_dealt * ev.hits_dealt
            + cfg.r_hit_taken * ev.hits_taken)
    if ev.killed_them:
        ev_r += cfg.r_kill
    if ev.was_killed:
        ev_r += cfg.r_death
    if ev.crashed:
        ev_r += cfg.r_crash
    if ev.arena_exit:
        ev_r += cfg.r_arena_exit
    total = shaped + ev_r
    return total, {"shaped": shaped, "events": ev_r, "phi": phi_now}


def terminal_reward(cfg: RewardConfig, result: str) -> float:
    """result in {win, loss, draw}"""
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
