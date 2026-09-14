"""The 1v1 guns-only dogfight environment.

Design notes
------------
* Physics runs at 50 Hz, decisions at 10 Hz (configurable).  Gun rounds are
  resolved at physics rate, so trigger timing is not limited by the policy rate.
* Both sides are symmetric: each gets its own geometry, its own potential and
  its own reward, computed from its own frame of reference.  Self-play is
  therefore literally the same fight viewed from two cockpits, which is what
  prevents "the agent" and "the opponent" from drifting into different physics.
* Everything is deterministic given (scenario seed, action sequence), which is
  required for reproducible replays and for black-box optimisers like CEM/ES.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .. import vecmath as vm
from . import geometry as geo
from .aircraft import Aircraft, AircraftParams, Command, State, default_f16
from .arena import Scenario
from .gun import Gun, GunState, HIT_DAMAGE
from .reward import RewardConfig, StepEvents, step_reward, terminal_reward


@dataclass
class EnvConfig:
    dt: float = 0.02                 # physics step
    decision_dt: float = 0.1         # brain / policy step
    max_time_s: float = 90.0
    arena_radius_m: float = 12000.0
    floor_m: float = 250.0
    ceiling_m: float = 13000.0
    start_hp: float = 1.0
    record_every: int = 5            # downsample factor for replay frames


@dataclass(eq=False)          # identity semantics: used as a dict key
class Side:
    name: str
    ac: Aircraft
    gun: Gun
    hp: float
    geom_to_other: Optional[geo.Geometry] = None
    phi: float = 0.0
    lead_angle: float = 0.0
    los_dir_prev: Optional[tuple] = None
    los_rate_az: float = 0.0
    los_rate_el: float = 0.0
    los_rate_rad: float = 0.0
    in_wez: bool = False
    reward: float = 0.0
    dead: bool = False
    cause_of_death: str = ""
    feats: List[list] = field(default_factory=list)
    fire_time_s: float = 0.0
    cmd: Optional[Command] = None        # last commanded control deflection

    @property
    def s(self) -> State:
        return self.ac.s


class Dogfight:
    """One 1v1 engagement."""

    def __init__(self, cfg: Optional[EnvConfig] = None,
                 reward_cfg: Optional[RewardConfig] = None,
                 p_blue: Optional[AircraftParams] = None,
                 p_red: Optional[AircraftParams] = None):
        self.cfg = cfg or EnvConfig()
        self.rc = reward_cfg or RewardConfig()
        self.p_blue = p_blue or default_f16()
        self.p_red = p_red or default_f16()
        self.scenario: Optional[Scenario] = None
        self.blue: Optional[Side] = None
        self.red: Optional[Side] = None
        self.frames: List[dict] = []
        self.event_log: List[dict] = []
        self.t = 0.0
        self.done = False
        self.result = ""
        self._step_count = 0
        self.rng_state = 12345

    # ------------------------------------------------------------------ reset
    def reset(self, scenario: Scenario):
        self.scenario = scenario
        a_state, t_state, (a_ac, t_ac) = scenario.build(self.p_blue, self.p_red)
        self.blue = Side("blue", a_ac, Gun(GunState(ammo=self.p_blue.gun_rounds),
                                           self.p_blue.gun_rps, seed=7),
                         self.cfg.start_hp)
        self.red = Side("red", t_ac, Gun(GunState(ammo=self.p_red.gun_rounds),
                                         self.p_red.gun_rps, seed=11),
                        self.cfg.start_hp)
        self._prime_geometry()
        self.t = 0.0
        self.done = False
        self.result = ""
        self._step_count = 0
        self.frames = []
        self.event_log = []
        self._record(force=True)
        return self.observe()

    def _prime_geometry(self):
        for me, other in ((self.blue, self.red), (self.red, self.blue)):
            me.geom_to_other = geo.compute(me.s, other.s)
            me.los_dir_prev = me.geom_to_other.azimuth_el
            me.lead_angle = _lead_angle(me.s, other.s)
            me.in_wez = geo.in_wez(me.geom_to_other)
            me.phi = self.rc.potential(me.geom_to_other, me.s, other.s,
                                       me.in_wez, me.lead_angle)

    # ------------------------------------------------------------------- step
    def step(self, cmd_blue: Command, cmd_red: Command):
        """Advance one decision interval. Returns (obs, rewards, done, info)."""
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        cfg = self.cfg
        n_sub = max(1, int(round(cfg.decision_dt / cfg.dt)))
        events = {self.blue: StepEvents(), self.red: StepEvents()}
        phi_prev = {self.blue: self.blue.phi, self.red: self.red.phi}

        self.blue.cmd = cmd_blue
        self.red.cmd = cmd_red
        for _ in range(n_sub):
            self.blue.ac.step(cmd_blue, cfg.dt)
            self.red.ac.step(cmd_red, cfg.dt)
            self.t += cfg.dt

            # gun resolution at physics rate
            fb, hb = self.blue.gun.fire(self.blue.s.pos, self.blue.s.vel,
                                        self.red.s.pos, self.red.s.vel,
                                        cfg.dt, cmd_blue.trigger and self.red.s.alive)
            fr, hr = self.red.gun.fire(self.red.s.pos, self.red.s.vel,
                                       self.blue.s.pos, self.blue.s.vel,
                                       cfg.dt, cmd_red.trigger and self.blue.s.alive)
            if hb:
                self.red.hp -= hb * HIT_DAMAGE
                events[self.blue].hits_dealt += hb
                events[self.red].hits_taken += hb
                self.blue.s.hits_dealt += hb
                self.red.s.hits_on_self += hb
                self._log(f"blue hits red x{hb} (hp={max(self.red.hp,0):.2f})")
            if hr:
                self.blue.hp -= hr * HIT_DAMAGE
                events[self.red].hits_dealt += hr
                events[self.blue].hits_taken += hr
                self.red.s.hits_dealt += hr
                self.blue.s.hits_on_self += hr
                self._log(f"red hits blue x{hr} (hp={max(self.blue.hp,0):.2f})")

            if self.t >= cfg.max_time_s:
                break

        # ---- geometry / potential update at decision rate
        for me, other in ((self.blue, self.red), (self.red, self.blue)):
            g = geo.compute(me.s, other.s)
            me.geom_to_other = g
            me.lead_angle = _lead_angle(me.s, other.s)
            me.in_wez = geo.in_wez(g)
            if me.los_dir_prev is not None and cfg.decision_dt > 0:
                # rotation rate of the LOS vector, signed relative to the nose.
                # This is the fly's lobula-plate-style "how fast is the target
                # sliding across my eye" signal, and the damping term for the
                # scripted steering law.
                ang = vm.angle_between(me.los_dir_prev, g.azimuth_el)
                axis = vm.cross(me.los_dir_prev, g.azimuth_el)
                sign = -1.0 if vm.dot(axis, me.s.vel) > 0 else 1.0
                me.los_rate_rad = sign * ang / cfg.decision_dt
                me.los_rate_az = me.los_rate_rad
                me.los_rate_el = ang / cfg.decision_dt
            me.los_dir_prev = g.azimuth_el
            me.phi = self.rc.potential(g, me.s, other.s, me.in_wez, me.lead_angle)

        # ---- deaths and terminations
        for me, other in ((self.blue, self.red), (self.red, self.blue)):
            if me.dead:
                continue
            if not me.s.alive:
                me.dead = True
                me.cause_of_death = "crash" if me.s.crashed else "control-loss"
                events[other].killed_them = True
                events[me].crashed = me.s.crashed
                self._log(f"{me.name} {me.cause_of_death}")
            elif me.hp <= 0.0:
                me.dead = True
                me.cause_of_death = "shot-down"
                events[other].killed_them = True
                self._log(f"{me.name} shot down")
            else:
                reason = _outside(me.s, cfg)
                if reason:
                    me.dead = True
                    me.cause_of_death = reason
                    events[me].arena_exit = True
                    self._log(f"{me.name}: {reason}")

        timeout = self.t >= cfg.max_time_s

        # ---- trigger discipline: rounds sent outside the envelope are waste
        for me, cmd in ((self.blue, cmd_blue), (self.red, cmd_red)):
            if cmd.trigger and not me.in_wez:
                events[me].trigger_waste += 1

        # ---- rewards
        rewards = {}
        for me in (self.blue, self.red):
            r, brk = step_reward(self.rc, phi_prev[me], me.phi, events[me])
            me.reward = r
            rewards[me.name] = r

        if self.blue.dead or self.red.dead:
            self.done = True
            if self.blue.dead and self.red.dead:
                self.result = "mutual"
            elif self.red.dead:
                self.result = "blue"
            else:
                self.result = "red"
            for me, other in ((self.blue, self.red), (self.red, self.blue)):
                if me.dead and not other.dead:
                    rewards[me.name] += terminal_reward(self.rc, "loss")
                elif other.dead and not me.dead:
                    rewards[me.name] += terminal_reward(self.rc, "win")
        elif timeout:
            self.done = True
            winner = self._points_decision()
            self.result = winner
            for me, other in ((self.blue, self.red), (self.red, self.blue)):
                if winner == "draw":
                    rewards[me.name] += terminal_reward(self.rc, "draw")
                elif winner == me.name:
                    rewards[me.name] += terminal_reward(self.rc, "win")
                else:
                    rewards[me.name] += terminal_reward(self.rc, "loss")

        self._step_count += 1
        self._record()
        return self.observe(), rewards, self.done, self.info()

    # ---------------------------------------------------------------- helpers
    def _points_decision(self) -> str:
        """Timeout = decision on points (BFM ruleset): damage, then position."""
        db = self.blue.hp - self.red.hp
        pos_b = math.cos(math.radians(self.blue.geom_to_other.taa_deg))
        pos_r = math.cos(math.radians(self.red.geom_to_other.taa_deg))
        score = db + 0.35 * (pos_b - pos_r)
        if abs(score) < 0.05:
            return "draw"
        return "blue" if score > 0 else "red"

    def observe(self) -> Dict[str, tuple]:
        return {"blue": self.observation_vector(self.blue),
                "red": self.observation_vector(self.red)}

    def observation_vector(self, me: Side) -> tuple:
        """Hand-designed feature vector (used by the non-brain baselines)."""
        other = self.red if me is self.blue else self.blue
        g = me.geom_to_other
        s = me.s
        de = math.tanh((s.specific_energy() - other.s.specific_energy()) / 1500.0)
        return (
            g.aa_deg / 60.0,                      # nose position, coarse
            g.aa_vert_deg / 60.0,
            math.cos(math.radians(g.taa_deg)),    # +1 offensive .. -1 defensive
            g.hca_deg / 180.0,
            min(g.range_m / 6000.0, 1.5),
            math.tanh(g.closure_ms / 200.0),
            me.los_rate_rad / 0.5,
            # Fine nose error.  This slot used to be a 0.0 placeholder, which
            # cost the linear policy the only thing that decides whether a round
            # connects: at 600 m the nose has to be inside about a degree, and a
            # feature that spans +-60 deg gives a linear policy a resolution of
            # 1/60 near zero -- indistinguishable from noise once the weights are
            # also being asked to fly the aircraft.
            _saturate(g.aa_deg / 5.0),
            s.v / 400.0,
            s.gamma / math.radians(60.0),
            math.sin(s.mu),
            math.cos(s.mu),
            s.n / 9.0,
            s.throttle,
            de,
            _saturate(me.lead_angle / 10.0),      # ballistic lead error, fine
            1.0 if me.in_wez else 0.0,
            1.0 if geo.overshoot_flag(g) else 0.0,
        )

    def info(self) -> dict:
        return {
            "t": self.t,
            "result": self.result,
            "blue_hp": max(self.blue.hp, 0.0),
            "red_hp": max(self.red.hp, 0.0),
            "blue_dead": self.blue.dead,
            "red_dead": self.red.dead,
            "blue_reward": self.blue.reward,
            "red_reward": self.red.reward,
            "scenario": self.scenario.describe() if self.scenario else "",
        }

    def summary(self) -> dict:
        """Episode-level metrics used by the evaluator."""
        return {
            "result": self.result,
            "t": round(self.t, 2),
            "blue_hp": round(max(self.blue.hp, 0.0), 3),
            "red_hp": round(max(self.red.hp, 0.0), 3),
            "blue_hits": self.blue.gun.g.hits,
            "red_hits": self.red.gun.g.hits,
            "blue_shots": self.blue.gun.g.shots,
            "red_shots": self.red.gun.g.shots,
            "blue_gun_time_s": round(self.blue.fire_time_s, 2),
            "blue_peak_g": round(self.blue.s.nz_max_seen, 2),
            "tag": self.scenario.tag if self.scenario else "",
        }

    def _log(self, msg: str):
        self.event_log.append({"t": round(self.t, 3), "msg": msg})

    # --------------------------------------------------------------- recording
    def _record(self, force: bool = False):
        if not force and self._step_count % self.cfg.record_every != 0:
            return
        self.frames.append({
            "t": round(self.t, 2),
            "b": _ac_snap(self.blue),
            "r": _ac_snap(self.red),
            "g": {
                "aa": round(self.blue.geom_to_other.aa_deg, 1),
                "taa": round(self.blue.geom_to_other.taa_deg, 1),
                "hca": round(self.blue.geom_to_other.hca_deg, 1),
                "rng": round(self.blue.geom_to_other.range_m, 1),
                "clo": round(self.blue.geom_to_other.closure_ms, 1),
                "wez": bool(self.blue.in_wez),
                "lead": round(self.blue.lead_angle, 1),
            },
            "hp": [round(max(self.blue.hp, 0), 3), round(max(self.red.hp, 0), 3)],
        })

    def to_replay(self) -> dict:
        return {
            "scenario": {"tag": self.scenario.tag, "desc": self.scenario.describe(),
                         "range_m": self.scenario.range_m, "taa_deg": self.scenario.taa_deg,
                         "hca_deg": self.scenario.hca_deg, "alt_m": self.scenario.alt_m,
                         "v_a": self.scenario.v_a, "v_t": self.scenario.v_t,
                         "seed": self.scenario.seed},
            "result": self.result,
            "duration_s": round(self.t, 2),
            "summary": self.summary(),
            "events": self.event_log,
            "frames": self.frames,
        }


def _ac_snap(side: Side) -> dict:
    s = side.s
    c = side.cmd
    return {"p": [round(x, 1) for x in s.pos],
            "psi": round(math.degrees(s.psi), 1),
            "gam": round(math.degrees(s.gamma), 1),
            "v": round(s.v, 1),
            "mu": round(math.degrees(s.mu), 1),
            "n": round(s.n, 2),
            "thr": round(s.throttle, 2),
            "ammo": side.gun.g.ammo,
            "cmd": [round(c.roll, 2), round(c.pull, 2), round(c.throttle, 2),
                    bool(c.trigger)] if c else None,
            "es": round(s.specific_energy(), 0)}


def _saturate(v: float, limit: float = 1.5) -> float:
    return max(-limit, min(limit, v))


def _lead_angle(me: State, other: State) -> float:
    from .gun import required_lead_angle_deg
    return required_lead_angle_deg(me.pos, me.vel, other.pos, other.vel)


def _outside(s: State, cfg: EnvConfig):
    """Returns a reason string if the aircraft has left the fight, else None."""
    r = math.hypot(s.pos[0], s.pos[1])
    if r > cfg.arena_radius_m:
        return "ran-away (arena wall)"
    if s.pos[2] < cfg.floor_m:
        return "ground"
    if s.pos[2] > cfg.ceiling_m:
        return "busted-the-ceiling"
    return None


# --------------------------------------------------------------------- match
def run_match(scen: Scenario, policy_blue, policy_red,
              cfg: Optional[EnvConfig] = None,
              rc: Optional[RewardConfig] = None,
              record: bool = True) -> Tuple[dict, list]:
    """Run one episode. Policies take (side_view, t) -> Command.

    `side_view` is a light dict with the geometry + own state, so a policy can
    be a scripted rule, a linear readout, or a connectome-backed brain.
    Returns (summary, blue_reward_trace).
    """
    env = Dogfight(cfg, rc)
    env.reset(scen)
    trace = []
    while not env.done:
        vb = env.blue
        vr = env.red
        cb = policy_blue(vb, env)
        cr = policy_red(vr, env)
        _, rewards, done, _ = env.step(cb, cr)
        trace.append(rewards["blue"])
        if not record:
            env.frames = []
    return env.summary(), trace, env
