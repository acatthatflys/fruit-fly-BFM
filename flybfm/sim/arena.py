"""Scenario generation and domain randomisation.

Every fight is defined by a small set of *meaningful* initial conditions rather
than by raw coordinates, because those are the axes you actually want to
randomise over to stop the agent from over-fitting a single opening:

    range_m        initial separation
    taa_deg        target aspect angle at the start (0 = attacker on the tail)
    nose_offset    (HCA - TAA): 0 = attacker's nose is pointed at the target
    dz_m           vertical offset of the attacker relative to the target
    alt_m          altitude of the fight
    v_a, v_t       initial airspeeds

This directly implements the "start it in different positions and speeds"
instinct, but with names that map onto the BFM literature so results are
comparable to published setups.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from .aircraft import AircraftParams, State, default_f16


@dataclass
class Scenario:
    range_m: float = 1500.0
    taa_deg: float = 0.0        # 0 = attacker on the target's six
    nose_offset_deg: float = 0.0
    dz_m: float = 0.0
    alt_m: float = 6000.0
    v_a: float = 250.0
    v_t: float = 250.0
    phase_deg: float = 0.0      # world orientation of the whole setup
    bank_a_deg: float = 0.0
    gamma_a_deg: float = 0.0
    gamma_t_deg: float = 0.0
    seed: int = 0
    tag: str = ""

    @property
    def hca_deg(self) -> float:
        return self.taa_deg + self.nose_offset_deg

    def attacker_is_offensive(self) -> bool:
        return self.taa_deg < 90.0

    def describe(self) -> str:
        return (f"R={self.range_m:.0f}m TAA={self.taa_deg:.0f} HCA={self.hca_deg:.0f} "
                f"dz={self.dz_m:+.0f}m alt={self.alt_m:.0f}m "
                f"Va={self.v_a:.0f} Vt={self.v_t:.0f} ({self.tag})")

    def build(self, p_a: AircraftParams | None = None,
              p_t: AircraftParams | None = None):
        """Return (attacker_state, target_state, (attacker, target)) Aircraft."""
        from .aircraft import Aircraft

        p_a = p_a or default_f16()
        p_t = p_t or default_f16()

        phase = math.radians(self.phase_deg)
        taa = math.radians(self.taa_deg)
        # target flies along +phase; its tail points back along phase+180
        tail_bearing = phase + math.pi
        # attacker sits at angle `taa` off that tail line (signed)
        los_dir = _unit2(tail_bearing + taa)
        t_pos = (0.0, 0.0, self.alt_m)
        a_pos = (t_pos[0] + self.range_m * los_dir[0],
                 t_pos[1] + self.range_m * los_dir[1],
                 t_pos[2] + self.dz_m)

        a_gamma = math.radians(self.gamma_a_deg)
        t_gamma = math.radians(self.gamma_t_deg)
        a_psi = phase + math.radians(self.hca_deg)
        t_psi = phase

        # centre the geometry on the arena centre, otherwise a 6 km offset
        # start plus 90 s of flying puts half the fights through the wall
        cx = 0.5 * (a_pos[0] + t_pos[0])
        cy = 0.5 * (a_pos[1] + t_pos[1])
        a_pos = (a_pos[0] - cx, a_pos[1] - cy, a_pos[2])
        t_pos = (t_pos[0] - cx, t_pos[1] - cy, t_pos[2])

        a_state = State(t=0.0, pos=a_pos, psi=a_psi, gamma=a_gamma, v=self.v_a,
                        mu=math.radians(self.bank_a_deg),
                        throttle=0.8, fuel=p_a.fuel_kg)
        t_state = State(t=0.0, pos=t_pos, psi=t_psi, gamma=t_gamma, v=self.v_t,
                        mu=0.0, throttle=0.8, fuel=p_t.fuel_kg)

        a = Aircraft(p_a, a_state, seed=self.seed * 2 + 1)
        t = Aircraft(p_t, t_state, seed=self.seed * 2 + 2)
        return a_state, t_state, (a, t)


def _unit2(bearing: float):
    return (math.cos(bearing), math.sin(bearing))


# --------------------------------------------------------------------------
@dataclass
class ScenarioSpace:
    """Uniform ranges to sample a Scenario from."""
    range_m: tuple = (600.0, 6000.0)
    taa_deg: tuple = (0.0, 180.0)
    nose_offset_deg: tuple = (-60.0, 60.0)
    dz_m: tuple = (-400.0, 400.0)
    alt_m: tuple = (3000.0, 8000.0)
    v_a: tuple = (150.0, 330.0)
    v_t: tuple = (150.0, 330.0)
    jitter_deg: float = 5.0
    jitter_speed: float = 10.0

    def sample(self, rng: random.Random, tag: str = "random") -> Scenario:
        s = Scenario(
            range_m=rng.uniform(*self.range_m),
            taa_deg=rng.uniform(*self.taa_deg),
            nose_offset_deg=rng.uniform(*self.nose_offset_deg) + rng.gauss(0, self.jitter_deg),
            dz_m=rng.uniform(*self.dz_m),
            alt_m=rng.uniform(*self.alt_m),
            v_a=rng.uniform(*self.v_a) + rng.gauss(0, self.jitter_speed),
            v_t=rng.uniform(*self.v_t) + rng.gauss(0, self.jitter_speed),
            phase_deg=rng.uniform(0, 360),
            bank_a_deg=rng.uniform(-45, 45),
            gamma_a_deg=rng.uniform(-25, 25),
            gamma_t_deg=rng.uniform(-25, 25),
            seed=rng.randrange(1, 2 ** 30),
            tag=tag,
        )
        s.v_a = max(120.0, min(340.0, s.v_a))
        s.v_t = max(120.0, min(340.0, s.v_t))
        s.range_m = max(300.0, s.range_m)
        return s


# Named canonical BFM starting setups.  These are the ones a human instructor
# actually uses, and they double as the held-out evaluation suite.
CANONICAL_SETUPS = {
    "perch":        Scenario(range_m=900,  taa_deg=0,   nose_offset_deg=0,   dz_m=50,   alt_m=6000, v_a=280, v_t=240, tag="perch"),
    "lag_45":       Scenario(range_m=1200, taa_deg=0,   nose_offset_deg=45,  dz_m=0,    alt_m=6000, v_a=260, v_t=260, tag="lag-45"),
    "lead_45":      Scenario(range_m=1200, taa_deg=0,   nose_offset_deg=-45, dz_m=0,    alt_m=6000, v_a=260, v_t=260, tag="lead-45"),
    "beam_90":      Scenario(range_m=1500, taa_deg=90,  nose_offset_deg=0,   dz_m=0,    alt_m=6000, v_a=260, v_t=260, tag="beam"),
    "head_on":      Scenario(range_m=4000, taa_deg=180, nose_offset_deg=0,   dz_m=100,  alt_m=6000, v_a=280, v_t=280, tag="head-on"),
    "defensive":    Scenario(range_m=1500, taa_deg=135, nose_offset_deg=0,   dz_m=-100, alt_m=6000, v_a=260, v_t=260, tag="defensive"),
    "neutral_mirror": Scenario(range_m=2000, taa_deg=90, nose_offset_deg=-90, dz_m=0,   alt_m=6000, v_a=260, v_t=260, tag="mirror"),
    "topgun_break": Scenario(range_m=800,  taa_deg=165, nose_offset_deg=-20, dz_m=150,  alt_m=5000, v_a=300, v_t=250, tag="gone-defensive"),
}


def canonical_curriculum():
    """A rough difficulty ramp: offensive -> neutral -> defensive starts."""
    return [
        ("perch", 1.0), ("lag_45", 1.0), ("lead_45", 1.0),
        ("beam_90", 1.0), ("defensive", 1.0), ("head_on", 1.0),
        ("neutral_mirror", 1.0), ("topgun_break", 1.0),
    ]
