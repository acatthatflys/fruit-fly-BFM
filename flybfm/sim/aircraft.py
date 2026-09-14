"""Point-mass energy-maneuverability (EM) flight model.

This is the standard "velocity vector" / energy model used for BFM studies and
for most air-combat RL environments that are not built on a full 6-DOF code
(JSBSim etc.).  It integrates four states:

    V     airspeed                       [m/s]
    psi   heading                        [rad, from +x toward +y]
    gamma flight-path angle              [rad, + = climbing]
    pos   position                       [m, z-up]

with three effective controls:

    roll_cmd      [-1, 1]   -> commanded roll rate
    pull_cmd      [-1, 1]   -> commanded load factor (n_z), -1 = max push
    throttle_cmd  [0, 1]    -> dry thrust .. max (A/B) thrust

The equations of motion are the classic point-mass set:

    V_dot   = g * ( (T - D) / W - sin(gamma) )
    gamma_dot = g * ( n * cos(mu) - cos(gamma) ) / V
    psi_dot   = g * n * sin(mu) / ( V * cos(gamma) )

with induced drag D_i = k * Cl^2 * q * S and Cl set by the *steady turn*
assumption Cl = n * W / (q * S).  That coupling is what makes energy management
emerge: pulling hard bleeds airspeed, which lowers q, which lowers the turn
rate available.  Corner speed falls out of the model rather than being scripted.

Aerodynamic limits are enforced:
  * Cl <= Cl_max  -> a low-q (slow) aircraft simply cannot pull n_max
  * structural n_max
  * thrust lapse with altitude (sigma) and Mach
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .. import vecmath as vm

G = 9.80665
RHO0 = 1.225          # ISA sea level density [kg/m^3]
T0 = 288.15           # ISA sea level temperature [K]
LAPSE = 0.0065        # K/m


def isa_density(h_m: float) -> float:
    """Tropospheric ISA density; clamped so the sim never blows up at h<0."""
    h = max(h_m, -500.0)
    h = min(h, 20000.0)
    T = T0 - LAPSE * h
    p_ratio = (1.0 - LAPSE * h / T0) ** 5.2561
    t_ratio = T / T0
    return RHO0 * p_ratio / t_ratio


@dataclass
class AircraftParams:
    name: str = "generic-fighter"

    mass_kg: float = 9200.0
    wing_area_m2: float = 27.87
    cd0: float = 0.019
    oswald_k: float = 0.12
    cl_max: float = 1.50

    thrust_dry_n: float = 76_000.0
    thrust_ab_n: float = 128_000.0

    n_max: float = 9.0
    n_min: float = -3.0

    max_roll_rate_dps: float = 240.0    # an F-16 rolls ~270 deg/s at low alpha
    roll_accel_dps2: float = 600.0
    g_onset_rate: float = 9.0           # load-factor slew [g/s]
    engine_tau_s: float = 1.4           # spool-up time constant

    min_speed_ms: float = 40.0          # below this the wing quits
    max_speed_ms: float = 700.0
    fuel_kg: float = 3200.0
    sfc_kg_per_n_s: float = 2.2e-5      # thrust-specific fuel consumption

    # gun / sensor envelope (see gun.py, geometry.py)
    gun_rounds: int = 511
    gun_rps: float = 100.0 / 6.0        # M61: ~100 rds/s over 0.6 s bursts

    def corner_speed(self, rho: float = RHO0) -> float:
        """Speed at which n_max is achievable at Cl_max (level, sea level)."""
        q = self.n_max * self.mass_kg * G / (self.wing_area_m2 * self.cl_max)
        return math.sqrt(2.0 * q / rho)

    def sustained_turn_speed(self, rho: float = RHO0) -> float:
        """Speed that minimises Ps=0 turn radius-ish: where T = D at n_max."""
        # Solve  T = q*S*cd0 + k*Cl_max^2*q*S  in q, then take the *high* root.
        a = self.wing_area_m2 * self.cd0
        b = self.oswald_k * self.wing_area_m2 * self.cl_max ** 2
        # T = q*(a+b)  ->  q = T/(a+b)   (n_max not assumed; this is the
        # max-lift-drag-free point, a useful proxy)
        q = self.thrust_dry_n / (a + b)
        return math.sqrt(2.0 * q / rho)


def default_f16() -> AircraftParams:
    """Roughly F-16C Block 50, clean, ~50% fuel, mil/AB, 9 g limit.

    Sanity check: corner speed comes out ~345 kt, real value ~330 kt.
    """
    return AircraftParams(name="f16c-like")


@dataclass
class State:
    t: float = 0.0
    pos: tuple = (0.0, 0.0, 6000.0)
    psi: float = 0.0
    gamma: float = 0.0
    v: float = 250.0
    mu: float = 0.0            # bank angle [rad]
    n: float = 1.0             # current load factor
    throttle: float = 0.7
    fuel: float = 3200.0
    alive: bool = True
    crashed: bool = False
    # bookkeeping for analysis
    g_load_cmd: float = 1.0
    roll_rate: float = 0.0
    nz_max_seen: float = 1.0
    shots_fired: int = 0
    hits_on_self: int = 0
    hits_dealt: int = 0

    def copy(self) -> "State":
        return replace(self)

    @property
    def vel(self):
        return vm.vel_from_angles(self.v, self.psi, self.gamma)

    @property
    def pos_vec(self):
        return self.pos

    def specific_energy(self) -> float:
        """Specific energy Es = h + V^2/(2g)  [metres]."""
        return self.pos[2] + self.v * self.v / (2.0 * G)


@dataclass
class Command:
    roll: float = 0.0        # [-1, 1]
    pull: float = 0.0        # [-1, 1]
    throttle: float = 0.7    # [0, 1]
    trigger: bool = False

    @staticmethod
    def neutral(p: AircraftParams) -> "Command":
        return Command(0.0, 0.0, 0.7, False)


class Aircraft:
    """Integrates one aircraft.  Pure state machine, no global state."""

    def __init__(self, params: AircraftParams, state: State, seed: int = 0):
        self.p = params
        self.s = state
        self._rng_seed = seed
        self._rng_state = seed * 2654435761 % (2 ** 32)
        self._spool = state.throttle

    # ---------------------------------------------------------------- random
    def _rand(self) -> float:
        """Small deterministic LCG so replays are bit-exact without `random`."""
        self._rng_state = (1103515245 * self._rng_state + 12345) % (2 ** 31)
        return self._rng_state / float(2 ** 31)

    # ------------------------------------------------------------- dynamics
    def q_dyn(self) -> float:
        return 0.5 * isa_density(self.s.pos[2]) * self.s.v * self.s.v

    def thrust_available(self) -> float:
        p, s = self.p, self.s
        sigma = isa_density(s.pos[2]) / RHO0
        # crude Mach lapse; AB loses relatively less
        mach = s.v / 300.0
        lapse = max(0.45, 1.0 - 0.30 * min(mach, 1.2))
        base = p.thrust_dry_n + (p.thrust_ab_n - p.thrust_dry_n) * _ab_frac(self._spool)
        return base * (sigma ** 0.75) * lapse

    def drag(self, n: float) -> float:
        p, s = self.p, self.s
        q = self.q_dyn()
        # Cl implied by the steady-turn assumption, capped by the wing
        qS = max(q * p.wing_area_m2, 1.0)
        cl = min(n * p.mass_kg * G / qS, p.cl_max)
        cd = p.cd0 + p.oswald_k * cl * cl
        # transonic drag rise above M0.9
        mach = s.v / 300.0
        if mach > 0.9:
            cd *= 1.0 + 3.5 * (mach - 0.9) ** 2
        return q * p.wing_area_m2 * cd, cl

    def n_available(self, cl_limited: bool = True) -> float:
        p, s = self.p, self.s
        q = self.q_dyn()
        n_aero = q * p.wing_area_m2 * p.cl_max / (p.mass_kg * G)
        if not cl_limited:
            return p.n_max
        return max(p.n_min, min(p.n_max, n_aero))

    def step(self, cmd: Command, dt: float) -> None:
        p, s = self.p, self.s
        if not s.alive:
            return

        # ---- roll dynamics: rate command, rate-limited
        target_rate = vm.clamp(cmd.roll, -1.0, 1.0) * math.radians(p.max_roll_rate_dps)
        max_dr = math.radians(p.roll_accel_dps2) * dt
        s.roll_rate += vm.clamp(target_rate - s.roll_rate, -max_dr, max_dr)
        s.mu = vm.wrap_pi(s.mu + s.roll_rate * dt)
        # rolling past +-pi wraps; snap the accumulated rate sign near vertical
        if abs(abs(s.mu) - math.pi) < 1e-3:
            s.mu = math.copysign(math.pi, s.mu)

        # ---- load factor: g-command with onset rate limit + aero limit
        n_cmd = p.n_min + (vm.clamp(cmd.pull, -1.0, 1.0) + 1.0) * 0.5 * (p.n_max - p.n_min)
        n_cmd = min(n_cmd, self.n_available(), p.n_max)
        n_cmd = max(n_cmd, p.n_min)
        dn = vm.clamp(n_cmd - s.n, -p.g_onset_rate * dt, p.g_onset_rate * dt)
        s.n += dn
        s.g_load_cmd = n_cmd
        s.nz_max_seen = max(s.nz_max_seen, abs(s.n))

        # ---- engine spool
        tau = p.engine_tau_s
        alpha = min(1.0, dt / max(tau, 1e-6))
        self._spool += (vm.clamp(cmd.throttle, 0.0, 1.0) - self._spool) * alpha
        s.throttle = self._spool

        # ---- forces
        T = self.thrust_available() * max(0.05, self._spool) if self._spool > 0.02 else 0.0
        D, cl = self.drag(max(s.n, 0.2))
        W = p.mass_kg * G
        s.fuel = max(0.0, s.fuel - T * p.sfc_kg_per_n_s * dt)
        if s.fuel <= 0.0:
            T = 0.0
        W = (p.mass_kg + s.fuel * 0.0) * G   # keep mass constant (ballast model)
        v_eff = max(s.v, 20.0)

        # ---- EOM
        v_dot = G * ((T - D) / W - math.sin(s.gamma))
        gamma_dot = G * (s.n * math.cos(s.mu) - math.cos(s.gamma)) / v_eff
        psi_dot = G * s.n * math.sin(s.mu) / (v_eff * max(math.cos(s.gamma), 0.15))

        s.v = vm.clamp(s.v + v_dot * dt, 0.0, p.max_speed_ms)
        s.gamma = vm.clamp(s.gamma + gamma_dot * dt, -math.radians(85.0), math.radians(85.0))
        s.psi = vm.wrap_pi(s.psi + psi_dot * dt)

        # ---- kinematics
        vel = vm.vel_from_angles(s.v, s.psi, s.gamma)
        s.pos = vm.add(s.pos, vm.scale(vel, dt))
        s.t += dt

        # ---- limits
        if s.pos[2] <= 0.0:
            s.pos = (s.pos[0], s.pos[1], 0.0)
            s.alive = False
            s.crashed = True
        if s.v < p.min_speed_ms:
            s.alive = False      # departed / stalled into the ground
        if s.pos[2] > 15000.0:
            s.pos = (s.pos[0], s.pos[1], 15000.0)

    # ------------------------------------------------------------- helpers
    def turn_rate_deg_s(self) -> float:
        """Instantaneous horizontal-ish turn rate for reporting."""
        if self.s.v < 1.0:
            return 0.0
        return math.degrees(G * self.s.n * math.sin(self.s.mu) /
                            (self.s.v * max(math.cos(self.s.gamma), 0.15)))

    def gs(self) -> float:
        return self.s.v


def _ab_frac(throttle: float) -> float:
    """Map throttle 0..1 onto dry..A/B with a small deadband at the gate."""
    if throttle <= 0.80:
        return 0.0
    return min(1.0, (throttle - 0.80) / 0.20)
