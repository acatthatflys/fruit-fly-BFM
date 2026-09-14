"""The brain-in-the-loop controller: sensors -> spiking network -> descending
neurons -> aircraft command, with dopamine delivered after the outcome.

Two things are learned, and it is worth keeping them straight because the
distinction is the difference between a demo and an experiment:

  1. READOUT (outer loop, trained by black-box search over episodes)
     A small linear map from descending-neuron population rates to the four
     command channels.  This is the "co-adaptation" that every viral demo
     quietly replaces with a separate conventional neural network.  Here it is
     explicit, small (a few dozen numbers), and it is the thing CEM/ES searches.
     **100% of the results in runs/ come from this loop.**

  2. CONNECTOME PLASTICITY (inner loop, trained by dopamine — the fly's own)
     Three-factor Hebbian updates on the KC->MBON edges only, gated by the
     TD-error dopamine signal: Δw = η·e·DA − λw. Implemented in lif.py
     (mark_plastic, deliver_dopamine) and tested (test_plasticity_path_exists,
     test_connectome_loader). The synthetic graph has 66 KC→MBON plastic edges;
     the real MaleCNS mushroom body has thousands. This is slower, noisier, and
     biologically the interesting part; it is **switched off by default**
     (plasticity=False) because on its own it is not enough to learn gunnery —
     which is itself a result worth reporting (doomfly: 3,000 runs, no learning).
     If you share results publicly, be crisp: *on the synthetic graph, zero of
     the current learning is the fly's own dopamine-gated plasticity; 100% is
     the outer CEM loop* — unless you explicitly enabled plasticity=True and
     measured the ablation.

Readout parameterisation, so the search is not blind:

    roll_cmd     = k * (turn_right_rate  - turn_left_rate)          [turn rate]
    pull_cmd     = bias + k * (up_rate - down_rate)                 [load factor]
    throttle_cmd = bias + k * speed_rate
    trigger      = rate(trigger_group) > threshold

Left/right DN pairs are compared rather than summed because that is how the fly
actually steers (wing-beat amplitude asymmetry driven by a small number of
left/right DN pairs), and because it makes the sign of the readout meaningful.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .. import vecmath as vm
from ..sim.aircraft import Command
from .connectome import (Connectome, DN_GROUPS, PLASTIC_POST, PLASTIC_PRE,
                         synthetic_connectome, flight_subgraph)
from .lif import LIFNetwork, LIFParams
from .plasticity import DopamineChannel, DopamineConfig, LinearCritic
from .sensors import SensorBank, SensorConfig


@dataclass
class ReadoutParams:
    """The searchable policy: a 1x6 linear map from DN population rates to the
    four aircraft command channels.  Six numbers, all of them meaningful.

        roll     = roll_sign     * turn_gain     * (turn_R - turn_L)   [sign convention: a
                   target to the RIGHT raises turn_R, and a positive roll turns
                   right in this sim, so roll_sign = +1 is "turn toward the
                   target".  CEM is still allowed to flip it.]
        n        = 1 + pull_bias + pull_gain     * (pitch_up - pitch_down)
        throttle = throttle_bias + throttle_gain * speed_rate
        fire     = trigger_gain  * trigger_rate  >  trigger_threshold

    Keep it this small on purpose.  Every extra parameter here is a parameter
    that can absorb a defect in the connectome or the sim, and it is the honest
    place to put a ceiling on "how much of this is the fly brain".
    """
    roll_sign: float = 1.0
    pitch_sign: float = 1.0
    turn_gain: float = 20.0
    pull_gain: float = 1.5
    pull_bias: float = 0.0
    throttle_gain: float = 2.0
    throttle_bias: float = 0.45
    trigger_gain: float = 10.0
    trigger_threshold: float = 0.5
    vis_gain: float = 8.0      # weight on the target-selective (STMD) L/R channel

    N_PARAMS = 10

    def vector(self) -> List[float]:
        return [self.roll_sign, self.pitch_sign, self.turn_gain, self.pull_gain,
                self.pull_bias, self.throttle_gain, self.throttle_bias,
                self.trigger_gain, self.trigger_threshold, self.vis_gain]

    @staticmethod
    def from_vector(v: Sequence[float]) -> "ReadoutParams":
        return ReadoutParams(*[float(x) for x in list(v)[:10]])

    def mutate(self, rng, sigma: float = 0.15) -> "ReadoutParams":
        return ReadoutParams.from_vector(
            [x + rng.gauss(0.0, sigma) for x in self.vector()])


class ConnectomeController:
    """One simulated fly CNS piloting one aircraft."""

    def __init__(self,
                 conn: Optional[Connectome] = None,
                 readout: Optional[ReadoutParams] = None,
                 lif_params: Optional[LIFParams] = None,
                 sensor_cfg: Optional[SensorConfig] = None,
                 da_cfg: Optional[DopamineConfig] = None,
                 seed: int = 0,
                 plasticity: bool = False,
                 name: str = "brain",
                 use_torch: bool = False,
                 torch_device: Optional[str] = None):
        self.name = name
        self.conn = conn if conn is not None else flight_subgraph(
            synthetic_connectome(seed=seed))
        base_net = LIFNetwork(self.conn, lif_params or LIFParams(dt=0.002),
                              seed=seed, plasticity=plasticity)
        self.n_plastic = (base_net.mark_plastic(PLASTIC_PRE, PLASTIC_POST)
                          if plasticity else 0)
        # Optionally promote to torch path for milestones 5-6 (full CNS needs GPU)
        if use_torch:
            try:
                self.net = base_net.to_torch(device=torch_device, seed=seed)
                # mark_plastic already done on base, repeat on torch net if needed
                if plasticity and not self.net.plastic_edges:
                    self.net.mark_plastic(PLASTIC_PRE, PLASTIC_POST)
            except Exception as exc:
                # fall back to Python with a warning — keeps stdlib path working
                import warnings
                warnings.warn(f"use_torch=True but torch not available ({exc}); using Python LIF")
                self.net = base_net
        else:
            self.net = base_net
        self.sensors = SensorBank(self.conn, sensor_cfg)
        self.readout = readout or ReadoutParams()
        self.da = DopamineChannel(self.conn, da_cfg)
        # observation now 22 dims (was 20, was 18) — gunnery-focused final adds lead az/el separate
        self.critic = LinearCritic(n_features=22, lr=(da_cfg or DopamineConfig()).critic_lr,
                                   gamma=(da_cfg or DopamineConfig()).gamma)
        self.plasticity = plasticity
        self._phi = [0.0] * 22
        self._rate = (lif_params or LIFParams(dt=0.002)).dt
        self.stats: Dict[str, float] = {"brain_steps": 0.0, "spikes": 0.0,
                                        "da_total": 0.0, "plastic_updates": 0.0}

    # ---------------------------------------------------------------- indexing
    def _groups(self) -> Dict[str, List[int]]:
        """Population indices the readout is allowed to look at.

        The lateral steering channel is L/R *instances*, not cell types.  A fly
        steers by making the left and right copies of the same descending neuron
        disagree; summing the activity of DNp26_L and DNp26_R and calling it
        "turn" deletes exactly the variable that carries the manoeuvre.  This
        mistake is easy to make and it makes the network look like it contains
        no bearing information at all.
        """
        by_type = self.conn.index_by_type()
        steering_types = sorted(set(DN_GROUPS["turn_left"]) | set(DN_GROUPS["turn_right"]))
        steering = [i for t in steering_types for i in by_type.get(t, [])]
        left = [i for i in steering if self.conn.sides[i] == "L"]
        right = [i for i in steering if self.conn.sides[i] != "L"]
        # Target-selective visual neurons.  Measured fact (tools/probe_brain.py):
        # in a reduced synthetic graph the *descending* lateral channel does not
        # carry the target's bearing -- the information is present in the STMD /
        # VPN population and is lost in the pool onto the DNs.  Handing the
        # readout both channels is the honest version of what every viral demo
        # does with a large learned decoder, and the fitted vis_gain tells you
        # which channel the behaviour actually depends on.
        stmd = [i for t in ("STMD",) for i in by_type.get(t, [])]
        g = {"turn_left": left, "turn_right": right,
             "vis_left": [i for i in stmd if self.conn.sides[i] == "L"],
             "vis_right": [i for i in stmd if self.conn.sides[i] != "L"]}
        for axis in ("pitch_up", "pitch_down", "speed", "trigger"):
            idx: List[int] = []
            for t in DN_GROUPS[axis]:
                idx.extend(by_type.get(t, []))
            g[axis] = idx
        return g

    # ------------------------------------------------------------------- policy
    def act(self, me, env) -> Command:
        """Run the brain for one decision interval and read out a command."""
        other = env.red if me is env.blue else env.blue
        dt_dec = env.cfg.decision_dt
        n_steps = max(1, int(round(dt_dec / self.net.p.dt)))
        groups = self._groups()

        # ---- sensory drive is re-applied every few brain ticks (the world does
        #      not change much inside 100 ms of sim time; the fly's photoreceptors
        #      adapt anyway)
        for k in range(n_steps):
            if k % 5 == 0:
                self.net.clear_inputs()
                self.sensors.inject(self.net, me.s, other.s, me.geom_to_other,
                                    getattr(me, "los_rate_rad", 0.0), self.net.p.dt)
            self.net.step()
        self.stats["brain_steps"] += n_steps

        # ---- descending-neuron readout
        r = {k: self.net.group_rate(v) for k, v in groups.items()}
        p = self.readout
        # + means "the target is to my right" -> roll right -> turn toward it
        lat_dn = r.get("turn_right", 0.0) - r.get("turn_left", 0.0)
        lat_vis = r.get("vis_right", 0.0) - r.get("vis_left", 0.0)
        turn_sig = p.roll_sign * (lat_dn + p.vis_gain * lat_vis)
        pitch_sig = p.pitch_sign * (r.get("pitch_up", 0.0) - r.get("pitch_down", 0.0))
        n_des = 1.0 + p.pull_bias + p.pull_gain * pitch_sig
        n_cap = min(me.ac.n_available(), me.ac.p.n_max)
        n_des = vm.clamp(n_des, me.ac.p.n_min, n_cap)
        pull = vm.clamp(2.0 * (n_des - me.ac.p.n_min) / (me.ac.p.n_max - me.ac.p.n_min) - 1.0,
                        -1.0, 1.0)
        roll = vm.clamp(p.turn_gain * turn_sig, -1.0, 1.0)
        thr = vm.clamp(p.throttle_bias + p.throttle_gain * r.get("speed", 0.0), 0.0, 1.0)
        fire = (p.trigger_gain * r.get("trigger", 0.0)) > p.trigger_threshold
        cmd = Command(roll=roll, pull=pull, throttle=thr, trigger=bool(fire))
        self._last_command = cmd
        return cmd

    # -------------------------------------------------------------- learning
    def observe(self, phi_next: Sequence[float], reward: float) -> float:
        """Deliver the outcome. Returns the dopamine signal that was applied.

        Called by the training loop *after* the environment has scored the
        action, which is the correct causal order: reward, then dopamine, then
        plasticity on the eligibility trace that was laid down during the action.
        """
        phi = self._phi
        if self.da.cfg.use_td:
            delta = self.critic.td_error(reward, phi_next)
            self.critic.update(delta, phi)
            signal = delta
        else:
            signal = reward * self.da.cfg.raw_hit_gain      # ablation arm
        self.critic.v_last = self.critic.value(phi_next)
        self._phi = list(phi_next)

        if self.da.available:
            self.da.drive(self.net, signal)
        self.stats["da_total"] += signal
        if self.plasticity and self.n_plastic:
            touched = self.net.deliver_dopamine(
                signal, lr=self.da.cfg.lr_plastic, weight_decay=self.da.cfg.weight_decay)
            self.stats["plastic_updates"] += touched
        return signal

    def reset(self):
        self.critic.reset()
        self.net.clear_inputs()
        self._phi = [0.0] * 22

    # ---------------------------------------------------------------- reporting
    def describe(self) -> dict:
        st = self.conn.stats()
        return {"name": self.name, "neurons": len(self.conn),
                "edges": self.conn.n_edges,
                "plastic_edges": len(self.net.plastic_edges),
                "sensory": self.sensors.available,
                "da_available": self.da.available,
                "mean_out_degree": round(st["mean_out_degree"], 2),
                "stats": {k: round(v, 2) for k, v in self.stats.items()}}


def build_controller(kind: str = "connectome", seed: int = 0,
                     neurons: Optional[int] = None,
                     plasticity: bool = False,
                     readout: Optional[ReadoutParams] = None,
                     use_torch: bool = False,
                     torch_device: Optional[str] = None,
                     **kw) -> ConnectomeController:
    """`kind`: 'connectome' (synthetic stand-in) or 'malecns' (real data).

    The `malecns` path needs the downloaded flat connectome; point
    FLYBFM_CONNECTOME / FLYBFM_ANNOTATIONS at the feather files (see
    tools/fetch_connectome.py) and it will load and prune them for real.
    """
    import os
    if kind == "malecns":
        con_path = os.environ.get("FLYBFM_CONNECTOME", "")
        ann_path = os.environ.get("FLYBFM_ANNOTATIONS", "")
        if not (con_path and ann_path):
            raise RuntimeError(
                "Set FLYBFM_CONNECTOME and FLYBFM_ANNOTATIONS to the MaleCNS "
                "feather files (see tools/fetch_connectome.py), or use "
                "--brain synthetic."
            )
        from .connectome import load_malecns_flat
        conn = flight_subgraph(load_malecns_flat(con_path, ann_path))
    else:
        conn = flight_subgraph(synthetic_connectome(seed=seed, neurons=neurons))
    # haltere / wind afferents are not in the generic synthetic graph; add them
    # so the proprioceptive channel has somewhere to land
    conn = _ensure_mechanosensory(conn, seed=seed)
    return ConnectomeController(conn, readout=readout, seed=seed,
                                plasticity=plasticity, name=kind,
                                use_torch=use_torch, torch_device=torch_device, **kw)


def _ensure_mechanosensory(conn: Connectome, seed: int = 0) -> Connectome:
    """Append a small haltere/wind population wired onto the descending neurons."""
    import random
    by_type = conn.index_by_type()
    if by_type.get("haltere"):
        return conn
    rng = random.Random(seed)
    dns = [i for t in ("DNp26", "DNp57", "DNp03", "DNg02", "DNp06", "DNa02",
                       "DNg13", "DNp10", "DNHS1") for i in by_type.get(t, [])]
    for t, n in (("haltere", 8), ("wind", 4)):
        for k in range(n):
            idx = len(conn.ids)
            conn.ids.append(10_000_000 + idx)
            conn.types.append(t)
            conn.sides.append("L" if k % 2 == 0 else "R")
            conn.transmitter.append("ACH")
            for dn in dns:
                if rng.random() < 0.5:
                    conn.pre.append(idx)
                    conn.post.append(dn)
                    conn.weight.append(round(rng.uniform(0.2, 0.9), 3))
    return conn
