"""Leaky integrate-and-fire network over a connectome, stdlib-only.

    tau_m dV/dt = -(V - V_rest) + R I_syn + R I_ext
    I_syn(t+dt) = I_syn(t) * exp(-dt/tau_syn) + sum_j w_ij * s_j(t)

Current-based synapses rather than conductance-based, because conductances are
not in the connectome.  Spikes are the only thing crossing an edge, and the
weight is a per-edge scalar: this is exactly the parameterisation used by
Shiu et al. 2024 (whole-brain LIF, FlyWire) and by the Eon Systems demo.

Performance reality check, so nobody is surprised:
    * pure Python, CPython: ~1-3 M synaptic events/s.
    * a 1,500-neuron flight-circuit subgraph at 1 ms steps, 200 Hz of sim time
      is roughly 5-15% of one core  -> fine for training loops.
    * the full 166k-neuron CNS at 1 ms needs a GPU implementation
      (PyTorch/CuPy sparse matmul).  `to_torch()` documents that path; the
      scalings are the same, only the kernel changes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .connectome import Connectome, _apply_transmitter_signs


@dataclass
class LIFParams:
    dt: float = 0.001          # s
    tau_m: float = 0.020       # s
    tau_syn: float = 0.005     # s
    tau_ref: float = 0.002     # s
    v_rest: float = -65.0      # mV
    v_reset: float = -70.0
    v_thresh: float = -50.0
    r_in: float = 10.0         # mV per unit current
    noise_mv: float = 0.6      # per-step membrane noise
    tonic: float = 0.10        # weak drive to *every* neuron (spontaneous activity)
    w_scale: float = 3.0       # global gain on connectome weights (unidentifiable!)
                               # calibrated so the sensory->DN pathway carries signal;
                               # see docs/DESIGN.md 'what is not identifiable'


class LIFNetwork:
    """Sparse LIF network. Deterministic given (connectome, params, seed)."""

    def __init__(self, conn: Connectome, params: Optional[LIFParams] = None,
                 seed: int = 0, plasticity: bool = False):
        self.c = conn
        self.p = params or LIFParams()
        self.n = len(conn.ids)
        self.rng = random.Random(seed)
        self.out: List[List[Tuple[int, float]]] = conn.outgoing()

        self.v = [self.p.v_rest + self.rng.uniform(-2, 2) for _ in range(self.n)]
        self.i_syn = [0.0] * self.n
        self.i_ext = [0.0] * self.n
        self.refractory = [0.0] * self.n
        self.spikes: List[int] = []
        self.rate_ma = [0.0] * self.n          # moving-average firing rate [Hz]
        self.rate_alpha = self.p.dt / 0.05     # 50 ms window

        self.plasticity = plasticity
        # eligibility traces for the plastic edges, keyed by edge position
        self.elig: List[float] = [0.0] * len(conn.pre)
        self.plastic_edges: List[int] = []
        self.t = 0.0
        self._decay_syn = math.exp(-self.p.dt / self.p.tau_syn)

    # ------------------------------------------------------------------ setup
    def mark_plastic(self, pre_types: Sequence[str], post_types: Sequence[str]):
        """Flag edges whose identity matters for learning (KC -> MBON)."""
        pre_types, post_types = set(pre_types), set(post_types)
        self.plastic_edges = [
            k for k in range(len(self.c.pre))
            if self.c.types[self.c.pre[k]] in pre_types
            and self.c.types[self.c.post[k]] in post_types
        ]
        for k in self.plastic_edges:
            self.elig[k] = 0.0
        return len(self.plastic_edges)

    def set_input(self, index: int, current: float):
        self.i_ext[index] = current

    def add_input(self, index: int, current: float):
        self.i_ext[index] += current

    def clear_inputs(self):
        for i in range(self.n):
            self.i_ext[i] = 0.0

    # -------------------------------------------------------------------- step
    def step(self) -> List[int]:
        p = self.p
        dt = p.dt
        decay = self._decay_syn
        vth, vres, vr = p.v_thresh, p.v_reset, p.v_rest
        noise = p.noise_mv
        alpha = self.rate_alpha
        spikes: List[int] = []

        v = self.v
        i_syn = self.i_syn
        i_ext = self.i_ext
        refr = self.refractory
        rng = self.rng
        for i in range(self.n):
            if refr[i] > 0.0:
                refr[i] -= dt
                i_syn[i] *= decay
                v[i] = vres
                continue
            dv = dt / p.tau_m * (-(v[i] - vr) + p.r_in * (i_syn[i] + i_ext[i] + p.tonic))
            v[i] += dv + rng.gauss(0.0, noise)
            i_syn[i] *= decay
            if v[i] >= vth:
                spikes.append(i)
                v[i] = vres
                refr[i] = p.tau_ref

        # synaptic propagation + eligibility traces (pre x post coincidence)
        if self.plasticity and self.plastic_edges:
            ev = {}
            for k in self.plastic_edges:
                ev[k] = 0.0
            pre_counts = {}
            for i in spikes:
                pre_counts[i] = pre_counts.get(i, 0) + 1
            for k in self.plastic_edges:
                if pre_counts.get(self.c.pre[k], 0) and v[self.c.post[k]] > vr:
                    self.elig[k] += 1.0
            for k in self.plastic_edges:
                self.elig[k] *= 0.995

        for i in spikes:
            for (j, w) in self.out[i]:
                i_syn[j] += w * p.w_scale

        for i in spikes:
            self.rate_ma[i] += alpha * (1.0 - self.rate_ma[i])
        for i in range(self.n):
            if self.rate_ma[i] > 1e-6:
                self.rate_ma[i] *= (1.0 - alpha)

        self.spikes = spikes
        self.t += dt
        return spikes

    def run(self, n_steps: int) -> int:
        total = 0
        for _ in range(n_steps):
            total += len(self.step())
        return total

    # ------------------------------------------------------------------ readout
    def group_rate(self, indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        return sum(self.rate_ma[i] for i in indices) / len(indices)

    def group_spikes(self, indices: Sequence[int]) -> int:
        s = set(indices)
        return sum(1 for i in self.spikes if i in s)

    def population_rate(self) -> float:
        return sum(self.rate_ma) / max(self.n, 1)

    # ------------------------------------------------------- dopamine / learning
    def deliver_dopamine(self, da: float, lr: float = 0.02, weight_decay: float = 1e-4):
        """Three-factor plasticity: eligibility x global neuromodulator.

            dw_ij = lr * e_ij * DA(t)  - weight_decay * w_ij

        This is the fly-mushroom-body abstraction of reinforcement learning
        (dopamine as the third factor gating an eligibility trace), and it is
        the only place in this project where the connectome itself changes.
        """
        if not self.plastic_edges:
            return 0
        touched = 0
        for k in self.plastic_edges:
            e = self.elig[k]
            if abs(e) < 1e-6 and da == 0.0:
                continue
            w = self.c.weight[k]
            w_new = w + lr * e * da - weight_decay * w
            w_new = max(-3.0, min(3.0, w_new))
            if w_new != w:
                self.c.weight[k] = w_new
                touched += 1
            # rebind inside the adjacency used for propagation
            self._update_adjacency(k, w_new)
            if self.elig[k] > 0.5:
                self.elig[k] *= 0.9
        return touched

    def _update_adjacency(self, edge_k: int, w_new: float):
        a = self.c.pre[edge_k]
        b = self.c.post[edge_k]
        for idx, (j, _) in enumerate(self.out[a]):
            if j == b:
                self.out[a][idx] = (j, w_new)
                break

    def snapshot(self) -> dict:
        return {"t": self.t, "mean_rate": self.population_rate(),
                "spikes_last_step": len(self.spikes),
                "v_mean": sum(self.v) / max(self.n, 1)}

    # --------------------------------------------------------------- torch path
    def to_torch(self):
        """Reference for the GPU path (not used here; see docs/DESIGN.md).

        The same equations as one sparse matmul per timestep:

            import torch
            from flybfm.brain.connectome import Connectome
            idx = torch.tensor([c.pre, c.post])
            W = torch.sparse_coo_tensor(idx, torch.tensor(c.weight))
            # per step:  I = I*decay + W @ spikes ;  V += dt/tau*(-(V-vr) + R*I)
        """
        raise NotImplementedError(
            "Install torch and reimplement `step` with a sparse matmul over the "
            "same connectome; the equations are identical (see the docstring)."
        )
