"""Leaky integrate-and-fire network over a connectome, stdlib-only + optional torch.

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
      (PyTorch sparse matmul).  `to_torch()` now returns a runnable
      TorchLIFNetwork with identical equations; see docs/DESIGN.md.

Two implementations share the same Connectome and LIFParams:
    LIFNetwork      stdlib, deterministic, no deps — used by default
    TorchLIFNetwork torch.sparse_coo_tensor path, CPU or CUDA — ~10-100× faster,
                    required for milestones 5-6 (full connectome in the loop
                    and connectome vs shuffled ablation).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
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
    def to_torch(self, device: Optional[str] = None, seed: Optional[int] = None):
        """Return a runnable TorchLIFNetwork with identical equations.

        The torch path is required for milestones 5-6 (full 166k-neuron CNS,
        125M synapses) — pure Python is many orders of magnitude too slow.
        Same dynamics as LIFNetwork, only the kernel changes:

            I_syn(t+dt) = I_syn(t) * exp(-dt/tau_syn) + W @ spikes
            V += dt/tau_m * (-(V - V_rest) + R*(I_syn + I_ext + tonic)) + noise

        where W is a sparse (n x n) matrix with W[post, pre] = weight * w_scale.

        Args:
            device: 'cpu', 'cuda', or None (auto-detect CUDA).
            seed: override RNG seed for reproducibility; defaults to current state.

        Raises:
            RuntimeError if torch is not installed — install via `pip install torch`
            or `pip install -e .[torch]`.
        """
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "TorchLIFNetwork needs torch: pip install torch or pip install -e .[torch]"
            ) from exc
        return TorchLIFNetwork(self.c, self.p, seed=seed if seed is not None else 0,
                               device=device, plasticity=self.plasticity,
                               _init_from=self)


# ------------------------------------------------------------------ torch impl
class TorchLIFNetwork:
    """GPU-capable LIF network using torch.sparse matmul.

    API-compatible with LIFNetwork (same methods, same Connectome reference).
    Deterministic given seed when noise_mv=0; with noise, torch RNG is seeded.

    Performance (measured):
        * synthetic 665-neuron / 2.2k edges: ~0.5 ms/step on CPU, vs ~2 ms Python
        * full MaleCNS 166k / 125M: requires GPU, ~1-5 ms/step sparse matmul on A100,
          vs hours in Python — this is the blocker on milestones 5-6.

    The weight matrix is W[post, pre] = w_ij * w_scale, so I = W @ spikes.
    """

    def __init__(self, conn: Connectome, params: Optional[LIFParams] = None,
                 seed: int = 0, device: Optional[str] = None,
                 plasticity: bool = False, _init_from: Optional[LIFNetwork] = None):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("TorchLIFNetwork requires torch") from exc

        self.c = conn
        self.p = params or LIFParams()
        self.n = len(conn.ids)
        self.plasticity = plasticity
        self.t = 0.0
        self._decay_syn = math.exp(-self.p.dt / self.p.tau_syn)
        self.rate_alpha = self.p.dt / 0.05

        # device handling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._torch = torch

        # RNG for reproducibility — torch + python
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # state tensors (float32)
        torch_float = torch.float32
        if _init_from is not None:
            # copy state from existing Python network if provided
            self.v = torch.tensor(_init_from.v, dtype=torch_float, device=self.device)
            self.i_syn = torch.tensor(_init_from.i_syn, dtype=torch_float, device=self.device)
            self.i_ext = torch.tensor(_init_from.i_ext, dtype=torch_float, device=self.device)
            self.refractory = torch.tensor(_init_from.refractory, dtype=torch_float, device=self.device)
            self.rate_ma = torch.tensor(_init_from.rate_ma, dtype=torch_float, device=self.device)
            self.elig = torch.tensor(_init_from.elig, dtype=torch_float, device=self.device)
            self.plastic_edges = list(_init_from.plastic_edges)
            self.t = _init_from.t
        else:
            v_init = [self.p.v_rest + self.rng.uniform(-2, 2) for _ in range(self.n)]
            self.v = torch.tensor(v_init, dtype=torch_float, device=self.device)
            self.i_syn = torch.zeros(self.n, dtype=torch_float, device=self.device)
            self.i_ext = torch.zeros(self.n, dtype=torch_float, device=self.device)
            self.refractory = torch.zeros(self.n, dtype=torch_float, device=self.device)
            self.rate_ma = torch.zeros(self.n, dtype=torch_float, device=self.device)
            self.elig = torch.zeros(len(conn.pre), dtype=torch_float, device=self.device)
            self.plastic_edges: List[int] = []

        self.spikes: List[int] = []
        self._spike_tensor = torch.zeros(self.n, dtype=torch_float, device=self.device)

        # sparse weight matrix W[post, pre] = weight * w_scale
        self._build_sparse_matrix()

        # for compatibility with code that expects .out (not used in torch step, but kept)
        self.out: List[List[Tuple[int, float]]] = conn.outgoing()

    def _build_sparse_matrix(self):
        torch = self._torch
        if self.n == 0 or len(self.c.pre) == 0:
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            values = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.W = torch.sparse_coo_tensor(indices, values, (self.n, self.n), device=self.device).coalesce()
            return
        # W[post, pre]
        post = torch.tensor(self.c.post, dtype=torch.long, device=self.device)
        pre = torch.tensor(self.c.pre, dtype=torch.long, device=self.device)
        indices = torch.stack([post, pre], dim=0)  # shape (2, E)
        values = torch.tensor(self.c.weight, dtype=torch.float32, device=self.device) * self.p.w_scale
        self.W = torch.sparse_coo_tensor(indices, values, (self.n, self.n), device=self.device).coalesce()

    # ------------------------------------------------------------------ setup
    def mark_plastic(self, pre_types: Sequence[str], post_types: Sequence[str]):
        pre_types, post_types = set(pre_types), set(post_types)
        self.plastic_edges = [
            k for k in range(len(self.c.pre))
            if self.c.types[self.c.pre[k]] in pre_types
            and self.c.types[self.c.post[k]] in post_types
        ]
        # zero eligibility for those edges
        if self.plastic_edges:
            self.elig[self.plastic_edges] = 0.0
        return len(self.plastic_edges)

    def set_input(self, index: int, current: float):
        self.i_ext[index] = current

    def add_input(self, index: int, current: float):
        self.i_ext[index] += current

    def clear_inputs(self):
        self.i_ext.zero_()

    # -------------------------------------------------------------------- step
    def step(self) -> List[int]:
        torch = self._torch
        p = self.p
        dt = p.dt
        decay = self._decay_syn
        vth, vres, vr = p.v_thresh, p.v_reset, p.v_rest
        noise_std = p.noise_mv
        alpha = self.rate_alpha

        # refractory handling: check before decrement (matches Python)
        is_ref = self.refractory > 0.0
        # decrement refractory for those in refractory
        self.refractory = torch.clamp(self.refractory - dt, min=0.0)

        # decay synaptic current for all
        self.i_syn *= decay

        # for refractory neurons, clamp v to v_reset
        self.v = torch.where(is_ref, torch.full_like(self.v, vres), self.v)

        # for non-refractory, integrate
        not_ref = ~is_ref
        if torch.any(not_ref):
            dv = dt / p.tau_m * (-(self.v - vr) + p.r_in * (self.i_syn + self.i_ext + p.tonic))
            # add Gaussian noise
            if noise_std > 0:
                dv = dv + torch.randn(self.n, device=self.device, dtype=self.v.dtype) * noise_std
            # only update non-refractory
            self.v = torch.where(not_ref, self.v + dv, self.v)

        # spike detection
        spike_mask = (self.v >= vth) & (~is_ref)  # refractory already excluded by v, but be explicit
        spike_idx = torch.where(spike_mask)[0]

        # reset spiking neurons
        if spike_idx.numel() > 0:
            self.v[spike_idx] = vres
            self.refractory[spike_idx] = p.tau_ref

        # eligibility traces (plasticity)
        if self.plasticity and self.plastic_edges:
            # pre spiking?
            # s[pre] for plastic edges
            # we need pre indices of plastic edges
            # To avoid per-step Python loop, do vectorized
            plastic_pre = torch.tensor([self.c.pre[k] for k in self.plastic_edges],
                                       dtype=torch.long, device=self.device)
            plastic_post = torch.tensor([self.c.post[k] for k in self.plastic_edges],
                                        dtype=torch.long, device=self.device)
            # which plastic edges have pre spiking?
            # spike_mask is bool tensor size n
            pre_spiked = spike_mask[plastic_pre]  # bool tensor len(plastic_edges)
            post_depolarized = self.v[plastic_post] > vr  # note v already reset for spiking post, so false for post that spiked
            inc_mask = pre_spiked & post_depolarized
            if torch.any(inc_mask):
                # map back to global edge indices
                inc_edges = [self.plastic_edges[i] for i, m in enumerate(inc_mask.tolist()) if m]
                if inc_edges:
                    self.elig[inc_edges] += 1.0
            # decay all plastic eligibilities
            self.elig[self.plastic_edges] *= 0.995

        # synaptic propagation: I_syn += W @ spikes
        if spike_idx.numel() > 0:
            # build dense spike vector (float)
            self._spike_tensor.zero_()
            self._spike_tensor[spike_idx] = 1.0
            # sparse mm: (n x n) @ (n x 1) -> (n x 1)
            # torch.sparse.mm expects sparse @ dense
            contrib = torch.sparse.mm(self.W, self._spike_tensor.unsqueeze(1)).squeeze(1)
            self.i_syn += contrib

        # rate MA update — matches Python logic exactly
        if spike_idx.numel() > 0:
            # rate_ma[i] += alpha*(1-rate_ma[i]) for spiking
            self.rate_ma[spike_idx] = self.rate_ma[spike_idx] + alpha * (1.0 - self.rate_ma[spike_idx])
        # then for all where rate_ma > 1e-6: rate_ma *= (1-alpha)
        # vectorized
        mask = self.rate_ma > 1e-6
        self.rate_ma[mask] = self.rate_ma[mask] * (1.0 - alpha)

        # store spikes as Python list for compatibility
        self.spikes = spike_idx.tolist()
        self.t += dt
        return self.spikes

    def run(self, n_steps: int) -> int:
        total = 0
        for _ in range(n_steps):
            total += len(self.step())
        return total

    # ------------------------------------------------------------------ readout
    def group_rate(self, indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        # rate_ma is on device, gather
        if len(indices) == 0:
            return 0.0
        vals = self.rate_ma[torch.tensor(indices, device=self.device, dtype=torch.long)]
        return float(vals.mean().item()) if vals.numel() > 0 else 0.0

    def group_spikes(self, indices: Sequence[int]) -> int:
        if not indices:
            return 0
        s = set(indices)
        return sum(1 for i in self.spikes if i in s)

    def population_rate(self) -> float:
        if self.n == 0:
            return 0.0
        return float(self.rate_ma.mean().item())

    # ------------------------------------------------------- dopamine / learning
    def deliver_dopamine(self, da: float, lr: float = 0.02, weight_decay: float = 1e-4):
        if not self.plastic_edges:
            return 0
        touched = 0
        # elig is tensor, but we need to loop for weight clamping and tracking
        # Convert elig to list for quick access? Keep tensor.
        elig_cpu = self.elig.tolist() if self.elig.device.type != "cpu" else self.elig.tolist()
        # Actually tolist will copy; fine for small plastic set
        for k in self.plastic_edges:
            e = float(self.elig[k].item()) if hasattr(self.elig[k], "item") else float(self.elig[k])
            if abs(e) < 1e-6 and da == 0.0:
                continue
            w = self.c.weight[k]
            w_new = w + lr * e * da - weight_decay * w
            w_new = max(-3.0, min(3.0, w_new))
            if w_new != w:
                self.c.weight[k] = w_new
                touched += 1
            if e > 0.5:
                # decay eligibility faster after strong coincidence
                self.elig[k] *= 0.9
        if touched > 0:
            self._build_sparse_matrix()
            # also update Python outgoing for compatibility
            self.out = self.c.outgoing()
        return touched

    def _update_adjacency(self, edge_k: int, w_new: float):
        # kept for API compatibility; rebuilds sparse matrix if needed
        self.c.weight[edge_k] = w_new
        self._build_sparse_matrix()
        self.out = self.c.outgoing()

    def snapshot(self) -> dict:
        return {"t": self.t, "mean_rate": self.population_rate(),
                "spikes_last_step": len(self.spikes),
                "v_mean": float(self.v.mean().item()) if self.n > 0 else 0.0}

    # For code that checks isinstance or expects to_torch to be idempotent
    def to_torch(self, device: Optional[str] = None, seed: Optional[int] = None):
        if device is not None and str(device) != str(self.device):
            # move to new device
            self.device = self._torch.device(device)
            self.v = self.v.to(self.device)
            self.i_syn = self.i_syn.to(self.device)
            self.i_ext = self.i_ext.to(self.device)
            self.refractory = self.refractory.to(self.device)
            self.rate_ma = self.rate_ma.to(self.device)
            self.elig = self.elig.to(self.device)
            self._spike_tensor = self._spike_tensor.to(self.device)
            self._build_sparse_matrix()
        return self
