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
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
        The synaptic current used for dv is the value *before* decay, matching
        the stdlib path (decay happens after dv, in preparation for next step).

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

    Correctness: dv is computed from i_syn *before* decay, matching stdlib
    LIFNetwork.step() (decay happens after, in preparation for next step).
    See test_torch_matches_python_trajectory.

    Performance: avoids per-edge device syncs, caches plastic index tensors,
    and updates sparse values in-place instead of rebuilding on every DA delivery.

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

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._torch = torch

        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        torch_float = torch.float32
        if _init_from is not None:
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
        self._spike_idx: Optional[torch.Tensor] = None  # cached tensor, avoids tolist sync

        # cached plasticity tensors (avoid rebuilding every step)
        self._plastic_pre_tensor: Optional[torch.Tensor] = None
        self._plastic_post_tensor: Optional[torch.Tensor] = None
        self._plastic_edges_tensor: Optional[torch.Tensor] = None
        # mapping from original edge_k -> coalesced value index for in-place updates
        self._edge_to_coalesced: Dict[int, int] = {}
        self._coalesced_to_edge: Dict[int, int] = {}

        self._build_sparse_matrix()
        self.out: List[List[Tuple[int, float]]] = conn.outgoing()

        # if we were initialized from a python net that already had plastic edges, cache tensors
        if self.plastic_edges:
            self._cache_plasticity_tensors()

    def _build_sparse_matrix(self):
        torch = self._torch
        if self.n == 0 or len(self.c.pre) == 0:
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            values = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.W = torch.sparse_coo_tensor(indices, values, (self.n, self.n), device=self.device).coalesce()
            self._edge_to_coalesced = {}
            return
        post = torch.tensor(self.c.post, dtype=torch.long, device=self.device)
        pre = torch.tensor(self.c.pre, dtype=torch.long, device=self.device)
        indices = torch.stack([post, pre], dim=0)
        values = torch.tensor(self.c.weight, dtype=torch.float32, device=self.device) * self.p.w_scale
        # coalesce to sum duplicates and sort — required for efficient mm
        W = torch.sparse_coo_tensor(indices, values, (self.n, self.n), device=self.device)
        self.W = W.coalesce()

        # Build mapping from original edge_k -> coalesced index for in-place plasticity updates.
        # For non-duplicate graphs (typical), this is a permutation. For duplicate (post,pre),
        # coalesce sums values, so we map each original edge to the coalesced entry that holds the sum.
        # We build a dict from (post,pre) -> coalesced position.
        try:
            co_idx = self.W.indices()  # (2, E_coalesced)
            # dict from (post,pre) tuple -> coalesced idx
            # Use CPU for dict building to avoid many small GPU ops
            co_post = co_idx[0].cpu().tolist()
            co_pre = co_idx[1].cpu().tolist()
            mapping = {}
            for ci, (po, pr) in enumerate(zip(co_post, co_pre)):
                mapping[(po, pr)] = ci
            edge_map = {}
            # original edges
            for k, (pr, po) in enumerate(zip(self.c.pre, self.c.post)):
                # note c.pre = pre, c.post = post, but W is [post, pre]
                key = (po, pr)
                if key in mapping:
                    edge_map[k] = mapping[key]
            self._edge_to_coalesced = edge_map
            # reverse for debugging
            self._coalesced_to_edge = {v: k for k, v in edge_map.items()}
        except Exception:
            # fallback: no mapping, will trigger full rebuild on plasticity
            self._edge_to_coalesced = {}

    def _cache_plasticity_tensors(self):
        torch = self._torch
        if not self.plastic_edges:
            self._plastic_pre_tensor = None
            self._plastic_post_tensor = None
            self._plastic_edges_tensor = None
            return
        # cache as long tensors on device
        pre_list = [self.c.pre[k] for k in self.plastic_edges]
        post_list = [self.c.post[k] for k in self.plastic_edges]
        self._plastic_pre_tensor = torch.tensor(pre_list, dtype=torch.long, device=self.device)
        self._plastic_post_tensor = torch.tensor(post_list, dtype=torch.long, device=self.device)
        self._plastic_edges_tensor = torch.tensor(self.plastic_edges, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ setup
    def mark_plastic(self, pre_types: Sequence[str], post_types: Sequence[str]):
        pre_types, post_types = set(pre_types), set(post_types)
        self.plastic_edges = [
            k for k in range(len(self.c.pre))
            if self.c.types[self.c.pre[k]] in pre_types
            and self.c.types[self.c.post[k]] in post_types
        ]
        if self.plastic_edges:
            self.elig[self.plastic_edges] = 0.0
        self._cache_plasticity_tensors()
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

        # Match stdlib: check refractory before decrement, decay i_syn for all,
        # but dv uses i_syn *before* decay.
        is_ref = self.refractory > 0.0
        i_syn_old = self.i_syn.clone()  # for dv, before decay — fixes 18% bug

        # decay synaptic current for all neurons (in prep for next step)
        self.i_syn.mul_(decay)

        # decrement refractory only where >0, clamp to 0 (matches Python's >0 check)
        if torch.any(is_ref):
            self.refractory[is_ref] -= dt
            self.refractory.clamp_(min=0.0)

        # refractory -> v = v_reset
        self.v = torch.where(is_ref, torch.full_like(self.v, vres), self.v)

        # non-refractory integration using i_syn_old (not decayed)
        not_ref = ~is_ref
        if torch.any(not_ref):
            dv = dt / p.tau_m * (-(self.v - vr) + p.r_in * (i_syn_old + self.i_ext + p.tonic))
            if noise_std > 0:
                dv = dv + torch.randn(self.n, device=self.device, dtype=self.v.dtype) * noise_std
            self.v = torch.where(not_ref, self.v + dv, self.v)

        # spike detection
        spike_mask = (self.v >= vth) & not_ref
        spike_idx = torch.where(spike_mask)[0]
        self._spike_idx = spike_idx  # cache tensor to avoid extra tolist

        if spike_idx.numel() > 0:
            self.v[spike_idx] = vres
            self.refractory[spike_idx] = p.tau_ref

        # eligibility traces — use cached tensors, no per-step list comprehension
        if self.plasticity and self.plastic_edges:
            if self._plastic_pre_tensor is None:
                self._cache_plasticity_tensors()
            # pre spiked?
            pre_spiked = spike_mask[self._plastic_pre_tensor]
            post_depolarized = self.v[self._plastic_post_tensor] > vr
            inc_mask = pre_spiked & post_depolarized
            if torch.any(inc_mask):
                # gather global edge indices where inc
                # inc_mask is bool tensor len(plastic_edges)
                # Use masked select on cached edges tensor
                inc_edges = torch.masked_select(self._plastic_edges_tensor, inc_mask)
                # inc_edges is on device; for elig update we can do vectorized
                self.elig[inc_edges] += 1.0
            # decay all plastic eligibilities
            self.elig[self._plastic_edges_tensor] *= 0.995

        # synaptic propagation: I_syn += W @ spikes (using spikes from this step, for next step's dv)
        if spike_idx.numel() > 0:
            self._spike_tensor.zero_()
            self._spike_tensor[spike_idx] = 1.0
            contrib = torch.sparse.mm(self.W, self._spike_tensor.unsqueeze(1)).squeeze(1)
            self.i_syn += contrib

        # rate MA — matches Python exactly
        if spike_idx.numel() > 0:
            self.rate_ma[spike_idx] = self.rate_ma[spike_idx] + alpha * (1.0 - self.rate_ma[spike_idx])
        mask = self.rate_ma > 1e-6
        self.rate_ma[mask] = self.rate_ma[mask] * (1.0 - alpha)

        # keep Python list for compatibility, but only materialize when needed.
        # For perf, we avoid tolist() on CUDA hot path unless caller reads self.spikes.
        # Here we still provide list for backward compat, but we use cached tensor to avoid extra sync if possible.
        # tolist() on CPU is cheap; on CUDA it's a sync — we accept it for now but note in docs.
        # To reduce syncs, we could make spikes a property, but we keep list for compat.
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
        if len(indices) == 0:
            return 0.0
        vals = self.rate_ma[torch.tensor(indices, device=self.device, dtype=torch.long)]
        return float(vals.mean().item()) if vals.numel() > 0 else 0.0

    def group_spikes(self, indices: Sequence[int]) -> int:
        if not indices:
            return 0
        # Use cached tensor if available to avoid list conversion
        if self._spike_idx is not None and self._spike_idx.numel() > 0:
            # check membership via tensor is faster than Python set for large n?
            # Fall back to list for simplicity
            s = set(indices)
            return sum(1 for i in self.spikes if i in s)
        s = set(indices)
        return sum(1 for i in self.spikes if i in s)

    def population_rate(self) -> float:
        if self.n == 0:
            return 0.0
        return float(self.rate_ma.mean().item())

    # ------------------------------------------------------- dopamine / learning
    def deliver_dopamine(self, da: float, lr: float = 0.02, weight_decay: float = 1e-4):
        """Three-factor plasticity, vectorized to avoid per-edge CUDA syncs.

        Previously looped with .item() on CUDA tensor — each .item() is a device sync.
        Now we batch eligibility to CPU once, then loop in Python without syncs,
        and update sparse values in-place via cached mapping.
        """
        if not self.plastic_edges:
            return 0

        torch = self._torch
        # Batch eligibility to CPU once (single sync) instead of per-edge .item()
        # This dominates wall-clock when plastic edges = thousands (real MB)
        if self.device.type == "cuda":
            elig_plastic = self.elig[self._plastic_edges_tensor].cpu()
        else:
            elig_plastic = self.elig[self._plastic_edges_tensor]

        # Also need Python list of elig for quick iteration without .item()
        # elig_plastic is tensor on CPU
        elig_list = elig_plastic.tolist()

        touched = 0
        # Track which coalesced indices need updating for in-place value patch
        coalesced_updates: Dict[int, float] = {}

        for local_i, (global_k, e) in enumerate(zip(self.plastic_edges, elig_list)):
            if abs(e) < 1e-6 and da == 0.0:
                continue
            w = self.c.weight[global_k]
            w_new = w + lr * e * da - weight_decay * w
            # clamp
            if w_new > 3.0:
                w_new = 3.0
            elif w_new < -3.0:
                w_new = -3.0
            if w_new != w:
                self.c.weight[global_k] = w_new
                touched += 1
                # prepare in-place sparse value update
                if global_k in self._edge_to_coalesced:
                    ci = self._edge_to_coalesced[global_k]
                    coalesced_updates[ci] = w_new * self.p.w_scale
            if e > 0.5:
                # decay eligibility faster after strong coincidence
                # update on device tensor directly
                self.elig[global_k] *= 0.9

        if touched > 0:
            if coalesced_updates and self._edge_to_coalesced:
                # in-place update of sparse values — no full rebuild
                try:
                    vals = self.W._values()
                    for ci, new_val in coalesced_updates.items():
                        vals[ci] = new_val
                except Exception:
                    # fallback to full rebuild if in-place fails
                    self._build_sparse_matrix()
            else:
                # no mapping (e.g., after duplicate handling) — rebuild
                self._build_sparse_matrix()
            self.out = self.c.outgoing()

        return touched

    def _update_adjacency(self, edge_k: int, w_new: float):
        self.c.weight[edge_k] = w_new
        # try in-place
        if edge_k in self._edge_to_coalesced:
            try:
                ci = self._edge_to_coalesced[edge_k]
                self.W._values()[ci] = w_new * self.p.w_scale
                self.out = self.c.outgoing()
                return
            except Exception:
                pass
        self._build_sparse_matrix()
        self.out = self.c.outgoing()

    def snapshot(self) -> dict:
        return {"t": self.t, "mean_rate": self.population_rate(),
                "spikes_last_step": len(self.spikes),
                "v_mean": float(self.v.mean().item()) if self.n > 0 else 0.0}

    def to_torch(self, device: Optional[str] = None, seed: Optional[int] = None):
        if device is not None and str(device) != str(self.device):
            self.device = self._torch.device(device)
            self.v = self.v.to(self.device)
            self.i_syn = self.i_syn.to(self.device)
            self.i_ext = self.i_ext.to(self.device)
            self.refractory = self.refractory.to(self.device)
            self.rate_ma = self.rate_ma.to(self.device)
            self.elig = self.elig.to(self.device)
            self._spike_tensor = self._spike_tensor.to(self.device)
            if self._plastic_pre_tensor is not None:
                self._plastic_pre_tensor = self._plastic_pre_tensor.to(self.device)
                self._plastic_post_tensor = self._plastic_post_tensor.to(self.device)
                self._plastic_edges_tensor = self._plastic_edges_tensor.to(self.device)
            self._build_sparse_matrix()
        return self
