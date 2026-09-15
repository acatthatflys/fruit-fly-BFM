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
        self.rate_ma = [0.0] * self.n
        self.rate_alpha = self.p.dt / 0.05

        self.plasticity = plasticity
        self.elig: List[float] = [0.0] * len(conn.pre)
        self.plastic_edges: List[int] = []
        self.t = 0.0
        self._decay_syn = math.exp(-self.p.dt / self.p.tau_syn)

    def mark_plastic(self, pre_types: Sequence[str], post_types: Sequence[str]):
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

    def reset_state(self, seed: Optional[int] = None):
        """Genuine reset back to rest — not just external drive.

        Reinitializes membrane potentials, synaptic currents, refractory timers,
        firing-rate moving averages, spikes, and eligibility traces. Called at
        the top of each episode so episodes are independent samples (fixes
        episode leakage bug where episode 2..N inherited v/i_syn/refractory from
        previous fight). If seed given, re-seed rng for reproducibility.
        """
        if seed is not None:
            self.rng = random.Random(seed)
        # v_rest + uniform(-2,2) same as __init__
        self.v = [self.p.v_rest + self.rng.uniform(-2, 2) for _ in range(self.n)]
        self.i_syn = [0.0] * self.n
        self.i_ext = [0.0] * self.n
        self.refractory = [0.0] * self.n
        self.spikes = []
        self.rate_ma = [0.0] * self.n
        self.t = 0.0
        if self.plasticity:
            self.elig = [0.0] * len(self.c.pre)
            # keep plastic_edges list, just zero elig for them
            for k in self.plastic_edges:
                self.elig[k] = 0.0

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

    def group_rate(self, indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        return sum(self.rate_ma[i] for i in indices) / len(indices)

    def group_spikes(self, indices: Sequence[int]) -> int:
        s = set(indices)
        return sum(1 for i in self.spikes if i in s)

    def population_rate(self) -> float:
        return sum(self.rate_ma) / max(self.n, 1)

    def deliver_dopamine(self, da: float, lr: float = 0.02, weight_decay: float = 1e-4):
        """Three-factor Hebbian update on KC→MBON edges: Δw = η·e·DA − λw.

        e = eligibility trace laid down when pre spikes and post is depolarized
            (Hebbian coincidence, decays 0.995 per step). DA = dopamine signal
            from critic TD-error or raw reward. Only touches plastic_edges
            (KC→MBON, 66 in synthetic, thousands in MaleCNS). Tested in
            test_plasticity_path_exists and test_connectome_loader. This is the
            inner loop (fly's own dopamine), switched off by default
            (plasticity=False) — outer CEM is 100% of reported results unless
            explicitly enabled.
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

    def to_torch(self, device: Optional[str] = None, seed: Optional[int] = None):
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "TorchLIFNetwork needs torch: pip install torch or pip install -e .[torch]"
            ) from exc
        return TorchLIFNetwork(self.c, self.p, seed=seed if seed is not None else 0,
                               device=device, plasticity=self.plasticity,
                               _init_from=self)


class TorchLIFNetwork:
    """GPU-capable LIF network using torch.sparse matmul.

    Correctness: dv uses i_syn before decay, matching stdlib.
    Memory: previously built two Python dicts over all 125M edges (~132 bytes/edge
    → ~16GB each at MaleCNS scale, ~32GB total). Now builds mapping only for
    plastic edges (KC->MBON) and only when plasticity enabled, filtered to MBON
    incoming edges. Full CNS without plasticity uses 0 bytes for mapping, so 32GB
    machines don't crash. See docs/SETUP_LEVELS.md Level 2.
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
        self._spike_idx: Optional[torch.Tensor] = None

        self._plastic_pre_tensor: Optional[torch.Tensor] = None
        self._plastic_post_tensor: Optional[torch.Tensor] = None
        self._plastic_edges_tensor: Optional[torch.Tensor] = None
        self._edge_to_coalesced: Dict[int, int] = {}

        self._build_sparse_matrix()
        self.out: List[List[Tuple[int, float]]] = conn.outgoing()

        if self.plastic_edges:
            self._cache_plasticity_tensors()

    def _build_sparse_matrix(self):
        """Build W[post, pre] = weight * w_scale, coalesced.

        Does NOT build Python dicts over all edges — that costs ~132 bytes/edge
        (~16GB per dict at 125M synapses, ~32GB for both). Mapping is built lazily
        only for plastic edges via _build_plastic_mapping(), and only when
        plasticity=True. Full CNS without plasticity uses no mapping.
        """
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
        W = torch.sparse_coo_tensor(indices, values, (self.n, self.n), device=self.device)
        self.W = W.coalesce()
        self._edge_to_coalesced = {}

    def _build_plastic_mapping(self):
        """Build edge_k -> coalesced index mapping ONLY for plastic edges.

        Old code built two dicts over every edge (125M → ~16GB each). Now:
        - Only when plasticity=True and plastic_edges non-empty
        - Filters coalesced to posts in plastic set (MBONs), reducing 125M → ~1M
        - Single dict, size ~edges to MBONs or less, not 125M
        - Uses tensor isin + CPU dict for filtered only

        On 32GB machines, Level 2 without plasticity uses 0 bytes.
        """
        if not self.plasticity or not self.plastic_edges:
            self._edge_to_coalesced = {}
            return

        torch = self._torch
        try:
            co_idx = self.W.indices()
            plastic_posts = [self.c.post[k] for k in self.plastic_edges]
            plastic_post_set = set(plastic_posts)

            co_post = co_idx[0]
            try:
                post_set_tensor = torch.tensor(list(plastic_post_set), device=co_post.device)
                mask = torch.isin(co_post, post_set_tensor)
            except Exception:
                try:
                    import numpy as np
                    co_post_cpu = co_post.cpu().numpy()
                    mask_np = np.isin(co_post_cpu, list(plastic_post_set))
                    mask = torch.from_numpy(mask_np).to(co_post.device)
                except Exception:
                    co_post_cpu = co_post.cpu()
                    mask_cpu = torch.zeros(co_post_cpu.shape[0], dtype=torch.bool)
                    s = plastic_post_set
                    chunk = 1_000_000
                    for start in range(0, co_post_cpu.shape[0], chunk):
                        end = min(start + chunk, co_post_cpu.shape[0])
                        for i in range(start, end):
                            if int(co_post_cpu[i].item()) in s:
                                mask_cpu[i] = True
                    mask = mask_cpu.to(co_post.device)

            filtered_co_pos = torch.where(mask)[0]
            filtered_post = co_idx[0][filtered_co_pos]
            filtered_pre = co_idx[1][filtered_co_pos]

            f_post_cpu = filtered_post.cpu().tolist()
            f_pre_cpu = filtered_pre.cpu().tolist()
            f_co_pos_cpu = filtered_co_pos.cpu().tolist()

            mapping: Dict[Tuple[int, int], int] = {}
            for po, pr, co_pos in zip(f_post_cpu, f_pre_cpu, f_co_pos_cpu):
                mapping[(po, pr)] = co_pos

            edge_map: Dict[int, int] = {}
            for global_k in self.plastic_edges:
                po = self.c.post[global_k]
                pr = self.c.pre[global_k]
                key = (po, pr)
                if key in mapping:
                    edge_map[global_k] = mapping[key]

            self._edge_to_coalesced = edge_map
        except Exception as e:
            warnings.warn(f"plastic mapping build failed ({e}), falling back to full rebuild on DA")
            self._edge_to_coalesced = {}

    def _cache_plasticity_tensors(self):
        torch = self._torch
        if not self.plastic_edges:
            self._plastic_pre_tensor = None
            self._plastic_post_tensor = None
            self._plastic_edges_tensor = None
            self._edge_to_coalesced = {}
            return
        pre_list = [self.c.pre[k] for k in self.plastic_edges]
        post_list = [self.c.post[k] for k in self.plastic_edges]
        self._plastic_pre_tensor = torch.tensor(pre_list, dtype=torch.long, device=self.device)
        self._plastic_post_tensor = torch.tensor(post_list, dtype=torch.long, device=self.device)
        self._plastic_edges_tensor = torch.tensor(self.plastic_edges, dtype=torch.long, device=self.device)
        if self.plasticity:
            self._build_plastic_mapping()

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

    def reset_state(self, seed: Optional[int] = None):
        """Genuine reset — mirrors LIFNetwork.reset_state for torch path."""
        torch = self._torch
        if seed is not None:
            self.rng = random.Random(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        v_init = [self.p.v_rest + self.rng.uniform(-2, 2) for _ in range(self.n)]
        self.v = torch.tensor(v_init, dtype=torch.float32, device=self.device)
        self.i_syn.zero_()
        self.i_ext.zero_()
        self.refractory.zero_()
        self.rate_ma.zero_()
        self.spikes = []
        self._spike_tensor.zero_()
        self._spike_idx = None
        self.t = 0.0
        if self.plasticity:
            self.elig.zero_()

    def step(self) -> List[int]:
        torch = self._torch
        p = self.p
        dt = p.dt
        decay = self._decay_syn
        vth, vres, vr = p.v_thresh, p.v_reset, p.v_rest
        noise_std = p.noise_mv
        alpha = self.rate_alpha

        is_ref = self.refractory > 0.0
        i_syn_old = self.i_syn.clone()

        self.i_syn.mul_(decay)

        if torch.any(is_ref):
            self.refractory[is_ref] -= dt
            self.refractory.clamp_(min=0.0)

        self.v = torch.where(is_ref, torch.full_like(self.v, vres), self.v)

        not_ref = ~is_ref
        if torch.any(not_ref):
            dv = dt / p.tau_m * (-(self.v - vr) + p.r_in * (i_syn_old + self.i_ext + p.tonic))
            if noise_std > 0:
                dv = dv + torch.randn(self.n, device=self.device, dtype=self.v.dtype) * noise_std
            self.v = torch.where(not_ref, self.v + dv, self.v)

        spike_mask = (self.v >= vth) & not_ref
        spike_idx = torch.where(spike_mask)[0]
        self._spike_idx = spike_idx

        if spike_idx.numel() > 0:
            self.v[spike_idx] = vres
            self.refractory[spike_idx] = p.tau_ref

        if self.plasticity and self.plastic_edges:
            if self._plastic_pre_tensor is None:
                self._cache_plasticity_tensors()
            pre_spiked = spike_mask[self._plastic_pre_tensor]
            post_depolarized = self.v[self._plastic_post_tensor] > vr
            inc_mask = pre_spiked & post_depolarized
            if torch.any(inc_mask):
                inc_edges = torch.masked_select(self._plastic_edges_tensor, inc_mask)
                self.elig[inc_edges] += 1.0
            self.elig[self._plastic_edges_tensor] *= 0.995

        if spike_idx.numel() > 0:
            self._spike_tensor.zero_()
            self._spike_tensor[spike_idx] = 1.0
            contrib = torch.sparse.mm(self.W, self._spike_tensor.unsqueeze(1)).squeeze(1)
            self.i_syn += contrib

        if spike_idx.numel() > 0:
            self.rate_ma[spike_idx] = self.rate_ma[spike_idx] + alpha * (1.0 - self.rate_ma[spike_idx])
        mask = self.rate_ma > 1e-6
        self.rate_ma[mask] = self.rate_ma[mask] * (1.0 - alpha)

        self.spikes = spike_idx.tolist()
        self.t += dt
        return self.spikes

    def run(self, n_steps: int) -> int:
        total = 0
        for _ in range(n_steps):
            total += len(self.step())
        return total

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
        s = set(indices)
        return sum(1 for i in self.spikes if i in s)

    def population_rate(self) -> float:
        if self.n == 0:
            return 0.0
        return float(self.rate_ma.mean().item())

    def deliver_dopamine(self, da: float, lr: float = 0.02, weight_decay: float = 1e-4):
        """Three-factor update — torch path, same equation as Python path.

        See LIFNetwork.deliver_dopamine for full explanation: Δw = η·e·DA − λw
        on KC→MBON only, eligibility gated. Batched via tensor indexing for
        GPU; coalesced sparse values updated in-place when possible to avoid
        rebuilding 125M-edge matrix.
        """
        if not self.plastic_edges:
            return 0

        if self.device.type == "cuda":
            elig_plastic = self.elig[self._plastic_edges_tensor].cpu()
        else:
            elig_plastic = self.elig[self._plastic_edges_tensor]

        elig_list = elig_plastic.tolist()

        touched = 0
        coalesced_updates: Dict[int, float] = {}

        for global_k, e in zip(self.plastic_edges, elig_list):
            if abs(e) < 1e-6 and da == 0.0:
                continue
            w = self.c.weight[global_k]
            w_new = w + lr * e * da - weight_decay * w
            if w_new > 3.0:
                w_new = 3.0
            elif w_new < -3.0:
                w_new = -3.0
            if w_new != w:
                self.c.weight[global_k] = w_new
                touched += 1
                if global_k in self._edge_to_coalesced:
                    ci = self._edge_to_coalesced[global_k]
                    coalesced_updates[ci] = w_new * self.p.w_scale
            if e > 0.5:
                self.elig[global_k] *= 0.9

        if touched > 0:
            if coalesced_updates and self._edge_to_coalesced:
                try:
                    vals = self.W._values()
                    for ci, new_val in coalesced_updates.items():
                        vals[ci] = new_val
                except Exception:
                    self._build_sparse_matrix()
                    if self.plasticity:
                        self._build_plastic_mapping()
            else:
                self._build_sparse_matrix()
                if self.plasticity:
                    self._build_plastic_mapping()
            self.out = self.c.outgoing()

        return touched

    def _update_adjacency(self, edge_k: int, w_new: float):
        self.c.weight[edge_k] = w_new
        if edge_k in self._edge_to_coalesced:
            try:
                ci = self._edge_to_coalesced[edge_k]
                self.W._values()[ci] = w_new * self.p.w_scale
                self.out = self.c.outgoing()
                return
            except Exception:
                pass
        self._build_sparse_matrix()
        if self.plasticity and self.plastic_edges:
            self._build_plastic_mapping()
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
            if self.plasticity and self.plastic_edges:
                self._build_plastic_mapping()
        return self
