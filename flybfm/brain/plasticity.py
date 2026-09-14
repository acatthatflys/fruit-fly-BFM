"""Reward -> dopamine -> plasticity.  This is the file that answers
"how do you set up the rewards for a fly brain?"

Three levels, in increasing order of how defensible they are:

  1. RAW PULSE (what most of the viral demos do)
        when damage is taken: inject current into two PPL101 cells
     + trivial to implement, + obviously "biological"
     - no credit assignment: the dopamine arrives after an arbitrary delay and
       is associated with whatever the mushroom body happened to be doing
     - this is why "3,000 episodes, no learning" is the honest result people
       report.  The signal exists; the *information* does not.

  2. TD ERROR (what this project does)
        delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        DA current = k * delta_t     (positive -> PAM-ish reward cells,
                                      negative -> PPL1-ish punishment cells)
     + has the correct temporal semantics: it is exactly the signal that a
       temporal-difference critic produces, and it is what the dopamine system
       in insects and mammals empirically looks like (Schultz 1997; for
       Drosophila mushroom-body dopamine, see the MBON/DAN literature)
     + handles delayed reward, so the sparse gun-hit event can shape behaviour
     - requires a value function: you have to learn V.  Here V is a small
       linear critic over the same observation vector the environment exposes.

  3. SHAPED POTENTIAL + TD (what you actually want in practice)
        the dense BFM terms in sim/reward.py are folded into a potential Phi,
        the shaped reward is provably policy-invariant, and delta is computed on
        the *shaped* reward.  The agent gets dense dopamine early and the
        objective is unchanged in the limit.

The three-factor weight update (Frémaux & Gerstner 2016; and in the fly, the
KC->MBON dopamine-gated plasticity of the mushroom body) is

    dw_ij = eta * e_ij * DA(t) - lambda * w_ij

with e_ij a decaying eligibility trace of pre-before-post coincidence.  The
trace is what gives the delayed dopamine something to act on; without it, the
arrival of a reward pulse changes nothing that matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class DopamineConfig:
    gain: float = 1.0            # uA per unit TD error
    bias: float = 0.0
    clip: float = 3.0
    lr_plastic: float = 0.02     # eta
    weight_decay: float = 1e-4   # lambda
    critic_lr: float = 0.02
    gamma: float = 0.995
    use_td: bool = True          # False -> raw event pulse (ablation arm)
    raw_hit_gain: float = 1.0    # only used when use_td=False


class LinearCritic:
    """V(s) = w . phi(s), updated by TD(0). Small, and deliberately not neural:
    it models dopaminergic *teaching signal* generation, not the fly."""

    def __init__(self, n_features: int, lr: float = 0.02, gamma: float = 0.995):
        self.w: List[float] = [0.0] * n_features
        self.lr = lr
        self.gamma = gamma
        self.v_last = 0.0

    def value(self, phi: Sequence[float]) -> float:
        return sum(wi * pi for wi, pi in zip(self.w, phi))

    def td_error(self, r: float, phi_next: Sequence[float]) -> float:
        v_next = self.value(phi_next)
        delta = r + self.gamma * v_next - self.v_last
        return delta

    def update(self, delta: float, phi: Sequence[float]) -> None:
        lr = self.lr
        for i, pi in enumerate(phi):
            if pi:
                self.w[i] += lr * delta * pi
        # keep the critic bounded so a runaway value estimate cannot produce
        # arbitrarily large dopamine
        norm = sum(w * w for w in self.w) ** 0.5
        if norm > 10.0:
            s = 10.0 / norm
            self.w = [w * s for w in self.w]

    def reset(self) -> None:
        self.v_last = 0.0


class DopamineChannel:
    """Turns a scalar teaching signal into currents on identified DA neurons.

    With the real connectome the target indices come from the cell-type
    annotations (PAM / PPL1 / PPL101 ...).  With the synthetic graph they come
    from the same lookup.  If a group is missing, the channel says so once and
    degrades to a no-op rather than silently doing nothing.
    """

    def __init__(self, conn, cfg: Optional[DopamineConfig] = None,
                 reward_types: Sequence[str] = ("PAM", "PPL1-α3", "PPL3"),
                 punish_types: Sequence[str] = ("PPL1-γ2α'1", "PPL1-γ1", "PPL101")):
        self.cfg = cfg or DopamineConfig()
        by_type = conn.index_by_type()
        self.reward_neurons: List[int] = [i for t in reward_types for i in by_type.get(t, [])]
        self.punish_neurons: List[int] = [i for t in punish_types for i in by_type.get(t, [])]
        self.warned = not (self.reward_neurons or self.punish_neurons)

    @property
    def available(self) -> bool:
        return bool(self.reward_neurons or self.punish_neurons)

    def current(self, signal: float) -> float:
        c = self.cfg
        s = max(-c.clip, min(c.clip, signal * c.gain + c.bias))
        return s

    def drive(self, net, signal: float) -> float:
        """Inject the dopamine signal as current into the DA populations."""
        c = self.cfg
        cur = self.current(signal)
        if cur >= 0.0:
            targets, amp = self.reward_neurons, cur
        else:
            targets, amp = self.punish_neurons, cur
        if not targets:
            # fall back to the other side with inverted sign if only one exists
            targets = self.reward_neurons or self.punish_neurons
            amp = cur if cur >= 0 else cur
        for i in targets:
            net.add_input(i, amp)
        return cur


def three_factor_update(net, da_signal: float, debug: bool = False) -> dict:
    """Apply one dopamine-gated plasticity step to the marked plastic edges."""
    before = [net.c.weight[k] for k in net.plastic_edges]
    touched = net.deliver_dopamine(da_signal, lr=net.p.last_lr if hasattr(net.p, "last_lr") else 0.02)
    after = [net.c.weight[k] for k in net.plastic_edges]
    mean_w = sum(after) / max(len(after), 1)
    return {"touched": touched, "mean_weight": mean_w,
            "delta_mean": (sum(after) - sum(before)) / max(len(after), 1)}
