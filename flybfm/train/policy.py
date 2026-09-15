"""Learnable policies for the dogfight environment.

Three policy families, deliberately comparable:

  FeaturePolicy     linear (or 2-layer tanh) map from the 22 engineered
                    geometry features to the four command channels.  This is
                    the workhorse: fast enough to train by black-box search,
                    and a clean baseline for "how much of BFM is in the
                    observation vector".

  ReadoutPolicy     the connectome path: the 10-parameter descending-neuron
                    readout of brain/controller.py.  Same search algorithm.

  ScriptedPolicy    anything from sim/scripted.py (not learnable, used as
                    opponents).

All learnable policies expose a flat parameter vector and a `with_params`
constructor, so the trainer is agnostic to which one it is optimising.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .. import vecmath as vm
from ..sim.aircraft import Command


class FeaturePolicy:
    """Linear policy over the engineered observation, with saturation.

    Gunnery-focused final (2026-09): N_IN 22 adds lead_az and lead_el separate
    horizontal/vertical errors to ballistic lead point. Linear default 92 params.
    Trigger bias positive to avoid never-fire local optimum.
    """

    N_IN = 22
    N_OUT = 4             # roll, pull, throttle, trigger logit

    def __init__(self, params: Optional[Sequence[float]] = None, hidden: int = 0,
                 seed: int = 0, name: str = "feature-policy"):
        self.name = name
        self.hidden = hidden
        rng = random.Random(seed)
        self.n_params = self.N_OUT * (self.N_IN + 1)
        if hidden:
            self.n_params = (hidden * (self.N_IN + 1) + self.N_OUT * (hidden + 1))
        if params is None:
            params = [rng.gauss(0.0, 0.1) for _ in range(self.n_params)]
            # trigger bias +1.5 to avoid never-fire local optimum (suggestion 4, was 0.8)
            for o in range(self.N_OUT):
                if o == 3:  # trigger
                    params[o * (self.N_IN + 1) + self.N_IN] = 1.5
        self.params: List[float] = list(params)

    def copy(self):
        return FeaturePolicy(self.params, hidden=self.hidden, name=self.name)

    def with_params(self, params: Sequence[float]):
        return FeaturePolicy(params, hidden=self.hidden, name=self.name)

    def _forward(self, phi: Sequence[float]) -> List[float]:
        if not self.hidden:
            out = []
            for o in range(self.N_OUT):
                base = o * (self.N_IN + 1)
                acc = self.params[base + self.N_IN]
                for i in range(self.N_IN):
                    acc += self.params[base + i] * phi[i]
                out.append(acc)
            return out
        h = self.hidden
        hid = []
        for j in range(h):
            base = j * (self.N_IN + 1)
            acc = self.params[base + self.N_IN]
            for i in range(self.N_IN):
                acc += self.params[base + i] * phi[i]
            hid.append(math.tanh(acc))
        off = h * (self.N_IN + 1)
        out = []
        for o in range(self.N_OUT):
            base = off + o * (h + 1)
            acc = self.params[base + h]
            for j in range(h):
                acc += self.params[base + j] * hid[j]
            out.append(acc)
        return out

    def __call__(self, me, env) -> Command:
        phi = env.observation_vector(me)
        roll, pull, thr, trig = self._forward(phi)
        return Command(roll=math.tanh(roll),
                       pull=math.tanh(pull),
                       throttle=1.0 / (1.0 + math.exp(-vm.clamp(thr, -30, 30))),
                       trigger=trig > 0.0)


# --------------------------------------------------------------------------- #
# A quadratic feature map.
#
# Measured fact, and the reason this class exists: a *linear* readout of the 18
# geometry features reaches a 74% win rate and a median ballistic lead error of
# 122 degrees.  It wins on points and never points at the target.  The laws that
# actually aim a fixed-forward gun -- roll ~ lateral aim error, pull ~ vertical
# aim error *and* the closing geometry that sets the lead -- are products of the
# features, not the features themselves, so give the optimiser products to work
# with.  Squares of every feature plus cross terms among the flight-relevant
# ones (nose, aspect, range, closure, LOS rate, speed, lead error) is small
# enough that ridge regression can fit it from a few thousand demonstrated
# samples, and therefore small enough for CEM to refine.
#
# Gunnery-focused final (2026-09): N_IN 22 adds lead_az/el separate (18,19,20,21)
# for direction to ballistic lead point. Subset now includes 15 (lead mag) + 18-21.
# --------------------------------------------------------------------------- #
QUAD_SUBSET = (0, 1, 2, 4, 5, 6, 7, 8, 9, 15, 18, 19, 20, 21)


def quad_features(phi: Sequence[float], subset=QUAD_SUBSET) -> List[float]:
    f = [1.0] + list(phi)
    for i in subset:
        f.append(phi[i] * phi[i])
    for a in range(len(subset)):
        for b in range(a + 1, len(subset)):
            f.append(phi[subset[a]] * phi[subset[b]])
    return f


class PolyPolicy:
    """Linear readout on a quadratic expansion of the observation.

    Gunnery-focused final (2026-09): N_IN 22, includes lead az/el direction separate.
    22 + squares 14 + cross 91 = 127 features per output *4 = 508 params.
    Linear is now default (92 params) — only switch to poly after linear hits.
    """

    N_IN = 22
    N_OUT = 4

    def __init__(self, params: Optional[Sequence[float]] = None, seed: int = 0,
                 name: str = "poly-policy"):
        self.name = name
        rng = random.Random(seed)
        self.n_features = len(quad_features([0.0] * self.N_IN))
        self.n_params = self.N_OUT * self.n_features
        if params is None:
            params = [rng.gauss(0.0, 0.05) for _ in range(self.n_params)]
            # trigger bias +1.5 to avoid never-fire
            params[3 * self.n_features] = 1.5
        self.params: List[float] = list(params)

    def copy(self):
        return PolyPolicy(self.params, name=self.name)

    def with_params(self, params: Sequence[float]):
        return PolyPolicy(params, name=self.name)

    def _forward(self, phi: Sequence[float]) -> List[float]:
        f = quad_features(phi)
        out = []
        for o in range(self.N_OUT):
            base = o * self.n_features
            acc = 0.0
            for i in range(self.n_features):
                acc += self.params[base + i] * f[i]
            out.append(acc)
        return out

    def __call__(self, me, env) -> Command:
        phi = env.observation_vector(me)
        roll, pull, thr, trig = self._forward(phi)
        return Command(roll=math.tanh(roll),
                       pull=math.tanh(pull),
                       throttle=1.0 / (1.0 + math.exp(-vm.clamp(thr, -30, 30))),
                       trigger=trig > 0.0)


class RandomPolicy:
    """Uniform random controls: the 'learned nothing' floor."""

    def __init__(self, seed: int = 0, trigger_p: float = 0.3):
        self.rng = random.Random(seed)
        self.trigger_p = trigger_p
        self.name = "random"
        self.n_params = 0

    def __call__(self, me, env) -> Command:
        return Command(roll=self.rng.uniform(-1, 1), pull=self.rng.uniform(-1, 1),
                       throttle=self.rng.random(),
                       trigger=self.rng.random() < self.trigger_p)


class BrainPolicyAdapter:
    """Wraps a ConnectomeController so it drops into the same trainer."""

    def __init__(self, controller):
        self.c = controller
        self.name = f"connectome-{controller.name}"
        self.n_params = len(controller.readout.vector())

    @property
    def params(self) -> List[float]:
        return self.c.readout.vector()

    def with_params(self, params):
        from ..brain.controller import ReadoutParams
        self.c.readout = ReadoutParams.from_vector(params)
        return self

    def __call__(self, me, env) -> Command:
        return self.c.act(me, env)

    def reset(self, seed: Optional[int] = None):
        """Passthrough so trainer can reset brain state each episode."""
        if hasattr(self.c, "reset"):
            try:
                return self.c.reset(seed=seed)
            except TypeError:
                return self.c.reset()
        return None

    def observe_reward(self, phi_next, reward):
        return self.c.observe(phi_next, reward)
