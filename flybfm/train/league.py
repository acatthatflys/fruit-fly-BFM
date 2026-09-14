"""Opponent pools, leagues and non-stationarity control.

This file is the answer to "how do I stop it over-fitting to one opponent".

Three mechanisms, all standard in the air-combat RL literature (and in
AlphaStar/AlphaDogfight for good reason):

  1. HETEROGENEOUS POOL      -- sim/scripted.py supplies ten different
     strategies.  A learner that only fights nose-on-the-target learns to beat
     nose-on-the-target.

  2. PRIORITISED SAMPLING (PFSP) -- sample opponents in proportion to how much
     there still is to learn from them, not uniformly.  Uniform sampling spends
     most of its budget on opponents that are already solved.

  3. LEAGUE / FROZEN SNAPSHOTS -- keep every N generations a frozen copy of the
     learner and put it in the pool.  Self-play against a single live opponent
     cycles (the "red queen" effect: each side's improvements redefine the other's
     problem, so measured skill goes sideways).  Freezing agents breaks the cycle.

A fourth mechanism lives in sim/arena.py: domain randomisation over the starting
geometry and speed.  That is the one the user proposed, and it is the correct
instinct -- the fight must not always start in the same place.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..sim.scripted import default_pool, POOL_BY_NAME


@dataclass
class OpponentRecord:
    name: str
    policy: object
    wins_vs_learner: int = 0
    losses_vs_learner: int = 0
    draws: int = 0
    episodes: int = 0
    importance: float = 1.0      # sampling weight for PFSP

    @property
    def win_rate(self) -> float:
        return self.wins_vs_learner / max(self.episodes, 1)

    def register(self, result: str):
        """`result` is the outcome *from the learner's point of view*."""
        self.episodes += 1
        if result == "win":
            self.losses_vs_learner += 1
        elif result == "loss":
            self.wins_vs_learner += 1
        else:
            self.draws += 1
        # PFSP: importance = how often the opponent still beats the learner.
        # A little floor keeps a solved opponent in the rotation (avoids
        # catastrophic forgetting of a strategy you stopped practising).
        p = self.win_rate
        self.importance = 0.15 + 0.85 * p


class OpponentPool:
    def __init__(self, seed: int = 0, include_scripts: bool = True):
        self.rng = random.Random(seed)
        self.records: List[OpponentRecord] = []
        if include_scripts:
            for p in default_pool(seed=seed):
                self.records.append(OpponentRecord(name=p.name, policy=p))

    def add(self, name: str, policy) -> None:
        self.records.append(OpponentRecord(name=name, policy=policy))

    def sample(self, rng: Optional[random.Random] = None) -> OpponentRecord:
        rng = rng or self.rng
        total = sum(r.importance for r in self.records)
        x = rng.random() * total
        acc = 0.0
        for r in self.records:
            acc += r.importance
            if x <= acc:
                return r
        return self.records[-1]

    def summary(self) -> List[dict]:
        return [{"name": r.name, "episodes": r.episodes,
                 "learner_wins": r.losses_vs_learner,
                 "opponent_wins": r.wins_vs_learner,
                 "draws": r.draws,
                 "importance": round(r.importance, 3)}
                for r in sorted(self.records, key=lambda r: -r.importance)]


class League:
    """Keeps frozen checkpoints of the learner so self-play cannot cycle."""

    def __init__(self, freeze_every: int = 5, max_frozen: int = 8,
                 n_self_play: int = 2):
        self.freeze_every = freeze_every
        self.max_frozen = max_frozen
        self.n_self_play = n_self_play
        self.frozen: List[dict] = []
        self.generation = 0

    def maybe_freeze(self, generation: int, policy, score: float) -> Optional[dict]:
        if generation % self.freeze_every != 0:
            return None
        snap = {"generation": generation,
                "params": list(policy.params),
                "score": score,
                "name": f"snapshot-g{generation}"}
        self.frozen.append(snap)
        if len(self.frozen) > self.max_frozen:
            self.frozen.pop(0)
        return snap

    def sample_frozen(self, rng: random.Random):
        if not self.frozen:
            return None
        return rng.choice(self.frozen)
