"""Recognising whether the agent is actually flying BFM.

Win rate alone cannot tell you whether you trained a fighter or a lucky nose-
tracker.  This module labels the trajectory with the manoeuvres a BFM instructor
would call, using the same airframe-state features the sim already records:

    lead / pure / lag pursuit      angle-off relative to the turn circle
    high yo-yo                     offensive, climbing, closing decreasing
    low yo-yo                      offensive, descending, closing increasing
    break turn                     defensive, sustained high-G turn
    scissors                       repeated roll reversals at close range
    extension / disengage          defensive, range opening
    immelmann / split-S            vertical heading reversal
    vertical fight                 sustained climb/dive with a nose-high/low solution

The point of the module is the "manoeuvre content" number: what fraction of the
fight the agent spent executing recognisable tactics as opposed to pointing at
the target.  It is a heuristic, and it is stated as one -- but it is measurable,
comparable and does not require a human to watch replays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

#: Tactical manoeuvres.  The instantaneous pursuit labels (lead/pure/lag) are
#: deliberately NOT in here: every instant of every fight is one of the three,
#: so including them would make "manoeuvre content" trivially 1.0.  They are
#: reported separately as a distribution, which is the useful form.
MOVES = ["high_yoyo", "low_yoyo", "break_turn", "scissors", "extension",
         "immelmann", "split_s", "barrel_roll", "vertical_fight"]
PURSUIT = ["lead", "pure", "lag"]


@dataclass
class ManoeuvreStats:
    duration_s: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)
    time_in: Dict[str, float] = field(default_factory=dict)
    pursuit: Dict[str, int] = field(default_factory=dict)

    @property
    def manoeuvre_content(self) -> float:
        """Fraction of the fought time spent in a recognised manoeuvre."""
        covered = sum(self.time_in.get(m, 0.0) for m in MOVES)
        return min(1.0, covered / max(self.duration_s, 1e-6))

    @property
    def pursuit_mix(self) -> dict:
        tot = max(sum(self.pursuit.values()), 1)
        return {k: round(v / tot, 3) for k, v in self.pursuit.items()}

    def summary(self) -> dict:
        return {"duration_s": round(self.duration_s, 2),
                "pursuit_mix": self.pursuit_mix,
                "manoeuvre_content": round(self.manoeuvre_content, 3),
                "counts": self.counts,
                "seconds": {k: round(v, 2) for k, v in self.time_in.items() if v > 0.05},
                "pursuit_samples": self.pursuit}


def classify_trajectory(frames: Sequence[dict], dt: Optional[float] = None) -> ManoeuvreStats:
    """`frames` are the replay frames produced by Dogfight._record()."""
    stats = ManoeuvreStats()
    if len(frames) < 3:
        return stats
    ts = [f["t"] for f in frames]
    stats.duration_s = ts[-1] - ts[0]
    if dt is None:
        dts = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        dt = sum(dts) / len(dts) if dts else 0.1

    # --- instantaneous pursuit labels
    for f in frames:
        aa = f["g"]["aa"]
        if aa < -5.0:
            stats.pursuit["lead"] = stats.pursuit.get("lead", 0) + 1
        elif aa > 5.0:
            stats.pursuit["lag"] = stats.pursuit.get("lag", 0) + 1
        else:
            stats.pursuit["pure"] = stats.pursuit.get("pure", 0) + 1

    # --- temporal manoeuvres
    active: Dict[str, bool] = {}
    def mark(name: str, on: bool):
        prev = active.get(name, False)
        if on:
            stats.time_in[name] = stats.time_in.get(name, 0.0) + dt
            if not prev:
                stats.counts[name] = stats.counts.get(name, 0) + 1
        active[name] = on

    win = 12                          # ~1.2 s window at the record rate
    for i, f in enumerate(frames):
        j = max(0, i - win)
        p0, p1 = frames[j]["b"], f["b"]
        g = f["g"]
        dz = p1["p"][2] - p0["p"][2]
        dv = p1["v"] - p0["v"]
        dpsi = _wrap_deg(p1["psi"] - p0["psi"])
        clo = g["clo"]
        aa = g["aa"]
        mu = abs(p1["mu"])
        n = p1["n"]
        offensive = g["taa"] < 90.0
        defensive = g["taa"] > 110.0

        mark("high_yoyo", offensive and dz > 250.0 and clo < 40.0 and p1["gam"] > 12.0)
        mark("low_yoyo", offensive and dz < -250.0 and clo > 40.0 and p1["gam"] < -12.0)
        mark("break_turn", defensive and mu > 55.0 and n > 5.0)
        mark("extension", defensive and clo < -30.0)
        mark("immelmann", dpsi > 110.0 and dz > 220.0)
        mark("split_s", dpsi < -110.0 and dz < -220.0)
        mark("vertical_fight", abs(p1["gam"]) > 40.0 and abs(aa) < 35.0)
        mark("barrel_roll", mu > 100.0 and abs(p1["gam"]) > 25.0 and offensive)

    # scissors: >=3 roll reversals inside any 20 s window at close range
    reversals = []
    for a, b in zip(frames, frames[1:]):
        if a["b"]["mu"] * b["b"]["mu"] < 0 and abs(a["b"]["mu"]) > 30.0:
            reversals.append(b["t"])
    for t in reversals:
        near = [f for f in frames if t - 20.0 <= f["t"] <= t and f["g"]["rng"] < 1500.0]
        if len(near) > 3:
            n_rev = sum(1 for x in reversals if t - 20.0 <= x <= t)
            if n_rev >= 3:
                stats.time_in["scissors"] = stats.time_in.get("scissors", 0.0) + dt
                stats.counts["scissors"] = max(stats.counts.get("scissors", 0), 1)
                break
    return stats


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def side_frame_manoeuvres(frames: Sequence[dict], which: str = "b") -> ManoeuvreStats:
    """Classify a side's manoeuvres from frames, rewriting the frame dicts so the
    classifier always works in 'blue' terms (the recorded geometry is blue's)."""
    if which == "b":
        return classify_trajectory(frames)
    flipped = []
    for f in frames:
        g = dict(f["g"])
        g["aa"] = -g["aa"]
        g["taa"] = 180.0 - g["taa"]
        flipped.append({"t": f["t"], "b": f["r"], "r": f["b"], "g": g, "hp": f["hp"]})
    return classify_trajectory(flipped)
