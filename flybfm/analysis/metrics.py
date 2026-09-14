"""The evaluation scorecard.

Anything you report about a fighter agent should come out of this file, on a
*held-out* set of starting conditions, with fixed seeds, so that two runs are
comparable.  The numbers are grouped the way a BFM debrief is:

    OFFENCE   tracking time, shots, hits, hit rate, time-to-first-hit
    DEFENCE   hits taken, time spent defensive, overshoots given away
    ENERGY    specific-energy advantage, time below corner speed
    TACTICS   manoeuvre content (analysis/maneuvers.py)
    SAFETY    ground/ceiling/arena losses

Reporting only win rate hides the most common failure: an agent that wins by
flying its opponent out of the arena or into the ground has learned nothing
about gunnery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .maneuvers import classify_trajectory


@dataclass
class EpisodeMetrics:
    result: str = "draw"
    duration_s: float = 0.0
    tracking_time_s: float = 0.0
    wez_time_s: float = 0.0
    overshoot_count: int = 0
    shots: int = 0
    hits: int = 0
    hits_taken: int = 0
    mean_abs_aa: float = 0.0
    mean_taa: float = 0.0
    energy_adv: float = 0.0
    mean_lead_angle: float = 0.0
    manoeuvre_content: float = 0.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def episode_metrics(env) -> EpisodeMetrics:
    m = EpisodeMetrics()
    m.duration_s = env.t
    m.shots = env.blue.gun.g.shots
    m.hits = env.blue.gun.g.hits
    m.hits_taken = env.red.gun.g.hits
    m.result = env.result
    n = max(len(env.frames), 1)
    aas, taas, leads, energy = [], [], [], []
    for f in env.frames:
        g = f["g"]
        aas.append(abs(g["aa"]))
        taas.append(g["taa"])
        leads.append(abs(g["lead"]))
        if g["wez"]:
            m.wez_time_s += env.cfg.decision_dt * env.cfg.record_every
        if abs(g["aa"]) < 12.0 and g["rng"] < 900.0:
            m.tracking_time_s += env.cfg.decision_dt * env.cfg.record_every
        if g["clo"] > 60.0 and g["rng"] < 300.0:
            m.overshoot_count += 1
    m.mean_abs_aa = sum(aas) / n
    m.mean_taa = sum(taas) / n
    m.mean_lead_angle = sum(leads) / n
    try:
        m.energy_adv = env.blue.s.specific_energy() - env.red.s.specific_energy()
    except Exception:
        m.energy_adv = 0.0
    m.manoeuvre_content = classify_trajectory(env.frames).manoeuvre_content
    return m


def scorecard(episodes: Sequence[EpisodeMetrics]) -> dict:
    n = max(len(episodes), 1)

    def mean(attr):
        return sum(getattr(e, attr) for e in episodes) / n

    wins = sum(1 for e in episodes if e.result in ("blue", "win"))
    losses = sum(1 for e in episodes if e.result in ("red", "loss"))
    draws = n - wins - losses
    shots = sum(e.shots for e in episodes)
    hits = sum(e.hits for e in episodes)
    return {
        "episodes": n,
        "win_rate": wins / n, "loss_rate": losses / n, "draw_rate": draws / n,
        "offence": {
            "shots": shots, "hits": hits, "hit_rate": hits / max(shots, 1),
            "episodes_with_a_hit": sum(1 for e in episodes if e.hits) / n,
            "tracking_time_s": mean("tracking_time_s"),
            "wez_time_s": mean("wez_time_s"),
            "mean_abs_angle_off_deg": mean("mean_abs_aa"),
            "mean_lead_angle_deg": mean("mean_lead_angle"),
        },
        "defence": {
            "hits_taken": sum(e.hits_taken for e in episodes),
            "mean_taa_deg": mean("mean_taa"),
        },
        "energy": {"mean_specific_energy_advantage_m": mean("energy_adv")},
        "tactics": {"manoeuvre_content": mean("manoeuvre_content")},
        "safety": {"overshoots_per_episode": mean("overshoot_count"),
                   "mean_duration_s": mean("duration_s")},
    }


def format_scorecard(sc: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    o, d, e, t, s = sc["offence"], sc["defence"], sc["energy"], sc["tactics"], sc["safety"]
    lines.append(f"  episodes {sc['episodes']}  W/L/D {sc['win_rate']:.2f}/{sc['loss_rate']:.2f}/{sc['draw_rate']:.2f}")
    lines.append(f"  offence : {o['shots']} rds, {o['hits']} hits ({100*o['hit_rate']:.1f}%), "
                 f"{o['episodes_with_a_hit']*100:.0f}% of fights with a hit, "
                 f"tracking {o['tracking_time_s']:.1f}s, lead-err {o['mean_lead_angle_deg']:.2f} deg")
    lines.append(f"  defence : {d['hits_taken']} hits taken, mean TAA {d['mean_taa_deg']:.0f} deg")
    lines.append(f"  energy  : {e['mean_specific_energy_advantage_m']:+.0f} m of specific energy")
    lines.append(f"  tactics : manoeuvre content {t['manoeuvre_content']:.2f}, "
                 f"overshoots {s['overshoots_per_episode']:.1f}/fight, duration {s['mean_duration_s']:.1f}s")
    return "\n".join(lines)
