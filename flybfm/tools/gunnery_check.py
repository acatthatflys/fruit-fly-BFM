"""Can the learner actually aim?

Win rate is a bad instrument for gunnery: a policy can win on points, on
timeouts, and on the other aircraft's mistakes without ever putting the nose
where a round would connect.  This tool flies a trained policy on an *easy*
start -- straight astern of a cooperative target at point-blank range -- and
reports the quantity that a fixed-forward gun actually cares about:

    lead_angle_deg = the angle between the nose and the correct *ballistic*
                     lead point (gun.required_lead_angle_deg).

If a policy cannot hold that angle inside a degree from a stern start at 300 m,
no amount of extra shaping will teach it to shoot, and the answer is a policy
class or credit-assignment problem, not a reward-weight problem.

    python -m flybfm.tools.gunnery_check --params runs/features/training.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import List, Optional

from ..sim.arena import Scenario
from ..sim.dogfight import Dogfight, EnvConfig
from ..sim.gun import required_lead_angle_deg
from ..sim.scripted import POOL_BY_NAME, default_pool
from ..train.policy import FeaturePolicy, PolyPolicy


def _load_policy(path: str, seed: int = 0):
    """Load a policy, using whatever action basis the checkpoint was trained on."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "best" in data:
        params, basis = data["best"]["params"], data.get("basis", "linear")
    else:
        params, basis = data, "linear"
    return PolyPolicy(params) if basis == "poly" else FeaturePolicy(params)


def gunnery_check(args) -> int:
    pol = _load_policy(args.params)
    if args.opponent not in POOL_BY_NAME:
        print("unknown opponent %r; pool is: %s"
              % (args.opponent, ", ".join(sorted(POOL_BY_NAME))))
        return 2
    opp_factory = POOL_BY_NAME[args.opponent]

    shots = hits = 0
    results = {"blue": 0, "red": 0, "draw": 0}
    lead_all: List[float] = []
    lead_wez: List[float] = []
    lead_trigger: List[float] = []
    lead_late: List[float] = []          # shots after the opening 2 s
    late_shots = 0
    in_range_steps = 0

    for ep in range(args.episodes):
        # A little jitter per episode, otherwise a deterministic policy against a
        # deterministic target produces the *same fight* N times and the table
        # below is one sample wearing N hats.  (Measured: it was.  Every episode
        # fired exactly 16 rounds in the first second and stopped.)
        j = random.Random(args.seed + ep)
        scen = Scenario(range_m=args.range * j.uniform(0.85, 1.15),
                        taa_deg=args.taa + j.uniform(-10.0, 10.0),
                        nose_offset_deg=j.uniform(-8.0, 8.0),
                        dz_m=j.uniform(-60.0, 60.0), alt_m=6000.0,
                        phase_deg=0.0, seed=args.seed + ep)
        env = Dogfight(EnvConfig(max_time_s=args.max_time))
        env.reset(scen)
        while not env.done:
            cb = pol(env.blue, env)
            cr = opp_factory(seed=args.seed + ep)(env.red, env)
            b = env.blue
            la = b.lead_angle
            lead_all.append(abs(la))
            if b.geom_to_other.range_m <= 900.0 and b.geom_to_other.range_m >= 150.0:
                in_range_steps += 1
                lead_wez.append(abs(la))
            if cb.trigger:
                lead_trigger.append(abs(la))
                if env.t > 2.0:
                    lead_late.append(abs(la))
                    late_shots += 1
            env.step(cb, cr)
        summ = env.summary()
        results[summ["result"]] = results.get(summ["result"], 0) + 1
        shots += summ["blue_shots"]
        hits += summ["blue_hits"]

    def pct(xs: List[float], q: float) -> float:
        if not xs:
            return float("nan")
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    n = max(1, args.episodes)
    print("policy %s vs %s   %d fights, start R=%d m TAA=%.0f deg"
          % (args.params, args.opponent, args.episodes, args.range, args.taa))
    print("result  W %d / L %d / D %d        rounds fired %d, hits %d%s"
          % (results.get("blue", 0), results.get("red", 0), results.get("draw", 0),
             shots, hits, "   (%.2f%% of rounds)" % (100.0 * hits / shots) if shots else ""))
    print()
    print("ballistic lead error, |deg|  (a round needs about 1 deg at 600 m)")
    print("  all decision steps       median %5.2f   p90 %6.2f   n=%d"
          % (pct(lead_all, 0.5), pct(lead_all, 0.9), len(lead_all)))
    print("  inside the gun envelope  median %5.2f   p90 %6.2f   n=%d"
          % (pct(lead_wez, 0.5), pct(lead_wez, 0.9), len(lead_wez)))
    print("  while firing             median %5.2f   p90 %6.2f   n=%d"
          % (pct(lead_trigger, 0.5), pct(lead_trigger, 0.9), len(lead_trigger)))
    if lead_trigger:
        good = sum(1 for v in lead_trigger if v <= 1.5) / len(lead_trigger)
        print("  trigger pressed with |lead| <= 1.5 deg: %.1f%% of firing steps" % (100.0 * good))
        print("  shots after the first 2 s           %d of %d  %s"
              % (late_shots, len(lead_trigger),
                 "(%.1f%% - this is the number that says whether the policy can "
                  "*hold* or *acquire* a solution, or only fire when one is handed to it)"
                  % (100.0 * late_shots / len(lead_trigger))))
    if lead_late:
        print("  lead error, shots after 2 s         median %5.2f   p90 %6.2f   n=%d"
              % (pct(lead_late, 0.5), pct(lead_late, 0.9), len(lead_late)))
    if in_range_steps:
        print("  fraction of decision steps inside the envelope: %.1f%%"
              % (100.0 * in_range_steps / max(1, len(lead_all))))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="measure a policy's aiming error")
    ap.add_argument("--params", default="runs/features/training.json")
    ap.add_argument("--opponent", default="level")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--range", type=int, default=300)
    ap.add_argument("--taa", type=float, default=0.0)
    ap.add_argument("--v-a", type=float, default=280.0)
    ap.add_argument("--v-t", type=float, default=250.0)
    ap.add_argument("--max-time", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1000)
    return gunnery_check(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
