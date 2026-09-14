"""Cross-entropy-method / evolutionary-strategy trainer.

Why CEM/ES rather than PPO
--------------------------
  * The controller we most want to train sits behind a spiking network and a
    black-box flight sim.  BPTT through 166k LIF neurons is possible (surrogate
    gradients) but it is a separate research project; ES treats the whole loop
    as a function and just asks "is this parameter vector good?".
  * The searchable policy here has 10-100 parameters.  At that dimension ES is
    competitive with policy gradients and far more robust to reward scale,
    credit assignment and non-stationarity.
  * It parallelises trivially across processes, and it is deterministic given
    seeds, which makes every result in this repo reproducible.
  * The user-facing consequence: *rewards are compared across whole episodes*,
    not turned into a value-function learning problem.  The reward design has to
    be right, because there is nowhere for the optimiser to hide.

If you want gradients instead, the environment exposes `observation_vector`
and a dense reward; PPO/SAC will drop straight in and the reward design in
sim/reward.py (potential-based shaping) is exactly what they need.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..sim.arena import CANONICAL_SETUPS, Scenario, ScenarioSpace
from ..sim.dogfight import Dogfight, EnvConfig, run_match
from ..sim.reward import RewardConfig
from .league import League, OpponentPool


# ----------------------------------------------------------------------------
def run_episode(scenario: Scenario, learner, opponent, env_cfg: EnvConfig,
                reward_cfg: RewardConfig, learn: bool = True, record: bool = False):
    """One fight. `learner` always flies blue, the opponent flies red. Returns
    (info_dict, env)."""
    env = Dogfight(env_cfg, reward_cfg)
    env.reset(scenario)
    returns = 0.0
    while not env.done:
        cb = learner(env.blue, env)
        cr = opponent(env.red, env)
        obs, rewards, done, info = env.step(cb, cr)
        if learn and hasattr(learner, "observe_reward"):
            # dopamine arrives after the outcome: the TD-error teaching signal
            learner.observe_reward(env.observation_vector(env.blue), rewards["blue"])
        returns += rewards["blue"]
    outcome = {"blue": "win", "red": "loss", "draw": "draw", "mutual": "draw"}.get(env.result, "draw")
    info = {
        "result": outcome,
        "raw_result": env.result,
        "return": returns,
        "t": env.t,
        "blue_hp": max(env.blue.hp, 0.0),
        "red_hp": max(env.red.hp, 0.0),
        "blue_hits": env.blue.gun.g.hits,
        "red_hits": env.red.gun.g.hits,
        "blue_shots": env.blue.gun.g.shots,
        "tag": scenario.tag,
    }
    return info, env


def evaluate(policy, scenarios: Sequence[Scenario], opponents: Sequence,
             env_cfg: Optional[EnvConfig] = None,
             reward_cfg: Optional[RewardConfig] = None,
             seed: int = 0) -> dict:
    """Fixed-seed evaluation. Always used for comparisons between generations."""
    env_cfg = env_cfg or EnvConfig()
    reward_cfg = reward_cfg or RewardConfig()
    rng = random.Random(seed)
    wins = losses = draws = 0
    hits = shots = 0
    hit_events = 0
    t_first_hit: List[float] = []
    for sc in scenarios:
        for opp in opponents:
            info, env = run_episode(sc, policy, opp, env_cfg, reward_cfg, learn=False)
            if info["result"] == "win":
                wins += 1
            elif info["result"] == "loss":
                losses += 1
            else:
                draws += 1
            hits += info["blue_hits"]
            shots += info["blue_shots"]
            if info["blue_hits"]:
                hit_events += 1
                for ev in env.event_log:
                    if "blue hits" in ev["msg"]:
                        t_first_hit.append(ev["t"])
                        break
    n = max(wins + losses + draws, 1)
    return {"episodes": n, "win_rate": wins / n, "loss_rate": losses / n,
            "draw_rate": draws / n, "hits": hits, "shots": shots,
            "hit_rate": hits / max(shots, 1),
            "episodes_with_a_hit": hit_events / n,
            "mean_time_to_first_hit": (sum(t_first_hit) / len(t_first_hit)) if t_first_hit else None}


# ----------------------------------------------------------------------------
@dataclass
class CEMConfig:
    generations: int = 8
    population: int = 12
    elite_frac: float = 0.25
    sigma0: float = 0.5
    sigma_min: float = 0.05
    episodes_per_candidate: int = 3
    eval_every: int = 2
    max_time_s: float = 30.0
    decision_dt: float = 0.1
    seed: int = 0
    freeze_every: int = 3
    verbose: bool = True


class MixedSpace:
    """Scenario sampler that mixes fully random starts with gunnery-range starts.

    The gun needs the nose inside about a degree.  Uniform random starts spread
    the learner over the whole 12 km arena, so the fraction of training time spent
    anywhere near a firing solution is tiny and the shooting skill never gets a
    gradient.  This sampler deliberately oversamples close, roughly-aligned
    starts so the sparse event can be learned, while leaving *evaluation* on the
    fully random space: the curriculum buys signal, it does not buy the score.
    """

    def __init__(self, easy_frac: float = 0.5, seed: int = 0):
        self.easy_frac = easy_frac
        self.rng = random.Random(seed)
        self.space = ScenarioSpace()

    def sample(self, rng=None, tag: str = "train") -> Scenario:
        rng = rng or self.rng
        if rng.random() >= self.easy_frac:
            return self.space.sample(rng, tag=tag)
        return Scenario(range_m=rng.uniform(250.0, 900.0),
                        taa_deg=rng.uniform(-30.0, 30.0),
                        nose_offset_deg=rng.uniform(-30.0, 30.0),
                        dz_m=rng.uniform(-150.0, 150.0),
                        alt_m=rng.uniform(4000.0, 9000.0),
                        v_a=rng.uniform(220.0, 320.0),
                        v_t=rng.uniform(200.0, 320.0),
                        phase_deg=rng.uniform(0.0, 360.0),
                        seed=rng.randrange(1 << 30), tag=tag + "_gunnery")


def train(policy_factory: Callable[[Sequence[float]], object],
          n_params: int,
          space: ScenarioSpace,
          pool: Optional[OpponentPool] = None,
          cfg: Optional[CEMConfig] = None,
          init_params: Optional[Sequence[float]] = None,
          out_dir: Optional[str] = None,
          meta: Optional[dict] = None) -> dict:
    """Optimise `n_params` numbers to maximise mean episodic return."""
    cfg = cfg or CEMConfig()
    rng = random.Random(cfg.seed)
    pool = pool or OpponentPool(seed=cfg.seed)
    env_cfg = EnvConfig(decision_dt=cfg.decision_dt, max_time_s=cfg.max_time_s)
    rc = RewardConfig()

    mu = list(init_params) if init_params else [0.0] * n_params
    sigma = [cfg.sigma0] * n_params
    n_elite = max(1, int(round(cfg.population * cfg.elite_frac)))
    league = League(freeze_every=cfg.freeze_every)
    history: List[dict] = []
    best = {"score": -1e9, "params": list(mu)}

    for gen in range(cfg.generations):
        t0 = time.time()
        cand_params = [list(mu)]
        for _ in range(max(0, cfg.population - 1)):
            cand_params.append([mu[i] + rng.gauss(0.0, sigma[i]) for i in range(n_params)])
        scores: List[float] = []
        for params in cand_params:
            policy = policy_factory(params)
            total = 0.0
            for k in range(cfg.episodes_per_candidate):
                sc = space.sample(rng, tag="train")
                rec = pool.sample(rng)
                info, _ = run_episode(sc, policy, rec.policy, env_cfg, rc,
                                      learn=True, record=False)
                total += info["return"]
                # bookkeeping for PFSP: who is still beating the learner
                if hasattr(rec.policy, "observe_reward"):
                    rec.policy.observe_reward((0.0,) * 18, 0.0)
                rec.register(info["result"])
            scores.append(total / cfg.episodes_per_candidate)

        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        elite = [cand_params[i] for i in order[:n_elite]]
        elite_scores = [scores[i] for i in order[:n_elite]]
        if elite_scores[0] > best["score"]:
            best = {"score": elite_scores[0], "params": list(cand_params[order[0]])}
        # CEM update
        for i in range(n_params):
            vals = [e[i] for e in elite]
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / len(vals)
            sigma[i] = max(cfg.sigma_min, math.sqrt(v))
            mu[i] = m

        gen_info = {"generation": gen, "best_return": elite_scores[0],
                    "mean_return": sum(scores) / len(scores),
                    "wall_s": round(time.time() - t0, 2)}
        if cfg.eval_every and gen % cfg.eval_every == 0:
            # Evaluate what actually gets saved.  The distribution mean and the
            # best member of the population are different policies, and reporting
            # the mean while writing `best.params` to disk is how a run ends up
            # claiming a 0.46 win rate for a checkpoint that scores 0.03.
            gen_info["eval"] = evaluate(policy_factory(best["params"]), _eval_scenarios(),
                                        [r.policy for r in pool.records], env_cfg, rc,
                                        seed=cfg.seed + gen)
        snap = league.maybe_freeze(gen, type("P", (), {"params": mu})(), elite_scores[0])
        if snap:
            gen_info["frozen"] = snap["name"]
        history.append(gen_info)
        if cfg.verbose:
            ev = gen_info.get("eval", {})
            print(f"[gen {gen:2d}] best={elite_scores[0]:+8.2f} mean={gen_info['mean_return']:+8.2f} "
                  f"({gen_info['wall_s']:5.1f}s)"
                  + (f"  win={ev.get('win_rate', 0):.2f} hit%={100*ev.get('hit_rate', 0):.1f}" if ev else ""))

    result = {"best": best, "mu": mu, "sigma": sigma, "history": history,
              "pool": pool.summary()}
    if meta:
        result.update(meta)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "training.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    return result


def _eval_scenarios() -> List[Scenario]:
    """Held-out evaluation: the canonical human-instructor setups."""
    return list(CANONICAL_SETUPS.values())
