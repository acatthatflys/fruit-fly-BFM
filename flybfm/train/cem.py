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

Training improvements (2026-09):
  * Scenario distribution: 40% easy gunnery, 30% fully random, 30% defensive
    (defensive + topgun_break with jitter) — fly regularly sees genuinely
    defensive situations, not just neutral/easy.
  * Episode-length curriculum: 15-20s → 30-45s → 60-90s across generations,
    so evolution first learns acquisition/pursuit, then maintaining advantage
    and recovery.
  * Gunnery-focused curriculum (2026-09 fix for "wins on points, cannot shoot"):
    - scenario mix curriculum: early 70% easy / 10% defensive / 20% random,
      middle 40/30/30, late 20/40/40 — so learner first learns to track,
      then to handle defensive.
    - opponent curriculum: early only level/nose-on (non-maneuvering),
      middle level/guns/lag/lead, late all 10 scripted pilots — so learner
      first learns to hold 1° solution vs cooperative target, then vs maneuvering.
    - reward tuning: tracking term now lead_angle at 1° scale weight 8.0,
      dominates aspect/range/energy; hit 5.0, kill 100, good trigger +0.5,
      bad spray -0.3, points weight 0.35→0.05 — closes "win on points" loophole.
    - observation fine channels now lead/2° and aa/2° (was /10 and /5) — 5x gain.
  * Prioritize evaluations over length: thousands of fights, not few hundred.
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
    generations: int = 16
    population: int = 24
    elite_frac: float = 0.25
    sigma0: float = 0.5
    sigma_min: float = 0.05
    episodes_per_candidate: int = 8
    eval_every: int = 2
    max_time_s: float = 30.0
    decision_dt: float = 0.1
    seed: int = 0
    freeze_every: int = 3
    verbose: bool = True
    # scenario distribution and curriculum — gunnery-focused defaults
    easy_frac: float = 0.4
    defensive_frac: float = 0.3
    random_frac: float = 0.3
    use_curriculum: bool = True
    use_scenario_mix_curriculum: bool = True
    use_opponent_curriculum: bool = True
    curriculum: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.0, 15.0, 20.0),   # first 30% gens: 15-20s (acquisition & pursuit)
        (0.3, 30.0, 45.0),   # next 40%: 30-45s (skills emerge)
        (0.7, 60.0, 90.0),   # last 30%: 60-90s (maintain advantage, recovery)
    ])
    # scenario mix curriculum: (progress_start, easy, defensive, random)
    scenario_mix_curriculum: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (0.0, 0.7, 0.1, 0.2),  # early: 70% easy — learn to track vs cooperative
        (0.3, 0.4, 0.3, 0.3),  # middle: 40/30/30 balanced
        (0.7, 0.2, 0.4, 0.4),  # late: 20% easy, 40% defensive, 40% random — handle defensive
    ])
    # opponent curriculum: easy early, hard late
    opponent_curriculum: List[Tuple[float, List[str]]] = field(default_factory=lambda: [
        (0.0, ["level", "nose-on"]),  # early: non-maneuvering — learn 1° hold
        (0.3, ["level", "nose-on", "guns", "lag", "lead"]),  # middle: add maneuvering
        (0.7, ["level", "nose-on", "guns", "lag", "lead", "jinker", "bnz", "break", "vertical", "instructor"]),  # late: all 10
    ])
    # live training viewer support
    live_replay_path: Optional[str] = None
    live_status_path: Optional[str] = None
    live_every: int = 1
    live_max_time_s: float = 60.0


def _get_max_time_for_gen(gen: int, total_gens: int, cfg: CEMConfig, rng: random.Random) -> float:
    """Episode-length curriculum: sample max_time based on generation progress."""
    if not cfg.use_curriculum or not cfg.curriculum:
        return cfg.max_time_s
    progress = gen / max(total_gens - 1, 1)
    chosen = cfg.curriculum[0]
    for start, lo, hi in cfg.curriculum:
        if progress >= start:
            chosen = (start, lo, hi)
        else:
            break
    _, lo, hi = chosen
    return rng.uniform(lo, hi)


def _get_scenario_fractions_for_gen(gen: int, total_gens: int, cfg: CEMConfig) -> Tuple[float, float, float]:
    """Scenario mix curriculum: early more easy, late more defensive/random."""
    if not cfg.use_scenario_mix_curriculum or not cfg.scenario_mix_curriculum:
        return cfg.easy_frac, cfg.defensive_frac, cfg.random_frac
    progress = gen / max(total_gens - 1, 1)
    chosen = cfg.scenario_mix_curriculum[0]
    for entry in cfg.scenario_mix_curriculum:
        if progress >= entry[0]:
            chosen = entry
        else:
            break
    _, easy, defensive, rand = chosen
    tot = easy + defensive + rand
    if tot > 0:
        return easy / tot, defensive / tot, rand / tot
    return cfg.easy_frac, cfg.defensive_frac, cfg.random_frac


def _get_opponent_names_for_gen(gen: int, total_gens: int, cfg: CEMConfig) -> List[str]:
    """Opponent curriculum: easy early, hard late."""
    if not cfg.use_opponent_curriculum or not cfg.opponent_curriculum:
        return ["level", "nose-on", "guns", "lag", "lead", "jinker", "bnz", "break", "vertical", "instructor"]
    progress = gen / max(total_gens - 1, 1)
    chosen = cfg.opponent_curriculum[0]
    for entry in cfg.opponent_curriculum:
        if progress >= entry[0]:
            chosen = entry
        else:
            break
    return chosen[1]


class BalancedTrainingSpace:
    """Scenario sampler with 40% easy gunnery, 30% random, 30% defensive.

    Gunnery-focused curriculum (2026-09): supports dynamic fractions per gen
    via sample_with_fractions(), so early gens see 70% easy (learn to track
    vs cooperative target at 300m stern), late gens see 40% defensive.
    """

    def __init__(self, easy_frac: float = 0.4, defensive_frac: float = 0.3,
                 random_frac: float = 0.3, seed: int = 0,
                 defensive_tags: Tuple[str, ...] = ("defensive", "topgun_break")):
        total = easy_frac + defensive_frac + random_frac
        if total <= 0:
            easy_frac, defensive_frac, random_frac = 0.4, 0.3, 0.3
            total = 1.0
        self.easy_frac = easy_frac / total
        self.defensive_frac = defensive_frac / total
        self.random_frac = random_frac / total
        self.rng = random.Random(seed)
        self.space = ScenarioSpace()
        self.defensive_tags = defensive_tags

    def sample(self, rng=None, tag: str = "train") -> Scenario:
        rng = rng or self.rng
        return self.sample_with_fractions(rng, self.easy_frac, self.defensive_frac, self.random_frac, tag)

    def sample_with_fractions(self, rng, easy_frac: float, defensive_frac: float, random_frac: float, tag: str = "train") -> Scenario:
        roll = rng.random()
        if roll < easy_frac:
            return self._sample_easy(rng, tag)
        elif roll < easy_frac + defensive_frac:
            return self._sample_defensive(rng, tag)
        else:
            return self.space.sample(rng, tag=tag)

    def _sample_easy(self, rng: random.Random, tag: str) -> Scenario:
        # gunnery-focused: start closer, more aligned early — 250-900m ±30° as before,
        # but for very easy we bias to 250-500m ±15° half the time to give 1° gradient
        if rng.random() < 0.5:
            # super easy: 250-500m, ±15° — gives sparse gun event a strong gradient
            return Scenario(range_m=rng.uniform(250.0, 500.0),
                            taa_deg=rng.uniform(-15.0, 15.0),
                            nose_offset_deg=rng.uniform(-15.0, 15.0),
                            dz_m=rng.uniform(-80.0, 80.0),
                            alt_m=rng.uniform(4000.0, 7000.0),
                            v_a=rng.uniform(240.0, 300.0),
                            v_t=rng.uniform(220.0, 280.0),
                            phase_deg=rng.uniform(0.0, 360.0),
                            seed=rng.randrange(1 << 30), tag=tag + "_gunnery_easy")
        return Scenario(range_m=rng.uniform(250.0, 900.0),
                        taa_deg=rng.uniform(-30.0, 30.0),
                        nose_offset_deg=rng.uniform(-30.0, 30.0),
                        dz_m=rng.uniform(-150.0, 150.0),
                        alt_m=rng.uniform(4000.0, 9000.0),
                        v_a=rng.uniform(220.0, 320.0),
                        v_t=rng.uniform(200.0, 320.0),
                        phase_deg=rng.uniform(0.0, 360.0),
                        seed=rng.randrange(1 << 30), tag=tag + "_gunnery")

    def _sample_defensive(self, rng: random.Random, tag: str) -> Scenario:
        base_tag = rng.choice(self.defensive_tags)
        base = CANONICAL_SETUPS.get(base_tag)
        if base is None:
            base = CANONICAL_SETUPS["defensive"]
        taa = base.taa_deg + rng.gauss(0, 10.0)
        if base.taa_deg >= 90:
            taa = max(90.0, min(180.0, taa))
        else:
            taa = max(0.0, min(180.0, taa))

        return Scenario(
            range_m=max(300.0, base.range_m + rng.gauss(0, 200.0)),
            taa_deg=taa,
            nose_offset_deg=base.nose_offset_deg + rng.gauss(0, 15.0),
            dz_m=base.dz_m + rng.gauss(0, 120.0),
            alt_m=max(2500.0, min(9000.0, base.alt_m + rng.gauss(0, 400.0))),
            v_a=max(150.0, min(340.0, base.v_a + rng.gauss(0, 20.0))),
            v_t=max(150.0, min(340.0, base.v_t + rng.gauss(0, 20.0))),
            phase_deg=rng.uniform(0, 360),
            bank_a_deg=rng.uniform(-60, 60),
            gamma_a_deg=rng.uniform(-30, 30),
            gamma_t_deg=rng.uniform(-30, 30),
            seed=rng.randrange(1 << 30),
            tag=tag + f"_{base_tag}",
        )


class MixedSpace(BalancedTrainingSpace):
    """Backward compat: old name, now 40/30/30 by default."""

    def __init__(self, easy_frac: float = 0.4, seed: int = 0,
                 defensive_frac: float = 0.3, random_frac: float = 0.3):
        super().__init__(easy_frac=easy_frac, defensive_frac=defensive_frac,
                         random_frac=random_frac, seed=seed)


def train(policy_factory: Callable[[Sequence[float]], object],
          n_params: int,
          space: ScenarioSpace,
          pool: Optional[OpponentPool] = None,
          cfg: Optional[CEMConfig] = None,
          init_params: Optional[Sequence[float]] = None,
          out_dir: Optional[str] = None,
          meta: Optional[dict] = None) -> dict:
    """Optimise `n_params` numbers to maximise mean episodic return.

    Gunnery-focused defaults (2026-09 fix):
    16 gens × 24 pop × 8 eps = 3072 fights, 40/30/30 base but curriculum
    70/10/20→40/30/30→20/40/40, opponent curriculum level→all, episode
    15-20s→30-45s→60-90s, reward tracking 8.0 at 1° scale dominates.
    """
    cfg = cfg or CEMConfig()
    rng = random.Random(cfg.seed)
    pool = pool or OpponentPool(seed=cfg.seed)
    rc = RewardConfig()

    mu = list(init_params) if init_params else [0.0] * n_params
    sigma = [cfg.sigma0] * n_params
    n_elite = max(1, int(round(cfg.population * cfg.elite_frac)))
    league = League(freeze_every=cfg.freeze_every)
    history: List[dict] = []
    best = {"score": -1e9, "params": list(mu)}

    for gen in range(cfg.generations):
        t0 = time.time()
        # dynamic curricula for this gen
        easy_f, def_f, rand_f = _get_scenario_fractions_for_gen(gen, cfg.generations, cfg)
        opp_names = _get_opponent_names_for_gen(gen, cfg.generations, cfg)
        # filter pool records to allowed opponents for this gen
        allowed_recs = [r for r in pool.records if r.name in opp_names] or pool.records

        cand_params = [list(mu)]
        for _ in range(max(0, cfg.population - 1)):
            cand_params.append([mu[i] + rng.gauss(0.0, sigma[i]) for i in range(n_params)])
        scores: List[float] = []
        for params in cand_params:
            policy = policy_factory(params)
            total = 0.0
            for k in range(cfg.episodes_per_candidate):
                # scenario with curriculum fractions
                if isinstance(space, BalancedTrainingSpace):
                    sc = space.sample_with_fractions(rng, easy_f, def_f, rand_f, tag="train")
                else:
                    sc = space.sample(rng, tag="train")
                # opponent with curriculum
                rec = rng.choice(allowed_recs) if allowed_recs else pool.sample(rng)
                max_time = _get_max_time_for_gen(gen, cfg.generations, cfg, rng)
                env_cfg = EnvConfig(decision_dt=cfg.decision_dt, max_time_s=max_time)
                info, _ = run_episode(sc, policy, rec.policy, env_cfg, rc,
                                      learn=True, record=False)
                total += info["return"]
                if hasattr(rec.policy, "observe_reward"):
                    # observation now 20 dims (was 18) — use 20 for gunnery-focused
                    rec.policy.observe_reward((0.0,) * 20, 0.0)
                rec.register(info["result"])
            scores.append(total / cfg.episodes_per_candidate)

        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        elite = [cand_params[i] for i in order[:n_elite]]
        elite_scores = [scores[i] for i in order[:n_elite]]
        if elite_scores[0] > best["score"]:
            best = {"score": elite_scores[0], "params": list(cand_params[order[0]])}
        for i in range(n_params):
            vals = [e[i] for e in elite]
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / len(vals)
            sigma[i] = max(cfg.sigma_min, math.sqrt(v))
            mu[i] = m

        gen_info = {"generation": gen, "best_return": elite_scores[0],
                    "mean_return": sum(scores) / len(scores),
                    "wall_s": round(time.time() - t0, 2),
                    "max_time_s": round(_get_max_time_for_gen(gen, cfg.generations, cfg, random.Random(cfg.seed + gen)), 1),
                    "scenario_mix": {"easy": round(easy_f, 2), "defensive": round(def_f, 2), "random": round(rand_f, 2)},
                    "opponents": opp_names}
        if cfg.eval_every and gen % cfg.eval_every == 0:
            eval_cfg = EnvConfig(decision_dt=cfg.decision_dt, max_time_s=60.0)
            gen_info["eval"] = evaluate(policy_factory(best["params"]), _eval_scenarios(),
                                        [r.policy for r in pool.records], eval_cfg, rc,
                                        seed=cfg.seed + gen)
        snap = league.maybe_freeze(gen, type("P", (), {"params": mu})(), elite_scores[0])
        if snap:
            gen_info["frozen"] = snap["name"]
        history.append(gen_info)
        if cfg.verbose:
            ev = gen_info.get("eval", {})
            print(f"[gen {gen:2d}] best={elite_scores[0]:+8.2f} mean={gen_info['mean_return']:+8.2f} "
                  f"({gen_info['wall_s']:5.1f}s, max_time={gen_info['max_time_s']:.0f}s, "
                  f"mix {easy_f:.1f}/{def_f:.1f}/{rand_f:.1f} opp {len(opp_names)}) "
                  + (f" win={ev.get('win_rate', 0):.2f} hit%={100*ev.get('hit_rate', 0):.1f} hits={ev.get('hits',0)}" if ev else ""))

        # live status + live replay for viewer while training
        try:
            if cfg.live_status_path:
                os.makedirs(os.path.dirname(cfg.live_status_path) or ".", exist_ok=True)
                live_status = {
                    "generation": gen,
                    "total_generations": cfg.generations,
                    "best_return": elite_scores[0],
                    "mean_return": gen_info["mean_return"],
                    "best": best,
                    "history": history[-10:],
                    "pool": pool.summary(),
                    "config": {"generations": cfg.generations, "population": cfg.population,
                               "episodes_per_candidate": cfg.episodes_per_candidate,
                               "easy_frac": easy_f, "defensive_frac": def_f,
                               "random_frac": rand_f, "opponents": opp_names},
                    "eval": gen_info.get("eval"),
                    "timestamp": time.time(),
                }
                tmp = cfg.live_status_path + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(live_status, fh, indent=2)
                os.replace(tmp, cfg.live_status_path)
            if cfg.live_replay_path and (gen % cfg.live_every == 0 or gen == cfg.generations - 1):
                best_policy = policy_factory(best["params"])
                sc_tag = "live"
                # live replay uses current curriculum fractions
                if isinstance(space, BalancedTrainingSpace):
                    sc = space.sample_with_fractions(rng, easy_f, def_f, rand_f, tag=sc_tag)
                else:
                    sc = space.sample(rng, tag=sc_tag)
                opp_rec = rng.choice(allowed_recs) if allowed_recs else pool.sample(rng)
                env_cfg_live = EnvConfig(decision_dt=cfg.decision_dt, max_time_s=cfg.live_max_time_s)
                _, _, env_live = run_match(sc, best_policy, opp_rec.policy, env_cfg_live, record=True)
                replay = env_live.to_replay()
                replay["blue_name"] = f"gen{gen}_best"
                replay["red_name"] = opp_rec.name
                replay["generation"] = gen
                replay["scenario"]["desc"] = f"live gen {gen} best vs {opp_rec.name} — {sc.describe()} — mix {easy_f:.1f}/{def_f:.1f}/{rand_f:.1f}"
                try:
                    from ..analysis.maneuvers import classify_trajectory
                    replay["manoeuvres"] = classify_trajectory(env_live.frames).summary()
                except Exception:
                    pass
                try:
                    from ..analysis.metrics import episode_metrics
                    replay["metrics"] = episode_metrics(env_live).as_dict()
                except Exception:
                    pass
                os.makedirs(os.path.dirname(cfg.live_replay_path) or ".", exist_ok=True)
                tmp = cfg.live_replay_path + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(replay, fh, separators=(",", ":"))
                os.replace(tmp, cfg.live_replay_path)
        except Exception as exc:
            if cfg.verbose:
                print(f"[live] warning: failed to write live files: {exc}")

    result = {"best": best, "mu": mu, "sigma": sigma, "history": history,
              "pool": pool.summary(),
              "config": {"generations": cfg.generations, "population": cfg.population,
                         "episodes_per_candidate": cfg.episodes_per_candidate,
                         "easy_frac": cfg.easy_frac, "defensive_frac": cfg.defensive_frac,
                         "random_frac": cfg.random_frac, "use_curriculum": cfg.use_curriculum,
                         "use_scenario_mix_curriculum": cfg.use_scenario_mix_curriculum,
                         "use_opponent_curriculum": cfg.use_opponent_curriculum,
                         "curriculum": cfg.curriculum,
                         "scenario_mix_curriculum": cfg.scenario_mix_curriculum,
                         "opponent_curriculum": cfg.opponent_curriculum}}
    if meta:
        result.update(meta)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "training.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        if not cfg.live_replay_path:
            try:
                best_policy = policy_factory(best["params"])
                sc = _eval_scenarios()[0]
                opp = pool.records[0].policy if pool.records else None
                if opp:
                    _, _, env_final = run_match(sc, best_policy, opp,
                                                EnvConfig(decision_dt=cfg.decision_dt, max_time_s=60.0),
                                                record=True)
                    replay = env_final.to_replay()
                    replay["blue_name"] = "best"
                    replay["red_name"] = pool.records[0].name if pool.records else "opponent"
                    with open(os.path.join(out_dir, "best_replay.json"), "w") as fh:
                        json.dump(replay, fh, separators=(",", ":"))
            except Exception:
                pass
    return result


def _eval_scenarios() -> List[Scenario]:
    return list(CANONICAL_SETUPS.values())
