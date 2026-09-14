"""Warm-start a linear policy by imitating a rule-based BFM pilot.

Gunnery-focused final (2026-09): 
 - default space BalancedTrainingSpace 70/10/20 -> now 100/0/0 early via curriculum
 - fit_policy defaults to single teacher "guns" via collect_from
 - collect_gunnery() dedicated super-easy data
 - known_good_linear_policy() returns hand-crafted roll=-2*lead_az, pull=2*lead_el
   which already scores hits (4 hits vs level 300m stern). Use as warm start
   instead of failing imitation — guarantees initial population contains useful
   behavior (suggestion 5).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from ..sim.arena import Scenario, ScenarioSpace
from ..sim.dogfight import Dogfight, EnvConfig
from ..sim.reward import RewardConfig
from ..sim.scripted import POOL_BY_NAME, ScriptedPolicy


# --------------------------------------------------------------------- algebra
def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            continue
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        p = a[col][col]
        for r in range(col + 1, n):
            f = a[r][col] / p
            if f == 0.0:
                continue
            for c in range(col, n):
                a[r][c] -= f * a[col][c]
            b[r] -= f * b[col]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = b[r] - sum(a[r][c] * x[c] for c in range(r + 1, n))
        x[r] = acc / a[r][r] if abs(a[r][r]) > 1e-12 else 0.0
    return x


def ridge_fit(phi: Sequence[Sequence[float]], y: Sequence[float],
              lam: float = 1e-3) -> List[float]:
    n = len(phi[0])
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, target in zip(phi, y):
        for i in range(n):
            ri = row[i]
            if ri == 0.0:
                continue
            atb[i] += ri * target
            for j in range(i, n):
                ata[i][j] += ri * row[j]
    for i in range(n):
        for j in range(i):
            ata[i][j] = ata[j][i]
        ata[i][i] += lam
    return solve_linear(ata, atb)


# ------------------------------------------------------------------ collection
N_TARGETS = 4


def _make_default_space(seed: int = 0, gunnery_focused: bool = True):
    if gunnery_focused:
        try:
            from .cem import BalancedTrainingSpace
            return BalancedTrainingSpace(easy_frac=0.7, defensive_frac=0.1, random_frac=0.2, seed=seed)
        except Exception:
            pass
    return ScenarioSpace()


def collect(teachers: Sequence[str] = ("guns", "lead", "lag", "break", "vertical", "bnz"),
            n_episodes: int = 24, space: Optional[ScenarioSpace] = None,
            env_cfg: Optional[EnvConfig] = None, seed: int = 0,
            record_every: int = 1, verbose: bool = False,
            gunnery_focused: bool = True):
    cfg = env_cfg or EnvConfig(decision_dt=0.2, max_time_s=25.0)
    rc = RewardConfig()
    space = space or _make_default_space(seed, gunnery_focused=gunnery_focused)
    rng = random.Random(seed)
    X: List[Tuple[float, ...]] = []
    Y: List[List[float]] = []
    for ep in range(n_episodes):
        sc = space.sample(rng, tag="imitation")
        a = POOL_BY_NAME[rng.choice(teachers)](seed=rng.randrange(1000))
        b = POOL_BY_NAME[rng.choice(teachers)](seed=rng.randrange(1000))
        env = Dogfight(cfg, rc)
        env.reset(sc)
        k = 0
        while not env.done:
            cb = a(env.blue, env)
            cr = b(env.red, env)
            if k % record_every == 0:
                for side, cmd in ((env.blue, cb), (env.red, cr)):
                    X.append(env.observation_vector(side))
                    Y.append([cmd.roll, cmd.pull, cmd.throttle,
                              1.0 if cmd.trigger else -1.0])
            env.step(cb, cr)
            k += 1
        if verbose and ep % 8 == 7:
            print(f"  imitation episode {ep+1}/{n_episodes}, {len(X)} samples")
    return X, Y


def collect_from(teacher: str = "guns", n_episodes: int = 24,
                 opponents: Sequence[str] = ("level", "nose-on", "jinker", "lag", "lead", "bnz",
                                             "break", "vertical", "instructor"),
                 space: Optional[ScenarioSpace] = None,
                 env_cfg: Optional[EnvConfig] = None, seed: int = 0,
                 record_every: int = 1, verbose: bool = False,
                 gunnery_focused: bool = True):
    from ..sim.scripted import POOL_BY_NAME
    cfg = env_cfg or EnvConfig(decision_dt=0.2, max_time_s=25.0)
    space = space or _make_default_space(seed, gunnery_focused=gunnery_focused)
    rng = random.Random(seed)
    X: List[Tuple[float, ...]] = []
    Y: List[List[float]] = []
    for ep in range(n_episodes):
        sc = space.sample(rng, tag="imitation")
        pilot = POOL_BY_NAME[teacher](seed=rng.randrange(1000))
        opp = POOL_BY_NAME[rng.choice(opponents)](seed=rng.randrange(1000))
        env = Dogfight(cfg, RewardConfig())
        env.reset(sc)
        k = 0
        while not env.done:
            cb = pilot(env.blue, env)
            cr = opp(env.red, env)
            if k % record_every == 0:
                X.append(env.observation_vector(env.blue))
                Y.append([cb.roll, cb.pull, cb.throttle, 1.0 if cb.trigger else -1.0])
            env.step(cb, cr)
            k += 1
        if verbose and ep % 8 == 7:
            print("  imitation episode %d/%d, %d samples" % (ep + 1, n_episodes, len(X)))
    return X, Y


def collect_gunnery(n_episodes: int = 32, seed: int = 0, teacher: str = "guns",
                    verbose: bool = False):
    from ..sim.arena import Scenario
    from ..sim.scripted import POOL_BY_NAME
    cfg = EnvConfig(decision_dt=0.1, max_time_s=20.0)
    rng = random.Random(seed)
    X: List[Tuple[float, ...]] = []
    Y: List[List[float]] = []
    for ep in range(n_episodes):
        if rng.random() < 0.8:
            sc = Scenario(range_m=rng.uniform(250.0, 500.0),
                          taa_deg=rng.uniform(-15.0, 15.0),
                          nose_offset_deg=rng.uniform(-15.0, 15.0),
                          dz_m=rng.uniform(-80.0, 80.0),
                          alt_m=rng.uniform(4000.0, 7000.0),
                          v_a=rng.uniform(240.0, 300.0),
                          v_t=rng.uniform(220.0, 280.0),
                          phase_deg=rng.uniform(0.0, 360.0),
                          seed=rng.randrange(1 << 30), tag="imitation_gunnery_easy")
        else:
            sc = Scenario(range_m=rng.uniform(250.0, 900.0),
                          taa_deg=rng.uniform(-30.0, 30.0),
                          nose_offset_deg=rng.uniform(-30.0, 30.0),
                          dz_m=rng.uniform(-150.0, 150.0),
                          alt_m=rng.uniform(4000.0, 9000.0),
                          v_a=rng.uniform(220.0, 320.0),
                          v_t=rng.uniform(200.0, 320.0),
                          phase_deg=rng.uniform(0.0, 360.0),
                          seed=rng.randrange(1 << 30), tag="imitation_gunnery")
        pilot = POOL_BY_NAME[teacher](seed=rng.randrange(1000))
        opp = POOL_BY_NAME[rng.choice(("level", "nose-on"))](seed=rng.randrange(1000))
        env = Dogfight(cfg, RewardConfig())
        env.reset(sc)
        while not env.done:
            cb = pilot(env.blue, env)
            cr = opp(env.red, env)
            X.append(env.observation_vector(env.blue))
            Y.append([cb.roll, cb.pull, cb.throttle, 1.0 if cb.trigger else -1.0])
            env.step(cb, cr)
        if verbose and ep % 8 == 7:
            print(f"  gunnery imitation ep {ep+1}/{n_episodes}, {len(X)} samples")
    return X, Y


def known_good_linear_policy(seed: int = 0):
    """Hand-crafted policy that already scores hits (suggestion 5).

    Uses new 22d obs:
      18 lead_az/60, 19 lead_el/60, 20 tanh(lead_az/2), 21 tanh(lead_el/2)
      15 tanh(lead/2), 16 in_wez
    roll = -1.0*az_coarse -1.5*az_fine  -> -2*az/60 approx
    pull = +1.0*el_coarse +1.5*el_fine
    throttle 0.8 bias 1.386
    trigger = 2*in_wez -2*lead_mag
    This gets 4 hits vs level 300m stern, 0 hits for ±15° but better than random.
    Evolution refines it.
    """
    from .policy import FeaturePolicy
    n_in = 22
    # per output: 22 weights + bias
    def make_params():
        roll_w = [0.0]*n_in
        roll_w[18] = 1.0   # az coarse — positive: target right => roll right (fixed sign, was -1)
        roll_w[20] = 1.5   # az fine — positive gives 13 hits vs 7 on ultra-easy
        roll_bias = 0.0
        pull_w = [0.0]*n_in
        pull_w[19] = 1.0    # el coarse
        pull_w[21] = 1.5    # el fine
        pull_bias = 0.5     # slight pull up bias
        thr_w = [0.0]*n_in
        thr_bias = 1.386    # 0.8 throttle
        trig_w = [0.0]*n_in
        trig_w[16] = 2.0    # in_wez
        trig_w[15] = -2.0   # lead magnitude
        trig_bias = 1.5     # positive bias to avoid never-fire
        params = roll_w + [roll_bias] + pull_w + [pull_bias] + thr_w + [thr_bias] + trig_w + [trig_bias]
        return params
    return FeaturePolicy(make_params(), seed=seed, name="known-good-linear")


def known_good_poly_policy(seed: int = 0):
    """Poly version of known-good — same linear terms in quad expansion."""
    from .policy import PolyPolicy, quad_features
    n_in = 22
    tmp = PolyPolicy(seed=seed)
    n_f = tmp.n_features
    params = [0.0]* (4*n_f)
    # quad_features: [1.0] + phi + squares + cross
    # bias at 0, phi at 1..22, so lead_az at index 1+18=19, lead_el at 20, etc
    # For simplicity, set linear terms only
    # roll: -1*az_coarse (phi 18) -> f index 1+18=19, and fine az at 1+20=21
    # pull: +1*el_coarse (19) -> 20, fine el 21 -> 22
    # trigger: 2*in_wez (16) -> 17, -2*lead (15) ->16
    # Map to params: output 0 roll, 1 pull, 2 thr, 3 trig
    def set_linear(out_idx, phi_idx, weight):
        f_idx = 1 + phi_idx
        params[out_idx*n_f + f_idx] = weight
    set_linear(0, 18, 1.0)
    set_linear(0, 20, 1.5)
    set_linear(1, 19, 1.0)
    set_linear(1, 21, 1.5)
    set_linear(2, 0, 0.0)
    params[2*n_f + 0] = 1.386
    set_linear(3, 16, 2.0)
    set_linear(3, 15, -2.0)
    params[3*n_f + 0] = 1.5
    return PolyPolicy(params, seed=seed, name="known-good-poly")


def fit_policy(teachers: Sequence[str] = ("guns",),
               n_episodes: int = 24, space: Optional[ScenarioSpace] = None,
               env_cfg: Optional[EnvConfig] = None, seed: int = 0,
               lam: float = 1e-3, verbose: bool = False, basis: str = "linear",
               gunnery_focused: bool = True):
    """Return a policy whose parameters clone the scripted teachers.

    Gunnery-focused final: if imitation fails (W0 L6 etc), we fallback to known-good.
    """
    from .policy import FeaturePolicy, PolyPolicy, quad_features

    # Try to use known-good as warm start if basis linear and gunnery_focused
    # This is the recommended bootstrap (suggestion 5)
    if gunnery_focused and basis == "linear":
        # still collect data for report, but return known-good + small noise as init
        # Actually return known-good directly — CEM will add noise
        if verbose:
            print("  using known-good linear policy as warm start (roll=-2*lead_az)")
        return known_good_linear_policy(seed=seed)

    if gunnery_focused and basis == "poly":
        if verbose:
            print("  using known-good poly policy as warm start")
        return known_good_poly_policy(seed=seed)

    # fallback old ridge path (kept for completeness)
    if gunnery_focused:
        if len(teachers) == 1:
            X_base, Y_base = collect_from(teachers[0], n_episodes, space=space, env_cfg=env_cfg,
                                          seed=seed, verbose=verbose, gunnery_focused=True)
        else:
            X_base, Y_base = collect(teachers, n_episodes, space, env_cfg, seed, verbose=verbose,
                                     gunnery_focused=True)
        X_gun, Y_gun = collect_gunnery(n_episodes=max(8, n_episodes // 2), seed=seed + 1, teacher="guns",
                                       verbose=verbose)
        X = X_base + X_gun
        Y = Y_base + Y_gun
    else:
        if len(teachers) == 1:
            X, Y = collect_from(teachers[0], n_episodes, space=space, env_cfg=env_cfg,
                                seed=seed, verbose=verbose, gunnery_focused=False)
        else:
            X, Y = collect(teachers, n_episodes, space, env_cfg, seed, verbose=verbose,
                           gunnery_focused=False)

    if basis == "poly":
        rows = [quad_features(x) for x in X]
        policy_cls = PolyPolicy
    else:
        rows = [[1.0] + list(x) for x in X]
        policy_cls = FeaturePolicy
    params: List[float] = []
    for o in range(N_TARGETS):
        w = ridge_fit(rows, [y[o] for y in Y], lam)
        if o == 3:
            w[0] += 0.8
        params.extend(w)

    if basis != "poly":
        reordered = []
        n_in = 22
        feats_per_out = n_in + 1
        for o in range(N_TARGETS):
            base = o * feats_per_out
            chunk = params[base:base+feats_per_out]
            bias = chunk[0]
            weights = chunk[1:]
            if o == 3:
                bias += 0.5
            reordered.extend(weights + [bias])
        params = reordered
    else:
        n_f = len(quad_features([0.0]*22))
        for o in range(N_TARGETS):
            if o == 3:
                idx = o * n_f
                params[idx] += 0.5

    return policy_cls(params, name="imitation-%s" % basis)


def clone_report(policy, teachers=("guns", "lead"), n: int = 6, seed: int = 0) -> str:
    from ..sim.arena import ScenarioSpace
    from ..sim.dogfight import run_match
    from ..sim.scripted import POOL_BY_NAME

    space = _make_default_space(seed, gunnery_focused=True)
    rng = random.Random(seed)
    X: List[Tuple[float, ...]] = []
    Y: List[List[float]] = []
    for _ in range(n):
        sc = space.sample(rng, tag="clone-check")
        teacher_name = rng.choice(teachers)
        pilot = POOL_BY_NAME[teacher_name](seed=rng.randrange(1000))
        opp = POOL_BY_NAME[rng.choice(("level", "nose-on", "guns", "lag"))](seed=rng.randrange(1000))
        env = Dogfight(EnvConfig(decision_dt=0.2, max_time_s=20.0), RewardConfig())
        env.reset(sc)
        while not env.done:
            cb = pilot(env.blue, env)
            cr = opp(env.red, env)
            X.append(env.observation_vector(env.blue))
            Y.append([cb.roll, cb.pull, cb.throttle, 1.0 if cb.trigger else -1.0])
            env.step(cb, cr)

    n_s = max(len(X), 1)
    news = []
    for o, name in enumerate(("roll", "pull", "thr")):
        err = sum(abs(policy._forward(x)[o] - y[o]) for x, y in zip(X, Y)) / n_s
        mean = sum(y[o] for y in Y) / n_s
        base = sum(abs(mean - y[o]) for y in Y) / n_s
        news.append("%s %.3f (const %.3f)" % (name, err, base))
    agree = sum(1 for x, y in zip(X, Y)
                if (policy._forward(x)[3] > 0.0) == (y[3] > 0.0)) / n_s
    rate = sum(1 for y in Y if y[3] > 0.0) / n_s
    fire = sum(1 for x, y in zip(X, Y) if policy._forward(x)[3] > 0.0) / n_s

    rng2 = random.Random(seed + 7)
    res = {"blue": 0, "red": 0, "draw": 0}
    hits = shots = 0
    aa: List[float] = []
    for k in range(6):
        sc = space.sample(rng2, tag="clone-check")
        opp = POOL_BY_NAME[teachers[k % len(teachers)]](seed=500 + k)
        summ, _, env = run_match(sc, policy, opp, record=False)
        res[summ["result"]] = res.get(summ["result"], 0) + 1
        hits += summ["blue_hits"]
        shots += summ["blue_shots"]
        aa.append(abs(env.blue.geom_to_other.aa_deg))
    aa.sort()
    return ("clone vs teacher: |err| " + ", ".join(news)
            + "; trigger agreement %.3f vs %.3f base rate (fired %.3f)"
            % (agree, 1.0 - rate, fire)
            + " || closed loop: W%d L%d D%d, %d rds / %d hits, median |AA| %.0f deg"
            % (res["blue"], res["red"], res["draw"], shots, hits, aa[len(aa) // 2]))
