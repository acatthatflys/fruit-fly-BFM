"""Warm-start a linear policy by imitating a rule-based BFM pilot.

Why this exists: black-box search over a 76-parameter policy from a random
initialisation spends most of its budget rediscovering "roll to put the target
on the lift vector, then pull".  A rule-based pilot already knows that, so the
cheap thing to do is record what it does, fit the linear map in closed form, and
then let evolutionary search improve on the result.

This is also the honest analogue of what the connectome demos do: the fly brain
(or here, the scripted pilot) supplies the prior, and a small learned decoder
supplies the competence.

The fit is ridge-regularised least squares, done with Gaussian elimination so it
needs no numpy:

    min_W  || Phi W - C||^2 + lambda ||W||^2
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
    """Gaussian elimination with partial pivoting. `a` is modified in place."""
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
    """Return w minimising ||Phi w - y||^2 + lam||w||^2 (Phi needs a bias column)."""
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
#: command channels: roll, pull, throttle, trigger-logit
N_TARGETS = 4


def collect(teachers: Sequence[str] = ("guns", "lead", "lag", "break", "vertical", "bnz"),
            n_episodes: int = 24, space: Optional[ScenarioSpace] = None,
            env_cfg: Optional[EnvConfig] = None, seed: int = 0,
            record_every: int = 1, verbose: bool = False):
    """Fly scripted pilots against each other and record (observation, command).

    Both sides are recorded, from their own point of view, which doubles the
    data and covers both offensive and defensive geometry.
    """
    cfg = env_cfg or EnvConfig(decision_dt=0.2, max_time_s=25.0)
    rc = RewardConfig()
    space = space or ScenarioSpace()
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
                 opponents: Sequence[str] = ("level", "jinker", "lag", "lead", "bnz",
                                             "break", "vertical", "instructor"),
                 space: Optional[ScenarioSpace] = None,
                 env_cfg: Optional[EnvConfig] = None, seed: int = 0,
                 record_every: int = 1, verbose: bool = False):
    """Record one pilot's commands from its own seat, against many opponents.

    Never pool both seats into one regression: the two sides are flying
    *different control laws*, so a fit to the union learns their average, and the
    average of "turn left" and "turn right" is "fly straight".  Measured: pooling
    both sides of six teachers produced a warm start that flew like the random
    policy (median angle-off 150 deg, zero wins in 18 fights).
    """
    from ..sim.scripted import POOL_BY_NAME

    cfg = env_cfg or EnvConfig(decision_dt=0.2, max_time_s=25.0)
    space = space or ScenarioSpace()
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


def fit_policy(teachers: Sequence[str] = ("guns", "lead", "lag", "break", "vertical", "bnz"),
               n_episodes: int = 24, space: Optional[ScenarioSpace] = None,
               env_cfg: Optional[EnvConfig] = None, seed: int = 0,
               lam: float = 1e-3, verbose: bool = False, basis: str = "linear"):
    """Return a policy whose parameters clone the scripted teachers.

    `basis="linear"` fits the 18 features directly.  `basis="poly"` fits the
    quadratic expansion, which is what actually works: the teachers' laws are
    products (roll ~ lateral error, pull ~ vertical error x range), and a linear
    readout cannot represent them -- measured command-space error was 0.43 / 0.67
    (roll/pull) for the linear clone, i.e. the clone barely flies.
    """
    from .policy import FeaturePolicy, PolyPolicy, quad_features

    if len(teachers) == 1:
        X, Y = collect_from(teachers[0], n_episodes, space=space, env_cfg=env_cfg,
                            seed=seed, verbose=verbose)
    else:
        X, Y = collect(teachers, n_episodes, space, env_cfg, seed, verbose=verbose)
    if basis == "poly":
        rows = [quad_features(x) for x in X]
        policy_cls = PolyPolicy
    else:
        rows = [[1.0] + list(x) for x in X]          # bias column
        policy_cls = FeaturePolicy
    params: List[float] = []
    for o in range(N_TARGETS):
        params.extend(ridge_fit(rows, [y[o] for y in Y], lam))
    return policy_cls(params, name="imitation-%s" % basis)


def clone_report(policy, teachers=("guns", "lead"), n: int = 6, seed: int = 0) -> str:
    """How good is this warm start, judged the only way that counts.

    Two numbers, and the first exists only to stop the second from being
    over-trusted:

    * *skill over a constant predictor* on held-out samples.  Raw command-space
      MAE is a trap: the scripted pilots hold the trigger down 2% of the time, so
      "never fire" scores 0.98 agreement while aiming nothing, and a policy whose
      output is near the mean command can post a small MAE while flying nothing
      like the teacher.
    * *closed-loop result*: a handful of real fights.  A clone that cannot fly is
      not a warm start, however small its regression error is.
    """
    from ..sim.arena import ScenarioSpace
    from ..sim.dogfight import run_match
    from ..sim.scripted import POOL_BY_NAME

    X, Y = collect(teachers, n_episodes=n, seed=seed, verbose=False,
                   env_cfg=EnvConfig(decision_dt=0.2, max_time_s=20.0))
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

    rng = random.Random(seed + 7)
    space = ScenarioSpace()
    res = {"blue": 0, "red": 0, "draw": 0}
    hits = shots = 0
    aa: List[float] = []
    for k in range(6):
        sc = space.sample(rng, tag="clone-check")
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
