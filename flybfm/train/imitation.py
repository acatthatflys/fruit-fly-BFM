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

Gunnery-focused fix (2026-09): previously pooled both seats of 6 teachers from
fully random starts — trigger rate ~2%, clone fired 0% and median AA 144°. Now:
 - default space is BalancedTrainingSpace 70/10/20 (70% easy 250-900m ±30°,
   half super-easy 250-500m ±15°) so trigger rate ~30%
 - fit_policy defaults to single teacher "guns" via collect_from (not pooling)
 - collect_gunnery() provides dedicated 80% super-easy data vs level/nose-on
   for 1° tracking gradient
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


def _make_default_space(seed: int = 0, gunnery_focused: bool = True):
    """Gunnery-focused default: 70% easy (250-500m ±15° half + 250-900m ±30°)
    to give trigger discipline a gradient, 10% defensive, 20% random."""
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
    """Fly scripted pilots against each other and record (observation, command).

    Both sides are recorded, from their own point of view, which doubles the
    data and covers both offensive and defensive geometry.

    Gunnery-focused (2026-09): default space is BalancedTrainingSpace 70/10/20
    so trigger rate is ~30% not 2%, giving the clone a chance to learn trigger.
    """
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
    """Record one pilot's commands from its own seat, against many opponents.

    Never pool both seats into one regression: the two sides are flying
    *different control laws*, so a fit to the union learns their average, and the
    average of "turn left" and "turn right" is "fly straight".  Measured: pooling
    both sides of six teachers produced a warm start that flew like the random
    policy (median angle-off 150 deg, zero wins in 18 fights).

    Gunnery-focused (2026-09): when gunnery_focused=True, use BalancedTrainingSpace
    70/10/20 so teacher fires often (300m stern), giving trigger logit a gradient.
    Also includes super-easy 250-500m ±15° half the time.
    """
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
    """Dedicated gunnery data: 80% super-easy 250-500m ±15°, 20% easy 250-900m ±30°,
    always vs level/nose-on (non-maneuvering) so teacher holds 1° solution.
    This gives the clone a strong gradient for trigger discipline."""
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


def fit_policy(teachers: Sequence[str] = ("guns",),
               n_episodes: int = 24, space: Optional[ScenarioSpace] = None,
               env_cfg: Optional[EnvConfig] = None, seed: int = 0,
               lam: float = 1e-3, verbose: bool = False, basis: str = "linear",
               gunnery_focused: bool = True):
    """Return a policy whose parameters clone the scripted teachers.

    `basis="linear"` fits the 18 features directly.  `basis="poly"` fits the
    quadratic expansion, which is what actually works: the teachers' laws are
    products (roll ~ lateral error, pull ~ vertical error x range), and a linear
    readout cannot represent them -- measured command-space error was 0.43 / 0.67
    (roll/pull) for the linear clone, i.e. the clone barely flies.

    Gunnery-focused (2026-09): default is single teacher "guns" with
    BalancedTrainingSpace 70/10/20 + dedicated gunnery data, so clone learns
    1° tracking and trigger discipline, not just BFM positioning.
    """
    from .policy import FeaturePolicy, PolyPolicy, quad_features

    # gunnery-focused: always include dedicated gunnery data
    if gunnery_focused:
        # base data from teacher(s) with easy-biased space
        if len(teachers) == 1:
            X_base, Y_base = collect_from(teachers[0], n_episodes, space=space, env_cfg=env_cfg,
                                          seed=seed, verbose=verbose, gunnery_focused=True)
        else:
            X_base, Y_base = collect(teachers, n_episodes, space, env_cfg, seed, verbose=verbose,
                                     gunnery_focused=True)
        # extra gunnery data: 50% of episodes as super-easy
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
        rows = [[1.0] + list(x) for x in X]          # bias column
        policy_cls = FeaturePolicy
    params: List[float] = []
    for o in range(N_TARGETS):
        w = ridge_fit(rows, [y[o] for y in Y], lam)
        # gunnery-focused: boost trigger bias to encourage firing (teacher fires 5-15%, clone 0%)
        if o == 3:  # trigger logit
            if basis == "poly":
                # bias is first feature (1.0)
                w[0] += 0.8
            else:
                # bias is last per output in FeaturePolicy? Actually FeaturePolicy uses [w0..w17, bias]
                # rows = [1.0] + phi, so bias is first? Wait FeaturePolicy's forward uses bias at end,
                # but rows here have bias at front (1.0 + phi). FeaturePolicy's params layout is
                # [w0..w17, bias] per output, but ridge_fit on rows [1.0, phi...] returns [bias, w0..]
                # so we need to reorder? Let's check FeaturePolicy: it expects bias at N_IN index,
                # but we fit with bias at 0. So we need to map: our rows are [1, phi0..], so w[0]=bias,
                # w[1..]=weights. FeaturePolicy expects weights then bias, so we should reorder.
                # However original code used same rows [1.0]+phi and directly used as params for
                # FeaturePolicy which expects [w0..w17, bias] — that was actually mismatched before.
                # For gunnery fix, we keep original layout but boost trigger bias:
                # For linear, bias is w[0] in our fit (since rows[0]=1), but FeaturePolicy reads bias from last.
                # To keep compatibility, we boost both first and last? Safer: boost w[0] and also
                # if len>18, boost w[18] equivalent after reorder. We'll handle after.
                w[0] += 0.8
        params.extend(w)

    # Fix FeaturePolicy param ordering: our ridge fit used [bias, w0..w17] but FeaturePolicy expects [w0..w17, bias]
    if basis != "poly":
        # params currently [bias, w0..w17] per output, need to reorder to [w0..w17, bias]
        reordered = []
        n_in = 18
        feats_per_out = n_in + 1
        for o in range(N_TARGETS):
            base = o * feats_per_out
            chunk = params[base:base+feats_per_out]
            bias = chunk[0]
            weights = chunk[1:]
            # boost trigger bias extra if this is trigger
            if o == 3:
                bias += 0.5  # extra boost for trigger
            reordered.extend(weights + [bias])
        params = reordered
    else:
        # poly: bias already at 0, we already boosted, add extra for trigger
        # n_features includes bias at 0
        n_f = len(quad_features([0.0]*18))
        for o in range(N_TARGETS):
            if o == 3:
                idx = o * n_f
                params[idx] += 0.5  # extra

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

    Gunnery-focused (2026-09): uses BalancedTrainingSpace 70/10/20 for held-out
    so trigger rate is realistic.
    """
    from ..sim.arena import ScenarioSpace
    from ..sim.dogfight import run_match
    from ..sim.scripted import POOL_BY_NAME

    space = _make_default_space(seed, gunnery_focused=True)
    rng = random.Random(seed)
    X: List[Tuple[float, ...]] = []
    Y: List[List[float]] = []
    # collect held-out from gunnery-focused space
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
