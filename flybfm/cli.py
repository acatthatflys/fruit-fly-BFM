"""Command line entry point.

    python -m flybfm fight   --blue guns --red instructor --tag head_on --replay out.json
    python -m flybfm train   --policy features --generations 6 --population 10
    python -m flybfm eval    --checkpoint runs/features/training.json
    python -m flybfm probe   --brain synthetic          # what does the brain encode?
    python -m flybfm league                              # round-robin over the pool
    python -m flybfm replay  --dir web/replays           # rebuild the viewer manifest
    python -m flybfm serve   --port 8000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

from .analysis.metrics import episode_metrics, format_scorecard, scorecard
from .sim.arena import CANONICAL_SETUPS, Scenario, ScenarioSpace
from .sim.dogfight import Dogfight, EnvConfig, run_match
from .sim.reward import RewardConfig
from .sim.scripted import POOL_BY_NAME, default_pool


# --------------------------------------------------------------------- helpers
def _policy(name: str, seed: int = 0, params=None, basis: str = "linear"):
    if name in POOL_BY_NAME:
        return POOL_BY_NAME[name](seed=seed)
    if name == "random":
        from .train.policy import RandomPolicy
        return RandomPolicy(seed=seed)
    if name in ("features", "poly"):
        from .train.policy import FeaturePolicy, PolyPolicy
        if name == "poly" or basis == "poly":
            return PolyPolicy(params=params, seed=seed) if params else PolyPolicy(seed=seed)
        return FeaturePolicy(params=params, seed=seed) if params else FeaturePolicy(seed=seed)
    if name in ("brain", "connectome"):
        from .brain.controller import build_controller, ReadoutParams
        c = build_controller("synthetic", seed=seed,
                             readout=ReadoutParams.from_vector(params) if params else None)
        from .train.policy import BrainPolicyAdapter
        return BrainPolicyAdapter(c)
    raise SystemExit(f"unknown policy '{name}' (try: {', '.join(sorted(POOL_BY_NAME))}, "
                     f"random, features, brain)")


def _scenario(tag: str, seed: int, random_space: bool) -> Scenario:
    # --random wins over --tag: it is the whole point of the flag
    if random_space:
        return ScenarioSpace().sample(random.Random(seed), tag="random")
    if tag in CANONICAL_SETUPS:
        s = CANONICAL_SETUPS[tag]
        return Scenario(**{**s.__dict__, "seed": seed})
    raise SystemExit(f"unknown tag '{tag}'; known: {', '.join(CANONICAL_SETUPS)}")


# ------------------------------------------------------------------- commands
def cmd_fight(args) -> int:
    bck, rck = _load_checkpoint(args.blue_ckpt), _load_checkpoint(args.red_ckpt)
    blue = _policy(args.blue, seed=1, params=bck["params"], basis=bck["basis"])
    red = _policy(args.red, seed=2, params=rck["params"], basis=rck["basis"])
    scen = _scenario(args.tag, args.seed, args.random)
    cfg = EnvConfig(decision_dt=args.decision_dt, max_time_s=args.max_time)
    info, trace, env = run_match(scen, blue, red, cfg)
    print(f"scenario: {scen.describe()}")
    print(f"result:   {env.result}   (t={env.t:.1f}s)")
    print(json.dumps(env.summary(), indent=2))
    m = episode_metrics(env)
    print(format_scorecard(scorecard([m]), "scorecard"))
    for ev in env.event_log[:12]:
        print(f"  [{ev['t']:6.1f}s] {ev['msg']}")
    if args.replay:
        _write_replay(args.replay, env, blue_name=args.blue, red_name=args.red)
        print(f"replay -> {args.replay}")
    return 0


def cmd_train(args) -> int:
    from .train.cem import CEMConfig, train
    from .train.league import OpponentPool
    out_dir = args.out or os.path.join("runs", args.policy)
    pool = OpponentPool(seed=args.seed)
    if not args.include_frozen:
        pass
    basis = getattr(args, "basis", "linear")
    n_params = {"features": 4 * 19, "brain": 10}.get(args.policy)
    if args.policy == "brain":
        from .train.policy import BrainPolicyAdapter
        from .brain.controller import build_controller
        ctl = build_controller("synthetic", seed=args.seed)
        proto = BrainPolicyAdapter(ctl)
        factory = lambda p: BrainPolicyAdapter(build_controller("synthetic", seed=args.seed)).with_params(p)
        init = proto.params
    elif args.policy == "features":
        from .train.policy import FeaturePolicy, PolyPolicy
        resume = getattr(args, "resume", None)
        if resume:
            ck = _load_checkpoint(resume)
            basis = ck["basis"]
            init = ck["params"]
            print(f"resuming from {resume} ({basis} basis, {len(init)} parameters)")
        proto = PolyPolicy(seed=args.seed) if basis == "poly" else FeaturePolicy(seed=args.seed)
        n_params = proto.n_params
        factory = (lambda p: PolyPolicy(p)) if basis == "poly" else (lambda p: FeaturePolicy(p))
        if resume:
            pass
        elif args.init == "imitation":
            from .train.imitation import fit_policy, clone_report
            print("warm start: cloning the rule-based pilots ...")
            teacher = fit_policy(n_episodes=args.imitation_episodes, seed=args.seed,
                                 basis=basis)
            print("  " + clone_report(teacher, seed=args.seed + 1))
            init = list(teacher.params)
        else:
            init = None
    else:
        raise SystemExit("policy must be 'features' or 'brain'")
    if args.policy == "brain":
        init = globals().get("init", None)
    cfg = CEMConfig(generations=args.generations, population=args.population,
                    episodes_per_candidate=args.episodes, max_time_s=args.max_time,
                    decision_dt=args.decision_dt, seed=args.seed,
                    eval_every=args.eval_every)
    if args.sigma0 is not None:
        cfg.sigma0 = args.sigma0
    print(f"training {args.policy} ({basis} basis): {n_params} parameters, "
          f"{cfg.generations} generations x {cfg.population} candidates x "
          f"{cfg.episodes_per_candidate} fights")
    space = ScenarioSpace()
    if getattr(args, "easy_frac", 0.0) > 0.0:
        from .train.cem import MixedSpace
        space = MixedSpace(easy_frac=args.easy_frac, seed=args.seed)
        print(f"curriculum: {100*args.easy_frac:.0f}% of training starts are gunnery-range starts")
    res = train(factory, n_params, space, pool, cfg, init_params=init, out_dir=out_dir,
                meta={"basis": basis, "init": args.init,
                      "easy_frac": getattr(args, "easy_frac", 0.0)})
    print(f"best mean episode return: {res['best']['score']:+.2f}")
    print("opponent pool after training:")
    for row in res["pool"]:
        print(f"  {row['name']:12s} episodes={row['episodes']:3d} "
              f"learner_wins={row['learner_wins']:3d} opponent_wins={row['opponent_wins']:3d} "
              f"importance={row['importance']:.2f}")
    print(f"artifacts -> {out_dir}/training.json")
    return 0


def cmd_eval(args) -> int:
    from .train.cem import evaluate
    from .train.league import OpponentPool
    params = _load_params(args.checkpoint)
    policy = _policy(args.policy, seed=args.seed, params=params)
    scenarios = [s for s in CANONICAL_SETUPS.values()] if args.canonical else \
        [ScenarioSpace().sample(random.Random(i), tag="heldout") for i in range(args.n)]
    pool = OpponentPool(seed=args.seed)
    opps = [pool.records[i].policy for i in range(min(args.opponents, len(pool.records)))]
    t0 = time.time()
    res = evaluate(policy, scenarios, opps, EnvConfig(max_time_s=args.max_time), RewardConfig(),
                   seed=args.seed)
    print(f"evaluation: {res['episodes']} fights vs {len(opps)} opponents, {time.time()-t0:.1f}s")
    print(json.dumps(res, indent=2))
    return 0


def cmd_league(args) -> int:
    """Round-robin the scripted pool: is the pool actually discriminating?"""
    import itertools
    names = list(POOL_BY_NAME)
    rng = random.Random(args.seed)
    space = ScenarioSpace()
    table = {n: {"w": 0, "l": 0, "d": 0} for n in names}
    for a, b in itertools.combinations(names, 2):
        for k in range(args.per_pair):
            sc = space.sample(rng, tag="league")
            summ, _, _ = run_match(sc, POOL_BY_NAME[a](seed=k), POOL_BY_NAME[b](seed=100 + k),
                                   record=False)
            if summ["result"] == "blue":
                table[a]["w"] += 1; table[b]["l"] += 1
            elif summ["result"] == "red":
                table[b]["w"] += 1; table[a]["l"] += 1
            else:
                table[a]["d"] += 1; table[b]["d"] += 1
    print(f"{'policy':12s} {'W':>4s} {'L':>4s} {'D':>4s}  win%")
    for n, r in sorted(table.items(), key=lambda kv: -(kv[1]["w"])):
        tot = max(r["w"] + r["l"] + r["d"], 1)
        print(f"{n:12s} {r['w']:4d} {r['l']:4d} {r['d']:4d}  {100*r['w']/tot:5.1f}")
    return 0


def cmd_gunnery(args) -> int:
    """Does the learner ever put the pipper on the target?  Win rate cannot tell you."""
    from .tools.gunnery_check import gunnery_check
    args.params = args.checkpoint or os.path.join("runs", "features", "training.json")
    return gunnery_check(args)


def cmd_probe(args) -> int:
    from .tools.probe_brain import probe_brain
    return probe_brain(args)


def cmd_replay(args) -> int:
    return _rebuild_manifest(args.dir)


def cmd_serve(args) -> int:
    import http.server
    import socketserver
    os.chdir(args.dir)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"serving {os.getcwd()} on http://{args.host}:{args.port}/ (Ctrl-C to stop)")
        httpd.serve_forever()
    return 0


# ------------------------------------------------------------------- plumbing
def _load_checkpoint(path):
    """Return {params, basis} from a training.json."""
    if not path:
        return {"params": None, "basis": "linear"}
    with open(path) as fh:
        data = json.load(fh)
    params = None
    if isinstance(data, dict):
        if "best" in data:
            params = data["best"]["params"]
        elif "params" in data:
            params = data["params"]
    if params is None:
        raise SystemExit(f"{path}: expected a training.json with 'best.params'")
    basis = data.get("basis", "linear") if isinstance(data, dict) else "linear"
    return {"params": params, "basis": basis}


def _load_params(path):
    return _load_checkpoint(path)["params"]


def _write_replay(path: str, env, blue_name: str = "blue", red_name: str = "red"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = env.to_replay()
    data["red_name"] = red_name
    data["blue_name"] = blue_name
    from .analysis.maneuvers import classify_trajectory
    stats = classify_trajectory(env.frames)
    data["manoeuvres"] = stats.summary()
    m = episode_metrics(env)
    data["metrics"] = m.as_dict()
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    _rebuild_manifest(os.path.dirname(path) or ".")


def _rebuild_manifest(directory: str) -> int:
    os.makedirs(directory, exist_ok=True)
    entries = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json") or fn == "index.json":
            continue
        p = os.path.join(directory, fn)
        try:
            with open(p) as fh:
                d = json.load(fh)
            entries.append({"file": fn, "tag": d.get("scenario", {}).get("tag", "?"),
                            "result": d.get("result", "?"),
                            "duration_s": d.get("duration_s", 0),
                            "blue": d.get("blue_name", "blue"), "red": d.get("red_name", "red"),
                            "frames": len(d.get("frames", [])),
                            "manoeuvres": d.get("manoeuvres", {}).get("manoeuvre_content", 0)})
        except Exception as exc:
            print(f"  skipping {fn}: {exc}")
    with open(os.path.join(directory, "index.json"), "w") as fh:
        json.dump({"replays": entries}, fh, indent=2)
    print(f"manifest: {len(entries)} replay(s) in {directory}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flybfm", description="Connectome-constrained BFM")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fight", help="run one 1v1 guns-only engagement")
    f.add_argument("--blue", default="guns")
    f.add_argument("--red", default="instructor")
    f.add_argument("--tag", default="head_on")
    f.add_argument("--random", action="store_true", help="ignore --tag, sample a random start")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--max-time", type=float, default=90.0)
    f.add_argument("--decision-dt", type=float, default=0.1)
    f.add_argument("--blue-ckpt", default=None)
    f.add_argument("--red-ckpt", default=None)
    f.add_argument("--replay", default=None)
    f.set_defaults(func=cmd_fight)

    t = sub.add_parser("train", help="train a policy by CEM/ES")
    t.add_argument("--policy", default="features", choices=["features", "brain"])
    t.add_argument("--generations", type=int, default=6)
    t.add_argument("--population", type=int, default=10)
    t.add_argument("--episodes", type=int, default=3)
    t.add_argument("--max-time", type=float, default=30.0)
    t.add_argument("--decision-dt", type=float, default=0.1)
    t.add_argument("--eval-every", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init", default="imitation", choices=["imitation", "random"])
    t.add_argument("--basis", default="poly", choices=["linear", "poly"],
                   help="action basis; the linear one cannot aim (see docs/EXPERIMENTS.md 5)")
    t.add_argument("--easy-frac", type=float, default=0.5, dest="easy_frac",
                   help="fraction of training starts drawn near the gun envelope")
    t.add_argument("--resume", default=None,
                   help="continue from a training.json (keeps its basis; ignores --init)")
    t.add_argument("--sigma0", type=float, default=None,
                   help="initial CEM step size (default 0.5, smaller when resuming)")
    t.add_argument("--imitation-episodes", type=int, default=24)
    t.add_argument("--out", default=None)
    t.add_argument("--include-frozen", action="store_true", default=True)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="evaluate a checkpoint on held-out setups")
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--policy", default="features")
    e.add_argument("--canonical", action="store_true", default=True)
    e.add_argument("--n", type=int, default=16)
    e.add_argument("--opponents", type=int, default=4)
    e.add_argument("--max-time", type=float, default=45.0)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_eval)

    l = sub.add_parser("league", help="round-robin the scripted pool")
    l.add_argument("--per-pair", type=int, default=3)
    l.add_argument("--seed", type=int, default=0)
    l.set_defaults(func=cmd_league)

    gn = sub.add_parser("gunnery", help="measure a policy's aiming error (win rate hides it)")
    gn.add_argument("--checkpoint", default=None,
                    help="training.json; defaults to runs/features/training.json")
    gn.add_argument("--opponent", default="level")
    gn.add_argument("--episodes", type=int, default=6)
    gn.add_argument("--range", type=float, default=300.0, help="start range, m")
    gn.add_argument("--taa", type=float, default=0.0, help="0 = start astern of the target")
    gn.add_argument("--max-time", type=float, default=30.0)
    gn.add_argument("--seed", type=int, default=1000)
    gn.set_defaults(func=cmd_gunnery)

    pr = sub.add_parser("probe", help="measure what the connectome brain encodes")
    pr.add_argument("--brain", default="synthetic", choices=["synthetic", "malecns"])
    pr.add_argument("--seed", type=int, default=2)
    pr.add_argument("--steps", type=int, default=250)
    pr.add_argument("--out", default=None)
    pr.set_defaults(func=cmd_probe)

    rp = sub.add_parser("replay", help="rebuild the web viewer manifest")
    rp.add_argument("--dir", default=os.path.join("web", "replays"))
    rp.set_defaults(func=cmd_replay)

    sv = sub.add_parser("serve", help="serve the replay viewer")
    sv.add_argument("--dir", default="web")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="0.0.0.0")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
