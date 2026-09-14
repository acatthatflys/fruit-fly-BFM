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
import platform
import random
import sys
import time
import subprocess
import signal
from http import HTTPStatus
import http.server
import socketserver
import multiprocessing

from .analysis.metrics import episode_metrics, format_scorecard, scorecard
from .sim.arena import CANONICAL_SETUPS, Scenario, ScenarioSpace
from .sim.dogfight import Dogfight, EnvConfig, run_match
from .sim.reward import RewardConfig
from .sim.scripted import POOL_BY_NAME, default_pool


# --------------------------------------------------------------------- helpers
def _policy(name: str, seed: int = 0, params=None, basis: str = "linear",
            use_torch: bool = False, torch_device: str | None = None):
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
                             readout=ReadoutParams.from_vector(params) if params else None,
                             use_torch=use_torch, torch_device=torch_device)
        from .train.policy import BrainPolicyAdapter
        return BrainPolicyAdapter(c)
    raise SystemExit(f"unknown policy '{name}' (try: {', '.join(sorted(POOL_BY_NAME))}, "
                     f"random, features, brain)")


def _scenario(tag: str, seed: int, random_space: bool) -> Scenario:
    if random_space:
        return ScenarioSpace().sample(random.Random(seed), tag="random")
    if tag in CANONICAL_SETUPS:
        s = CANONICAL_SETUPS[tag]
        return Scenario(**{**s.__dict__, "seed": seed})
    raise SystemExit(f"unknown tag '{tag}'; known: {', '.join(CANONICAL_SETUPS)}")


def _bench_time_per_fight(n_fights: int = 8, max_time_s: float = 20.0, decision_dt: float = 0.1) -> dict:
    """Quick benchmark: run n_fights with random vs level to estimate fight cost."""
    from .train.policy import RandomPolicy
    import time as _time
    space = ScenarioSpace()
    rng = random.Random(0)
    env_cfg = EnvConfig(max_time_s=max_time_s, decision_dt=decision_dt)
    rc = RewardConfig()
    blue = RandomPolicy(seed=0)
    red = POOL_BY_NAME["level"](seed=1)
    times = []
    for i in range(n_fights):
        sc = space.sample(rng, tag="bench")
        t0 = _time.time()
        # run_match does full env
        run_match(sc, blue, red, env_cfg, rc, record=False)
        times.append(_time.time() - t0)
    avg = sum(times) / len(times) if times else 0.05
    return {
        "n_fights": n_fights,
        "max_time_s": max_time_s,
        "decision_dt": decision_dt,
        "avg_fight_s": avg,
        "fights_per_second": (1.0 / avg) if avg > 0 else 0,
        "total_fight_time_s": sum(times),
        "cpu_count": multiprocessing.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


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
    basis = getattr(args, "basis", "linear")
    # n_params now 84 linear (20*4+4) or 396 poly (99*4) — compute from proto later
    n_params = {"features": 84, "brain": 10}.get(args.policy, 84)
    if args.policy == "brain":
        from .train.policy import BrainPolicyAdapter
        from .brain.controller import build_controller
        use_torch = getattr(args, "use_torch", False)
        torch_device = getattr(args, "torch_device", None)
        ctl = build_controller("synthetic", seed=args.seed, use_torch=use_torch, torch_device=torch_device)
        proto = BrainPolicyAdapter(ctl)
        factory = lambda p: BrainPolicyAdapter(
            build_controller("synthetic", seed=args.seed, use_torch=use_torch, torch_device=torch_device)
        ).with_params(p)
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

    easy_frac = getattr(args, "easy_frac", 0.4)
    defensive_frac = getattr(args, "defensive_frac", 0.3)
    random_frac = getattr(args, "random_frac", 0.3)
    tot = easy_frac + defensive_frac + random_frac
    if tot > 0:
        easy_frac /= tot
        defensive_frac /= tot
        random_frac /= tot

    cfg = CEMConfig(generations=args.generations, population=args.population,
                    episodes_per_candidate=args.episodes, max_time_s=args.max_time,
                    decision_dt=args.decision_dt, seed=args.seed,
                    eval_every=args.eval_every,
                    easy_frac=easy_frac, defensive_frac=defensive_frac, random_frac=random_frac,
                    use_curriculum=not getattr(args, "no_curriculum", False),
                    use_scenario_mix_curriculum=not getattr(args, "no_scenario_mix_curriculum", False),
                    use_opponent_curriculum=not getattr(args, "no_opponent_curriculum", False),
                    live_replay_path=getattr(args, "live_replay", None),
                    live_status_path=getattr(args, "live_status", None),
                    live_every=getattr(args, "live_every", 1),
                    live_max_time_s=getattr(args, "live_max_time", 60.0))
    if args.sigma0 is not None:
        cfg.sigma0 = args.sigma0
    total_fights = cfg.generations * cfg.population * cfg.episodes_per_candidate
    print(f"training {args.policy} ({basis} basis): {n_params} parameters, "
          f"{cfg.generations} gens x {cfg.population} pop x {cfg.episodes_per_candidate} ep = {total_fights} fights")
    print(f"scenario mix: easy {100*cfg.easy_frac:.0f}% gunnery (250-900m ±30°, super-easy 250-500m ±15° half), "
          f"defensive {100*cfg.defensive_frac:.0f}% (defensive/topgun_break jitter), "
          f"random {100*cfg.random_frac:.0f}% fully random")
    if cfg.use_curriculum:
        print(f"curriculum: episode 15-20s→30-45s→60-90s, scenario mix 70/10/20→40/30/30→20/40/40, opponent level→all")
        print(f"  reward: tracking lead_angle 1° weight 12.0 dominates, hit 10 kill 100, good trigger +2.0 bad -0.05, points 0.05, obs 20d lead AA direction + tanh fine")
    else:
        print(f"curriculum: off, fixed max_time {cfg.max_time_s}s")
    if cfg.live_replay_path:
        print(f"live replay -> {cfg.live_replay_path}, status -> {cfg.live_status_path}")

    # quick bench for estimated time
    try:
        bench = _bench_time_per_fight(n_fights=4, max_time_s=20.0, decision_dt=cfg.decision_dt)
        est_s = total_fights * bench["avg_fight_s"]
        # curriculum makes early fights shorter (15-20s) so ~0.7x avg
        est_s *= 0.75
        print(f"bench: avg fight {bench['avg_fight_s']*1000:.0f}ms ({bench['fights_per_second']:.1f} fights/s) on {bench['cpu_count']} CPUs")
        print(f"estimated training time: {est_s/60:.1f} min ({est_s:.0f}s) for {total_fights} fights — prioritizes evals over length")
    except Exception as e:
        print(f"bench failed: {e}")

    from .train.cem import BalancedTrainingSpace
    space = BalancedTrainingSpace(easy_frac=cfg.easy_frac, defensive_frac=cfg.defensive_frac,
                                  random_frac=cfg.random_frac, seed=args.seed)
    res = train(factory, n_params, space, pool, cfg, init_params=init, out_dir=out_dir,
                meta={"basis": basis, "init": args.init,
                      "easy_frac": cfg.easy_frac, "defensive_frac": cfg.defensive_frac,
                      "random_frac": cfg.random_frac})
    print(f"best mean episode return: {res['best']['score']:+.2f}")
    print("opponent pool after training:")
    for row in res["pool"]:
        print(f"  {row['name']:12s} episodes={row['episodes']:3d} "
              f"learner_wins={row['learner_wins']:3d} opponent_wins={row['opponent_wins']:3d} "
              f"importance={row['importance']:.2f}")
    print(f"artifacts -> {out_dir}/training.json")
    try:
        if cfg.live_status_path:
            pid_path = os.path.join(os.path.dirname(cfg.live_status_path), "pid.txt")
            if os.path.exists(pid_path):
                with open(pid_path) as pf:
                    try:
                        pid_in_file = int(pf.read().strip())
                        if pid_in_file == os.getpid():
                            os.remove(pid_path)
                    except Exception:
                        pass
    except Exception:
        pass
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
    from .tools.gunnery_check import gunnery_check
    args.params = args.checkpoint or os.path.join("runs", "features", "training.json")
    return gunnery_check(args)


def cmd_probe(args) -> int:
    from .tools.probe_brain import probe_brain
    return probe_brain(args)


def cmd_replay(args) -> int:
    return _rebuild_manifest(args.dir)


# ------------------------------------------------------------------ serve with live training API + bench
_bench_cache = None
_bench_time = 0

def _get_bench():
    global _bench_cache, _bench_time
    now = time.time()
    if _bench_cache is None or now - _bench_time > 30:
        try:
            _bench_cache = _bench_time_per_fight(n_fights=6, max_time_s=20.0, decision_dt=0.1)
            _bench_time = now
        except Exception as e:
            _bench_cache = {"error": str(e), "avg_fight_s": 0.05, "fights_per_second": 20,
                            "cpu_count": multiprocessing.cpu_count(), "platform": platform.platform()}
            _bench_time = now
    return _bench_cache


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            for line in fh:
                if line.startswith("State:"):
                    if "Z" in line or "X" in line or "zombie" in line.lower():
                        return False
                    break
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            if not fh.read().strip():
                return False
    except Exception:
        pass
    return True


def _read_live_status(status_path: str, pid_path: str, log_path: str) -> dict:
    running = False
    pid = None
    if os.path.exists(pid_path):
        try:
            with open(pid_path) as fh:
                pid = int(fh.read().strip())
            running = _is_pid_alive(pid)
            if not running:
                try:
                    os.remove(pid_path)
                except Exception:
                    pass
                pid = None
        except Exception:
            pid = None
            running = False
    status = {}
    if os.path.exists(status_path):
        try:
            with open(status_path) as fh:
                status = json.load(fh)
        except Exception as e:
            status = {"error": f"failed to read status: {e}"}
    return {"running": running, "pid": pid, "status": status,
            "log_exists": os.path.exists(log_path),
            "status_path": status_path}


class LiveTrainingHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, web_dir=None, runs_dir=None, **kwargs):
        self.web_dir = web_dir or "web"
        self.runs_dir = runs_dir or "runs/live"
        super().__init__(*args, directory=self.web_dir, **kwargs)

    def log_message(self, format, *args):
        sys.stderr.write(f"{self.client_address[0]} - - [{self.log_date_time_string()}] {format % args}\n")

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._handle_api_get()
        else:
            return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._handle_api_post()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_api_get(self):
        if self.path.startswith("/api/train/status"):
            status_path = os.path.join(self.runs_dir, "status.json")
            pid_path = os.path.join(self.runs_dir, "pid.txt")
            log_path = os.path.join(self.runs_dir, "train.log")
            data = _read_live_status(status_path, pid_path, log_path)
            self._send_json(data)
            return
        if self.path.startswith("/api/train/live"):
            live_path = os.path.join(self.web_dir, "replays", "live.json")
            alt_path = os.path.join(self.runs_dir, "live.json")
            path = live_path if os.path.exists(live_path) else alt_path
            if not os.path.exists(path):
                self._send_json({"error": "no live replay yet", "path": path}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                with open(path) as fh:
                    data = json.load(fh)
                self._send_json(data)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if self.path.startswith("/api/train/log"):
            log_path = os.path.join(self.runs_dir, "train.log")
            lines = 200
            try:
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                if "lines" in qs:
                    lines = int(qs["lines"][0])
            except Exception:
                pass
            if not os.path.exists(log_path):
                self._send_json({"log": "", "error": "no log file"})
                return
            try:
                with open(log_path, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    block = 8192
                    data = b""
                    while len(data.splitlines()) <= lines and size > 0:
                        read_size = min(block, size)
                        size -= read_size
                        fh.seek(size)
                        data = fh.read(read_size) + data
                        if size == 0:
                            break
                    text = data.decode(errors="ignore")
                    out_lines = text.splitlines()[-lines:]
                self._send_json({"log": "\n".join(out_lines)})
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if self.path.startswith("/api/bench"):
            bench = _get_bench()
            # also estimate for current live config if available
            status_path = os.path.join(self.runs_dir, "status.json")
            try:
                if os.path.exists(status_path):
                    with open(status_path) as fh:
                        st = json.load(fh)
                    cfg = st.get("config", {})
                    gens = cfg.get("generations", 12)
                    pop = cfg.get("population", 24)
                    eps = cfg.get("episodes_per_candidate", 8)
                    bench["current_estimated_total_s"] = gens * pop * eps * bench["avg_fight_s"] * 0.75
            except Exception:
                pass
            self._send_json(bench)
            return
        if self.path.startswith("/api/system"):
            self._send_json({
                "cpu_count": multiprocessing.cpu_count(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "bench": _get_bench(),
            })
            return
        if self.path.startswith("/api/replays"):
            idx_path = os.path.join(self.web_dir, "replays", "index.json")
            if os.path.exists(idx_path):
                with open(idx_path) as fh:
                    self._send_json(json.load(fh))
            else:
                self._send_json({"replays": []})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "API endpoint not found")

    def _handle_api_post(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if self.path.startswith("/api/train/start"):
            status_path = os.path.join(self.runs_dir, "status.json")
            pid_path = os.path.join(self.runs_dir, "pid.txt")
            log_path = os.path.join(self.runs_dir, "train.log")
            if os.path.exists(pid_path):
                try:
                    with open(pid_path) as fh:
                        pid = int(fh.read().strip())
                    if _is_pid_alive(pid):
                        self._send_json({"error": "training already running", "pid": pid},
                                        status=HTTPStatus.CONFLICT)
                        return
                except Exception:
                    pass
            os.makedirs(self.runs_dir, exist_ok=True)
            os.makedirs(os.path.join(self.web_dir, "replays"), exist_ok=True)

            generations = int(data.get("generations", 16))
            population = int(data.get("population", 24))
            episodes = int(data.get("episodes", 8))
            policy = data.get("policy", "features")
            basis = data.get("basis", "poly")
            seed = int(data.get("seed", 0))
            easy_frac = float(data.get("easy_frac", 0.4))
            defensive_frac = float(data.get("defensive_frac", 0.3))
            random_frac = float(data.get("random_frac", 0.3))
            max_time = float(data.get("max_time", 30.0))
            no_curriculum = bool(data.get("no_curriculum", False))
            no_scenario_mix = bool(data.get("no_scenario_mix_curriculum", False))
            no_opponent = bool(data.get("no_opponent_curriculum", False))

            live_replay = os.path.abspath(os.path.join(self.web_dir, "replays", "live.json"))
            live_status = os.path.abspath(status_path)
            runs_abs = os.path.abspath(self.runs_dir)

            cmd = [
                sys.executable, "-u", "-m", "flybfm", "train",
                "--policy", policy,
                "--basis", basis,
                "--generations", str(generations),
                "--population", str(population),
                "--episodes", str(episodes),
                "--seed", str(seed),
                "--easy-frac", str(easy_frac),
                "--defensive-frac", str(defensive_frac),
                "--random-frac", str(random_frac),
                "--max-time", str(max_time),
                "--out", runs_abs,
                "--live-replay", live_replay,
                "--live-status", live_status,
                "--live-every", "1",
            ]
            if no_curriculum:
                cmd.append("--no-curriculum")
            if no_scenario_mix:
                cmd.append("--no-scenario-mix-curriculum")
            if no_opponent:
                cmd.append("--no-opponent-curriculum")
            if data.get("init"):
                cmd.extend(["--init", str(data["init"])])

            try:
                log_fh = open(log_path, "w")
                proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                        start_new_session=True, cwd=os.getcwd())
                with open(pid_path, "w") as pf:
                    pf.write(str(proc.pid))
                self._send_json({"started": True, "pid": proc.pid, "cmd": " ".join(cmd),
                                 "log": log_path, "live_replay": live_replay})
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path.startswith("/api/train/stop"):
            pid_path = os.path.join(self.runs_dir, "pid.txt")
            if not os.path.exists(pid_path):
                self._send_json({"error": "no pid file, not running"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                with open(pid_path) as fh:
                    pid = int(fh.read().strip())
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if _is_pid_alive(pid):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except Exception:
                        os.kill(pid, signal.SIGKILL)
                try:
                    os.remove(pid_path)
                except Exception:
                    pass
                self._send_json({"stopped": True, "pid": pid})
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "API endpoint not found")

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def cmd_serve(args) -> int:
    web_dir = args.dir
    runs_dir = getattr(args, "runs_dir", os.path.join("runs", "live"))
    os.makedirs(web_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(os.path.join(web_dir, "replays"), exist_ok=True)

    handler = lambda *a, **kw: LiveTrainingHandler(*a, web_dir=web_dir, runs_dir=runs_dir, **kw)
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"serving {os.path.abspath(web_dir)} on http://{args.host}:{args.port}/")
        print(f"  live training API: POST http://{args.host}:{args.port}/api/train/start")
        print(f"  status: GET http://{args.host}:{args.port}/api/train/status")
        print(f"  live replay: GET http://{args.host}:{args.port}/api/train/live")
        print(f"  log tail: GET http://{args.host}:{args.port}/api/train/log?lines=200")
        print(f"  bench: GET http://{args.host}:{args.port}/api/bench")
        print(f"  training out dir: {os.path.abspath(runs_dir)}")
        print(f"  training survives browser close (runs in terminal/server process)")
        print(f"  Ctrl-C to stop server (training subprocess will keep running unless stopped via API)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nserver stopped")
    return 0


# ------------------------------------------------------------------- plumbing
def _load_checkpoint(path):
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
    t.add_argument("--generations", type=int, default=16,
                   help="number of generations (default 16, ~3072 fights with gunnery-focused defaults)")
    t.add_argument("--population", type=int, default=24,
                   help="candidates per generation (default 24)")
    t.add_argument("--episodes", type=int, default=8,
                   help="fights per candidate (default 8, prioritize evals over length)")
    t.add_argument("--max-time", type=float, default=30.0,
                   help="max time per episode when curriculum off (default 30s)")
    t.add_argument("--decision-dt", type=float, default=0.1)
    t.add_argument("--eval-every", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init", default="imitation", choices=["imitation", "random"])
    t.add_argument("--basis", default="poly", choices=["linear", "poly"],
                   help="action basis; the linear one cannot aim (see docs/EXPERIMENTS.md 5)")
    t.add_argument("--easy-frac", type=float, default=0.4, dest="easy_frac",
                   help="fraction easy gunnery starts 250-900m ±30° (default 0.4)")
    t.add_argument("--defensive-frac", type=float, default=0.3, dest="defensive_frac",
                   help="fraction defensive starts from defensive/topgun_break + jitter (default 0.3)")
    t.add_argument("--random-frac", type=float, default=0.3, dest="random_frac",
                   help="fraction fully random starts (default 0.3)")
    t.add_argument("--no-curriculum", action="store_true",
                   help="disable episode-length curriculum (15-20s→30-45s→60-90s)")
    t.add_argument("--no-scenario-mix-curriculum", action="store_true",
                   help="disable scenario mix curriculum 70/10/20→40/30/30→20/40/40")
    t.add_argument("--no-opponent-curriculum", action="store_true",
                   help="disable opponent curriculum level→all")
    t.add_argument("--live-replay", default=None,
                   help="write live best-vs-random replay each gen for viewer (e.g. web/replays/live.json)")
    t.add_argument("--live-status", default=None,
                   help="write live status json each gen (e.g. runs/live/status.json)")
    t.add_argument("--live-every", type=int, default=1,
                   help="how often to write live replay (generations)")
    t.add_argument("--live-max-time", type=float, default=60.0,
                   help="max time for live replay fights")
    t.add_argument("--resume", default=None,
                   help="continue from a training.json (keeps its basis; ignores --init)")
    t.add_argument("--sigma0", type=float, default=None,
                   help="initial CEM step size (default 0.5, smaller when resuming)")
    t.add_argument("--imitation-episodes", type=int, default=24)
    t.add_argument("--out", default=None)
    t.add_argument("--include-frozen", action="store_true", default=True)
    t.add_argument("--use-torch", action="store_true", help="use TorchLIFNetwork (CPU/CUDA) for brain policy — required for full CNS")
    t.add_argument("--torch-device", default=None, help="torch device: cpu, cuda, cuda:0, etc (auto-detect if omitted)")
    t.add_argument("--plasticity", action="store_true", help="enable inner KC->MBON dopamine-gated plasticity (off by default; 0%% of reported results use it)")
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
    pr.add_argument("--use-torch", action="store_true", help="use TorchLIFNetwork (CPU/CUDA) — ~10-100x faster, needed for full CNS")
    pr.add_argument("--torch-device", default=None, help="torch device: cpu, cuda, etc")
    pr.add_argument("--plasticity", action="store_true", help="enable KC->MBON plasticity (off by default)")
    pr.set_defaults(func=cmd_probe)

    rp = sub.add_parser("replay", help="rebuild the web viewer manifest")
    rp.add_argument("--dir", default=os.path.join("web", "replays"))
    rp.set_defaults(func=cmd_replay)

    sv = sub.add_parser("serve", help="serve the replay viewer + live training API")
    sv.add_argument("--dir", default="web", help="web root to serve (default web)")
    sv.add_argument("--runs-dir", default=os.path.join("runs", "live"), help="dir for live training status/log")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="0.0.0.0")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
