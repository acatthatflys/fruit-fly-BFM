# fruit-fly-BFM — can a fly brain learn basic fighter manoeuvres?

This repo is a **connectome-constrained BFM simulator**: a leaky integrate-and-fire network over the *Drosophila* CNS (166k neurons, ~125M synapses in the real MaleCNS v1.0 release, 665 neurons / 2.2k synapses in the in-repo synthetic stand-in) drives a point-mass energy fighter (F-16-like EM model) through a 12 km guns-only arena at 50 Hz physics / 10 Hz decisions. The opponent pool has ten scripted pilots (level, guns, lead, lag, break, vertical, boom-and-zoom, jinker, instructor, nose-on) plus PFSP self-play and a CEM/ES outer learner.

Short answer on feasibility: **yes**, you can build a full-brain-scale LIF simulation, drive it with a synthetic retina, and read out descending neurons to fly an aircraft — that runs today. **No**, you should not expect the fly brain itself to invent energy management, lag/lead pursuit switching, or the high yo-yo. In this repo the fly brain is a fixed nonlinear reservoir and the thing that actually learns is a small readout policy (6–296 parameters) trained by cross-entropy method; the inner mushroom-body dopamine-gated plasticity (`KC → MBON`, `Δw = η·e·DA − λw`) is implemented, has 66 plastic edges in the synthetic graph, but is **off by default** and inert unless you pass `plasticity=True` and load a real mushroom-body subgraph — 100% of the results in `runs/` come from the outer CEM loop. That's intentional and documented.

Design, reward shaping, data sources, measured results, and the full roadmap with pass/fail gates are in [`docs/DESIGN.md`](docs/DESIGN.md). Setup at each allocation level (tiny slice via neuPrint, full MaleCNS needing many RAM/GPU, very simple baseline) is in [`docs/SETUP_LEVELS.md`](docs/SETUP_LEVELS.md). Start with DESIGN, then SETUP_LEVELS, then [`docs/RUNNING.md`](docs/RUNNING.md) for commands.

## Quickstart (stdlib only, ~1 s)

```bash
git clone https://github.com/acatthatflys/fruit-fly-BFM && cd fruit-fly-BFM
python3 -m unittest discover -s tests          # 32 tests, no deps
python3 -m flybfm fight --blue guns --red instructor --tag head_on
python3 -m flybfm fight --blue brain --red level --tag perch --replay web/replays/brain_perch.json  # records brain firing
python3 -m flybfm gunnery --checkpoint runs/poly/training.json
python3 -m flybfm serve --port 8000            # viewer at http://localhost:8000/ — 3 windows: 3D fight, brain firing, fly stick & throttle
```

## Install (optional acceleration)

```bash
pip install -e .                               # stdlib core, no torch/pandas needed
pip install -e ".[torch]"                      # GPU path: TorchLIFNetwork, ~100× faster
pip install -e ".[data]"                       # pandas/pyarrow for MaleCNS flat loader
pip install -e ".[dev]"                        # all + neuprint-python
```

- **Level -1 very simple**: `random` or `guns` — no brain, floor baseline.
- **Level 0 synthetic brain** (default, stdlib only): `python -m flybfm probe --brain synthetic` — 665 neurons, no download. See `SETUP_LEVELS.md` Level 0.
- **Level 1 small slice via neuPrint** (visual→DN subgraph, not full file): needs `NEUPRINT_TOKEN`, pulls only requested cell types (e.g., R1-R6, T4/T5, STMD, VPN, DNs, KC/MBON). ~1k-5k neurons, 0.5-2 GB RAM, runs in Python or torch. See `SETUP_LEVELS.md` Level 1.
- **Level 2 full MaleCNS v1.0** (1.1 GB flat connectome, muchos RAM/GPU): `python -m flybfm.tools.fetch_connectome download` then `export FLYBFM_CONNECTOME=... FLYBFM_ANNOTATIONS=...` and `python -m flybfm probe --brain malecns --use-torch --torch-device cuda`. See `SETUP_LEVELS.md` Level 2 and [`docs/DATA.md`](docs/DATA.md).
- **Torch path** (milestones 5–6): `LIFNetwork.to_torch()` returns runnable `TorchLIFNetwork` using `torch.sparse_coo_tensor` — same equations as Python loop, CPU or CUDA. `python -m flybfm train --policy brain --use-torch --generations 2` auto-uses torch if installed, else falls back.

## What this repo measures

- Turn performance at corner speed vs analytic, pursuit mix (pure/lead/lag), manoeuvre detection (`analysis/maneuvers.py`), gunnery lead error (not just win rate), and **connectome vs shuffled connectome** ablation (the study that would actually get cited).
- Honest status: learned policies **win on points and cannot shoot** — `runs/poly` fires 215 rounds for 0 hits jittered, 0% inside 1.5°; scripted `guns` hits 3.4%. See `docs/EXPERIMENTS.md` §5 and `docs/RUNNING.md` §3.1.

## Packaging & CI

- `pyproject.toml` / `requirements.txt` — stdlib core, optional `torch`, `pandas`, `pyarrow`.
- GitHub Actions runs `python -m unittest discover -s tests` on every push (see `.github/workflows/ci.yml`).
- `load_malecns_flat` column-sniffing has a regression test with a tiny synthetic `.feather` fixture (`tests/test_connectome_loader.py`), because column names move between MaleCNS releases (flagged in `DATA.md`).

## Viewer — 3 windows + live training

`python -m flybfm serve --dir web --port 8000` → http://localhost:8000/

- **Top: 3D fight** — chase-blue/red, ground plane, WEZ bubble, tracers.
- **Middle-left: brain firing** — when replay has `brain` field (auto-recorded for `--blue brain`), shows population rates per group: `turn_left/right` (DN steering L/R), `vis_left/right` (STMD L/R), `pitch_up/down`, `speed`, `trigger`, plus `per_type` top-10 when full CNS is loaded. Bars = Hz, color = L vs R. No brain field? Shows pseudo from commands.
- **Middle-right: fly stick & throttle** — cartoon fly manipulating stick based on actual DN output: stick X = roll (`turn_gain*(R−L + vis_gain*STMD_R−L)`), Y = pull (`pitch_up−down`), throttle lever = throttle, trigger button. Linked to what the fly brain is actually doing (pulling up, rolling right, etc.). Fly wings flap with throttle.

Replays with brain: `python -m flybfm fight --blue brain --red level --replay web/replays/brain_perch.json`

**Live training** (new): `flybfm serve` now has `/api/train/*` endpoints. Click ⚙ train in header to open training bar — configurable gens/pop/episodes, policy/basis, scenario mix (40% easy gunnery 250-900m ±30°, 30% defensive jittered defensive/topgun_break, 30% random), curriculum 15-20s→30-45s→60-90s, evals prioritized (defaults 12×24×8=2304 fights). Training runs in terminal (server subprocess, `start_new_session=True`), survives window close; viewer shows live match of a random training fly while waiting, plus best result. See `docs/LIVE_TRAINING.md` for feasibility and `python -m flybfm train --help`.

Training quickstart:
```bash
python -m flybfm serve --port 8000 &
python -m flybfm train --live-replay web/replays/live.json --live-status runs/live/status.json --generations 12 --population 24 --episodes 8
# open http://localhost:8000, click 👁 watch live fly
```

## Layout

```
flybfm/sim/       point-mass F-16 EM model, geometry, gun ballistics, env, reward, scripted pool
flybfm/brain/     connectome, LIF (+ TorchLIFNetwork), retinal encoder, plasticity, DN readout
flybfm/train/     CEM/ES loop, PFSP league, imitation warm start, policies
flybfm/analysis/  manoeuvre classifier, episode metrics
flybfm/tools/     probe_brain, gunnery_check, fetch_connectome
web/              replay viewer (static) — 3 windows: fight, brain firing, fly stick
tests/            brain, sim, loader regression
docs/             DESIGN, REWARD, DATA, EXPERIMENTS, RUNNING, SETUP_LEVELS (allocation tiers)
```

License: MIT.
