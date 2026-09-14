# fruit-fly-BFM — can a fly brain learn basic fighter manoeuvres?

This repo is a **connectome-constrained BFM simulator**: a leaky integrate-and-fire network over the *Drosophila* CNS (166k neurons, ~125M synapses in the real MaleCNS v1.0 release, 665 neurons / 2.2k synapses in the in-repo synthetic stand-in) drives a point-mass energy fighter (F-16-like EM model) through a 12 km guns-only arena at 50 Hz physics / 10 Hz decisions. The opponent pool has ten scripted pilots (level, guns, lead, lag, break, vertical, boom-and-zoom, jinker, instructor, nose-on) plus PFSP self-play and a CEM/ES outer learner.

Short answer on feasibility: **yes**, you can build a full-brain-scale LIF simulation, drive it with a synthetic retina, and read out descending neurons to fly an aircraft — that runs today. **No**, you should not expect the fly brain itself to invent energy management, lag/lead pursuit switching, or the high yo-yo. In this repo the fly brain is a fixed nonlinear reservoir and the thing that actually learns is a small readout policy (6–296 parameters) trained by cross-entropy method; the inner mushroom-body dopamine-gated plasticity (`KC → MBON`, `Δw = η·e·DA − λw`) is implemented, has 66 plastic edges in the synthetic graph, but is **off by default** and inert unless you pass `plasticity=True` and load a real mushroom-body subgraph — 100% of the results in `runs/` come from the outer CEM loop. That's intentional and documented.

Design, reward shaping, data sources, measured results, and the full roadmap with pass/fail gates are in [`docs/DESIGN.md`](docs/DESIGN.md). Start there, then [`docs/RUNNING.md`](docs/RUNNING.md) for commands.

## Quickstart (stdlib only, ~1 s)

```bash
git clone https://github.com/acatthatflys/fruit-fly-BFM && cd fruit-fly-BFM
python3 -m unittest discover -s tests          # 27 tests, no deps
python3 -m flybfm fight --blue guns --red instructor --tag head_on
python3 -m flybfm gunnery --checkpoint runs/poly/training.json
python3 -m flybfm serve --port 8000            # viewer at http://localhost:8000/
```

## Install (optional acceleration)

```bash
pip install -e .                               # stdlib core, no torch/pandas needed
pip install -e ".[torch]"                      # GPU path: TorchLIFNetwork, ~100× faster
pip install -e ".[data]"                       # pandas/pyarrow for MaleCNS flat loader
pip install -e ".[dev]"                        # all + neuprint-python
```

- **Synthetic brain** (default): `python -m flybfm probe --brain synthetic` — no download.
- **Real MaleCNS v1.0** (1.1 GB flat connectome): `python -m flybfm.tools.fetch_connectome download` then `export FLYBFM_CONNECTOME=... FLYBFM_ANNOTATIONS=...` and `python -m flybfm probe --brain malecns` — see [`docs/DATA.md`](docs/DATA.md).
- **Torch path** (milestones 5–6): `LIFNetwork.to_torch()` now returns a runnable `TorchLIFNetwork` using `torch.sparse_coo_tensor` — same equations as the Python loop, CPU or CUDA. `python -m flybfm train --policy brain --generations 2` will auto-use torch if installed, else falls back to Python.

## What this repo measures

- Turn performance at corner speed vs analytic, pursuit mix (pure/lead/lag), manoeuvre detection (`analysis/maneuvers.py`), gunnery lead error (not just win rate), and **connectome vs shuffled connectome** ablation (the study that would actually get cited).
- Honest status: learned policies **win on points and cannot shoot** — `runs/poly` fires 215 rounds for 0 hits jittered, 0% inside 1.5°; scripted `guns` hits 3.4%. See `docs/EXPERIMENTS.md` §5 and `docs/RUNNING.md` §3.1.

## Packaging & CI

- `pyproject.toml` / `requirements.txt` — stdlib core, optional `torch`, `pandas`, `pyarrow`.
- GitHub Actions runs `python -m unittest discover -s tests` on every push (see `.github/workflows/ci.yml`).
- `load_malecns_flat` column-sniffing has a regression test with a tiny synthetic `.feather` fixture (`tests/test_connectome_loader.py`), because column names move between MaleCNS releases (flagged in `DATA.md`).

## Layout

```
flybfm/sim/       point-mass F-16 EM model, geometry, gun ballistics, env, reward, scripted pool
flybfm/brain/     connectome, LIF (+ TorchLIFNetwork), retinal encoder, plasticity, DN readout
flybfm/train/     CEM/ES loop, PFSP league, imitation warm start, policies
flybfm/analysis/  manoeuvre classifier, episode metrics
flybfm/tools/     probe_brain, gunnery_check, fetch_connectome
web/              replay viewer (static)
tests/            brain, sim, loader regression
docs/             DESIGN, REWARD, DATA, EXPERIMENTS, RUNNING
```

License: MIT.
