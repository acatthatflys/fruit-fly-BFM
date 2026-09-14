# Setup levels — from zero deps to full MaleCNS

This repo runs at four levels of biological fidelity. Each level is a superset of the previous: same physics, same BFM scoring, same CEM trainer — only the brain changes. Pick the level that matches your RAM, GPU, and patience.

---

## Level -1 — No brain at all (random / scripted baseline)

**What it is:** No connectome, no LIF, just the point-mass F-16 and ten hand-written opponents. Used to measure "did learning do anything?".

**Cost:** ~0.05 s per 30 s fight, 5 min for a full 24×24×8 CEM run on 2 cores. Zero dependencies.

**Run:**
```bash
python3 -m unittest discover -s tests
python3 -m flybfm fight --blue guns --red level --tag perch
python3 -m flybfm fight --blue random --red instructor --random --replay web/replays/random.json
python3 -m flybfm league --per-pair 3
python3 -m flybfm serve --port 8000   # viewer
```

**What you get:** `guns` hits 3.4% of rounds, `random` hits 0%, win rates on points. This is the floor every learned policy must beat.

---

## Level 0 — Synthetic minimal brain (default, stdlib only)

**What it is:** Reduced synthetic connectome with realistic degree statistics, not MaleCNS. 665 neurons / 2.2k edges after `flight_subgraph` pruning, including 66 `KC→MBON` plastic edges for testing the dopamine-gated rule. LIF implemented in pure Python (`LIFNetwork`), deterministic, no torch/numpy.

**Cost:** Same as Level -1 for `features` policies. Brain policy is ~24 ms per decision, ~74 s per 12 s fight (~6× slower than real time) — use 2-3 generations for training. No download, no extra pip.

**Run:**
```bash
pip install -e .   # stdlib core
python3 -m flybfm probe --brain synthetic --steps 250
python3 -m flybfm fight --blue brain --red level --tag perch --replay web/replays/brain_vs_level.json
python3 -m flybfm train --policy brain --generations 3 --population 10 --episodes 3 --out runs/brain_small
python3 -m flybfm train --policy brain --use-torch --generations 3 --population 10 --episodes 3 --out runs/brain_torch_cpu  # same brain, torch-accelerated
```

**What you get:** Retina tracks bearing (`probe` shows `retina peak cell` following target), `STMD` L/R difference orders with bearing, DN L/R diff is bearing-invariant in the reduced graph (reported, not hidden — `vis_gain` tells you which channel the behaviour rides on). All numbers in `runs/` and `docs/EXPERIMENTS.md` come from outer CEM only; inner plasticity is off by default.

**When to use:** Development, unit tests, CI. `python -m unittest discover -s tests` → 32 tests.

---

## Level 1 — Small neural slice via neuPrint (visual → DN, no 1.1 GB download)

**What it is:** Pull a *subgraph* live from neuPrint instead of downloading the flat connectome. You ask for exactly the cell types you need (e.g., `R1-R6, T4, T5, LC4, LPLC2, STMD, VPN, DNp26, DNg02, DNp06, KC, MBON, PAM, PPL101`) and get back only those neurons and their edges. No segmentation volumes, no skeletons, just the induced subgraph.

**Cost:** ~100-5k neurons, ~10k-500k synapses depending on types requested. ~50-200 MB RAM, runs in pure Python at ~0.2-2 s per fight, or ~0.5 ms/step with `TorchLIFNetwork` on CPU. Needs free neuPrint account + token.

**Setup:**
```bash
pip install -e .[data]   # pandas for annotations, neuprint-python optional
export NEUPRINT_TOKEN="your_free_token_from_neuprint.janelia.org"
python3 - << 'PY'
from flybfm.brain.connectome import load_neuprint, flight_subgraph
# visual → descending slice
conn = load_neuprint(dataset="male-cns:v1.0", token="...", 
                     cell_types=["R1-R6","T4","T5","LC4","LPLC2","STMD","VPN",
                                 "DNp26","DNp57","DNp03","DNg02","DNp06","DNa02","DNg13","DNp10","DNHS1",
                                 "KC","MBON","PAM","PPL101"])
print(conn.stats())
PY
```

**Or via CLI (uses env vars for real data path if you cached a slice):**
```bash
# fetch a tiny slice and cache as feather (optional)
python3 -m flybfm.tools.fetch_connectome list
python3 -m flybfm probe --brain malecns --use-torch --steps 100
# If you have a pruned feather from a previous download:
export FLYBFM_CONNECTOME=data/malecns_pruned/connectome.feather
export FLYBFM_ANNOTATIONS=data/malecns_pruned/annotations.feather
python3 -m flybfm probe --brain malecns --steps 250 --use-torch --plasticity
```

**What you get:** Real transmitter signs (GABA/GLUT negative), real degree distribution, real mushroom-body `KC→MBON` counts (hundreds to thousands, not 66). The inner plasticity path is now biologically grounded; you can ablate `plasticity=True` vs `False` and measure whether dopamine-gated learning adds anything over outer CEM.

**When to use:** When you want real wiring without paying 1.1 GB + 16 GB RAM. This is the recommended level for Track B experiments (connectome vs shuffled).

---

## Level 2 — Full MaleCNS v1.0 map (166k neurons, 125M synapses)

**What it is:** The complete male *Drosophila* CNS, brain **and VNC**, including descending and ascending neurons. Flat edge list: `body-annotations` (13 MB) + `body-neurotransmitters` (42 MB) + `connectome-weights` (1.1 GB) = ~1.2 GB total, CC-BY 4.0, hosted on GCS and HTTPS.

**Cost:** Requires muchos RAM or GPU — this is not a laptop-without-torch job.
- **RAM:** 16-32 GB for pruned visual→DN→motor subgraph (`fetch_connectome prune`), 64 GB+ for full graph in Python, ~8-12 GB for sparse torch tensor (125M edges × 12 bytes ≈ 1.5 GB + overhead).
- **Compute:** Pure Python LIF is ~1-3 M synaptic events/s → hours per fight. **Torch path is required**: `TorchLIFNetwork` with `torch.sparse_coo_tensor` does `I = W @ spikes` at ~1-5 ms/step on A100, ~10-30 ms/step on CPU for pruned subgraph.
- **Download:** One-time 1.2 GB, needs `pandas`/`pyarrow`.

**Setup:**
```bash
pip install -e .[all]   # torch + pandas + pyarrow + neuprint-python
python3 -m flybfm.tools.fetch_connectome list
python3 -m flybfm.tools.fetch_connectome download --out data/malecns
# prune to flight-relevant subgraph (visual → DN → premotor) to make it tractable
python3 -m flybfm.tools.fetch_connectome prune --source data/malecns --out data/malecns_pruned
export FLYBFM_CONNECTOME=data/malecns_pruned/connectome-weights-male-cns-v1.0-minconf-0.5.feather
export FLYBFM_ANNOTATIONS=data/malecns_pruned/body-annotations-male-cns-v1.0-minconf-0.5.feather

# probe with torch (CPU or CUDA)
python3 -m flybfm probe --brain malecns --use-torch --steps 250
python3 -m flybfm probe --brain malecns --use-torch --torch-device cuda --steps 250 --plasticity

# fight with full CNS in the loop (needs GPU for real-time)
python3 -m flybfm fight --blue brain --red instructor --tag head_on --use-torch --torch-device cuda --replay web/replays/full_cns.json

# train readout (outer loop) + optional inner plasticity
python3 -m flybfm train --policy brain --use-torch --torch-device cuda --generations 6 --population 12 --episodes 4 --plasticity --out runs/full_cns_torch
```

**What you get:**
- Real `DNp26/DNp57/DNp03` saccade hubs, `DNg02` steering, `DNp06` heading, `DNp10` landing, with true left/right copies (lateral steering is `R−L`, not sum).
- Mushroom-body `KC→MBON` with thousands of plastic edges; `deliver_dopamine` now acts on a real learning substrate. Ablation `plasticity=True` vs `False` is the honest inner-vs-outer experiment.
- Viewer shows **brain firing panel** (population rates per type, L/R diff) and **fly stick & throttle panel** (stick position = `turn_gain*(R−L + vis_gain*STMD_L/R)`, pull = `pull_gain*(up−down)`, throttle = `throttle_bias + gain*speed_rate`, trigger = `trigger_gain*trigger_rate > thresh`) when replay contains `brain` field (auto-recorded for `--blue brain` fights).

**When to use:** Milestones 5 (full connectome in loop, latency budget documented) and 6 (connectome vs shuffled, ≥100 held-out fights with error bars). This is the study that would actually get cited.

---

## Level 3 — Full CNS + biomechanical body (optional, not in this repo yet)

**What it is:** Replace point-mass F-16 with `FlyGym / NeuroMechFly v2` flapping fly body (Apache-2.0, ~2× real time on CPU). Then the fly's sensorimotor repertoire (saccadic fixation, optomotor stabilization, looming escape) is *relevant* and the Reynolds mismatch goes away. Engagement shrinks to fly scale: 1 g-class MAV, speeds m/s, ranges tens of meters, laser/airsoft "gun".

**Cost:** FlyGym install + MuJoCo, GPU for CNS + CPU for body.

**Why mention:** Track A in `DESIGN.md` §7 — honest about the body. Not implemented here, but compatible by construction: NeuroMechFly v2 splits brain/VNC with explicit descending/ascending interface, which is exactly the interface this repo reads out.

---

## Quick chooser

| Level | Brain | Deps | RAM | Speed | Use for |
|---|---|---|---|---|---|
| -1 | none / scripted | stdlib | 100 MB | 0.05 s/fight | floor, win-rate sanity |
| 0 | synthetic 665-neuron | stdlib | 200 MB | 0.05 s (features) / 74 s (brain py) / 3 s (brain torch) | dev, CI, `probe` |
| 1 | neuPrint slice 1k-5k neurons | pandas + token | 0.5-2 GB | 0.5-2 s py / 0.5 ms/step torch | real wiring without 1.1 GB |
| 2 | full MaleCNS 166k/125M | torch + pandas + 1.2 GB dl | 16-64 GB | hours py / 1-5 ms/step torch A100 | milestones 5-6, paper |
| 3 | full CNS + FlyGym body | + flygym + mujoco | 16 GB+ | ~0.5× real time | micro-BFM, MAV |

## Packaging note

```bash
pip install -e .              # Level -1,0
pip install -e .[torch]       # + torch for Level 0/1/2 torch path
pip install -e .[data]        # + pandas/pyarrow for Level 1/2 flat loader
pip install -e .[all]         # everything
python -m unittest discover -s tests   # 32 tests, no deps needed for core
```

All levels log `blue_name`/`red_name` in replay JSON so the viewer HUD shows which brain flew which side.
