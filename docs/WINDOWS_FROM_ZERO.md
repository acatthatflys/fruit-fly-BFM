# Windows from zero — Level -1, 0, 1, 2 + localhost viewer

This guide assumes **nothing installed**: no Python, no Git, no pip. Windows 10/11, PowerShell (recommended) or CMD. Every command is copy-pasteable. Levels are cumulative: same physics/BFM, only the brain changes.

> **Quick chooser**
> | Level | Brain | RAM | Time | What you prove |
> |---|---|---|---|---|
> | -1 | none / scripted | 100 MB | 0.05 s/fight | floor baseline |
> | 0 | synthetic 665-neuron | 200 MB | 0.05 s features / 3 s brain-torch | dev, CI, probe |
> | 1 | neuPrint slice 1k-5k | 0.5-2 GB | 0.5 ms/step torch | real wiring, no 1.1 GB dl |
> | 2 | full MaleCNS 166k/125M | 16-64 GB | 1-5 ms/step torch A100, hours py | paper milestones |

---

## 0. Prerequisites — install once

### 0.1 Install Git for Windows
1. Go to https://git-scm.com/download/win → download 64-bit.
2. Run installer: leave defaults, **ensure** “Git from the command line and also from 3rd-party software” checked.
3. Verify in **PowerShell**:
```powershell
git --version
```

### 0.2 Install Python 3.11 or 3.12
1. https://www.python.org/downloads/ → Download Python 3.11.9 (or 3.12).
2. **Important:** Check **“Add python.exe to PATH”** at bottom of installer.
3. Also check “pip”.
4. Verify:
```powershell
python --version
pip --version
# if python not found, try py or python3:
py --version
```
If `python` fails, use `py` everywhere below (Windows launcher). This guide uses `python`.

### 0.3 (Optional but recommended) Windows Terminal + VS Code
- Windows Terminal from Microsoft Store gives tabs for server + training.
- VS Code for editing.

---

## 1. Clone the repo

```powershell
cd $HOME
# or cd C:\Users\YourName\Documents
git clone https://github.com/acatthatflys/fruit-fly-BFM.git
cd fruit-fly-BFM
dir
```

You should see `flybfm/`, `web/`, `tests/`, `docs/`, `pyproject.toml`.

---

## Level -1 — No brain at all (random / scripted baseline)

**What:** No connectome, no LIF, just point-mass F-16 + 10 scripted opponents. Zero pip dependencies. Measures “did learning do anything?”.

**Cost:** ~0.05 s per 30 s fight, 5 min for full CEM on 2 cores.

### Run tests (stdlib only)
```powershell
python -m unittest discover -s tests -v
# or
python -m pytest tests -q   # if pytest installed
```
Expected: 27 passed, 8 skipped (or 32 tests if data deps installed).

### Watch one fight
```powershell
python -m flybfm fight --blue guns --red level --tag perch
python -m flybfm fight --blue random --red instructor --random --replay web/replays/random.json
python -m flybfm league --per-pair 3
```
Output: scenario, result (`blue`/`red`/`draw`), JSON summary, scorecard, first events. `guns` hits ~3.4% rounds, `random` 0%.

### Viewer (Level -1 works too)
```powershell
python -m flybfm serve --port 8000
# default --host 127.0.0.1 = localhost only. Use --host 0.0.0.0 to expose on LAN (no auth — anyone on WiFi can start/stop training via /api/train/start|stop).
```
Open **http://localhost:8000/** in Chrome/Edge. You should see 3D fight, ground plane, WEZ bubble. If `web/replays/` empty, generate one with `--replay` above, then click ↻ in header.

**Leave this PowerShell running** for Level 2 localhost training — training survives window close because it runs as server subprocess, not browser thread.

---

## Level 0 — Synthetic minimal brain (default, stdlib only)

**What:** Reduced synthetic connectome with realistic degree stats, 665 neurons / 2.2k edges after pruning, 66 KC→MBON plastic edges for dopamine test. Pure Python LIF, deterministic, no torch/numpy.

**Cost:** Features policy same as -1. Brain policy ~24 ms/decision, ~74 s per 12 s fight in Python (~6× slower than real time). Torch path ~3 s/fight on CPU.

### Install core (stdlib, no torch/pandas)
```powershell
pip install -e .
# or if python not in PATH:
py -m pip install -e .
```

### Probe — does retina track?
```powershell
python -m flybfm probe --brain synthetic --steps 250
```
Look for: `population_hz`, `retina peak cell`, `visual L/R diff` ordering with bearing. `descending L/R diff` expected to FAIL on reduced graph — reported, not hidden (vis_gain tells you which channel behavior rides).

### Fight with brain
```powershell
python -m flybfm fight --blue brain --red level --tag perch --replay web/replays/brain_perch.json
python -m flybfm serve --port 8000
# reload http://localhost:8000 — select brain_perch.json
# you now get 3 windows: 3D fight + brain firing + fly stick & throttle
```

### Train tiny (outer CEM only, inner plasticity OFF by default)
```powershell
python -m flybfm train --policy features --basis linear --init known-good --generations 8 --population 48 --episodes 8 --out runs/linear_small --easy-frac 0.4 --defensive-frac 0.3 --random-frac 0.3
# linear 92 params, 8*48*8=3072 fights, ~3 min, bench prints avg fight ms + fights/s + ETA
# known-good: roll=+1*az_coarse+1.5*az_fine (fixed sign), gets 4 hits vs level 300m stern

python -m flybfm train --policy brain --generations 3 --population 10 --episodes 3 --out runs/brain_small
# Python LIF slow — use 2-3 gens only
```

### Torch-accelerated same brain (CPU)
```powershell
pip install -e ".[torch]"
python -m flybfm probe --brain synthetic --use-torch --steps 250
python -m flybfm train --policy brain --use-torch --generations 3 --population 10 --episodes 3 --out runs/brain_torch_cpu
```
`TorchLIFNetwork` uses `torch.sparse_coo_tensor`, `I = W @ spikes`, identical equations, 10-100× faster. `LIFNetwork.to_torch()` is entry point.

**Honest status:** All results in `runs/` are 100% outer CEM, 0% inner plasticity unless you pass `--plasticity`. Synthetic has 66 plastic edges but controller defaults `plasticity=False`.

---

## Level 1 — Small neural slice via neuPrint (visual→DN, no 1.1 GB download)

**What:** Pull induced subgraph live from neuPrint — only cell types you request, no volumes/skeletons. Real transmitter signs (GABA/GLUT negative), real degree distribution, hundreds-thousands KC→MBON. **No feather files, no env vars** — uses in-memory `Connectome`.

**Cost:** 100-5k neurons, 10k-500k synapses, 50-200 MB RAM, 0.2-2 s/fight py, 0.5 ms/step torch CPU. Needs free neuPrint account + token.

### Get token
1. https://neuprint.janelia.org → Register (free) → Account → copy token.

### Install data deps
```powershell
pip install -e ".[data]"
# includes pandas, pyarrow, neuprint-python optional
```

### Set token (PowerShell)
```powershell
$env:NEUPRINT_TOKEN="your_token_here"
# to persist:
[Environment]::SetEnvironmentVariable("NEUPRINT_TOKEN","your_token_here","User")
```

### Fetch slice — correct path (in-memory, no CLI feather cache)

`fetch_connectome.py` currently only has `list` / `download` / `prune` — **no slice-to-feather** subcommand. CLI `--brain malecns` is full-download only (expects the 3-file official naming). Don't use `FLYBFM_CONNECTOME` env var for slices — it won't work.

Instead use `load_neuprint()` which returns an in-memory `Connectome`, and pass it directly to `ConnectomeController(conn=...)`. The controller has **no** `.step(bearing_deg=..., range_m=..., dt=...)` or `.last_groups` — use the low-level loop `probe_brain.py` actually uses:

```powershell
python - << 'PY'
from flybfm.brain.connectome import load_neuprint
from flybfm.brain.controller import ConnectomeController
from flybfm.sim.arena import Scenario
from flybfm.sim.dogfight import Dogfight, EnvConfig

# 1. pull slice live — no files written
conn = load_neuprint(dataset="male-cns:v1.0",
                     cell_types=["R1-R6","T4","T5","LC4","LPLC2","STMD","VPN",
                                 "DNp26","DNp57","DNp03","DNg02","DNp06","DNa02","DNg13","DNp10","DNHS1",
                                 "KC","MBON","PAM","PPL101"])
print(conn.stats())
# e.g. 2.1k neurons, 180k edges, 1.2k KC->MBON plastic

# 2. use directly — param is conn=, not connectome=
ctl = ConnectomeController(conn=conn, plasticity=False)
# ctl = ConnectomeController(conn=conn, plasticity=True) for inner dopamine

# 3. open-loop drive like probe_brain.py does
env = Dogfight(EnvConfig())
env.reset(Scenario(range_m=500.0, taa_deg=0.0, nose_offset_deg=-30.0, alt_m=6000.0, seed=0))
me = env.blue
other = env.red

groups = ctl._groups()  # DN groups the readout looks at
for _ in range(250):
    ctl.net.clear_inputs()
    ctl.sensors.inject(ctl.net, me.s, other.s, me.geom_to_other,
                       getattr(me, "los_rate_rad", 0.0), ctl.net.p.dt)
    ctl.net.step()

rates = {k: ctl.net.group_rate(v) for k, v in groups.items()}
print("DN rates:", rates)
print("pop Hz:", ctl.net.population_rate())
PY
```

Probe with torch, same low-level loop:

```powershell
python - << 'PY'
from flybfm.brain.connectome import load_neuprint
from flybfm.brain.controller import ConnectomeController
from flybfm.sim.arena import Scenario
from flybfm.sim.dogfight import Dogfight, EnvConfig

conn = load_neuprint(dataset="male-cns:v1.0",
                     cell_types=["R1-R6","T4","T5","LC4","LPLC2","STMD","DNp26","DNp57","DNp03","DNg02","KC","MBON"])
ctl = ConnectomeController(conn=conn, use_torch=True, plasticity=True)

env = Dogfight(EnvConfig())
env.reset(Scenario(range_m=300.0, taa_deg=0.0, nose_offset_deg=-10.0, alt_m=6000.0, seed=1))
me, other = env.blue, env.red

for _ in range(100):
    ctl.net.clear_inputs()
    ctl.sensors.inject(ctl.net, me.s, other.s, me.geom_to_other, 0.0, ctl.net.p.dt)
    ctl.net.step()

print({k: ctl.net.group_rate(v) for k, v in ctl._groups().items()})
PY
```

Fight example — use controller directly as policy (no CLI flag):

```powershell
python - << 'PY'
from flybfm.brain.connectome import load_neuprint
from flybfm.brain.controller import ConnectomeController
from flybfm.train.policy import BrainPolicyAdapter
from flybfm.sim.arena import CANONICAL_SETUPS
from flybfm.sim.dogfight import run_match, EnvConfig
import json, os

conn = load_neuprint(dataset="male-cns:v1.0",
                     cell_types=["R1-R6","T4","T5","LC4","LPLC2","STMD","DNp26","DNp57","KC","MBON"])
ctl = ConnectomeController(conn=conn, use_torch=True)
brain = BrainPolicyAdapter(ctl)  # wraps act() + reset() + observe_reward()

sc = CANONICAL_SETUPS["perch"]
_, _, env = run_match(sc, brain, "level", EnvConfig(max_time_s=30.0), record=True)
os.makedirs("web/replays", exist_ok=True)
with open("web/replays/brain_slice.json", "w") as fh:
    json.dump(env.to_replay(), fh)
print("wrote web/replays/brain_slice.json")
PY
```



### Why not env-var feather?

`load_malecns_flat()` derives the neurotransmitter file via string-replace `connectome-weights` → `body-neurotransmitters`, so it only works with the official 3-file naming from `download`:
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0-minconf-0.5.feather`

If you write a slice as `connectome.feather` / `annotations.feather`, the replace fails and transmitter signs are missing. That's why Level 1 must stay in-memory.

### Future fix — proposed `fetch-slice` subcommand

Longer-term we will add:

```powershell
python -m flybfm.tools.fetch_connectome fetch-slice --cell-types R1-R6,T4,T5,LC4,DNp26,DNp57,KC,MBON --out data/slice_pruned --dataset male-cns:v1.0
# writes:
#   data/slice_pruned/connectome-weights-male-cns-v1.0-minconf-0.5.feather
#   data/slice_pruned/body-annotations-male-cns-v1.0-minconf-0.5.feather
#   data/slice_pruned/body-neurotransmitters-male-cns-v1.0-minconf-0.5.feather
# with expected naming so load_malecns_flat() works, plus a manifest.json with query
```

Until then, use the short script above — no `FLYBFM_CONNECTOME` / `FLYBFM_ANNOTATIONS`.

**When to use:** Real wiring without 1.1 GB. Recommended for Track B (connectome vs shuffled).

---

## Level 2 — Full MaleCNS v1.0 (166k neurons, 125M synapses) + localhost training window

**What:** Complete male Drosophila CNS, brain + VNC. Flat edge list: annotations 13 MB + transmitters 42 MB + weights 1.1 GB = ~1.2 GB, CC-BY 4.0.

**Cost:**
- RAM: 16-32 GB pruned visual→DN→motor, 64 GB+ full Python, 8-12 GB sparse torch (125M edges ×12 bytes ≈1.5 GB)
- Compute: Python LIF ~1-3 M events/s → hours/fight. **Torch required**.
- Download: one-time 1.2 GB.

### Install all
```powershell
pip install -e ".[all]"
# torch + pandas + pyarrow + neuprint-python
# If torch CUDA needed:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Download + prune
```powershell
python -m flybfm.tools.fetch_connectome list
python -m flybfm.tools.fetch_connectome download --out data/malecns
python -m flybfm.tools.fetch_connectome prune --source data/malecns --out data/malecns_pruned
dir data/malecns_pruned
```

### Set env vars (PowerShell, session)
```powershell
$env:FLYBFM_CONNECTOME="data/malecns_pruned/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
$env:FLYBFM_ANNOTATIONS="data/malecns_pruned/body-annotations-male-cns-v1.0-minconf-0.5.feather"
# persist:
[Environment]::SetEnvironmentVariable("FLYBFM_CONNECTOME",$env:FLYBFM_CONNECTOME,"User")
[Environment]::SetEnvironmentVariable("FLYBFM_ANNOTATIONS",$env:FLYBFM_ANNOTATIONS,"User")
```

### Probe full CNS (torch CPU or CUDA)
```powershell
python -m flybfm probe --brain malecns --use-torch --steps 250
# with CUDA:
python -m flybfm probe --brain malecns --use-torch --torch-device cuda --steps 250 --plasticity
```

### Fight with full CNS in loop
```powershell
python -m flybfm fight --blue brain --red instructor --tag head_on --use-torch --torch-device cuda --replay web/replays/full_cns.json
# CPU fallback:
python -m flybfm fight --blue brain --red level --tag perch --use-torch --replay web/replays/full_cns_cpu.json
```

### Train readout (outer CEM) + optional inner plasticity
```powershell
python -m flybfm train --policy brain --use-torch --torch-device cuda --generations 6 --population 12 --episodes 4 --plasticity --out runs/full_cns_torch
# CPU:
python -m flybfm train --policy brain --use-torch --generations 4 --population 10 --episodes 3 --out runs/full_cns_cpu
```

**What you get:** Real DNp26/57/03 saccade hubs, DNg02 steering, DNp06 heading, DNp10 landing, true L/R copies (steering = R−L). Mushroom-body KC→MBON thousands edges; `deliver_dopamine` acts on real substrate. Viewer shows brain firing panel (Hz per type, L/R diff) + fly stick & throttle panel when replay has `brain` field (auto-recorded).

---

## Level 2 localhost window — training that survives close

This is the new part (2026-09). `flybfm serve` now has `/api/train/*` and spawns training as **terminal subprocess** (`start_new_session=True`), not browser thread. Closing browser doesn't kill training.

### Terminal 1 — serve
```powershell
cd C:\Users\YourName\Documents\fruit-fly-BFM
python -m flybfm serve --port 8000 --runs-dir runs/live
# default binds 127.0.0.1 (localhost only). For LAN exposure use --host 0.0.0.0 (no auth on /api/train/* — anyone on network can start/stop jobs).
# prints:
# serving ... on http://127.0.0.1:8000/
#   live training API: POST .../api/train/start
#   status: GET .../api/train/status
#   live replay: GET .../api/train/live
#   log tail: GET .../api/train/log?lines=200
#   bench: GET .../api/bench
```

### Browser — open viewer
Open **http://localhost:8000/** → header has ⚙ train button.

Click ⚙ → training bar opens:
- policy `features (linear)` / `brain`, basis `linear` (92 params, best) / `poly` (460)
- gens 16, pop 64, ep/cand 8 → 8192 fights (default final)
- easy% 40 (250-900m ±30°, ultra-easy 250-350±5° 50% + super-easy 250-500±15° 25%), def% 30 (defensive/topgun_break jitter), rand% 30
- curriculum: episode 15-20s→30-45s→60-90s + scenario 100/0/0→80/5/15→40/30/30→20/40/40 + opponent level→all
- reward: tracking lead 1° w12 +30° w2 inverse-quad, hit 20 kill 200, good trigger +2 bad -0.01, good_solution +0.5 in_wez +0.05
- est row: bench now 60s fights with curriculum mix (was 20s random), shows avg fight ms, fights/s, CPU count, mix, ETA. Example: 90ms/fight, 8-11 fights/s on 2 CPUs, 8192 fights ≈10-15 min laptop.

Click **▶ start training** → POST `/api/train/start` → server spawns:
```
python -u -m flybfm train --policy features --basis linear --generations 16 --population 64 --episodes 8 --easy-frac 0.4 --defensive-frac 0.3 --random-frac 0.3 --out runs/live --live-replay web/replays/live.json --live-status runs/live/status.json --live-every 1
```
Log streams to `runs/live/train.log`, pid to `pid.txt`, status to `status.json` each gen, live replay to `web/replays/live.json`.

### Watch live fly
- While training: viewer auto-shows live match of random training fly (if you clicked start or 👁 watch live fly). Badge: `LIVE gen X`.
- Training bar status: `running pid ... gen 3/16 best +14.0 mean -0.5`.
- Progress panel: shows `gen best mean max_time_s win% hit%` + `live X fights/s, ETA Y for Z fights remaining` computed from `history[].wall_s`.
- Log box: tail of `train.log` (120 lines).
- **Close browser → training keeps running** (server subprocess). Reopen http://localhost:8000 → status still running, live replay still updates.
- Stop: ■ stop button → POST `/api/train/stop` → SIGTERM then SIGKILL pid group.

### Terminal 2 — alternative manual training (same effect)
```powershell
python -m flybfm train --live-replay web/replays/live.json --live-status runs/live/status.json --generations 16 --population 64 --episodes 8 --out runs/live
# then serve in other terminal, open browser, click watch live
```

### After training — best result
When training finishes, best replay auto-loads: `👁 show best result replay` button. File `runs/live/training.json` has `best.params`, full history, pool importances. Generate viewer replay:
```powershell
python -m flybfm fight --blue features --red guns --blue-ckpt runs/live/training.json --tag perch --replay web/replays/best_perch.json
python -m flybfm replay --dir web/replays
```

### Bench endpoint
```powershell
curl http://localhost:8000/api/bench
# or in browser: http://localhost:8000/api/bench
```
Returns:
```json
{
  "n_fights": 6,
  "max_time_s": 60.0,
  "avg_fight_s": 0.09,
  "fights_per_second": 11.0,
  "cpu_count": 8,
  "easy_frac": 0.4,
  "current_estimated_total_s": 552
}
```
Web uses it for ETA: `total_fights * avg_fight_s *0.75`.

---

## Troubleshooting Windows

| symptom | fix |
|---|---|
| `python` not found | Use `py` or `py -3` — Windows launcher. `py -m flybfm ...` |
| `ModuleNotFoundError: flybfm` | Run from repo root, not subfolder. `cd fruit-fly-BFM` |
| `0 rounds fired` | Not in WEZ. Check `frac inside envelope` in `gunnery`. Easy frac 1.0 for training. |
| win high hits 0 | Wins on points — run `python -m flybfm gunnery --checkpoint ...` — lead error median should be <10° not 122° |
| brain silent 0 Hz | `bg_current`/`gain_photo` low — run `probe` |
| viewer “no replay” | `dir web/replays` empty — generate with `--replay` |
| torch CUDA not found | `pip install torch --index-url https://download.pytorch.org/whl/cu121` + check `nvidia-smi` |
| pandas not found Level 1/2 | `pip install -e ".[data]"` |
| PowerShell env vars not persisting | Use `[Environment]::SetEnvironmentVariable(...,"User")` then restart terminal |
| Port 8000 in use | `python -m flybfm serve --port 8001` |
| Serve exposed on LAN | Default is now `127.0.0.1` (localhost only). `--host 0.0.0.0` exposes `/api/train/start|stop` with no auth — only use on trusted LAN. |
| Training already running | `http://localhost:8000/api/train/status` → `running pid` → ■ stop or `taskkill /PID <pid> /T /F` |

---

## One-liner copy-paste for impatient Windows user

```powershell
# 0. prerequisites already installed (git + python with PATH)
git clone https://github.com/acatthatflys/fruit-fly-BFM.git; cd fruit-fly-BFM
pip install -e ".[torch]"; python -m unittest discover -s tests
python -m flybfm fight --blue guns --red level --tag perch --replay web/replays/perch.json
python -m flybfm serve --port 8000
# open http://localhost:8000/ in browser, click ⚙ train → ▶ start training → 👁 watch live fly
```

For full MaleCNS (Level 2):
```powershell
pip install -e ".[all]"
python -m flybfm.tools.fetch_connectome download --out data/malecns
python -m flybfm.tools.fetch_connectome prune --source data/malecns --out data/malecns_pruned
$env:FLYBFM_CONNECTOME="data/malecns_pruned/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
$env:FLYBFM_ANNOTATIONS="data/malecns_pruned/body-annotations-male-cns-v1.0-minconf-0.5.feather"
python -m flybfm probe --brain malecns --use-torch --steps 250
python -m flybfm serve --port 8000
```

Done — you now have floor baseline, synthetic brain, real slice, full CNS, and localhost window where training survives close and shows live fly.
