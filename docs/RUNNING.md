# Running it

Everything here runs from the repository root, on stock Python 3.11+, with **no
numpy, no torch, no simulators to install**. The physics, the connectome model
and the learning loop are stdlib only. Commands are `python3 -m flybfm ...`.

```
git clone <this repo> && cd fruit-fly-BFM
python3 -m flybfm --help
```

## Who is who (read this first)

* **BLUE is the fly.** It is our side, the connectome-constrained controller (or
  the learner we are training). In a replay the HUD prints it as
  `BLUE · FLY <policy name>`, and the viewer's `blue_name` comes from whatever
  you passed to `--blue`.
* **RED is the opponent** — a scripted pilot, another policy, or a frozen copy of
  the fly.
* The engagement is **symmetric**; you can put any policy in either seat
  (`--blue level --red brain` is legal). The convention only fixes the
  *reporting*: every number in `docs/` — win rate, hit %, rounds per hit, lead
  error — is **blue's**. `summary()` fields (`blue_hits`, `blue_shots`, …) are
  likewise the fly's.
* Rounds are 20 mm, 511 per fight, ~16.7/s, and a hit does 1/6 damage, so it
  takes ~6 hits on target to kill. Guns only: no missiles anywhere in the repo.

Quick health check, ~1 second, 27 tests:

```
python3 -m unittest discover -s tests
```

---

## 1. Watch one fight

```
python3 -m flybfm fight --blue guns --red instructor --tag head_on
```

Prints the scenario, the result, a JSON summary, a scorecard (offence / defence /
energy), and the first events. Wall time ~0.05 s.

Useful forms:

```
# canonical BFM starts: perch lag_45 lead_45 beam_90 head_on defensive
#                        neutral_mirror topgun_break
python3 -m flybfm fight --blue guns --red level       --tag perch
python3 -m flybfm fight --blue instructor --red bnz   --random --seed 7

# a trained policy in either seat (name = file-stem of --out, so "poly")
python3 -m flybfm fight --blue poly --red guns --blue-ckpt runs/poly/training.json

# save it for the viewer
python3 -m flybfm fight --blue guns --red level --tag perch \
    --replay web/replays/perch.json
```

`--random` samples a fresh start from the scenario space (random position,
heading, speed, altitude). `--replay` writes the frame trace and refreshes
`web/replays/index.json`.

## 2. The viewer

```
python3 -m flybfm serve --dir web --port 8000
```

Then open `http://localhost:8000/` (on a hosted sandbox, the port is exposed as a
live preview automatically — the server already binds `0.0.0.0`). It is a single
`index.html` + `app.js` + `style.css`, no build step, no dependencies.

What is on screen:

| element | meaning |
|---|---|
`BLUE · FLY` | our side, with the policy name from the replay |
`RED · OPP` | the opponent |
ground plane | a real surface at **z = 0 m** (250 m below the hard floor), 2 km grid, 10 km bold lines, labelled `GROUND z = 0 m` |
altitude leaders | dashed line from each aircraft down to the ground with a caret on the surface and an **`AGL`** readout — up is up, down is down |
bank line | the wing line rotates with bank angle, so you can see a hard turn |
`GUN WEZ` | green when the target is inside 150–900 m *and* the nose is inside the tracking cone |
amber tracer | drawn whenever the trigger is down |
dashed ring | the 12 km arena boundary |

Controls: drag to orbit, wheel to zoom, click the timeline to scrub, play/pause,
0.25×–4× speed, camera chase-blue / chase-red / top-down / free. The side panel
shows telemetry, the policy's commanded roll/pull/throttle/trigger, the
manoeuvre mix, and the event log.

Note the icons are drawn at fixed size — a real F-16 is 9 m long in a 12 km
arena, so a to-scale aircraft is invisible. Positions, headings and altitudes
are exact; the drawings are symbols.

The blue aircraft in a replay is the fly.  Generate a replay of a trained policy
with:

```
python3 -m flybfm fight --blue poly --red guns --blue-ckpt runs/poly/training.json \
    --tag perch --max-time 60 --replay web/replays/poly_perch.json
```

## 3. Train

```
python3 -m flybfm train --policy features --basis poly --init imitation \
    --easy-frac 0.5 --generations 24 --population 24 --episodes 8 \
    --max-time 30 --seed 1 --out runs/poly
```

| flag | what it does |
|---|---|
`--policy features` | linear policy over the 18 geometry features (with `--basis poly`, a quadratic expansion of them) |
`--policy brain` | the connectome controller; trains its readout instead |
`--basis linear\|poly` | **`poly` matters**; `linear` cannot aim (see below) |
`--init imitation\|random` | warm-start by ridge-cloning the scripted pilots, or from noise |
`--resume PATH` | continue from a `training.json` (keeps its basis) |
`--sigma0 x` | CEM step size; smaller (0.2) when refining a good policy, 0.5 from scratch |
`--easy-frac 0.0–1.0` | fraction of *training* starts drawn at gunnery range (250–900 m, ±30° off nose). Evaluation stays uniformly random |
`--generations/population/episodes` | CEM: candidates per generation × fights per candidate |
`--eval-every` | generations between evaluation runs; the printed `win=` is the eval against the whole pool, not the training return |

What you should expect to see printed.  The warm-start line is deliberately
harsh about itself — the second half of it is a *closed-loop* result, because
command-space error alone says almost nothing here:

```
warm start: cloning the rule-based pilots ...
  clone vs teacher: |err| roll 0.31 (const 0.26), pull 0.05 (const 0.19),
    thr 0.22 (const 0.27); trigger agreement 0.98 vs 0.98 base rate (fired 0.000)
    || closed loop: W0 L6 D0, 0 rds / 0 hits, median |AA| 161 deg
training features (poly basis): 296 parameters, 24 generations x 24 candidates x 8 fights
[gen  0] best=   +3.38 mean=   -6.54 ( 12.8s)  win=0.49 hit%=0.0
...
best mean episode return: +9.86
opponent pool after training:  (PFSP importances: how hard each opponent still is)
```

That `W0 L6 D0 ... median |AA| 161 deg` is the state of the art for the
imitation warm start: it does **not** produce a pilot.  It is kept because it
still initialises the search near the mean command, and CEM does the rest — but
do not expect it to fly, and do not trust a small "|err|" without its constant
baseline (see `docs/EXPERIMENTS.md` §5.1).

Timing on two CPU cores: **~12 s per generation** for the poly policy, so a
24-generation run is about **5 minutes** and roughly 4,600 fights, including 288
evaluation fights. The brain policy is ~24 ms per decision, about 25× slower —
use 3–4 generations for that.

Artifacts: `runs/<out>/training.json` with `best.params`, the full `history`, the
opponent-pool table, and the `basis`/`init`/`easy_frac` it was trained with. A run
only writes that file at the end, so an interrupted run leaves nothing behind.

### 3.1 What a training run actually produces

Straight answer: **a policy that wins fights on points and cannot shoot.** That is
the measured state of this repository, not a guess, and it is the single most
important thing to expect. The numbers behind it:

| run | action basis | canonical win (points) | rounds fired | hits | gunnery verdict |
|---|---|---|---|---|---|
`runs/features_runA` | linear | 0.74 | 21,000 | 0 | never aims (median lead error 122°) |
`runs/features` (run D) | linear + fine-nose features | 0.47 | 9,267 | 0 | median lead error 29° while firing |
`runs/poly` | quadratic | 0.03 | 2,035 (eval) | 0 | 215 rounds / 0 hits when jittered; 0% of firing steps inside 1.5° |
`runs/poly_easy` | quadratic + gunnery curriculum | — | 0 | 0 | never pulls the trigger |
`runs/poly_refine` | quadratic, resumed from `runs/poly` | 0.03 | 657 (gunnery) | 1 | sprays harder: 77% of rounds after 2 s, median lead error 114° |

So two different failure modes trade off, and the reward decides which one you
get: the linear basis learns *points* BFM (be behind the other aircraft at the
bell) and sprays 20,000 rounds at nothing; the quadratic basis learns a slightly
better trigger gate, sprays less, and flies the general engagement badly. **No
learned policy in this repo has ever acquired or held a gun solution.** The
scripted pilots are the only agents that shoot: `guns` hits 3.4% of its rounds.

If you want to improve on that, the instrument to watch is the *late-fire* line
of `flybfm gunnery` (rounds fired after the first 2 s, and their lead error) —
not the win rate. A run that raises the win rate while that line stays at 140°
has learned nothing about gunnery. `runs/poly_refine` is the demonstration: it
resumed the gun-capable policy, trained 20 more generations at a small step
size, and came out firing *three times as many rounds* (657) for *one* hit,
with 77% of them fired more than 2 s in at a median lead error of 114°.

The honest reading of that is that the bottleneck is not search effort or model
capacity — both were increased and the aim did not move. It is the objective:
the reward's tracking term (max +1.5, saturating at 3°) is small next to the
points terms, so the optimiser buys wins, not solutions. Making tracking the
dominant, sharply-peaked term near the gun envelope — or supervising the aim law
directly with a closed-loop metric — is the next experiment, and `flybfm gunnery`
is how you tell whether it worked.

Timing, measured on two cores:

| policy | per fight (30 s of sim) | 24 generations × 24 candidates × 8 fights |
|---|---|---|
`features` (linear, 76 params) | ~0.05 s / 40 ms | ~5 min |
`features --basis poly` (296 params) | ~0.05 s | ~5 min (12 s/generation) |
`brain` (665-neuron connectome) | **~74 s per 12 s fight (~6× slower than real time)** | days — use 2-3 generations |

The brain path runs end-to-end (verified), and `probe` is how you check it is
alive, but it is not where any result in this repo came from, and its readout is
inert until a real mushroom-body subgraph is loaded. Long runs: always
`python3 -u`.

## 4. Evaluate and compare

```
python3 -m flybfm eval --checkpoint runs/poly/training.json --canonical
python3 -m flybfm eval --checkpoint runs/poly/training.json --n 32 --opponents 4
python3 -m flybfm league --per-pair 3          # scripted pool round-robin
```

`eval --canonical` flies the eight canonical BFM setups against the pool;
`--n 32` uses randomly sampled held-out starts instead. `league` ignores your
policy and ranks the ten scripted pilots against each other — it is the control
that tells you whether the pool is actually discriminating.

## 5. Check the gunnery, not the win rate

```
python3 -m flybfm gunnery --checkpoint runs/poly/training.json --opponent level
```

This is the honest instrument. It starts blue near-astern of a cooperative target
at 300 m (jittered per episode — a deterministic policy on a deterministic fight
would otherwise be measured on the *same* engagement N times) and reports the
**ballistic lead error** (`gun.required_lead_angle_deg`): the angle between the
nose and the point a round must actually be aimed at.  A fixed-forward gun needs
roughly:

| range | nose error for a hit |
|---|---|
300 m | ~3° |
600 m | ~1° |
900 m | ~0.5° |

and a policy can win 74% of fights with a **median lead error of 122°**, firing
twenty thousand rounds and hitting nothing — that is what `runs/features_runA`
did.  Read the gunnery number before believing any win rate.

The load-bearing line is the second-to-last one:

```
  shots after the first 2 s           59 of 126  (46.8%)
  lead error, shots after 2 s         median 143.68   p90 156.01   n=59
```

A policy that only fires when a perfect solution is handed to it at t = 0 scores
well on "lead error while firing" and cannot fight.  A policy that fires with the
target 144° behind it is spraying.  Everything measured in this repository so
far is in the second category: `runs/poly`, the best of them, fires 215 rounds
for 0 hits once the fights are jittered, with 0% of its firing steps inside
1.5°.  The scripted pilots are the only things here that hold a solution.

## 6. Probe the brain

```
python3 -m flybfm probe --steps 250 --seed 2 --out runs/probe_brain.json
```

Parks a target at a known angle off the nose and reports what the retina, the
target-selective cells and the descending neurons do. Use it whenever you change
anything in `brain/`: the four ways the brain goes silent all look identical from
the outside, and this prints the three numbers that separate them (population
rate → retinal peak cell → visual L/R difference). The `descending L/R diff`
verdict is expected to *fail* on the reduced synthetic graph; that is a real
limitation, reported rather than hidden.

## 7. Real connectome data (optional)

Everything above runs on a reduced **synthetic** graph. To use the actual
MaleCNS v1.0 connectome:

```
python3 -m flybfm.tools.fetch_connectome list
python3 -m flybfm.tools.fetch_connectome download --out data/malecns   # needs pandas
python3 -m flybfm.tools.fetch_connectome prune --source data/malecns --out data/malecns_pruned
export FLYBFM_CONNECTOME=data/malecns_pruned
export FLYBFM_ANNOTATIONS=data/malecns_pruned/annotations.feather
python3 -m flybfm probe --steps 250
```

The flat connectome is a 42 MB annotation table plus a ~1.1 GB edge list; the
download is the one step that needs third-party packages (pandas/pyarrow). The
loader is `brain.connectome.load_malecns_flat(...)`, and it prefers the
`FLYBFM_*` environment variables, so no code changes are needed to switch.
`prune` keeps the visual→descending→premotor subgraph, which is what makes a
full-brain graph tractable in pure Python. Details: `docs/DATA.md`.

## 8. Where things live

```
flybfm/sim/       point-mass F-16 EM model, geometry, gun ballistics, env, reward, scripted pool
flybfm/brain/     connectome, LIF, retinal encoder, three-factor plasticity, DN readout
flybfm/train/     CEM/ES loop, PFSP opponent league, imitation warm start, policies
flybfm/analysis/  manoeuvre classifier, episode metrics, scorecard
flybfm/tools/     probe_brain, gunnery_check, fetch_connectome
web/              the replay viewer (static; replays in web/replays/)
runs/             training artifacts (git-ignored)
docs/             DESIGN, REWARD, DATA, EXPERIMENTS, RUNNING (this file)
```

---

## Minor notes and gotchas

* **Run long jobs with `python3 -u`.** Piecewise-linear buffering hides the
  progress of a five-minute run, and half the time the log simply looks empty.
* **`decision_dt` is not a lever.** 0.1 s vs 0.02 s changes nothing
  qualitatively; the physics steps at 0.02 s either way.
* Kill counts are rare and that is fine: across 60 sampled fights, 51 were decided
  on **points**, not kills. Points go to whoever is *behind* the other at the
  bell (the moment the aspect sign was fixed, the `level` opponent went from the
  hardest in the pool to the easiest — a passive target cannot be hard).
* The environment's `t` overshoots `max_time_s` by one 0.02 s step. Tests are
  written accordingly; don't "fix" it.
* Round ground speed is ~1050 m/s and rounds live 3 s, so the 900 m envelope is
  not a range limit — it is a *standard* WEZ. Extend it and hit rates fall off a
  cliff because the required lead error shrinks with range.
* The aircraft leaves the arena at 12 km radius and dies below 250 m AGL or above
  13 km. Both are worth −20/−30 reward; a policy that scores well by fleeing is
  being scored on points, not on flying.
* `run_match(...)` returns `(summary, trace, env)`; the training path's
  `run_episode(...)` returns `(info, env)`. Read the signature before scripting.
* `Geometry` has no `.range`/`.rng`; it is `range_m`, `aa_deg`, `closure_ms`,
  `taa_deg`. Distances in the summaries are metres, speeds m/s, angles degrees
  (the modules use radians internally).
* Everything is seeded (`--seed`); two runs of the same command give identical
  numbers. The scripted pilots hold their own `random.Random`, so fights are
  reproducible but the pool is not frozen across code changes.
* `web/replays/*.json` are generated, not hand-written. After adding one, the
  manifest picks it up automatically (or run `python3 -m flybfm replay --dir
  web/replays`).

## Troubleshooting

| symptom | cause |
|---|---|
`0 rounds fired` in a fight | the round accumulator, or the policy never enters the WEZ. Check `blue_shots` in the summary and `frac of steps inside the envelope` in `gunnery` |
win rate high, hits 0 | the policy is flying for points; run `gunnery` |
brain silent (all rates 0) | `bg_current` / `gain_photo` too low; check with `probe` |
brain spikes but does not track bearing | the blob is smaller than the receptor spacing, or the mosaic is not laid out as its own receptive fields |
`descending L/R diff` flat | left and right copies of one DN type summed together — split by `conn.sides` |
viewer shows "no replay" | empty `web/replays/`; generate one with `--replay` |
`ModuleNotFoundError: flybfm` | run from the repository root, not from inside a subpackage |
