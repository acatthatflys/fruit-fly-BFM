# Can a fly brain learn basic fighter manoeuvres?

*Design document for `fruit-fly-BFM`. Written 2026-09-14.*

---

## 0. Short answer

**Feasible:** building a full-brain-scale simulation of the male *Drosophila*
CNS, driving it with a synthetic retina, reading out descending neurons, and
having a point-mass fighter fly the result. All of that runs today, and this
repo does a reduced version of it.

**Not feasible (yet, and probably not the right framing):** expecting the fly
brain *itself* to invent basic fighter manoeuvres — energy management, lag and
lead pursuit, the vertical fight. A fly is a 1 mg animal that flies at 3 m/s and
possesses no gun, no radar, no energy state worth managing and no evolutionary
pressure toward 9-g turning. Nothing in the connectome encodes "high yo-yo".

**What the viral demos actually are.** Read them carefully and every one of them
is the same construction:

| Layer | What it is | Learned? |
|---|---|---|
| Input | photoreceptors stimulated by a hand-written encoder mapping game/sim state → R1-R6 activity | hand-designed |
| Wiring | the real connectome, treated as a fixed sparse graph | not learnable |
| Neurons | leaky integrate-and-fire with *guessed* parameters, uniform *guessed* efficacies | guessed |
| Output | some descending (or arbitrary) neurons read out, usually by a conventional ML model or a hand-wired mapping | learned / hand-wired |
| Reward | current injected into two dopaminergic neurons | hand-wired |

So the fly brain is a **fixed nonlinear reservoir**, and the thing that is
actually learning to play Doom / trade stocks / fly the aircraft is a small
conventional model sitting on top of it. `doomfly` is explicit about its own
result: after 3,000 runs, no learning. That is not a bug in their code; it is
the expected outcome of injecting reward into two cells and hoping that
eligibility traces over guessed weights do the rest.

That is not a reason not to do it. It is a reason to be precise about *which*
part learns, and to measure it. This document does that.

---

## 1. What Basic Fighter Manoeuvres actually are

BFM is the doctrine of winning a close-in, guns-only engagement. It is not "fly
at the enemy". It is a set of coupled control laws over a state that includes
your energy and his:

* **Pursuit geometry.** With the line-of-sight (LOS) fixed between you:
  *pure pursuit* points the nose at the target (works only from dead astern),
  *lead pursuit* points ahead of him (needed to shoot and to cut him off),
  *lag pursuit* points behind him (needed to preserve turning room and avoid an
  overshoot). Switching between them, at the right times, is the core skill.
* **Energy management.** Specific energy `E_s = h + V²/2g` and its rate `P_s`.
  Every hard turn costs energy; the aircraft that arrives at the merge with more
  energy wins the subsequent turn. Corner speed is where instantaneous and
  sustained turn performance cross.
* **One-circle and two-circle geometry**, nose-to-nose or nose-to-tail, which
  determines whether you should roll *into* the bandit or away from him.
* **The named manoeuvres** — high yo-yo, low yo-yo, barrel roll attack,
  break turn, scissors, extension, Immelmann, split-S.
* **The gun equation.** A fixed forward gun needs the nose *and the aircraft's
  own velocity vector* slaved to a lead point, with gravity drop compensated.
  You must be inside a ~10° cone at short range, which in practice means flying
  smoothly at high g, which costs energy. Everything above exists to buy that
  second of tracking.

A learned agent that only ever points its nose at the target does not fly BFM.
Measurable consequence, used throughout this repo: track **pursuit mix** and
**manoeuvre content** (see `flybfm/analysis/maneuvers.py`) alongside win rate.

---

## 2. Feasibility, split by claim

| Claim | Verdict | Why |
|---|---|---|
| Simulate the whole male CNS as LIF neurons | **Yes, today** | 166k neurons / ~125M synapses; GitHub `doomfly` and `philshiu/Drosophila_brain_model` (Shiu et al., *Nature* 2024) already do whole-brain LIF; a GPU tensor implementation runs it at or faster than real time |
| Drive it with a retina and read out DNs | **Yes** | R1-R6 (≈3.3k) and R8 (≈0.8k) are annotated in MaleCNS; the descending-neuron catalogue is complete and `neuPrint` queries it |
| Let the fly *fly* a body | **Yes, with a body sim** | NeuroMechFly v2 / FlyGym gives a biomechanical fly with wings and halteres; this repo instead flies a point-mass aircraft because BFM is an aircraft manoeuvre |
| Learn to shoot guns from dopamine into PPL101 | **No** | Sparse reward, credit assignment over 166k neurons, guessed plasticity rules. Ablate it and see: `doomfly`'s 3,000 fruitless runs is the honest baseline |
| Learn BFM at all | **Yes, but by a conventional learner** | A small policy over geometry features learns pursuit geometry in minutes (this repo); AlphaDogfight-class results need hierarchical RL over millions of episodes |
| The fly brain adds value over a random reservoir | **Open question worth answering** | Test: same learner, same feature budget, controller = real connectome vs shuffled connectome vs random MLP vs linear. If the connectome doesn't beat the shuffled version, you have measured something real |
| A fly is a better fighter than an F-16 pilot | **Category error** | Different sensor, different scale, different dynamics; see §7 |

### The three specific engineering problems

1. **No synapse weights exist.** The connectome gives *synapse counts*, not
   conductances. Doomfly's approach — uniform efficacy per connection, sign from
   neurotransmitter identity — is defensible, but it means your dynamics are a
   guess, and a 25-condition guess on a chaotic system. Anything you learn rides
   on that guess. Report it as a guess.
2. **The readout carries the information, and the readout is yours.** In the
   reduced connectome in this repo the target's bearing is clearly present in
   the photoreceptors and the small-target (STMD) layer, but the **descending**
   lateral channel (`left − right` DN rate) is essentially bearing-invariant
   (`python -m flybfm probe`). Viral demos hit the same wall and solve it the
   same way: a learned decoder reads activity from wherever it can. Do that, but
   *say* that you did.
3. **Reward is a single scalar delivered to a handful of cells.** Dopamine
   teaches an eligibility trace. With guessed weights, the trace points
   somewhere unhelpful. See §6 for the three-level workaround.

---

## 3. Where to actually get the brain

Everything is public and CC-BY. Details, URLs and loader code:
[`DATA.md`](DATA.md). Summary:

| Source | Content | Access |
|---|---|---|
| **MaleCNS v1.0** (`male-cns:v1.0`) | the Sept 2026 complete male CNS: ~166k neurons, ~125M synapses, brain **and VNC**, with descending and *ascending* neurons | `neuPrint` (Python/R), flat-connectome `.feather` on GCS, Cell Type Explorer, Neuroglancer scene |
| **FlyWire (female brain)** | 139k neurons of the female brain, no VNC, the dataset behind most published whole-brain models | `codex.flywire.ai`, `flywire_annotations` (CC-BY), Zenodo dump |
| **MANC** | male VNC, the motor side: DN → wing/haltere/leg motor neurons | via MaleCNS v1.0 or the MANC release |
| **FlyGym / NeuroMechFly v2** | biomechanical fly body (Apache-2.0) | `pip install flygym` |
| **FlyVision** | connectome-constrained visual system (lamina → lobula plate), ready-made | `TuragaLab/flyvis` |

The important structural point for this project: **male CNS includes the VNC and
the ascending neurons.** FlyWire-only demos have to bolt on a fake interface
between "brain" and body. Here the real descending pathway — DN axons leaving
the brain for the wing motor neurons — is in the same dataset, so the
brain→wings interface is anatomically grounded, which is exactly the interface
this project reads out.

---

## 4. Architecture of this repo

```
   engagement geometry ─┐
                        ├─► sim/aircraft.py   point-mass energy fighter (F-16-ish, configurable)
   guns, WEZ, lead ─────┘        │
                                 ▼
                     sim/dogfight.py  50 Hz physics / 10 Hz decisions, guns-only, arena
                                 │                                   ▲
              ┌──────────────────┴───────────────┐                   │
              ▼                                  ▼                   │
   brain/sensors.py                      sim/scripted.py       sim/reward.py
   retina-style encoder                  opponent pool         events + potential
   (azimuth × elevation, R1-R6/R8)       (10 strategies)       (policy-invariant)
              │                                  │                   │
              ▼                                  │                   ▼
   brain/lif.py  sparse LIF + eligibility traces │           brain/plasticity.py
              │                                  │           TD error → reward /
              ▼                                  │           punishment DA cells
   brain/readout.py  descending neurons → controls                   │
   (DNp26/57/03 saccade hubs, DNg02, DNp06, ...)                     │
              │                                                       │
              ▼                                                       │
   train/policy.py  learnable readout ◄── train/cem.py (ES/CEM) ◄── train/league.py
                                            │                        (PFSP + frozen
                                            ▼                         snapshots)
                                    analysis/metrics.py, analysis/maneuvers.py
```

Everything in the loop is swappable, because the interesting experiments are the
ablations: connectome on/off, plasticity on/off, DA shaping level 1/2/3,
opponent pool uniform vs PFSP.

**Two layers of learning, deliberately separated — and which one actually learns:**

1. **Inner (biological, the fly's own).** Mushroom-body-style three-factor plasticity:
   `Δw_ij = η · e_ij · DA(t) − λ w_ij`, with the eligibility trace a decaying
   pre×post coincidence. This is the fly's own learning rule and it is implemented
   in `brain/lif.py` (`LIFNetwork.mark_plastic`, `deliver_dopamine`) and in
   `brain/plasticity.py`. The synthetic graph has **66 KC→MBON plastic edges**
   (measured: `build_controller(plasticity=True).n_plastic == 66`), and the
   torch path (`TorchLIFNetwork`) implements the same rule with a sparse matmul.

   **Crucially, it is off by default and inert in all reported results.** The
   controller is constructed with `plasticity=False` unless you explicitly pass
   `plasticity=True`; the training loop in `train/cem.py` never enables it; and
   every number in `runs/` and in `docs/EXPERIMENTS.md` comes from the outer loop
   alone. On the synthetic graph, enabling it does change weights (see
   `tests/test_brain.py::test_plasticity_path_exists`), but the signal is weak,
   noisy, and insufficient to learn gunnery on its own — which is itself a result
   worth reporting, and exactly what `doomfly`'s "3,000 runs, no learning" honest
   baseline shows. If you share results publicly, say: *zero of the current
   learning is the fly's own dopamine-gated plasticity; 100% is the outer CEM
   loop* — unless you have explicitly enabled the inner loop and measured it.

2. **Outer (engineering, what actually learns).** Cross-entropy method over the
   readout parameters. Chosen over PPO/SAC for three reasons: (a) the spiking
   controller cannot be back-propagated through without surrogate gradients and
   a torch rewrite (now available as `TorchLIFNetwork`, but still non-differentiable
   without surrogate gradients); (b) the search space is 10–100 numbers, where ES
   is competitive and far more robust to reward scale; (c) it is deterministic
   given a seed, which makes every number in the repo reproducible. If you want
   gradients, the env already exposes `observation_vector` and a dense
   potential-based reward — PPO drops straight in.

---

## 5. The engagement environment

* **Flight model.** Point-mass energy model (the standard
  energy-manoeuvrability formulation), because BFM is *about* energy:
  `V̇ = g((T−D)/W − sin γ)`, `γ̇ = g(n cos μ − cos γ)/V`,
  `ψ̇ = g n sin μ / (V cos γ)`. Drag includes induced drag from the current
  load factor, thrust lapses with density and Mach, load factor is limited by
  both the structural limit and `Cl_max`. Default parameters are F-16-like:
  corner speed ≈176 m/s, 9 g, 76/128 kN dry/AB — i.e. the model reproduces the
  real aircraft's corner-speed behaviour, which is what makes turns at the right
  speed matter.
* **Guns.** Fixed forward, 6,000 rds/min, 1050 m/s muzzle, gravity drop,
  dispersion, and — critically — the bullet's velocity is *muzzle in the
  shooter's frame*, so the shooter's own velocity is in the solution. Hit
  radius 3 m. Tracking shots require the nose to be inside ~1° at 900 m; that
  number sets the entire difficulty of the task.
* **Arena.** 12 km radius, 250 m floor, 13 km ceiling, 90 s default cap.
  Leaving or crashing costs the episode.
* **Decision rate.** 10 Hz for the policy, 50 Hz physics — so gun resolution
  never depends on the decision rate. The brain can be run at any internal rate
  (default 2 kHz LIF inside a decision interval).
* **Starting conditions.** `Scenario` covers the classic setups (perch, lag 45,
  lead 45, beam 90, head-on, defensive) and `ScenarioSpace` samples the same
  parameters uniformly, with jitter — this is the domain randomisation that
  §8 leans on.

---

## 6. Reward design

Full detail, including the exact constants, in [`REWARD.md`](REWARD.md). The
summary you asked for — *how are people setting up rewards?* — has three levels,
and it is worth being blunt about which is which.

**Level 1 — what the demos do.** Inject current into the two PPL101 neurons when
something bad happens (and / or a reward cluster when something good happens).
No temporal credit assignment, no value function, no shaping. This is the setup
that produces "3,000 runs, zero learning".

**Level 2 — the biologically correct version.** Dopamine *is* a temporal
difference error signal (Schultz); PPL1 encodes the aversive side and PAM the
appetitive side. So deliver `δ_t = r_t + γV(s_{t+1}) − V(s_t)` into the DA
clusters, where `V` is a small learned critic. That gets the temporal semantics
right — eligibility traces are gated by prediction error, not by raw outcome —
and it is implementable in a few dozen lines (`brain/plasticity.py`).

**Level 3 — what makes it train at all.** Sparse gun events give an RL agent
almost nothing to learn from: a hit happens perhaps once per 1,500 rounds in
this sim's baseline. So add **policy-invariant potential-based shaping**:

```
r_t = events + γΦ(s_{t+1}) − Φ(s_t)
```

with `Φ` built from BFM quantities — required lead angle, target aspect, range
relative to the gun envelope, specific-energy advantage, closure. Ng, Harada &
Russell (1999) proved that this form leaves the optimal policy unchanged, so it
is dense guidance without a hand-written script: **you cannot shape the fly into
a marionette, only accelerate it toward the same optimum.** That is the property
to insist on if you care about the agent's behaviour being *learned*.

Two more pieces matter:

* **Event symmetry.** Reward hits dealt and punish hits taken; reward a kill far
  more than any shaping term; punish crashing and leaving the arena, or the
  agent learns to win by geometry abuse.
* **Timeout rule.** On the clock, award the fight on damage first, then on
  position (`cos TAA`). Otherwise "run away forever" is optimal.

---

## 7. The scale problem, stated plainly

| | Fly | F-16 |
|---|---|---|
| mass | ~1 mg | 9,200 kg |
| speed | 2–3 m/s | 250 m/s |
| sensor | ~700 ommatidia/eye, 4.5° inter-ommatidial angle | radar + HUD, arc-second resolution |
| control | 200 Hz wingbeat, saccades every ~50 ms | control surfaces, 9 g limit, 1–2 s energy constant |
| mission | hover, courtship, escape | intercept and destroy |

A fly's sensorimotor repertoire is saccadic fixation, optomotor stabilisation,
looming escape and object avoidance. All of these are **useful primitives** for
pursuit (and `DNp26/DNp57/DNp03` really are the saccade hubs, `DNg02` steers,
`DNp06` holds heading, `DNp10` lands) — so a fly-brain controller *can* plausibly
produce "align, fixate, track". What it cannot produce is the doctrine.

Two honest ways to spend the effort:

* **Track A — "micro-BFM".** Shrink the whole engagement to fly scale: a
  1 g-class MAV with a laser or airsoft-scale "gun", speeds in metres per
  second, engagement ranges in tens of metres. Then the fly's sensorimotor
  repertoire is *relevant*, the Reynolds-number mismatch goes away (use a
  flapping-flight body model), and the result — "a connectome-constrained
  controller learns to intercept a manoeuvring target" — is a real contribution
  to MAV autonomy.
* **Track B — "connectome as architecture".** Keep the F-16-class engagement and
  ask a scientific question instead of building a warhead: *does a
  connectome-constrained controller learn BFM faster, or more robustly, than a
  parameter-matched MLP?* Shuffle the connectome as a control. This is the study
  that would actually get cited, and it cannot fail to produce a result.

Do both, in that order: Track A first because it is honest about the body, Track
B because it is honest about the question.

---

## 8. Preventing over-tuning

Your instinct — randomise the start, and give the opponent a repertoire — is
correct and it is what the air-combat RL literature converged on independently.
The full kit, in the order it starts to matter:

1. **Domain randomisation of the initial condition** (already in
   `ScenarioSpace`): range, aspect, heading crossing angle, vertical offset,
   altitude band 3–8 km, and both aircraft's speeds. Jitter every canonical
   setup, and never let a training episode start from the same numbers twice.
2. **Opponent pool.** Ten scripted strategies (level, nose-on, lead, lag, gun
   solution, break, vertical, boom-and-zoom, jinker, instructor). A learner that
   only ever sees one gets a counter-strategy, not a skill.
3. **Prioritised fictitious self-play (PFSP).** Sample opponents in proportion
   to how often they still beat you, with a floor so that you keep rehearsing
   what you have already solved. Implemented in `train/league.py`. Uniform
   sampling wastes most of its budget on solved opponents.
4. **League with frozen snapshots.** Every *k* generations, freeze the learner
   and put it in the pool. Self-play against a single live opponent cycles
   (the Red Queen effect): both sides improve while measured skill goes
   sideways. Freezing breaks the cycle.
5. **Randomise your own dynamics, a little.** ±10% on thrust, drag, mass, gun
   dispersion, sensor latency. An agent tuned to exact parameters is a memoriser
   of those parameters. This is also the cheapest robustness win available.
6. **Sensor noise and delay.** 50–100 ms of visual latency (the fly's real
   lobula latency is of this order) plus noise. It makes tracking behaviour
   smooth rather than twitchy, which is what makes the gun work.
7. **Held-out evaluation that is actually held out.** Fixed seeds, scenarios the
   trainer never samples, and a scorecard — not just win rate. Report pursuit
   mix, tracking time, hit rate, energy advantage, and how often the agent wins
   *by damaging the opponent* versus by running the clock.
8. **Regularise the readout.** L2 on the policy weights; keep the parameter
   count small (10–100, not 100k) so that ES explores rather than memorises.
9. **Early stopping on the held-out scorecard**, never on the training return.
10. **Watch for the degenerate strategies** explicitly: running away, abusing the
    arena boundary, camping the floor, flying an opponent into the ground. Each
    gets an explicit penalty and a metric, because each is a way to "win" while
    learning nothing.

The single most valuable of these is #2 + #3, and the single most-often-omitted
is #7.

---

## 9. Roadmap with pass/fail gates

| # | Milestone | Pass condition |
|---|---|---|
| 0 | Sim runs, no anomalies | No crashes/exits in pooled fights; turn performance at corner speed matches the analytic value |
| 1 | Rule-based pilots fight | Pool hit rate > 5%, engagement decides > 80% of fights |
| 2 | Learner beats the random floor | Trained policy's held-out scorecard beats a random policy on tracking time and hit rate |
| 3 | Learner beats a scripted pilot | Held-out win rate > 0.5 against the pool, with hit rate > the best scripted pilot's |
| 4 | Maneuvres appear | Pursuit mix contains lead *and* lag pursuit; at least one high yo-yo or break turn per fight is detected |
| 5 | Full connectome in the loop | `male-cns:v1.0` subgraph (visual → DN) drives the same policy interface; latency budget documented |
| 6 | The scientific question | Connectome vs shuffled-connectome at matched parameter count, ≥ 100 held-out fights, reported with error bars |

Milestones 0–3 run on this machine. 4 is measurable with
`analysis/maneuvers.py`. 5 needs the 1.1 GB flat connectome and ~16 GB RAM or a
GPU. 6 needs the loop to be fast, i.e. torch.

---

## 10. Limitations of this repo, explicitly

* The default "brain" is a **reduced synthetic connectome** with realistic
  degree statistics, not MaleCNS. It exercises the whole pipeline offline; the
  real loader is in `tools/fetch_connectome.py` and needs the download. The
  synthetic graph has 665 neurons / 2.2k edges and 66 KC→MBON plastic edges —
  enough to unit-test the plasticity path, not enough to learn BFM biologically.
* Synaptic weights are **not** identifiable from the connectome; the model uses
  uniform magnitudes with signs from transmitter identity, and a global gain
  that is calibrated, not measured.
* The mushroom-body plasticity is **implemented, tested, and off by default**.
  `LIFNetwork.mark_plastic` finds KC→MBON edges in both synthetic and real graphs
  (see `tests/test_brain.py` and `tests/test_connectome_loader.py`), and
  `deliver_dopamine` applies `Δw = η·e·DA − λw`. However, `ConnectomeController`
  defaults to `plasticity=False`, the CEM trainer never enables it, and all
  results in `runs/` come from the outer loop. The inner loop is weak, noisy,
  and on its own insufficient to learn gunnery — which is why `doomfly` reports
  "3,000 runs, no learning" and why this repo is explicit: **if you quote a win
  rate, 0% of that learning is the fly's own dopamine-gated plasticity unless
  you explicitly enabled it**. The honest ablation is `plasticity=True` vs
  `False` on the real MaleCNS mushroom-body subgraph.
* The manoeuvre classifier is a **heuristic**. It is stated as one, and it is
  meant to be replaced by a learned classifier once there is data to train it.
* The readout consumes upstream visual activity as well as DN activity, because
  the DN channel in the reduced graph is bearing-invariant. That is disclosed in
  `controller._groups()` and is the same compromise every viral demo makes
  silently. `vis_gain` in `ReadoutParams` tells you which channel the behaviour
  actually depends on.
* Torch path: `LIFNetwork.to_torch()` now returns a runnable `TorchLIFNetwork`
  (CPU or CUDA, sparse matmul `W[post,pre] @ spikes`), not a `NotImplementedError`.
  Pure-Python LIF is ~1-3M synaptic events/s; the full 166k/125M CNS needs GPU
  and runs ~1-5 ms/step on A100 — this was the blocker on milestones 5-6 and is
  now unblocked.
* Packaging: `pyproject.toml` + `requirements.txt` + CI (`.github/workflows/ci.yml`
  runs `python -m unittest discover -s tests` on every push). The loader's
  column-sniffing has a regression test with a tiny synthetic `.feather` fixture
  (`tests/test_connectome_loader.py`) because `DATA.md` flags that columns move
  between releases.
* This is a simulation of air combat for reinforcement-learning research. It is
  a toy: no real vehicle, no real weapon, no export-controlled content.

### 10.1 Current status, measured

The learnable policies in this repo **win fights on points and cannot shoot**.
Best measured case, `runs/poly`: a quadratic readout over the 18 geometry
features reaches a few percent of rounds on target in training evaluations, and
in the controlled gunnery test fires 215 rounds for **0 hits**, with 0% of its
firing steps inside 1.5° of the required lead angle and 47% of its rounds fired
with the target more than 90° behind it. The scripted rule-based pilots are the
only agents here that hold a firing solution. The instrument that shows this —
and the three ways earlier metrics hid it — is `docs/EXPERIMENTS.md` §5, and it
is worth reading before trusting any win rate quoted anywhere in this repo.

**How to run any of it, and what to expect:** `docs/RUNNING.md`.
