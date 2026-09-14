# Measurements

Produced on a 2-core, 3 GB sandbox with no numpy and no torch. The point of the
wall-clock numbers is that the whole training loop runs on a laptop, because the
part that needs a GPU (166k-neuron simulation) is deliberately kept out of the
inner loop.

Reproduce each block with the command shown. `python -m unittest discover -s tests`
checks the numbers that are labelled as assertions.

---

## 0. Two bugs worth more than any tuning

Both were found by *measuring*, and neither announced itself as a bug — each
announced itself as "the learner keeps doing something stupid", which is exactly
how a geometry or ballistics error disguises itself.

### 0.1 The aspect angle was inverted (`geometry.py`)

```python
back = vm.scale(vm.sub(shooter.pos, target.pos), -1.0)   # WRONG: that is the LOS
```

`(shooter - target) * -1` is the line of sight *from the shooter to the target*,
not the line from the target back to the shooter, so `taa_deg` came out inverted:

| geometry | TAA reported | TAA correct |
|---|---|---|
tail chase (shooter on the target's six) | 180° | **0°** |
head-on | 0° | **180°** |

Aspect is not a diagnostic number in this project — it is load-bearing:

* the reward's aspect term rewarded being **in front of** the opponent;
* the timeout points decision awarded the fight to whoever was **in front**;
* the scripted pilots' offensive/defensive branches were flipped;
* the manoeuvre classifier labelled break turns as extensions.

The observable symptom was that a wings-level target that never fired a shot was
the *hardest* opponent in the pool (5.6% win rate for everyone who fought it) and
that a learner could reach a 74% win rate while firing 21,000 rounds for zero
hits. After the fix, `level` is the weakest opponent in the pool, as it should be.

### 0.2 The gun disagreed with itself about gravity (`gun.py`)

`lead_solution` compensated for the round's drop; `_intersects` did not model it,
and the drop term's sign was wrong in both places. The two errors do *not* cancel
— the residual miss grows quadratically with range:

| range | miss before the fix |
|---|---|
500 m | 2.3 m |
700 m | 4.5 m |
900 m | 7.5 m |

against a 4 m hit radius. The gun was, in effect, a gun that could only hit at
short range, and every conclusion drawn about "the policy cannot shoot" was drawn
through it.

Both fixed, with the analytic-lead solver, the perfect-aim hit rate is now a
smooth physical function of range (table in §2).

---

## 1. Flight model (`tests/test_sim.py`)

| check | analytic | model |
|---|---|---|
corner speed (9 g, Cl_max 1.5, 9200 kg, 27.87 m², sea level) | 178.1 m/s | 178.1 m/s |
sustained-turn speed at 128 kN AB | — | 124.1 m/s |
turn rate at 9 g / 250 m/s, banked for level flight | 20.10 °/s | 20.12 °/s |
90° of heading at that bank | ≈4.5 s steady | 4.5 s + 1.5 s of g-onset |

## 2. Gun ballistics (`tests/test_gun.py`)

Synthetic perfect aim — the nose is slaved exactly to the ballistic lead point —
1,000 rounds per condition, hit radius 4 m, dispersion 2.5 mrad:

| geometry | range | hit rate |
|---|---|---|
stern (own 300 m/s, target 250 m/s) | 300 m | 100% |
 | 500 m | 99.8% |
 | 700 m | 93.8% |
 | 900 m | 80.5% |
 | 1200 m | 58.6% |
 | 1500 m | 40.4% |
head-on (both 300 m/s) | 900 m | 98.3% (the round is fired into the target's motion) |

That table *is* the difficulty curve of the task: a round must be aimed inside
about a degree at 600 m, and the tolerance halves every time the range doubles.

## 3. Connectome probe (`python -m flybfm probe --seed 2 --steps 250`)

Bearing sweep, target at 1,200 m, 500 ms of brain time per bearing, using the
*measured* nose error:

| measured nose error | R1-R6 peak `pref_az` | photoreceptor spikes | population rate |
|---|---|---|---|
+60° | +45° | 40 / 1346 | 0.030 Hz |
0° | −15° | 40 / 1365 | 0.028 Hz |
−60° | −75° | 40 / 1359 | 0.029 Hz |

The retina **tracks the target's bearing** — that is now an assertion in
`tests/test_brain.py::test_retina_tracks_bearing`, and it is the cheapest test
that catches the entire class of "the brain is silent / has no bearing sense"
bugs. Four distinct causes produced that symptom during development, all of which
look identical from outside:

1. the target's image was smaller than the receptor spacing, so the retina
   emitted the same spike train for every bearing;
2. the sensor mosaic's coordinates were not the receptive fields;
3. propagation gain was below spike threshold;
4. the readout summed left and right copies of the same descending neuron.

Diagnose in that order: population rate → retinal peak cell → small-target cell →
descending groups by side.

**Downstream, in the reduced synthetic graph, the descending lateral channel is
still bearing-invariant** — the differential signal is at the noise floor
(`R−L` between +0.005 and +0.017 Hz across the sweep, not monotone). That is why
`controller._groups()` reads the target-selective (STMD) population as well as
the descending neurons, and it is disclosed rather than hidden. Replacing the
synthetic graph with a real `male-cns:v1.0` subgraph is milestone 5.

## 4. Scripted pool baseline (`python -m flybfm league --per-pair 3`)

135 fights (all 45 pairings × 3), 32 s. Ranking under the fixed conventions:

| policy | W | L | D | win % |
|---|---|---|---|---|
lead | 22 | 5 | 0 | 81.5 |
guns | 21 | 5 | 1 | 77.8 |
lag | 19 | 7 | 1 | 70.4 |
jinker | 19 | 8 | 0 | 70.4 |
nose-on | 18 | 8 | 1 | 66.7 |
bnz | 10 | 14 | 3 | 37.0 |
instructor | 7 | 19 | 1 | 25.9 |
break | 6 | 19 | 2 | 22.2 |
vertical | 6 | 20 | 1 | 22.2 |
level | 2 | 25 | 0 | **7.4** |

Before the aspect fix, `level` was near the top of that table. A passive target
cannot be the hardest target.

Most fights end with both aircraft alive and are decided on the points rule;
3 of 66 sampled fights ended in a shoot-down. That is the honest baseline: even a
carefully written rule-based pilot only lands a hit in a few percent of its
rounds, which is why sparse-event rewards do not train and potential shaping is
not optional.

## 5. Learning to fight

```
python -m flybfm train --policy features --basis poly --init imitation \
    --generations 24 --population 24 --episodes 8 --max-time 30 --seed 1 --out runs/poly
```

Two measurement instruments, and they disagree in an informative way:

* **win rate** — `flybfm eval --checkpoint ... --canonical`, or the `win=` printed
  every `--eval-every` generations.  This is *points* BFM: who is behind whom at
  the bell.
* **gunnery** — `flybfm gunnery --checkpoint ...`.  This is *guns* BFM: the
  ballistic lead error, i.e. whether a round could connect at all.

| run | basis | warm start | canonical win | rounds | hits | median lead error while firing |
|---|---|---|---|---|---|---|
`runs/features_runA` | linear | both-sides clone | 0.74 *(a)* | 21,000 | 0 | — |
`runs/features` (run D) | linear | both-sides clone | 0.47 | 9,267 (eval) | 0 | 29° |
`runs/poly` | quadratic | both-sides clone | **0.03** | 2,035 (eval) | 0 | 10° |
`runs/poly_easy` | quadratic | clone + curriculum | — | 0 | 0 | — |

*(a)* reported by the training loop at the time, i.e. the win rate of the CEM
distribution mean, not of the saved artifact — see the inconsistencies listed at
the end of this section.  Treat pre-`poly` win rates as indicative rather than
comparable.

Note the scale of the spray even where hits do happen: the *in-training*
evaluations of `runs/poly` (288 fights against the whole pool, distribution-mean
policy) recorded 2, 7, 1, 3, 2, 4 and 3 hits — out of 14,094, 22,027, 12,865,
13,029, 12,222, 11,601 and 10,062 rounds respectively.  That is 0.02-0.06% of
rounds.  The scripted `guns` pilot, by contrast, hits 3.4% of its rounds.

**What this table says.** Every learnable policy in this repository wins fights
on points and cannot shoot.  The quadratic basis produces a *slightly* better
trigger gate (it fires when it is already aligned and stays silent otherwise) but
its general BFM is poor.  No policy in the table — linear or quadratic, imitated
or evolved — has ever **acquired** a gun solution.

**How that claim was measured, because the first three versions of this table
were wrong.**  `flybfm gunnery` now reports the number that settles it:

```
policy runs/poly/training.json vs level   8 fights, start R=300 m TAA=0 deg
result  W 6 / L 0 / D 2        rounds fired 215, hits 0   (0.00% of rounds)
ballistic lead error |deg|   all steps median 72.9   in envelope median 32.0
  while firing             median 10.13   p90 150.51   n=126
  trigger pressed with |lead| <= 1.5 deg: 0.0% of firing steps
  shots after the first 2 s           59 of 126  (46.8%)
  lead error, shots after 2 s         median 143.68   p90 156.01   n=59
```

Read that bottom line: 47% of the rounds are fired more than two seconds in,
with a **median lead error of 144°** — the target is behind the aircraft and the
trigger is still down.  The policy is not tracking; it is spraying.

### 5.1 Three ways the measurement lied first

Every wrong number in this project was produced by a metric that looked
reasonable.  They are worth recording, because each one is a trap in any
sparse-event, fast-dynamics domain:

1. **Trigger agreement against a base rate.**  The scripted pilots hold the
   trigger down 2% of the time (0.4% in mixed-teacher data).  A warm start that
   *never fires* therefore scores 0.98 agreement.  `clone_report` now prints the
   agreement *and* the base rate it must beat.
2. **Command-space MAE with no constant-predictor baseline.**  Pull MAE of 0.05
   looks like a good clone until you notice the best constant predictor scores
   0.19 and the *roll* channel of the same clone is worse than a constant
   (0.31 vs 0.26).  A clone can fit the mean command, fly like the random policy
   (median angle-off 150°, 0 wins in 18 fights me), and still look precise.
3. **A deterministic policy evaluated on a deterministic fight.**  Starting the
   poly policy 300 m dead astern of a *straight-flying* target gives the same
   fight every time; the first version of the gunnery table reported 6 identical
   fights as `W 6 / 0 lost / 24 hits / 25% of rounds` — the 25% was the policy
   pulling the trigger at t=0 with a perfect solution handed to it, and the
   target's dispersion-driven 4 hits.  Add per-episode jitter (range ±15°,
   TAA ±10°, nose offset ±8°, dz ±60 m) and the same policy lands **0 hits from
   215 rounds**.  `gunnery_check` jitters now.

### 5.2 Resuming the gun-capable policy made it worse, which is the point

`runs/poly_refine` resumed `runs/poly` (quadratic basis, the only checkpoint that
ever landed hits) for 20 more generations at `--sigma0 0.2`:

| | `runs/poly` | `runs/poly_refine` |
|---|---|---|
rounds fired (jittered gunnery, 8 fights) | 215 | 657 |
hits | 0 | 1 |
shots after the first 2 s | 46.8% | 77.3% |
lead error on those late shots (median) | 143.7° | 114.4° |
canonical win rate | 0.03 | 0.03 |

More optimisation, more firing, one accidental hit, no acquisition. Combined with
the basis experiment (quadratic did not beat linear on the *aiming* measure) and
the imitation experiment (warm start flies like random), the conclusion is that
the bottleneck is the objective, not the search: the tracking term in Phi is
worth at most +1.5 against points-shaped terms worth more, so the optimiser
purchases wins instead of gun solutions.  Any fix has to make the near-envelope
tracking error dominant *and* sharply peaked (1°-scale), and `flybfm gunnery`'s
late-fire line is the acceptance test.

Two more inconsistencies worth knowing about, both fixed:

* the training loop printed the win rate of the CEM **distribution mean** while
  writing the **best** member to disk (0.46 printed vs 0.03 for the artifact);
  it now evaluates the parameters it actually saves;
* `--random` was silently ignored whenever a canonical `--tag` was also passed.

## 6. Speed

| loop | cost |
|---|---|
physics + policy, 25 s fight @ 10 Hz decisions | ~33 ms |
physics + policy, 30 s fight | ~40 ms |
connectome brain (665 neurons, 2 kHz LIF) + policy, 20 s fight | ~5 s (~24 ms/decision) |
80-fight evaluation sweep | ~3 s |
