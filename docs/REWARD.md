# Reward design for a connectome-constrained fighter

You asked how the reward is set up in projects like these, and what the right
answer is. This is the right answer, with the measurements that justify it.

The whole file is implemented in `flybfm/sim/reward.py` and
`flybfm/brain/plasticity.py`. Constants below are the defaults.

---

## 0. The three levels

Everything runs through the fly's dopamine system eventually — that is the only
learning signal a fly has. The question is what you *put into it*.

### Level 1 — raw outcome into the dopamine cells (what the demos do)

```
if something bad happened:  I_DA(PPL101) += k
if something good happened: I_DA(reward cluster) += k
```

No temporal credit assignment, no critic, no shaping. This is `doomfly`'s setup
(2 PPL101 neurons receive the game's punishment signal) and it is the setup
behind the reported "3,000 runs, no learning". Not a bug in their code: a global
scalar arriving after the fact cannot assign credit to the ~10^8 synapses that
produced the action, and with guessed weights the eligibility trace points
somewhere unhelpful anyway.

### Level 2 — prediction error into the dopamine cells (biologically right)

Dopamine is a *temporal difference error* signal (Schultz; and in flies the
PPL1 and PAM clusters do an aversive/appetitive split). So deliver

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

into the dopamine cells, with `V` a small learned critic. Implemented in
`brain/plasticity.py::DopamineChannel` (`use_td=True`), critic = linear TD(0)
over the 18-dim observation, `gamma=0.995`, `critic_lr=0.02`. `use_td=False`
reproduces Level 1, so the ablation is one flag.

### Level 3 — potential-based shaping (what makes it trainable at all)

Gun events are *sparse*. In this repo's baseline, pooled scripted fights fire
~6% hits — a hit is roughly one event per 1,500 rounds. A learner cannot find
that by accident. But naive dense shaping rewrites the objective and you end up
training a script, not a fighter.

The fix that keeps the objective honest is **potential-based reward shaping**
(Ng, Harada & Russell, ICML 1999):

```
r_t = events_t + gamma * Phi(s_{t+1}) - Phi(s_t)
```

for *any* bounded `Phi`. It is provable that this leaves the optimal policy
unchanged — the shaping changes how fast you get there, not where "there" is.
So it is dense guidance without a hand-written script. Use it.

---

## 1. Events (the objective)

| term | value | note |
|---|---|---|
`r_hit_dealt` | +1.00 per hit | 1 hit = 1/6 of the target's health |
`r_hit_taken` | −1.00 per hit | symmetric: the agent is punished for being hit, not just rewarded for hitting |
`r_kill` | +30 | terminal |
`r_death` | −30 | terminal |
`r_crash` | −30 | terminal; ground, or stall/control departure |
`r_arena_exit` | −20 | terminal; leaving the 12 km arena |
`r_timeout_win` | +5 | decision on points |
`r_timeout_loss` | −5 | |
`r_timeout_draw` | 0 | |
`r_trigger_waste` | −0.01 per decision step | trigger down with no firing solution — ammo discipline |

Two deliberate choices:

* **A kill is worth 6 hits and 6× a points win.** Otherwise the agent optimises
  the scorecard instead of the fight.
* **Leaving the arena is worse than losing on points.** Otherwise "run away" is
  a valid strategy, and you will get an agent that wins by exiting the fight.

## 2. Potential `Phi(s)` (the shaping)

All terms are bounded, so `Phi` is bounded (`max_potential = 4.0` for the sum of
the older terms; the tracking term is bounded by construction).

| term | weight | form | BFM meaning |
|---|---|---|---|
`w_lead_angle` | 3.5 | `-(min(|lead|/12°, 1))²` | how far the nose is from the *ballistic* firing direction, saturated at 12° |
`w_track` | 1.5 | `exp(-(AA/3°)²)` inside the envelope | nose-on-target at the scale the gun actually needs |
`w_aspect` | 0.6 | `0.5 cos(TAA) + 0.5` | offensive vs defensive geometry |
`w_range` | 0.4 | peaked at 400 m, negative inside 150 m | be *in* the gun envelope, and not overrunning it |
`w_energy` | 0.35 | `tanh(ΔE_s / 1500 m)` | specific-energy advantage, saturating so "climb and run" cannot farm it |
`w_closure` | 0.25 | closing when far, refusing closure when close | anti-overshoot |

### Why there is a `w_track` term, and why it is at 3°

This one is empirical. Training with only the terms above produced an agent that
won 74% of its fights and **never hit anything** — over 80 evaluation fights it
fired 21,000 rounds for 0 hits. It learned to sit in the gun envelope with the
nose within ~10°, collect the shaping reward, and win on the points decision.
None of the other terms is sensitive at the scale the gun needs: at 600 m a
round needs the nose inside roughly one degree.

This is the single most instructive result in the repo. It says:
* a shaping term that looks reasonable in units of degrees can be two orders of
  magnitude too soft for the thing you actually care about;
* and win rate is not the metric that catches it — the round count did.

## 3. Timeout rule

A 90 s cap would otherwise be resolved by a coin flip. On timeout the fight is
decided as a BFM judge would decide it:

```
score = (blue_hp - red_hp) + 0.35 * (cos TAA_blue - cos TAA_red)
```

damage first, then position. This matters more than it looks: in baseline
fights between scripted pilots most episodes end on the clock, so the timeout
rule *is* the reward signal for a large fraction of the data.

## 4. What the dopamine actually does

`brain/lif.py::deliver_dopamine` applies the three-factor rule to every marked
plastic edge:

```
dw_ij = lr * e_ij * DA(t) - weight_decay * w_ij
e_ij  = decaying pre x post coincidence (eligibility trace)
```

with `lr = 0.02`, `weight_decay = 1e-4`, weights clipped to ±3.

Honest status: this is the fly's own learning rule and it is *weak*. In the
reduced synthetic connectome it is inert (no mushroom-body edges exist in the
subgraph). Its insufficiency is a result worth reporting, not a bug to hide, and
the Level 2/3 machinery exists precisely because relying on it is not enough.

## 5. Anti-degeneracy checklist

Every entry was either observed in this repo's own runs or is a standard failure
of this class of task. Each is cheap to monitor; all of them are in
`analysis/metrics.py`:

1. **Winning without shooting** — the 74%-zero-hits agent above. Monitor rounds,
   hit rate, and hits-per-fight, not just win rate.
2. **Running the clock** — check the distribution of termination reasons.
3. **Arena abuse** — the agent "wins" if the opponent leaves; ensure exits are
   penalised for both sides and reported.
4. **Ground-hugging / camping the floor** — the floor is 250 m, and a low fight
   is a legitimate tactic up to a point; report mean altitude and use the
   dive-rate-dependent recovery margin so that neither side dies for free.
5. **Energy suicide** — winning the turn but arriving at 90 m/s. Track the
   specific-energy difference and the time below corner speed.
6. **Trigger spam** — measured directly (`r_trigger_waste`).
7. **Overfitting the opponent** — see `docs/DESIGN.md` §8; PFSP + league.
