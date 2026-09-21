# Metrics: what every logged key measures

Every run writes `metrics.jsonl` (one sparse JSON object per line, key ->
value, plus `step`) and, unless overridden, streams the same values to
[wandb](https://wandb.ai). This is a reference for every key that appears
there: what it measures, its units, and how to read it. Grouped by what part
of the system produces it. Keys are exactly as logged — copy them into a
wandb chart or a `jq` filter as-is.

Two things worth knowing before reading any number below:

- **`epstats/log/*` are rates, not counts**, and it is easy to get this wrong
  (I did, once, live — see the M2 postmortem in `notes/devlog.md`). Details in
  [§2](#2-episode-outcomes-epstatslog).
- **World-model losses, KL terms and actor/critic losses are all in different,
  incomparable units.** A `train/loss/lidar` of 5 and a `train/loss/policy` of
  5 mean nothing relative to each other. Read each family against its own
  history, never across families.

---

## 1. Episode return: `episode/score`, `episode/length`

One row per finished episode (not batched into a window like everything under
`epstats/`), written the instant any of the parallel envs ends one.

| key | measures | units |
|---|---|---|
| `episode/score` | the env's own summed reward for that episode — `info["episode_reward"]` from `carnav`, exactly | reward units (see rl-env3d's INTEGRATION.md: −0.1/step time penalty, +100/waypoint, −100 crash, −50 red light entry) |
| `episode/length` | steps survived | env steps (after `action_repeat`) |

This is the number the whole project is ultimately judged on, and the
noisiest one: individual episodes swing from about −100 (never moving) to
+400 (all three waypoints, no mistakes). Always read it as a rolling mean over
tens of episodes, never as a single point — `scripts/plot_metrics.py` uses a
20-episode window for this reason, and a wandb line chart should be smoothed
the same way.

Two length values are diagnostic on their own, independent of score:

- `length == max_episode_steps` (1000 by default) — a `timeout`, i.e. the
  episode ran to its cap without succeeding, crashing, or getting stuck.
- `length == stuck_steps` (150 by default) *and* the episode has never moved
  above 0.5 m/s — a `stuck` ending, which charges the *entire remaining* time
  penalty as a lump sum (see §2). A run whose episodes all sit at exactly 151
  steps has a policy that isn't moving at all; that was the M2 run's first
  ~10,000 steps.

## 2. Episode outcomes: `epstats/log/*`

Per-step scalars attached to every observation by
`carnav_dreamer/env.py::CarNav._obs` under the `log/` prefix (stripped before
the agent sees them, per `embodied`'s convention): `success`, `waypoint`,
`crash` (split into `crash_building`/`crash_vehicle`/`crash_pedestrian`),
`red_light`, `stuck`, `timeout`, `dist_to_target`, `speed`,
`pedestrians_on_road`. The last three of those are always in `LOG_KEYS`
regardless of task, so `crash_pedestrian`/`pedestrians_on_road` are simply
zero on any task with `pedestrians=False` rather than being schema that
varies per task. Each becomes three keys —
`epstats/log/<name>/{avg,max,sum}` — through two stacked aggregations that
matter to understand correctly:

1. **Per episode** (`embodied/run/train.py::logfn`, an `elements.Agg` scoped
   to one episode): `episode.add('log/<name>/sum', value, agg='sum')` etc.
   sums (or maxes, or averages) that scalar over every step of *one* episode.
   For an indicator like `crash` that is 0 every step except the one it
   happens on, the per-episode sum is 0 or 1 — "did this episode crash".
2. **Across episodes in a window** (the module-level `epstats` aggregator,
   flushed to the log every `log_every` seconds): `epstats.add(result)` is
   called with **no explicit `agg=`**, so `elements.Agg`'s default rule for a
   scalar applies — **mean** (`elements/agg.py`, `RULES`). So
   `epstats/log/crash/sum` is not "how many crashes in this window" — it is
   the **mean, across episodes that ended in this window, of each episode's
   own crash-sum**, i.e. the *fraction of episodes that crashed*, a number in
   `[0, 1]`. The `/sum` suffix names where the value came from (step 1), not
   what happened in step 2.

Concretely, over the M2 run's last logged window, `epstats/log/stuck/sum ==
1.0` means "every episode in that window got stuck", not "1 stuck episode
total". Getting a true whole-run count requires weighting each window's rate
by how many episodes actually closed inside it — done once, as a sanity
check, in `notes/devlog.md`'s M2 postmortem; not needed for day-to-day
reading, where the rate is exactly what you want anyway.

| key (rate, `[0,1]`, over recent episodes) | what an episode ending this way means |
|---|---|
| `epstats/log/success/sum` | reached all `n_targets` waypoints — the number to watch |
| `epstats/log/waypoint/sum` | mean **count** of waypoints reached per episode (not a rate — `targets_reached_this_step` sums above 1 across a multi-waypoint episode), so this one can exceed 1 |
| `epstats/log/crash/sum` | hit a building or a vehicle |
| `epstats/log/crash_building/sum`, `epstats/log/crash_vehicle/sum`, `epstats/log/crash_pedestrian/sum` | the same, split by what was hit; the three always partition `crash`'s per-episode sum. `crash_pedestrian` needs `pedestrians=True` to ever be nonzero — costs `pedestrian_penalty` (300, bigger than a vehicle crash's 100) |
| `epstats/log/stuck/sum` | idled below 0.5 m/s for `stuck_steps` — charged the *entire remaining* time penalty as one lump sum, reported as `terminated` (not `truncated`) so the value function is never also bootstrapped past it (INTEGRATION.md, "Episode endings") |
| `epstats/log/timeout/sum` | ran to `max_episode_steps` without any of the above — `truncated`, so the critic *does* bootstrap here |
| `epstats/log/red_light/sum` | mean **count** of red-light entries per episode (diffed from the env's cumulative counter in the adapter, so this is a per-step event count, not cumulative) |

`success + crash + stuck + timeout` should sum to ~1.0 in any window with
enough episodes (they're mutually exclusive endings); it's a good sanity
check that nothing is being miscounted.

Two more, privileged ground truth per INTEGRATION.md ("for logging and
diagnostics only", never observation):

- `epstats/log/dist_to_target/avg` — average metres to the *current* waypoint
  over the episode. Falling within an episode's own step trace would mean
  "driving toward the goal"; this aggregate falling release-over-release means
  episodes are, on average, ending closer to a waypoint than before.
- `epstats/log/speed/avg` — average speed in m/s over the episode. The
  cheapest way to tell "not moving" apart from "moving without progress" —
  exactly the ambiguity that made the M2 run's flat score misleading (length
  went from 151 to 400+ while score got *worse*: the car started driving, just
  not toward anything yet).
- `epstats/log/pedestrians_on_road/avg` — average count of people currently
  on a zebra crossing (`info["pedestrians_on_road"]`), only ever nonzero with
  `pedestrians=True`. A rough measure of how often the episode actually
  presented a pedestrian-avoidance situation at all, useful for sanity-
  checking that a pedestrian-enabled run is seeing them with any regularity
  before reading anything into `crash_pedestrian`.
- `epstats/log/target_at_intersection/avg` — fraction of this episode's
  waypoints that landed at a genuine intersection rather than a dead end or a
  bare through-point, constant for the whole episode (all waypoints are
  chained at reset, not resampled mid-episode) and only meaningful with
  `env.carnav.intersection_targets=True`
  (`carnav_dreamer.targets.patch_intersection_targets`) — 0.0 always without
  it, since the flag being off means the field was never populated. Should
  sit near 1.0 when the flag is on; a value drifting down over many episodes
  would mean the fallback ladder (through-points, then the original sampler)
  is being hit more often than expected for the configured map size / target
  distance range.

`epstats/reward_rate` (outside `log/`, upstream's own): fraction of
consecutive step-reward pairs in an episode that differ by ≥0.01 — a coarse
"is the reward actually varying, or is this policy generating a constant
stream" check, unrelated to our env-specific metrics.

## 3. World model: reconstruction, reward, continuation

One number per training batch (chapter 2–2.6 of `notes/dreamerv3-walkthrough.md`
has the derivations; this is the summary for reading the log).

| key | measures | good direction | notes |
|---|---|---|---|
| `train/loss/lidar`, `train/loss/dynamics`, `train/loss/nav`, and (when the task has them) `train/loss/traffic_light`, `train/loss/traffic` | decoder reconstruction error for that observation *block*, summed over its elements, in `symlog` units | down | one per block **because** `carnav_dreamer/env.py` splits the vector into named observation keys — without that, this would be a single undifferentiated `train/loss/vector` and the 7 traffic-light dims would be invisible against 32 LIDAR dims |
| `train/loss/rew` | reward-prediction loss: cross-entropy of the two-hot reward head against the actual reward, in a log-spaced 255-bin grid (`heads.py::symexp_twohot`) | down | its scale does not grow with reward magnitude — this is the mechanism that makes CarNav's ±100 spikes trainable with no reward scaling |
| `train/loss/con` | continue-head loss: binary cross-entropy against `~is_terminal` (discounted by `1 - 1/horizon` when `contdisc: True`) | down, usually small (crashes/success/stuck are rare per step) | a `con` near 0 mostly means "predicting 'still going' every step is easy", not evidence of anything more |
| `train/loss/dyn` | KL(posterior ‖ prior), stop-gradient on the posterior — trains the **prior** to anticipate what the encoder will see | down to the `free_nats` floor (1.0), then flat | flat at ~1.0 is the *expected* steady state, not stalled training — see §3.1 |
| `train/loss/rep` | KL(posterior ‖ prior), stop-gradient on the prior — trains the **posterior** to stay predictable (the other half of KL balancing) | down to 1.0, then flat | same floor, same expected flatness |

### 3.1 Why `dyn`/`rep` flatten at ~1.0 and that's fine

`free_nats: 1.0` clips both KL terms from below (`jnp.maximum`) — there is
*no gradient* once either drops under one nat. The M2 run's `dyn`/`rep` sat at
1.0–1.3 for the entire 146k steps; that is the free-bits floor doing its job
(stopping the posterior from collapsing onto the prior on easy inputs), not a
sign the world model stopped learning. Watch `dyn_ent`/`rep_ent` (§4) instead
if you want to know whether the *latent itself* is still changing.

### 3.2 Latent and policy entropy

| key | measures | units |
|---|---|---|
| `train/dyn_ent` | entropy of the **prior** latent distribution (32 categoricals, `classes` categories each, summed) | nats |
| `train/rep_ent` | entropy of the **posterior** latent distribution | nats |
| `train/ent/action` | differential entropy of the Gaussian action policy | nats (unbounded sign for a continuous distribution) |

`unimix: 0.01` puts a floor under every category's probability, so
`dyn_ent`/`rep_ent` can fall but never all the way to 0. Both sitting near
`log(classes) * stoch` (uniform) — 40–43 for `classes=4, stoch=32` — means the
discrete latent hasn't specialized on anything yet; falling over training
means specific latent codes are becoming reliably predictable. `ent/action`
staying flat means the actor hasn't narrowed its action distribution — check
this alongside `train/adv_std` (§4): entropy has no reason to drop while the
advantage signal is still noise.

## 4. Actor and critic (learning in imagination)

Everything here comes from rolling the *prior* forward under the current
policy (chapter 3 of the walkthrough). Reading order: `rew`/`con` (what the
imagined rollout predicts) → `val`/`tar`/`ret` (what the critic says it's
worth) → `adv` (the signal the actor actually climbs).

| key | measures |
|---|---|
| `train/rew` | mean predicted reward over the imagined rollout |
| `train/con` | mean predicted continue-probability over the imagined rollout |
| `train/val` | the online critic's value estimate |
| `train/slowval` | the slow (EMA) critic's value estimate — regularises `val`, not a second independent number to track |
| `train/tar` | the λ-return target the critic is regressed toward (`lambda_return`, chapter 3.3) |
| `train/ret` | the same λ-return, after the percentile return-normalisation (`retnorm`) |
| `train/ret_min`, `train/ret_max` | min/max of that normalised return in the batch |
| `train/ret_rate` | fraction of normalised returns with `\|·\| >= 1` — how often the return normaliser's own 5th/95th-percentile scale is actually being hit; near 0 early in training just means returns haven't spread out yet |
| `train/adv` | the actor's advantage signal, `(return − value) / scale` |
| `train/adv_std`, `train/adv_mag` | spread and mean absolute value of that advantage — this is the number to watch for "has the critic found any signal yet". Near 0 (std and mag both tiny) means the actor is being pushed in essentially random directions — the M2 run's whole 146k steps, since every episode scored within a few points of the same stuck-detector floor, so there was no reward *variation* for a critic to explain |
| `train/weight` | the discounted-continuation weight (`cumprod(con)`) applied to every imagined step — down-weights steps the model thinks are unlikely to still be inside the episode |
| `train/loss/policy` | the (negative) actor objective: `-(logπ(a)·advantage + entropy_bonus)` |
| `train/loss/value` | critic loss: two-hot cross-entropy of `val` against the λ-return target, plus a slow-critic regulariser |
| `train/loss/repval` | a **second** critic loss, computed on real replayed states rather than imagined ones, bootstrapped from the imagined return at that state (chapter 3.4) — this is what lets the critic anchor to real transitions even where the world model is wrong |

**The single most informative pair for "has training found anything yet" is
`adv_std`/`adv_mag` next to `episode/score`.** If the score is flat and these
are near zero, the model has correctly learned that it doesn't yet know
anything useful about which actions are better — that is the honest state of
the M2 run throughout, not a bug.

## 5. Optimizer: `train/opt/*`

One shared optimizer updates the world model, both heads, and the
actor-critic together every step (`embodied/jax/opt.py::Optimizer`).

| key | measures | healthy shape |
|---|---|---|
| `train/opt/loss` | the single scalar being minimised: `Σ mean(loss_k) * scale_k` over every loss in §3–4 | trending down, but dominated by whichever loss has the largest scale × magnitude — not comparable across runs with different task/size |
| `train/opt/grad_norm` | global L2 norm of the gradient, **after** adaptive gradient clipping | should not be pinned at the clip threshold for long stretches — that means the pre-clip gradient is consistently huge |
| `train/opt/grad_rms`, `train/opt/update_rms`, `train/opt/param_rms` | RMS (root-mean-square, over all parameters) of the gradient, the applied update, and the parameters themselves | `update_rms` a few orders of magnitude below `param_rms` is the normal regime; `update_rms ≈ param_rms` means the optimizer is moving weights by a large fraction of their own scale every step, usually a sign of instability |
| `train/opt/param_count` | total parameter count of the optimized modules | constant for a given size tier — 643,123 for `size1m` on the plain task, confirmed against the restored M2 checkpoint |
| `train/opt/updates` | cumulative optimizer steps taken | monotonic counter, useful for computing effective `train_ratio` post hoc |

## 6. Replay buffer: `replay/*`

`embodied/core/replay.py::Replay.stats()`, reset to zero every time it's read
(so these are per-`log_every`-window rates/counts, like the aggregator's own
counters, not cumulative totals).

| key | measures |
|---|---|
| `replay/items` | number of sampleable start positions currently in the buffer |
| `replay/chunks` | number of 1024-step storage chunks currently held (bounded by `capacity`) |
| `replay/streams` | number of active per-worker write streams — equals `run.envs` |
| `replay/ram_gb` | memory the buffer's chunk arrays occupy |
| `replay/inserts` | new items registered since the last read |
| `replay/samples` | training batches drawn since the last read |
| `replay/updates` | steps' worth of posterior state written back since the last read (`replay_context`, walkthrough §1.5) |
| `replay/replay_ratio` | `length * samples / inserts` — the **achieved** train ratio; compare against the `run.train_ratio` you configured. M2 configured 256, logged ~260 — on target |

## 7. Throughput and resource usage

| key | measures |
|---|---|
| `fps/policy` | env steps per second — how fast new experience is collected. Bottlenecked by `train_ratio` on a single-process CPU run: the M2 run averaged 9.9, because for every env step the trainer does `train_ratio` (256) replayed-steps worth of gradient work first |
| `fps/train` | replayed steps per second — the actual training throughput. M2 averaged ~2,500, occasionally dipping under 10 during a checkpoint save or a JIT recompile; that's a momentary stall, not a leak, as long as it recovers |
| `usage/psutil/proc_cpu_usage`, `proc_ram_gb`, `proc_ram_frac` | this process's own CPU/RAM | |
| `usage/psutil/total_*` | whole-machine CPU/RAM, for noticing when something else on the box is competing for it | |

## 8. What "the pipeline is correctly set up" actually means, checked

Distinct from "did it learn to drive" (it didn't, and was never expected to
on `size1m`/CPU/146k steps — see `notes/devlog.md`'s M2 postmortem). What was
checked, and where in this document each check lives:

- Every step for 146,352 steps produced finite values everywhere (§3–7): zero
  `NaN`/`Inf` in 1,132 logged rows.
- All four episode endings occurred and were logged correctly:
  success/crash/stuck/timeout summed to exactly the 652 real episodes once
  weighted by episodes-per-window (§2) — every termination code path in
  `carnav_dreamer/env.py` was exercised without error, not just the common
  one.
- Per-block reconstruction losses (§3) all fell by more than an order of
  magnitude and stayed finite — the encoder/decoder split is wired correctly
  end to end.
- `replay/replay_ratio` tracked the configured `train_ratio` (§6) — the
  training loop is running at the rate it was told to, not silently starved
  or runaway.
- The checkpoint written at step 146,352 restores through
  `carnav_dreamer/agent.py::DreamerAgent` (643,123 params, matching
  `train/opt/param_count`) and drives real episodes to completion with finite,
  in-range actions — the save/load/inference path the GPU runs and the viewer
  both depend on is exercised on a real, non-trivial checkpoint, not just the
  tiny one-minute model the test suite trains.

None of that requires the score to have moved — it is exactly the "wiring is
correct" checklist a short CPU run is for, before spending GPU time on a
run meant to actually learn something.
