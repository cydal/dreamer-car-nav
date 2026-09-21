# dreamer-car-nav

[DreamerV3](https://github.com/danijar/dreamerv3) trained on
[CarNav](https://github.com/cydal/Car-Navigation-Env), a procedurally generated
3D city in which a car has to reach a sequence of waypoints from a 46- to 73-D
ego-centric vector observation (LIDAR, dynamics, waypoint bearings, traffic
lights, traffic) with a 3-D continuous action (throttle, brake, steer).

Two design choices shape the repo:

- **The reference implementation is used unmodified.** `scripts/fetch_upstream.sh`
  clones `danijar/dreamerv3` at a pinned commit into the gitignored `dreamerv3/`
  and installs it. Everything of ours lives in `carnav_dreamer/`; see
  [UPSTREAM.md](UPSTREAM.md).
- **The environment is a dependency, not a fork.** rl-env3d is installed as a
  package; its [INTEGRATION.md](https://github.com/cydal/Car-Navigation-Env/blob/main/INTEGRATION.md)
  is the contract the adapter implements.

Work in progress: see the milestones below.

## Setup

```bash
conda env create -f environment.yml          # python 3.11 + requirements.txt (CPU JAX)
conda activate dreamer-carnav
pip install -e "../rl-env3d[gym]"            # the env, from a sibling checkout
scripts/fetch_upstream.sh                    # upstream DreamerV3 at the pinned sha
pip install -e .                             # makes `carnav_dreamer` importable
pytest                                       # ~30 s: adapter contract + tiny train/restore
```

On a Linux NVIDIA machine, `scripts/setup_gpu.sh` does all of this with
`requirements-cuda.txt` (CUDA 12 wheels, env pinned to a git sha).

### Logging: wandb by default

Runs log to [wandb](https://wandb.ai) (project `dreamer-car-nav`) alongside
the local `metrics.jsonl` every run always writes. Put your API key in a
`.env` file at the repo root (gitignored):

```
WANDB_API_KEY=...
```

(`carnav_dreamer/train.py::_setup_wandb_env` also accepts a key named
`WANDB_API`, and defaults `WANDB_PROJECT` if you haven't set one.) To train
with no network calls at all — what the test suite does — override the
logger:

```bash
python -m carnav_dreamer.train ... --logger.outputs jsonl
```

## Train

```bash
# Real run (GPU): plain 46-D task, 12M-parameter model, 2M env steps
scripts/train_gpu.sh                                  # = python -m carnav_dreamer.train --configs carnav size12m
SIZE=size25m scripts/train_gpu.sh --task carnav_full  # harder task, bigger model

# Laptop (CPU): pipeline validation with the 1M model
scripts/train_local.sh                                # = --configs carnav carnav_mac size1m --run.steps 3e5

# Smoke test in ~15 s
python -m carnav_dreamer.train --configs carnav carnav_mac debug --run.steps 3000 --logdir logdir/smoke
```

Runs compose upstream config blocks with ours (`carnav_dreamer/configs.yaml`):
`carnav` sets the task and run schedule, `carnav_mac` forces CPU/float32,
`size1m`…`size400m` and `debug` are upstream's. Any config key can be overridden
as a flag (`--run.train_ratio 256`, `--env.carnav.max_episode_steps 500`);
upstream's parser only accepts keys that already exist, so the env knobs you
can set this way are the ones listed under `env.carnav` in the yaml.

Tasks: `carnav_plain` (46-D, no traffic, no lights), `carnav_lights` (53-D),
`carnav_traffic` (66-D), `carnav_full` (73-D). Same map, spawn and waypoints
for a given seed across all four.

**Size tier is chosen explicitly, and by task.** `size1m` (`--configs carnav
size1m`) is the tier upstream itself uses for proprioceptive continuous
control (`dmc_proprio`) with state dims in the same range as `plain`/`lights`
(46/53-D vs. DMC's ~24-60-D) — well precedented, not a guess. `traffic`/`full`
add eight independently controlled vehicles and a signal-phase state machine
to predict forward, a much richer *process* than the extra observation dims
suggest, so `size12m` is the more defensible starting point there — reasoning
from task structure rather than measurement, and worth an ablation once GPU
time is available. `scripts/train_gpu.sh` picks the tier from `--task`
automatically (override with `SIZE=`); see the comment block above the
`carnav` entry in `carnav_dreamer/configs.yaml` for the numbers.

## Evaluate and watch

```bash
python scripts/evaluate.py logdir/<run>            # 300 paired episodes vs the scripted baseline, seeds 5000-5299
python scripts/plot_metrics.py logdir/<run> [...]  # metrics.jsonl -> metrics.png (several runs overlay)
python ../rl-env3d/main.py serve --agent configs/agents/dreamer.json --traffic none   # drive in the browser viewer
```

`carnav_dreamer.agent.DreamerAgent` restores a run from its logdir and
implements rl-env3d's `reset()/act()/diagnostics()` agent contract, so the
trained policy is a drop-in wherever the env accepts an agent.

**The viewer also draws the world model's imagination.** rl-env3d's
`diagnostics()["imagined_trajectories"]` contract (INTEGRATION.md,
"Visualising a world model's imagination") is implemented for real: every
tick, `DreamerAgent` rolls the *prior* forward from the current latent state
under the current policy (`n_samples` times, `horizon` steps), decodes the
already-trained `dynamics` block (speed, yaw rate, slip -- no retraining, no
observation change) and integrates it with the exact kinematics
`env/car.py::Car.step` uses, anchored at the car's current pose. Shows up in
the viewer as translucent lines fanning out from the car and an `imagining
×N · H steps` chip next to the agent's name, both gated behind the `sensors`
overlay toggle. ~5ms per call on CPU (`size1m`, defaults `n_samples=3,
horizon=16`), jitted once at construction so it holds up at the viewer's
20 Hz tick. Constructor args, or edit `configs/agents/dreamer.json`'s
`kwargs`. Details and the three JAX/ninjax gotchas it took to get there:
walkthrough chapter 4.5.

## What is in the observation, and how the adapter uses it

The env's vector is a concatenation of named blocks (`env.obs_slices`):
`lidar` (32), `dynamics` (5), `nav` (9), and, when enabled, `traffic_light` (7)
and `traffic` (20). The adapter exposes each block as its own observation key.
DreamerV3's encoder concatenates them anyway, but its decoder emits one
reconstruction head per key, so the logs carry `train/loss/lidar`,
`train/loss/nav`, … separately and the 7 traffic-light channels are not drowned
by 32 LIDAR channels. Episode outcomes (`success`, `waypoint`, `crash` split into
`crash_building`/`crash_vehicle`, `red_light`, `stuck`, `timeout`) plus
`dist_to_target` and `speed` (both privileged ground truth, explicitly
sanctioned by the env's own docs for logging, never fed back into the
observation) ride along as `log/` scalars that upstream aggregates per
episode, so `epstats/log/success/sum` is the success rate over recent
episodes. **[METRICS.md](METRICS.md)** explains every logged key in detail,
including the aggregation semantics that make `/sum` a rate rather than a
count here — worth reading before trusting a number from a run's logs.

The env's `terminated` (success, crash, stuck) maps to `is_terminal`, and its
`truncated` (timeout) to `is_last` only, so the value function bootstraps
exactly where the env's documentation says it should.

## Milestones

- [x] M0 environment, pinned upstream, tests
- [x] M1 adapter, config layering, smoke run (12 s on an M3 Pro)
- [x] M2 pipeline validation on the laptop (`size1m`, 146k steps, ~5h CPU) —
      not meant to learn to drive, and it didn't; confirms every termination
      path, loss family and the checkpoint save/restore/inference path are
      wired correctly before spending GPU time. Details in
      [METRICS.md §8](METRICS.md#8-what-the-pipeline-is-correctly-set-up-actually-means-checked)
      and `notes/devlog.md`.
- [ ] M3 GPU runs on `carnav_plain`, paired evaluation against the scripted baseline (+222 reward, 44% full route)
- [ ] M4 harder tasks (`lights`, `full`)
- [x] M5a imagined-trajectory overlay in the viewer -- decodes the existing
      `dynamics` head, no retraining; verified against the live viewer and
      its raw websocket payload
- [ ] M5b image observations

## License

MIT for this repository. The upstream DreamerV3 code it fetches is MIT,
Copyright (c) 2023 Danijar Hafner. The environment has its own license in its
repository.
