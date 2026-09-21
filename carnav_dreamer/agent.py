"""A trained DreamerV3 policy behind rl-env3d's `Agent` interface.

rl-env3d's decision-maker contract (`agents/base.py`) is `reset()`,
`act(obs, info=None)` and an optional `diagnostics()`; anything satisfying it
can drive the live browser viewer (`main.py serve --agent <json>`), the
screenshot tool, or a plain evaluation loop. This class wraps an upstream
`dreamerv3.agent.Agent` restored from a training logdir so the world model
policy is a drop-in for the scripted controller.

    agent = DreamerAgent(env, logdir='logdir/plain_12m')
    obs, info = env.reset(seed=5000); agent.reset()
    action = agent.act(obs, info)

Or, for the viewer, `configs/agents/dreamer.json`:

    {"module": "carnav_dreamer.agent", "factory": "DreamerAgent",
     "kwargs": {"logdir": "logdir/plain_12m"}}

How it works, in upstream terms:

- the run's `config.yaml` (saved by `train.py`) is reloaded, so the network
  sizes match the checkpoint; JAX is forced onto CPU with no train/report
  precompilation since only `policy()` is needed;
- observation/action spaces come from a throwaway `CarNav` built with the
  run's own `env.carnav` settings, exactly as `dreamerv3/main.py::make_agent`
  does, so the observation keys (the vector blocks) line up with training;
- the checkpoint's `agent.pkl` is loaded through `elements.Checkpoint`, the
  same path `embodied/run/eval_only.py` takes;
- `act()` builds a batch-of-one observation dict (`is_first=True` on the step
  after `reset()`), calls `agent.policy(carry, obs, mode='eval')`, keeps the
  returned recurrent carry, and hands back the 3-D action clipped to [-1, 1].

Note that DreamerV3 samples its action from the policy distribution in every
mode (`dreamerv3/agent.py::Agent.policy` never reads `mode`), so evaluation is
stochastic, as in the paper.

## Imagination: `diagnostics()["imagined_trajectories"]`

rl-env3d's viewer draws an agent's imagined future if `diagnostics()` returns
`{"imagined_trajectories": [{"x": [...], "y": [...]}, ...]}` -- absolute
world-frame metres, one dict per rollout (INTEGRATION.md, "Visualising a
world model's imagination"). `_imagine()` below produces it *without* any
retraining or change to the checkpoint, by reusing a decoder head the model
already has:

1. Tile the current recurrent state (`deter`, `stoch`, from the last `act()`
   call) across `n_samples` copies and roll `dyn.imagine` forward `horizon`
   steps under the policy -- the same call `agent.py::Agent.loss` makes
   during training's imagination phase (walkthrough chapter 3.2), just
   outside a training step.
2. Decode each imagined state's `dynamics` block with `dec` -- speed, yaw
   rate and slip, the same five numbers `env/sensors.py::dynamics_features`
   puts in every observation and the same head `train/loss/dynamics`
   measures. `symlog_mse` predicts in symlog space, so `.pred()` needs
   `nets.symexp` before the numbers mean m/s and rad/s again.
3. Integrate those with the *exact* kinematics `env/car.py::Car.step` uses --
   `x += speed * cos(heading + slip) * dt`, `y += speed * sin(heading + slip)
   * dt`, `heading += yaw_rate * dt` -- anchored at the pose `info` handed
   over on the last `act()` call.

No observation, decoder, or training change was needed: `dynamics` is decoded
during training regardless (it's part of `self.blocks`, chapter 2.6 of the
walkthrough), so this reuses that head exactly as it already exists in every
checkpoint from this project. Called every server tick, so the pure call is
jitted once at construction and reused -- retracing per step would be too
slow for a 20 Hz viewer.
"""

import functools
import pathlib

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import carnav_dreamer  # noqa: F401  (puts the upstream clone on sys.path)
import dreamerv3.main as upstream
import embodied.jax.nets as nets
from dreamerv3.agent import Agent
from dreamerv3.agent import sample as _sample_dist

from .env import CarNav


class DreamerAgent:

  def __init__(self, env, logdir, checkpoint=None, platform='cpu', seed=0,
               n_samples=3, horizon=16):
    self.logdir = pathlib.Path(logdir)
    self.config = elements.Config.load(str(self.logdir / 'config.yaml'))
    self.n_samples = n_samples
    self.horizon = horizon

    # Physical constants for integrating imagined (speed, yaw_rate, slip)
    # into world-frame positions -- read from the real env when we have one
    # (env/car.py::CarParams, env/nav_env.py::EnvConfig), else the same
    # defaults those classes use, so `diagnostics()` still returns something
    # sensible without a live env (e.g. constructed for offline inspection).
    car = getattr(env, 'car', None)
    cfg = getattr(env, 'cfg', None)
    self._max_speed = float(car.p.max_speed) if car else 22.0
    self._max_steer = float(car.p.max_steer) if car else np.radians(32.0)
    self._dt = float(cfg.dt * cfg.action_repeat) if cfg else 0.05

    # Spaces from an adapter built like the training one (task + env kwargs).
    suite, task = self.config.task.split('_', 1)
    assert suite == 'carnav', self.config.task
    env_kwargs = dict(self.config.env.get('carnav', {}))
    env_kwargs.pop('use_seed', None)
    probe = upstream.wrap_env(CarNav(task, **env_kwargs), self.config)
    self.obs_space = {
        k: v for k, v in probe.obs_space.items() if not k.startswith('log/')}
    self.act_space = {k: v for k, v in probe.act_space.items() if k != 'reset'}
    self.blocks = probe.blocks
    probe.close()
    self.live_slices = self._match_layout(env)

    # Same construction as dreamerv3/main.py::make_agent, minus training-only
    # JAX work: `precompile` would JIT the train and report functions.
    jax_cfg = dict(self.config.jax)
    jax_cfg.update(platform=platform, precompile=False, prealloc=False)
    self.agent = Agent(self.obs_space, self.act_space, elements.Config(
        **self.config.agent,
        logdir=str(self.logdir),
        seed=seed,
        jax=jax_cfg,
        batch_size=self.config.batch_size,
        batch_length=self.config.batch_length,
        replay_context=self.config.replay_context,
        report_length=self.config.report_length,
        replica=0,
        replicas=1,
    ))

    cp = elements.Checkpoint()
    cp.agent = self.agent
    if checkpoint is None:
      cp = elements.Checkpoint(str(self.logdir / 'ckpt'))
      cp.agent = self.agent
      cp.load(keys=['agent'])                       # the `latest` save
    else:
      cp.load(str(checkpoint), keys=['agent'])

    # A jitted, side-effect-free call to the model's own dyn.imagine + dec --
    # bypassing embodied.jax.Agent's policy()/train() entry points, which are
    # the only ones upstream jits, since neither does what we need here. Built
    # once: retracing on every server tick would defeat the point of jitting.
    # `create`/`modify` gate plain Python control flow inside `nj.pure`
    # (`assert context.create or not create`, ...), so they have to be fixed
    # Python bools *before* `jax.jit` traces anything -- passing them as jit
    # call-time kwargs instead raises `TracerBoolConversionError`.
    #
    # `modify=True` despite this being a read-only call: `nj.scan`'s own
    # optimisation (`ninjax.py::scan`) slices the *live* state down to just
    # the keys `_prerun` finds accessed/modified before entering the loop --
    # but `_prerun` itself short-circuits to "nothing accessed" whenever both
    # `create` and `modify` are False (`_prerun`'s literal first line), which
    # then hands the scan body an *empty* state and every parameter lookup
    # inside `dyn.imagine`'s scan fails as if it needed creating. We never
    # read the modified state back out (we only ever ask for the second
    # return value), so allowing writes is free.
    self._imagine_jit = (
        jax.jit(functools.partial(
            nj.pure(self._imagine_fn), create=False, modify=True))
        if 'dynamics' in self.blocks else None)
    # `self.agent.params` carries the training mesh's sharding (`train_mesh`
    # in `embodied/jax/agent.py`, `memory_kind='unpinned_host'` under CPU),
    # which a plain array built here (`carry`, `seed`) does not have; jit
    # then refuses the implicit transfer ("Disallowed host-to-device
    # transfer"). One device_get/device_put round trip strips that mesh
    # placement down to plain default placement -- cheap, done once, and
    # avoids fighting the multi-device training machinery for what is a
    # single-CPU inference call.
    self._imagine_params = (
        jax.device_put(jax.device_get(self.agent.params))
        if self._imagine_jit else None)
    self._diag_calls = 0

    self._carry = None
    self._first = True
    self._last_action = np.zeros(3, np.float32)
    self._pose = (0.0, 0.0, 0.0)
    self.reset()

  # ---------------------------------------------------- rl-env3d Agent API

  def reset(self):
    self._carry = self.agent.init_policy(1)
    self._first = True

  def act(self, obs, info=None):
    batch = self._obs(np.asarray(obs, np.float32))
    self._carry, acts, _ = self.agent.policy(self._carry, batch, mode='eval')
    self._first = False
    action = np.clip(np.asarray(acts['action'][0], np.float32), -1.0, 1.0)
    self._last_action = action
    if info is not None and 'x' in info:
      # Absolute world-frame pose, privileged and never observed (env docs,
      # "things that will bite you" #5) -- fine here, since it only anchors
      # a diagnostic overlay the policy itself never sees.
      self._pose = (float(info['x']), float(info['y']), float(info['heading']))
    return action

  def diagnostics(self):
    """`imagined_trajectories` for the viewer -- see the module docstring.
    Never raises: this runs on every server tick, and a bug in a diagnostic
    overlay taking down a live demo would be a worse outcome than a missing
    overlay."""
    if self._imagine_jit is None or self._carry is None:
      return {}
    try:
      return {'imagined_trajectories': self._imagine()}
    except Exception as e:                                        # noqa: BLE001
      if self._diag_calls == 0:
        print(f'DreamerAgent.diagnostics: imagination disabled after error: {e!r}')
      self._imagine_jit = None
      return {}
    finally:
      self._diag_calls += 1

  # -------------------------------------------------------------- helpers

  def _match_layout(self, env):
    """Where each trained-on block sits in *this* env's observation.

    Blocks are matched by name through the env's `obs_slices`, not by
    position, so a policy trained on the 46-D plain task can drive an env that
    exposes the 73-D layout (the live viewer always does, padding the traffic
    blocks when traffic is off): it simply reads the `lidar`, `dynamics` and
    `nav` blocks it knows and ignores the rest. What cannot be papered over is
    a block whose width differs (e.g. a different `n_beams`) or is absent.
    """
    slices = getattr(env, 'obs_slices', None) if env is not None else None
    if slices is None:
      return dict(self.blocks)          # no env to ask; assume training layout
    live = {}
    for name, sl in self.blocks.items():
      want = sl.stop - sl.start
      have = slices.get(name)
      if have is None or have.stop - have.start != want:
        got = 'absent' if have is None else f'{have.stop - have.start}-wide'
        raise ValueError(
            f"observation block '{name}' is {want}-wide in the checkpoint "
            f"(task {self.config.task}) but {got} in this env; build the env "
            f"with the same sensor settings")
      live[name] = have
    return live

  def _obs(self, vector):
    out = {name: vector[sl][None] for name, sl in self.live_slices.items()}
    out.update(
        reward=np.zeros((1,), np.float32),
        is_first=np.array([self._first], bool),
        is_last=np.zeros((1,), bool),
        is_terminal=np.zeros((1,), bool),
    )
    return out

  # -------------------------------------------------- imagination (viewer)

  def _imagine_fn(self, carry):
    """The impure half: plain ninjax/JAX calls against `self.agent.model`,
    wrapped in `nj.pure` + `jax.jit` by the caller. `carry` is `dyn_carry`
    (deter/stoch) already tiled to `n_samples` rows. Mirrors what
    `dreamerv3/agent.py::Agent.loss` does for its imagination rollout
    (walkthrough chapter 3.2), minus the training-only bookkeeping."""
    model = self.agent.model
    policyfn = lambda feat: _sample_dist(model.pol(model.feat2tensor(feat), 1))
    _, feat, _ = model.dyn.imagine(carry, policyfn, self.horizon, training=False)
    reset = jnp.zeros(feat['deter'].shape[:2], bool)          # (n_samples, horizon)
    _, _, recons = model.dec({}, feat, reset, training=False)
    return recons['dynamics'].pred()          # symlog-space; caller symexps it

  def _imagine(self):
    # `self._carry` came back from `agent.policy()` through
    # `embodied.jax.Agent._split`, which turns every array into a Python
    # list of per-device shards (`list(x)`) for its multi-device machinery --
    # on our single CPU device that's just `[array_without_the_batch_dim]`.
    # `_stack` (the inverse the same class uses before its own `_policy`
    # call) restores a plain batched array before we tile it ourselves.
    dyn_carry = self.agent._stack(self._carry[1])   # also on the training mesh's sharding
    carry = jax.tree.map(
        lambda x: jnp.repeat(x, self.n_samples, axis=0), dyn_carry)
    carry = jax.device_put(jax.device_get(carry))    # strip that sharding too, see __init__
    rng = np.random.default_rng(seed=[int(self.config.seed), self._diag_calls])
    seed = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    seed = jax.device_put(seed)                      # same reason as carry, above
    _, dyn_symlog = self._imagine_jit(self._imagine_params, carry, seed=seed)
    dyn_norm = np.asarray(nets.symexp(dyn_symlog))      # (n_samples, horizon, 5)

    # Same three channels and the same integration `env/car.py::Car.step`
    # uses -- see the module docstring. speed/yaw_rate/slip are channels
    # 0/1/4 of the dynamics block (env/sensors.py::dynamics_features).
    speed = dyn_norm[..., 0] * self._max_speed
    yaw_rate = dyn_norm[..., 1] * 2.0
    slip = dyn_norm[..., 4] * self._max_steer

    x0, y0, heading0 = self._pose
    x = np.full(self.n_samples, x0, np.float64)
    y = np.full(self.n_samples, y0, np.float64)
    heading = np.full(self.n_samples, heading0, np.float64)
    trajs = [{'x': [], 'y': []} for _ in range(self.n_samples)]
    for h in range(self.horizon):
      x += speed[:, h] * np.cos(heading + slip[:, h]) * self._dt
      y += speed[:, h] * np.sin(heading + slip[:, h]) * self._dt
      heading += yaw_rate[:, h] * self._dt
      for i in range(self.n_samples):
        trajs[i]['x'].append(float(x[i]))
        trajs[i]['y'].append(float(y[i]))
    return trajs
