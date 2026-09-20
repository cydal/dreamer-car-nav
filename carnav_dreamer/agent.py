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
"""

import pathlib

import elements
import numpy as np

import carnav_dreamer  # noqa: F401  (puts the upstream clone on sys.path)
import dreamerv3.main as upstream
from dreamerv3.agent import Agent

from .env import CarNav


class DreamerAgent:

  def __init__(self, env, logdir, checkpoint=None, platform='cpu', seed=0):
    self.logdir = pathlib.Path(logdir)
    self.config = elements.Config.load(str(self.logdir / 'config.yaml'))

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

    self._carry = None
    self._first = True
    self._last_action = np.zeros(3, np.float32)
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
    return action

  def diagnostics(self):
    # `imagined_trajectories` (decoded imagination rollouts for the viewer)
    # is planned; see the M5 notes. Nothing to draw yet.
    return {}

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
