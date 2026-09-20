"""CarNav as an `embodied.Env` for the vendored DreamerV3.

Why this exists instead of upstream's `embodied/envs/from_gym.py`: that adapter
targets the old `gym` API (4-tuple step, `reset()` returning only obs) and
imports `gym`, so it cannot wrap a Gymnasium 5-tuple env. The `embodied.Env`
contract is small (see `embodied/core/base.py`) and `embodied/envs/dummy.py`
shows the shape; this file follows both.

Contract recap (what the driver and agent rely on):

- `obs_space` / `act_space` are dicts of `elements.Space`. Observations must
  include `reward`, `is_first`, `is_last`, `is_terminal`. Keys starting with
  `log/` are stripped before the agent sees them and aggregated per episode
  (avg / max / sum) by `embodied/run/train.py::logfn`; they must be scalars.
- `act_space` must include `reset`. `step(action)` resets when
  `action['reset']` is true or the previous step ended an episode, and
  returns the first observation with `is_first=True` and zero reward.

Mapping from Gymnasium semantics (rl-env3d INTEGRATION.md, "Episode endings"):

    terminated (success / crash / stuck)  ->  is_last=True, is_terminal=True
    truncated  (timeout)                  ->  is_last=True, is_terminal=False

`stuck` is reported as terminated by the env on purpose -- it charges the
remaining time penalty as a lump sum, so bootstrapping V(s') as well would
double count -- and `is_terminal=True` is therefore right, not a shortcut.

Observation layout: by default the vector is split into its `obs_slices`
blocks (`lidar`, `dynamics`, `nav`, `traffic_light`, `traffic`, ...) as
separate observation keys. The encoder concatenates them anyway, but the
decoder emits one reconstruction head per key, so this gives per-block
reconstruction losses and metrics (`train/loss/lidar`, `train/loss/nav`, ...)
for free -- which INTEGRATION.md's "Notes for world models" asks for, because
32 LIDAR channels would otherwise drown the 7 traffic-light channels that
carry the decision. `split_blocks=False` gives a single `vector` key.

`reward_scale` multiplies the reward handed to the agent (default 1.0); it is
the switch for the reward-scaling ablation and nothing else reads it.
"""

import functools

import elements
import embodied
import numpy as np

import carnav

from .presets import PRESETS


class CarNav(embodied.Env):

  # Per-step scalars exposed as `log/<name>`; the trainer aggregates each per
  # episode, so `log/success/sum` is the success indicator, `log/waypoint/sum`
  # the waypoints reached, `log/crash/sum` the crash indicator, and so on.
  LOG_KEYS = (
      'success', 'waypoint', 'crash', 'red_light', 'stuck', 'timeout',
      'dist_to_target')

  def __init__(self, task='plain', seed=None, split_blocks=True,
               reward_scale=1.0, **kwargs):
    if task not in PRESETS:
      raise KeyError(f'unknown carnav task {task!r}; one of {sorted(PRESETS)}')
    settings = dict(PRESETS[task])
    settings.update(kwargs)
    self._env = carnav.make(obs_type='vector', seed=seed, **settings)
    # Multiplies the reward the agent sees; the env's own bookkeeping
    # (info['episode_reward'], the log/ scalars) is untouched. 1.0 is the
    # default and the claim under test: DreamerV3's symlog two-hot heads and
    # return normalisation should make scaling unnecessary. Set 0.01 to run
    # the ablation. Same effect as rl-env3d's RewardOverrideWrapper with a
    # multiplicative reward_fn, kept in the adapter so it is one config flag.
    self._reward_scale = float(reward_scale)
    if split_blocks:
      self._blocks = {
          name: sl for name, sl in self._env.obs_slices.items()
          if sl.stop > sl.start}
    else:
      self._blocks = {'vector': slice(0, self._env.vector_dim)}
    self._done = True
    self._info = None
    self._prev_red = 0

  @property
  def env(self):
    """The underlying `CarNavEnv`, for tests and for the inference agent."""
    return self._env

  @property
  def info(self):
    """The most recent Gymnasium `info` dict (privileged x/y/heading live here)."""
    return self._info

  @property
  def blocks(self):
    return dict(self._blocks)

  @functools.cached_property
  def obs_space(self):
    spaces = {
        name: elements.Space(np.float32, (sl.stop - sl.start,), -1.0, 1.0)
        for name, sl in self._blocks.items()}
    spaces.update({
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    })
    spaces.update({f'log/{k}': elements.Space(np.float32) for k in self.LOG_KEYS})
    return spaces

  @functools.cached_property
  def act_space(self):
    return {
        'action': elements.Space(np.float32, (3,), -1.0, 1.0),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      self._prev_red = 0
      obs, self._info = self._env.reset()
      return self._obs(obs, 0.0, self._info, is_first=True)
    act = np.asarray(action['action'], np.float32)
    obs, reward, terminated, truncated, self._info = self._env.step(act)
    self._done = bool(terminated or truncated)
    return self._obs(
        obs, reward * self._reward_scale, self._info,
        is_last=self._done, is_terminal=bool(terminated))

  def _obs(self, vector, reward, info,
           is_first=False, is_last=False, is_terminal=False):
    out = {
        name: np.asarray(vector[sl], np.float32)
        for name, sl in self._blocks.items()}
    reason = info.get('reason')
    red = int(info.get('red_light_violations', 0))
    logs = {
        'success': reason == 'success',
        'waypoint': info.get('targets_reached_this_step', 0),
        'crash': bool(info.get('crashed', False)),
        # The env's count is cumulative over the episode; diff it so the
        # per-episode `sum` aggregate is the number of violations.
        'red_light': max(0, red - self._prev_red),
        'stuck': reason == 'stuck',
        'timeout': reason == 'timeout',
        'dist_to_target': info.get('dist_to_target', 0.0),
    }
    self._prev_red = red
    out.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    out.update({f'log/{k}': np.float32(v) for k, v in logs.items()})
    return out

  def close(self):
    close = getattr(self._env, 'close', None)
    if close:
      close()
