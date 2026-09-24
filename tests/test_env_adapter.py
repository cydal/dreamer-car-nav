"""The seam between rl-env3d and DreamerV3: `carnav_dreamer.env.CarNav`.

Every check here guards a way the adapter could be silently wrong while still
producing a perfectly valid-looking training run: a block sliced off by one,
a `stuck` ending bootstrapped, a `log/` value with a shape the logger rejects
only after an hour of training.
"""

import numpy as np
import pytest

import carnav
import embodied
import elements

from carnav_dreamer.env import CarNav
from carnav_dreamer.presets import PRESETS

SEED = 1234


def fixed_actions(n, seed=0):
  rng = np.random.default_rng(seed)
  return rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)


def drive(env, actions):
  """Reset once, then apply `actions`; return the list of observations."""
  obs = [env.step({'action': np.zeros(3, np.float32), 'reset': True})]
  for a in actions:
    obs.append(env.step({'action': a, 'reset': False}))
  return obs


# ---------------------------------------------------------------- spaces

@pytest.mark.parametrize('task', sorted(PRESETS))
def test_blocks_match_obs_slices(task):
  env = CarNav(task, seed=SEED)
  raw = env.env
  nonempty = {k: s for k, s in raw.obs_slices.items() if s.stop > s.start}
  block_keys = {k for k in env.obs_space if k in raw.obs_slices}
  assert block_keys == set(nonempty), (task, block_keys, set(nonempty))
  total = sum(env.obs_space[k].shape[0] for k in block_keys)
  assert total == raw.vector_dim
  for k in block_keys:
    assert env.obs_space[k].dtype == np.float32
    assert env.obs_space[k].shape == (nonempty[k].stop - nonempty[k].start,)


def test_single_vector_mode():
  env = CarNav('full', seed=SEED, split_blocks=False)
  assert 'vector' in env.obs_space
  assert env.obs_space['vector'].shape == (env.env.vector_dim,)
  assert not any(k in env.env.obs_slices for k in env.obs_space)


def test_required_keys_present():
  env = CarNav('plain', seed=SEED)
  for k in ('reward', 'is_first', 'is_last', 'is_terminal'):
    assert k in env.obs_space
  assert set(env.act_space) == {'action', 'reset'}
  assert env.act_space['action'].shape == (3,)
  assert not env.act_space['action'].discrete


def test_blocks_concat_equals_raw_vector():
  """Concatenating the blocks must reproduce the env's own vector exactly,
  step for step, so nothing is sliced off by one or reordered."""
  actions = fixed_actions(60)
  env = CarNav('full', seed=SEED)
  ref = carnav.make(seed=SEED, **PRESETS['full'])
  ref_obs, _ = ref.reset()
  outs = drive(env, actions)
  order = sorted(env.blocks, key=lambda k: env.blocks[k].start)
  concat = lambda o: np.concatenate([o[k] for k in order])
  np.testing.assert_array_equal(concat(outs[0]), ref_obs)
  for a, o in zip(actions, outs[1:]):
    ref_obs, *_ = ref.step(a)
    np.testing.assert_array_equal(concat(o), ref_obs)


# ------------------------------------------------------------ semantics

def test_is_first_after_reset_action_and_after_episode_end():
  env = CarNav('plain', seed=SEED, max_episode_steps=20)
  outs = drive(env, fixed_actions(19))
  assert outs[0]['is_first'] and outs[0]['reward'] == 0.0
  assert not any(o['is_first'] for o in outs[1:])
  assert not any(o['is_last'] for o in outs[:-1])
  # 20 steps is the limit, so step 20 (the 20th action) is the timeout.
  last = env.step({'action': np.zeros(3, np.float32), 'reset': False})
  assert last['is_last'] and not last['is_terminal']
  assert env.info['reason'] == 'timeout'
  # Whatever the driver sends next, the adapter must reset.
  nxt = env.step({'action': np.ones(3, np.float32), 'reset': False})
  assert nxt['is_first'] and not nxt['is_last']


def test_terminal_flag_tracks_reason_over_many_episodes():
  """terminated -> is_terminal, truncated -> not. Checked on real episodes
  driven by a random policy until every ending type has been seen at least
  once (timeout is forced with a short limit in half of the envs)."""
  seen = set()
  for i, limit in enumerate([1000, 60] * 4):
    env = CarNav('plain', seed=SEED + i, max_episode_steps=limit)
    rng = np.random.default_rng(i)
    obs = env.step({'action': np.zeros(3, np.float32), 'reset': True})
    while not obs['is_last']:
      # Bias towards driving so crashes actually happen.
      a = rng.uniform(-1, 1, 3).astype(np.float32)
      a[0] = abs(a[0]); a[1] = -1.0
      obs = env.step({'action': a, 'reset': False})
    reason = env.info['reason']
    seen.add(reason)
    if reason == 'timeout':
      assert not obs['is_terminal']
    else:
      assert reason in ('success', 'crash', 'stuck')
      assert obs['is_terminal']
    assert obs[f'log/{reason}'] == 1.0
  assert 'timeout' in seen and ('crash' in seen or 'stuck' in seen), seen


def test_log_values_are_float32_scalars():
  env = CarNav('full', seed=SEED)
  for o in drive(env, fixed_actions(5)):
    for k, v in o.items():
      if k.startswith('log/'):
        assert isinstance(v, np.floating) and v.dtype == np.float32
        assert np.ndim(v) == 0, (k, v)
        assert k in env.obs_space


def test_crash_split_is_mutually_exclusive_and_sums_to_crash():
  """crash_building / crash_vehicle partition crash: never both, and their
  sum equals it every step, over episodes biased toward actually crashing."""
  saw_a_crash = False
  for i in range(6):
    env = CarNav('full', seed=SEED + i, max_episode_steps=400)
    rng = np.random.default_rng(i)
    obs = env.step({'action': np.zeros(3, np.float32), 'reset': True})
    while not obs['is_last']:
      a = rng.uniform(-1, 1, 3).astype(np.float32)
      a[0] = abs(a[0]); a[1] = -1.0
      obs = env.step({'action': a, 'reset': False})
      b, v, c = obs['log/crash_building'], obs['log/crash_vehicle'], obs['log/crash']
      assert not (b and v)
      assert b + v == c
      saw_a_crash = saw_a_crash or bool(c)
  assert saw_a_crash


def test_speed_log_matches_info():
  """log/speed is exactly info['speed'] every step, not re-derived."""
  env = CarNav('plain', seed=SEED)
  env.step({'action': np.zeros(3, np.float32), 'reset': True})
  o = None
  for _ in range(10):
    o = env.step({'action': np.array([0.8, -1.0, 0.0], np.float32), 'reset': False})
    assert o['log/speed'] == np.float32(env.info['speed'])
  assert o['log/speed'] > 0.0   # accelerating forward for 10 steps actually moved it


def test_red_light_log_is_per_step_not_cumulative():
  env = CarNav('lights', seed=SEED)
  outs = drive(env, fixed_actions(300, seed=3))
  per_step = np.array([o['log/red_light'] for o in outs])
  assert set(np.unique(per_step)) <= {0.0, 1.0}
  assert per_step.sum() == env.info['red_light_violations']


def test_reward_scale_only_touches_the_agent_reward():
  actions = fixed_actions(40, seed=7)
  a = drive(CarNav('plain', seed=SEED), actions)
  b = drive(CarNav('plain', seed=SEED, reward_scale=0.01), actions)
  ra = np.array([o['reward'] for o in a]); rb = np.array([o['reward'] for o in b])
  assert ra[0] == rb[0] == 0.0
  np.testing.assert_allclose(rb, ra * 0.01, rtol=1e-5, atol=1e-7)
  for k in ('log/dist_to_target', 'lidar'):
    np.testing.assert_array_equal(a[-1][k], b[-1][k])


# ------------------------------------------------------- reproducibility

def test_same_seed_same_trajectory():
  actions = fixed_actions(80)
  a = drive(CarNav('full', seed=SEED), actions)
  b = drive(CarNav('full', seed=SEED), actions)
  for x, y in zip(a, b):
    for k in x:
      np.testing.assert_array_equal(x[k], y[k], err_msg=k)


def test_different_seed_different_trajectory():
  actions = fixed_actions(5)
  a = drive(CarNav('full', seed=SEED), actions)
  b = drive(CarNav('full', seed=SEED + 1), actions)
  assert not np.array_equal(a[0]['lidar'], b[0]['lidar'])


# ---------------------------------------------- the real upstream stack

def test_wrapped_env_runs_under_driver_with_random_agent():
  """The exact wrapper stack `dreamerv3/main.py::wrap_env` applies, driven by
  `embodied.Driver` with `embodied.RandomAgent`, must survive 2,000 steps.
  `CheckSpaces` raises on the first value outside its declared space, so
  this is the test that the [-1, 1] bounds and dtypes are honest."""
  from dreamerv3.main import wrap_env
  env = wrap_env(CarNav('full', seed=SEED, max_episode_steps=200), config=None)
  obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  agent = embodied.RandomAgent(obs_space, act_space)
  driver = embodied.Driver([lambda: env], parallel=False)
  counts = {'steps': 0, 'episodes': 0, 'firsts': 0}

  def on_step(tran, worker):
    counts['steps'] += 1
    counts['episodes'] += int(tran['is_last'])
    counts['firsts'] += int(tran['is_first'])
    assert set(tran) >= set(env.obs_space) | {'action'}
    assert tran['action'].shape == (3,)

  driver.on_step(on_step)
  driver.reset(agent.init_policy)
  driver(agent.policy, steps=2000)
  assert counts['steps'] == 2000
  assert counts['episodes'] >= 5, counts   # 200-step cap forces endings
  assert counts['firsts'] == counts['episodes'] + 1 or \
      counts['firsts'] == counts['episodes']
