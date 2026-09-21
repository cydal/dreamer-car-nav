"""`DreamerAgent.diagnostics()["imagined_trajectories"]` -- rl-env3d's
world-model imagination contract (INTEGRATION.md, "Visualising a world
model's imagination"), decoded from the model's own already-trained
`dynamics` head with no retraining and no observation change. See the
docstring at the top of carnav_dreamer/agent.py for how; this file is about
what the result has to satisfy, using `tiny_logdir`'s real (if tiny)
checkpoint from tests/conftest.py.
"""

import numpy as np

import carnav
from carnav_dreamer.agent import DreamerAgent
from carnav_dreamer.presets import PRESETS


def _agent(tiny_logdir, **kwargs):
  env = carnav.make(width=48, height=48, max_episode_steps=200, **PRESETS['plain'])
  return env, DreamerAgent(env, str(tiny_logdir), **kwargs)


def _drive_one_step(env, agent, seed=0):
  obs, info = env.reset(seed=seed)
  agent.reset()
  agent.act(obs, info)                    # populates _carry and _pose
  return obs, info


def test_shape_matches_n_samples_and_horizon(tiny_logdir):
  env, agent = _agent(tiny_logdir, n_samples=5, horizon=9)
  _drive_one_step(env, agent)
  diag = agent.diagnostics()
  trajs = diag['imagined_trajectories']
  assert len(trajs) == 5
  for t in trajs:
    assert set(t) == {'x', 'y'}
    assert len(t['x']) == len(t['y']) == 9


def test_finite_and_anchored_at_current_pose(tiny_logdir):
  """Absolute world-frame metres (INTEGRATION.md), starting from the *next*
  predicted step -- so the first point is near, not exactly at, the anchor;
  a debug-sized model over one 0.05s*action_repeat step can't move far."""
  env, agent = _agent(tiny_logdir, n_samples=4, horizon=10)
  _drive_one_step(env, agent, seed=1)
  x0, y0, _ = agent._pose
  for t in agent.diagnostics()['imagined_trajectories']:
    xs, ys = np.array(t['x']), np.array(t['y'])
    assert np.isfinite(xs).all() and np.isfinite(ys).all()
    # One step at up to max_speed (22 m/s) * dt (0.05s) is at most ~1.1 m;
    # give it a generous margin rather than pin the exact kinematics here.
    assert abs(xs[0] - x0) < 5.0 and abs(ys[0] - y0) < 5.0


def test_samples_diverge(tiny_logdir):
  """The whole point of >1 sample is to show the spread of a stochastic
  model, not draw the same line n_samples times."""
  env, agent = _agent(tiny_logdir, n_samples=6, horizon=12)
  _drive_one_step(env, agent, seed=2)
  trajs = agent.diagnostics()['imagined_trajectories']
  finals = np.array([[t['x'][-1], t['y'][-1]] for t in trajs])
  spread = finals.std(axis=0).sum()
  assert spread > 0, 'all rollouts ended at the exact same point'


def test_reproducible_given_the_same_call_sequence(tiny_logdir):
  """Same run seed, same sequence of act()/diagnostics() calls -> identical
  rollouts (carnav_dreamer.agent uses config.seed + a call counter, never an
  unseeded RNG), the same reproducibility discipline as the rest of the
  project."""
  results = []
  for _ in range(2):
    env, agent = _agent(tiny_logdir, n_samples=3, horizon=8)
    _drive_one_step(env, agent, seed=3)
    results.append(agent.diagnostics()['imagined_trajectories'])
  for a, b in zip(results[0], results[1]):
    np.testing.assert_array_equal(a['x'], b['x'])
    np.testing.assert_array_equal(a['y'], b['y'])


def test_disabled_when_no_dynamics_block_is_available(tiny_logdir):
  """Guards the one real precondition: imagination decodes the `dynamics`
  block specifically, so an agent whose checkpoint has no such head (not a
  real task here -- `dynamics` is never gated off -- but the adapter's
  `split_blocks=False` single-`vector` mode has no per-block heads at all)
  must degrade to `{}` rather than raise or decode nonsense. Exercised at the
  unit level (fake the missing block on an otherwise-real agent) rather than
  training a second checkpoint just to get one without a `dynamics` head."""
  env, agent = _agent(tiny_logdir, n_samples=3, horizon=8)
  _drive_one_step(env, agent, seed=6)
  assert agent._imagine_jit is not None            # sanity: normally enabled
  agent._imagine_jit = None                        # simulate no dynamics head
  assert agent.diagnostics() == {}


def test_does_not_mutate_trained_parameters(tiny_logdir):
  """`modify=True` is required for `nj.scan`'s internal bookkeeping (see the
  long comment in agent.py's __init__), but the *output* state must never be
  written back -- otherwise repeated calls could drift the checkpoint's own
  weights during a live demo."""
  env, agent = _agent(tiny_logdir, n_samples=3, horizon=8)
  _drive_one_step(env, agent, seed=4)
  before = {k: np.array(v) for k, v in agent._imagine_params.items()}
  for _ in range(5):
    agent.diagnostics()
  for k, v in agent._imagine_params.items():
    np.testing.assert_array_equal(before[k], np.array(v), err_msg=k)


def test_survives_a_broken_imagine_call_without_crashing(tiny_logdir, monkeypatch):
  """`diagnostics()` runs on every server tick (serve/server.py::
  agent_diagnostics has no try/except of its own); a bug here must degrade to
  `{}` forever after, not take down a live viewer session."""
  env, agent = _agent(tiny_logdir, n_samples=3, horizon=8)
  _drive_one_step(env, agent, seed=5)

  def boom(*a, **k):
    raise RuntimeError('synthetic failure')
  monkeypatch.setattr(agent, '_imagine', boom)

  assert agent.diagnostics() == {}
  assert agent._imagine_jit is None            # disabled, not retried every tick
  assert agent.diagnostics() == {}             # still safe on a second call
