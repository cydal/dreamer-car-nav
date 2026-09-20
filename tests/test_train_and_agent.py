"""End to end: train a tiny model for a few hundred steps, then restore it
through `DreamerAgent` and drive the real env with it.

Training runs in a subprocess because JAX platform settings are per process
and upstream's `internal.setup` applies them once; the restore-and-act half
runs in-process. ~25 s on a laptop CPU. `log_every` is wall-clock seconds, so
the run must outlive one tick for train metrics to be flushed.

`--logger.outputs jsonl` overrides the project default (which includes
`wandb`, see carnav_dreamer/configs.yaml) so this test never makes a network
call or depends on a WANDB_API_KEY being configured.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import carnav
from carnav_dreamer.presets import PRESETS

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def tiny_logdir(tmp_path_factory):
  logdir = tmp_path_factory.mktemp('run') / 'tiny'
  cmd = [
      sys.executable, '-m', 'carnav_dreamer.train',
      '--configs', 'carnav', 'carnav_mac', 'debug',
      '--run.steps', '1500', '--run.envs', '2', '--run.log_every', '1',
      '--env.carnav.max_episode_steps', '60',
      '--logger.outputs', 'jsonl',
      '--logdir', str(logdir)]
  proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
  assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
  return logdir


def test_training_writes_config_metrics_and_checkpoint(tiny_logdir):
  assert (tiny_logdir / 'config.yaml').exists()
  assert (tiny_logdir / 'ckpt' / 'latest').exists()
  rows = [json.loads(l) for l in (tiny_logdir / 'metrics.jsonl').open()]
  keys = {k for r in rows for k in r}
  # Per-block reconstruction losses and the episode-level log aggregates are
  # the two things our adapter adds over a stock upstream run.
  assert {'train/loss/lidar', 'train/loss/nav', 'train/loss/dynamics'} <= keys, keys
  assert 'epstats/log/success/sum' in keys or 'episode/score' in keys, keys
  assert any('episode/score' in r for r in rows)


def test_agent_restores_and_drives(tiny_logdir):
  from carnav_dreamer.agent import DreamerAgent
  env = carnav.make(width=48, height=48, max_episode_steps=40, **PRESETS['plain'])
  agent = DreamerAgent(env, str(tiny_logdir))
  for seed in (1, 2):
    obs, info = env.reset(seed=seed)
    agent.reset()
    actions = []
    while True:
      a = agent.act(obs, info)
      assert a.shape == (3,) and a.dtype == np.float32
      assert np.all(a >= -1) and np.all(a <= 1) and np.isfinite(a).all()
      actions.append(a)
      obs, reward, terminated, truncated, info = env.step(a)
      if terminated or truncated:
        break
    assert len(actions) <= 40
    assert agent.diagnostics() == {}


def test_agent_reads_its_blocks_by_name_from_a_wider_env(tiny_logdir):
  """The live viewer always exposes the 73-D layout; a 46-D plain-task policy
  must still drive it by picking its blocks by name."""
  from carnav_dreamer.agent import DreamerAgent
  wide = carnav.make(width=48, height=48, max_episode_steps=10, **PRESETS['full'])
  agent = DreamerAgent(wide, str(tiny_logdir))
  assert set(agent.live_slices) == {'lidar', 'dynamics', 'nav'}
  assert agent.live_slices == {k: wide.obs_slices[k] for k in agent.live_slices}
  obs, info = wide.reset(seed=3)
  agent.reset()
  a = agent.act(obs, info)
  assert a.shape == (3,)


def test_agent_refuses_a_block_of_different_width(tiny_logdir):
  from carnav_dreamer.agent import DreamerAgent
  wrong = carnav.make(n_beams=16, **PRESETS['plain'])     # lidar 16 vs 32
  with pytest.raises(ValueError, match="'lidar'"):
    DreamerAgent(wrong, str(tiny_logdir))
