import json
import pathlib
import subprocess
import sys

import pytest

# Make `carnav_dreamer` importable when pytest is run from anywhere. Importing
# it also puts the upstream clone (`dreamerv3/`) on the path, see its __init__.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
import carnav_dreamer  # noqa: E402,F401


@pytest.fixture(scope='session')
def tiny_logdir(tmp_path_factory):
  """A real (if tiny) trained checkpoint, shared by every test module that
  needs one -- training it is the expensive part (~25s), so it happens once
  per test session rather than once per file.

  Runs in a subprocess because JAX platform settings are per process and
  upstream's `internal.setup` applies them once. `--logger.outputs jsonl`
  overrides the project default (which includes `wandb`) so this never makes
  a network call or depends on a `WANDB_API_KEY`. `log_every` is wall-clock
  seconds, so the run must outlive one tick for train metrics to be flushed.
  """
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
