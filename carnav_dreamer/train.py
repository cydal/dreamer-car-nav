"""Entry point: DreamerV3 (upstream, unmodified) on CarNav.

This is upstream's `dreamerv3/main.py::main` rewritten so that two things can
be injected without touching the vendored tree:

1. an extra config file (`carnav_dreamer/configs.yaml`) merged on top of
   upstream's `dreamerv3/configs.yaml`, and
2. an env constructor for the `carnav` suite, routed through upstream's own
   `wrap_env` so the action/dtype/space wrappers are identical to every other
   task.

Everything else -- agent, replay, stream, logger, the run scripts -- is
upstream's, called by reference. Read this file alongside
`dreamerv3/main.py`; the structure is the same on purpose.

    python -m carnav_dreamer.train --configs carnav size12m --logdir logdir/{timestamp}
    python -m carnav_dreamer.train --configs carnav carnav_mac debug --run.steps 3000 --logdir logdir/smoke
"""

import os
import pathlib
from functools import partial as bind

import elements
import embodied
import portal
import ruamel.yaml as yaml

import dreamerv3.main as upstream

from .env import CarNav

HERE = pathlib.Path(__file__).parent
UPSTREAM_CONFIGS = pathlib.Path(upstream.folder) / 'configs.yaml'
LOCAL_CONFIGS = HERE / 'configs.yaml'


def _deep_merge(base, extra):
  """Recursively merge `extra` into a copy of `base` (dicts only)."""
  out = dict(base)
  for key, value in extra.items():
    if isinstance(value, dict) and isinstance(out.get(key), dict):
      out[key] = _deep_merge(out[key], value)
    else:
      out[key] = value
  return out


def load_configs():
  """Upstream's config blocks with ours layered on top.

  `defaults` is deep-merged (so `env.carnav` becomes a known key and can be
  overridden from the command line); every other block is a named override
  that simply joins the pool selectable with `--configs`.
  """
  loader = yaml.YAML(typ='safe')
  ours = loader.load(LOCAL_CONFIGS.read_text())
  theirs = loader.load(UPSTREAM_CONFIGS.read_text())
  merged = dict(theirs)
  merged['defaults'] = _deep_merge(theirs['defaults'], ours.pop('defaults', {}))
  clash = set(ours) & set(merged)
  assert not clash, f'local config blocks shadow upstream blocks: {sorted(clash)}'
  merged.update(ours)
  return merged


def make_env(config, index, **overrides):
  """`carnav_*` tasks go to our adapter; anything else falls through to
  upstream so `--task dummy_disc` etc. still work from this entry point."""
  suite, task = config.task.split('_', 1)
  if suite != 'carnav':
    return upstream.make_env(config, index, **overrides)
  kwargs = dict(config.env.get('carnav', {}))
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    # Same recipe as upstream (DMLab): deterministic per (run seed, env index).
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  env = CarNav(task, **kwargs)
  return upstream.wrap_env(env, config)


def main(argv=None):
  from dreamerv3.agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = load_configs()
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  # Upstream's make_agent builds one env to read the spaces via the module
  # global `make_env`; pointing that global at ours is the one hook needed.
  upstream.make_env = make_env

  if config.script == 'train':
    embodied.run.train(
        bind(upstream.make_agent, config),
        bind(upstream.make_replay, config, 'replay'),
        bind(make_env, config),
        bind(upstream.make_stream, config),
        bind(upstream.make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(upstream.make_agent, config),
        bind(upstream.make_replay, config, 'replay'),
        bind(upstream.make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(upstream.make_stream, config),
        bind(upstream.make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(upstream.make_agent, config),
        bind(make_env, config),
        bind(upstream.make_logger, config),
        args)

  else:
    raise NotImplementedError(
        f'{config.script!r}: only train / train_eval / eval_only are wired '
        'here; the distributed parallel_* scripts live in dreamerv3/main.py.')


if __name__ == '__main__':
  main()
