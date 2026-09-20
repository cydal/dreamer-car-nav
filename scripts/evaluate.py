"""Paired evaluation: a DreamerV3 checkpoint vs the scripted baseline.

    python scripts/evaluate.py logdir/plain_12m                 # 300 episodes, seeds 5000-5299
    python scripts/evaluate.py logdir/plain_12m --episodes 40   # quick look
    python scripts/evaluate.py logdir/plain_12m --agents dreamer scripted random

Protocol (rl-env3d README, "Baseline"): 3 waypoints, 48x48-tile map, seeds
5000-5299, every agent on the *same* seeds. The env draws map, spawn and
waypoints from RNG streams that a wrapper or policy cannot perturb, so each
seed is the same episode for every agent and the differences are paired: the
standard error of "Dreamer minus scripted" is a fraction of the unpaired one.
The scripted rows are re-run here rather than copied from the README so a
config change (task, map size) cannot silently un-pair the comparison.

Task settings come from the checkpoint's own config (task preset + env.carnav
kwargs), so the policy is evaluated on what it was trained on.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import carnav  # noqa: E402
from baselines.scripted import GapFollower  # noqa: E402

import carnav_dreamer  # noqa: E402,F401
from carnav_dreamer.presets import PRESETS  # noqa: E402


class RandomAgent:
  def __init__(self, env):
    self.space = env.action_space
  def reset(self): pass
  def act(self, obs, info=None): return self.space.sample()
  def diagnostics(self): return {}


def run_episode(env, agent, seed):
  obs, info = env.reset(seed=seed)
  agent.reset()
  steps = 0
  while True:
    obs, reward, terminated, truncated, info = env.step(agent.act(obs, info))
    steps += 1
    if terminated or truncated:
      return dict(
          seed=seed, reward=float(info['episode_reward']), steps=steps,
          waypoints=int(info['targets_reached']), n_targets=int(info['n_targets']),
          success=bool(info['is_success']), reason=info['reason'],
          crash_with=info.get('crash_with'),
          red_lights=int(info.get('red_light_violations', 0)))


def summarize(rows):
  n = len(rows)
  return dict(
      episodes=n,
      mean_reward=float(np.mean([r['reward'] for r in rows])),
      waypoints=float(np.mean([r['waypoints'] for r in rows])),
      success_rate=float(np.mean([r['success'] for r in rows])),
      crash_building=sum(r['crash_with'] == 'building' for r in rows),
      crash_vehicle=sum(r['crash_with'] == 'vehicle' for r in rows),
      stuck=sum(r['reason'] == 'stuck' for r in rows),
      timeout=sum(r['reason'] == 'timeout' for r in rows),
      red_lights=float(np.mean([r['red_lights'] for r in rows])),
      mean_steps=float(np.mean([r['steps'] for r in rows])),
  )


def paired(a, b, key):
  d = np.array([x[key] for x in a], float) - np.array([x[key] for x in b], float)
  return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('logdir')
  p.add_argument('--checkpoint', default=None, help='specific ckpt folder; default: latest')
  p.add_argument('--episodes', type=int, default=300)
  p.add_argument('--first-seed', type=int, default=5000)
  p.add_argument('--agents', nargs='+', default=['dreamer', 'scripted'],
                 choices=['dreamer', 'scripted', 'random'])
  p.add_argument('--out', default=None, help='JSON results path; default <logdir>/eval.json')
  args = p.parse_args()

  from carnav_dreamer.agent import DreamerAgent
  import elements
  config = elements.Config.load(str(pathlib.Path(args.logdir) / 'config.yaml'))
  task = config.task.split('_', 1)[1]
  env_kwargs = dict(config.env.get('carnav', {}))
  for k in ('use_seed', 'split_blocks'):
    env_kwargs.pop(k, None)
  env_kwargs.update(PRESETS[task])
  env_kwargs.setdefault('n_targets', 3)
  print(f'task carnav_{task}: carnav.make({env_kwargs})')
  env = carnav.make(**env_kwargs)
  seeds = list(range(args.first_seed, args.first_seed + args.episodes))

  agents = {}
  for name in args.agents:
    if name == 'dreamer':
      agents[name] = DreamerAgent(env, args.logdir, checkpoint=args.checkpoint)
    elif name == 'scripted':
      agents[name] = GapFollower.for_env(env)
    else:
      agents[name] = RandomAgent(env)

  results = {}
  for name, agent in agents.items():
    t0 = time.time()
    rows = [run_episode(env, agent, s) for s in seeds]
    results[name] = dict(summary=summarize(rows), episodes=rows)
    s = results[name]['summary']
    print(f"{name:>9}: reward {s['mean_reward']:+7.1f}  waypoints {s['waypoints']:.2f}/3  "
          f"full route {100 * s['success_rate']:5.1f}%  crashes b/v {s['crash_building']}/{s['crash_vehicle']}  "
          f"stuck {s['stuck']}  timeout {s['timeout']}  red {s['red_lights']:.2f}  "
          f"({time.time() - t0:.0f}s)")

  if 'dreamer' in results and 'scripted' in results:
    a, b = results['dreamer']['episodes'], results['scripted']['episodes']
    dr, dr_se = paired(a, b, 'reward')
    ds, ds_se = paired(a, b, 'success')
    dw, dw_se = paired(a, b, 'waypoints')
    print(f'\npaired dreamer - scripted over {len(a)} episodes:')
    print(f'  reward     {dr:+.1f} ± {dr_se:.1f}')
    print(f'  full route {100 * ds:+.1f} ± {100 * ds_se:.1f} points')
    print(f'  waypoints  {dw:+.2f} ± {dw_se:.2f}')
    results['paired'] = dict(reward=[dr, dr_se], success=[ds, ds_se], waypoints=[dw, dw_se])

  out = pathlib.Path(args.out or pathlib.Path(args.logdir) / 'eval.json')
  out.write_text(json.dumps(dict(
      logdir=args.logdir, checkpoint=args.checkpoint, task=f'carnav_{task}',
      env_kwargs=env_kwargs, seeds=[seeds[0], seeds[-1]], results=results), indent=1))
  print('wrote', out)


if __name__ == '__main__':
  main()
