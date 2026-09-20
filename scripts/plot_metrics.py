"""Plot a run's metrics.jsonl: score, episode outcomes, per-block losses.

    python scripts/plot_metrics.py logdir/m2_plain_size1m [more logdirs ...] --out figures/m2.png

Each logdir's `metrics.jsonl` is a stream of sparse rows `{"step": int, key:
value, ...}` written by upstream's JSONL logger: episode rows appear once per
finished episode, train/epstats rows once per `log_every` seconds. Several
logdirs on one figure make ablations (split_blocks on/off, reward scaling)
directly comparable.

wandb (see carnav_dreamer/configs.yaml, on by default) is the live/primary
place to watch a run now -- every `log/` scalar shows up there automatically,
no plotting code needed. This script stays as the offline fallback: old runs,
a headless box with no wandb account, or a quick local look before deciding
whether a run is worth naming on wandb at all.
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = [
    # (title, [keys], kind)
    ('Episode score', ['episode/score'], 'episodes'),
    ('Episode length', ['episode/length'], 'episodes'),
    ('Outcome rate per episode',
     ['epstats/log/success/sum', 'epstats/log/crash/sum',
      'epstats/log/stuck/sum', 'epstats/log/timeout/sum'], 'lines'),
    ('Crash type (of episodes that crashed)',
     ['epstats/log/crash_building/sum', 'epstats/log/crash_vehicle/sum'], 'lines'),
    ('Waypoints reached per episode', ['epstats/log/waypoint/sum'], 'lines'),
    ('Distance to current waypoint (m, avg over episode)',
     ['epstats/log/dist_to_target/avg'], 'lines'),
    ('Speed (m/s, avg over episode)', ['epstats/log/speed/avg'], 'lines'),
    ('Reconstruction loss per block',
     ['train/loss/lidar', 'train/loss/dynamics', 'train/loss/nav',
      'train/loss/traffic_light', 'train/loss/traffic', 'train/loss/vector'],
     'lines'),
    ('World model: KL and heads',
     ['train/loss/dyn', 'train/loss/rep', 'train/loss/rew', 'train/loss/con'],
     'lines'),
    ('Actor / critic', ['train/loss/policy', 'train/loss/value',
                        'train/loss/repval'], 'lines'),
    ('Policy entropy and latent entropies',
     ['train/ent/action', 'train/dyn_ent', 'train/rep_ent'], 'lines'),
    ('Throughput (steps / s)', ['fps/policy', 'fps/train'], 'lines'),
]


def load(logdir):
  rows = [json.loads(l) for l in (pathlib.Path(logdir) / 'metrics.jsonl').open()]
  return pd.DataFrame(rows).sort_values('step')


def rolling(df, key, window):
  s = df[['step', key]].dropna()
  if s.empty:
    return None
  return s['step'].values, s[key].rolling(window, min_periods=1).mean().values, s[key].values


def main():
  p = argparse.ArgumentParser()
  p.add_argument('logdirs', nargs='+')
  p.add_argument('--out', default=None, help='PNG path; default shows nothing and writes <first logdir>/metrics.png')
  p.add_argument('--window', type=int, default=20, help='episodes in the rolling mean')
  args = p.parse_args()

  runs = {pathlib.Path(d).name: load(d) for d in args.logdirs}
  ncol = 3
  nrow = int(np.ceil(len(PANELS) / ncol))
  fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.6 * nrow))
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

  for ax, (title, keys, kind) in zip(axes.flat, PANELS):
    ci = 0
    for name, df in runs.items():
      for key in keys:
        if key not in df:
          continue
        r = rolling(df, key, args.window if kind == 'episodes' else 1)
        if r is None:
          continue
        steps, smooth, raw = r
        label = f'{name}: {key.split("/")[-1]}' if len(keys) > 1 else name
        color = colors[ci % len(colors)]
        ci += 1
        if kind == 'episodes':
          ax.plot(steps, raw, '.', ms=2, alpha=0.25, color=color)
        ax.plot(steps, smooth, color=color, label=label, lw=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('env steps')
    ax.grid(alpha=0.3)
    if ci:
      ax.legend(fontsize=7)
    if title.startswith('Throughput') or title.startswith('Reconstruction'):
      ax.set_yscale('log')
  for ax in list(axes.flat)[len(PANELS):]:
    ax.axis('off')

  fig.suptitle(' vs '.join(runs), fontsize=12)
  fig.tight_layout()
  out = args.out or str(pathlib.Path(args.logdirs[0]) / 'metrics.png')
  pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=120)
  print('wrote', out)


if __name__ == '__main__':
  main()
