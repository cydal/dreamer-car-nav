# Upstream provenance

The reference DreamerV3 implementation, https://github.com/danijar/dreamerv3,
is **not** checked into this repository. `scripts/fetch_upstream.sh` clones it
into `./dreamerv3` (gitignored) at a pinned commit and installs it into the
active environment with `pip install --no-deps -e dreamerv3/`, which makes the
`dreamerv3` and `embodied` packages importable:

| | |
|---|---|
| commit | `e3f02248693a79dc8b0ebd62c93683888ddaccfe` |
| date | 2026-05-25, "Fix Atari frame maxpooling on reset (#213)" |
| license | MIT, Copyright (c) 2023 Danijar Hafner (`dreamerv3/LICENSE` after fetching) |

## Zero local modifications

The upstream tree is used byte-for-byte. Everything specific to this project
lives in `carnav_dreamer/`:

| ours | replaces / hooks |
|---|---|
| `carnav_dreamer/train.py` | a rewrite of `dreamerv3/main.py::main` that loads an extra config file and routes the `carnav` task suite to our env constructor; agent, replay, stream, logger and run scripts are upstream's, called by reference |
| `carnav_dreamer/configs.yaml` | merged on top of `dreamerv3/configs.yaml`: an `env.carnav` default plus the `carnav` and `carnav_mac` blocks |
| `carnav_dreamer/env.py` | `embodied.Env` adapter for the Gymnasium-style CarNav env (upstream's `embodied/envs/from_gym.py` targets the old `gym` API) |

The only run-time hook into upstream is `train.py` assigning
`dreamerv3.main.make_env = carnav_dreamer.train.make_env`, because upstream's
`make_agent` reads observation/action spaces through that module global.

`scripts/fetch_upstream.sh` warns if `dreamerv3/` has local edits; if a change
to upstream ever becomes unavoidable, it goes in as a patch file applied by that
script and is listed here.
