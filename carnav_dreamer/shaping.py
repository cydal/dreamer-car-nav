"""Path-distance progress shaping: fix for the dead-end trap.

`CarNavEnv`'s own progress reward is potential-based on **straight-line**
distance to the current waypoint (`env/nav_env.py::_dist_to_target`, plain
`np.hypot`). That is exactly right when the shortest path to the target *is*
roughly a straight line, and wrong whenever it isn't: backing out of a
dead-end corridor to turn around necessarily *increases* straight-line
distance for a few steps, even though it is the only way to eventually
reduce it. The env charges that as negative progress regardless -- a
correctly-executed recovery maneuver looks identical, reward-wise, to
driving the wrong way. That is a shaping trap independent of whether the
policy has "learned to reverse": even a perfect driver takes the reward hit,
which then biases exploration away from ever finding the correct behaviour.

`env.city.road_nodes` / `env.city.node_links` (`env/world.py::
_build_road_graph`) already exist for a different reason (keeping rule-based
traffic on the road) but are exactly what's needed here: a lattice of
corridor-centreline crossings with 4-neighbour adjacency, explicitly built to
"keep T-junctions and dead ends too". `PathDistanceShaper` runs Dijkstra over
that graph from the current waypoint, and substitutes graph-path distance
for straight-line distance in the progress term -- and nothing else. Every
other reward component (time, crash, red light, waypoint bonus, pedestrian,
speeding) passes through unchanged, via `info['reward_components']`, the
same exact per-step breakdown INTEGRATION.md documents.

No env change: this is a `reward_fn` for rl-env3d's own
`wrappers.RewardOverrideWrapper`, used exactly as INTEGRATION.md's
"Overriding the reward" section describes -- the env computes its usual
reward internally first, unaffected, and `info["episode_reward"]` keeps
tracking *that* (the one gotcha that section calls out). Only what
`carnav_dreamer.env.CarNav` hands to the agent changes; `scripts/evaluate.py`
builds a fresh, unwrapped env and is untouched by this, so evaluation against
the published baseline stays on the env's own ground-truth formula.

    import carnav
    from wrappers import RewardOverrideWrapper
    from carnav_dreamer.shaping import PathDistanceShaper

    raw = carnav.make(...)
    env = RewardOverrideWrapper(raw, PathDistanceShaper(raw))
"""

import heapq

import numpy as np


class PathDistanceShaper:
    """Stateful `reward_fn` for `wrappers.RewardOverrideWrapper`.

    Rebuilds its distance field (Dijkstra from the current waypoint over the
    road graph) whenever the map regenerates or the active waypoint changes --
    detected cheaply and robustly via object identity
    (`env.city.road_nodes` is a *new* array each `city.generate()` call, even
    though `env.city` itself is the same long-lived object across resets) and
    `env.target_idx`, rather than trying to infer episode boundaries from
    `info` (which has no `is_first`-style flag of its own -- that's an
    `embodied.Env` convention `carnav_dreamer.env.CarNav` adds on top, and
    this shaper sits *below* that adapter, wrapping the raw `CarNavEnv`).
    Dijkstra runs at most a handful of times per 1000-step episode (once per
    map regen, once per waypoint advance) over a graph of a few hundred
    nodes -- negligible next to a training step.
    """

    def __init__(self, env, weight=None):
        self.env = env
        self.weight = weight            # None -> env.cfg.progress_weight, live
        self._nodes_id = None
        self._target_idx = None
        self._dist_field = None         # None entries or np.inf -> unreachable node
        self._prev_path_dist = None

    def __call__(self, obs, action, reward, terminated, truncated, info):
        env = self.env
        weight = self.weight if self.weight is not None else env.cfg.progress_weight

        nodes = env.city.road_nodes
        if id(nodes) != self._nodes_id or env.target_idx != self._target_idx:
            self._nodes_id = id(nodes)
            self._target_idx = env.target_idx
            self._dist_field = self._distance_field(env)
            self._prev_path_dist = None     # don't credit/charge the very first lookup

        path_dist = self._path_dist(env, info['x'], info['y'])
        if self._prev_path_dist is None or path_dist is None:
            path_progress = 0.0
        else:
            path_progress = (self._prev_path_dist - path_dist) * weight
        self._prev_path_dist = path_dist

        old_progress = info['reward_components']['progress']
        return reward - old_progress + path_progress

    def _distance_field(self, env):
        """Dijkstra from the current waypoint's nearest node; None if there
        are no more waypoints (episode already completed all of them) or no
        road graph (shouldn't happen, but fail soft rather than crash a
        training run over a degenerate map)."""
        nodes, links = env.city.road_nodes, env.city.node_links
        if nodes is None or len(nodes) == 0 or env.target_idx >= len(env.targets):
            return None
        tx, ty = env.targets[env.target_idx]
        start = int(np.argmin(np.hypot(nodes[:, 0] - tx, nodes[:, 1] - ty)))

        dist = np.full(len(nodes), np.inf)
        dist[start] = 0.0
        heap = [(0.0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v in links[u]:
                if v < 0:
                    continue
                w = float(np.hypot(nodes[v, 0] - nodes[u, 0], nodes[v, 1] - nodes[u, 1]))
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def _path_dist(self, env, x, y):
        if self._dist_field is None:
            return None      # no waypoints left / degenerate map: caller skips shaping this step
        nodes = env.city.road_nodes
        idx = int(np.argmin(np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)))
        graph_dist = self._dist_field[idx]
        if not np.isfinite(graph_dist):
            return None      # nearest node isn't connected to the target's component; skip rather than lie
        last_mile = float(np.hypot(nodes[idx, 0] - x, nodes[idx, 1] - y))
        return graph_dist + last_mile
