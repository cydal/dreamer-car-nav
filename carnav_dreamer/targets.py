"""Waypoint placement that avoids dead ends -- the map-side half of the fix
(`carnav_dreamer.shaping.PathDistanceShaper` is the reward-side half).

`env/nav_env.py::_sample_targets` chains all `n_targets` waypoints for an
episode in one call at `reset()`, each drawn from `env/world.py::
sample_point_near`: a uniformly random road tile filtered only by
straight-line distance from the previous point. Nothing there distinguishes
a through-street tile from the very tip of a dead-end corridor -- contrast
with spawn placement (`sample_free_pose`), which explicitly probes for room
before accepting a pose, precisely because an unrecoverable start "is an
instant crash the agent cannot be blamed for". Waypoints never got the same
treatment.

`patch_intersection_targets(env)` replaces `env._sample_targets` (on that one
instance, not the class -- other envs, and rl-env3d itself, are untouched)
with a version constrained to `env.city.road_nodes` / `node_links`
(`env/world.py::_build_road_graph`) filtered to node **degree** -- how many
of a node's 4 lattice neighbours are actually road:

    degree >= 3   a true intersection: whichever way the car arrived, there
                  is always at least one way to *continue* without a U-turn.
    degree == 2   a through-point: not a junction, but still never a dead
                  end -- there are two ways out, not one.
    degree == 1   a dead end. This is what's being avoided.

Falls back gracefully, in order, if a stricter tier has no candidate within
the configured distance annulus: intersections -> through-points -> the
original (dead-end-permitting) sampler -> a fully random road point. The
episode is never left without a target; it's just no longer allowed to be a
dead end unless the map genuinely has nothing else at the right distance.

Reproducibility is preserved on purpose: every draw here comes from
`env.city.rng`, the same per-subsystem stream `sample_point_near`/
`sample_road_point` already use (INTEGRATION.md, "Seeding and
reproducibility") -- same seed still gives the same episode, just a
different (intersection-constrained) one than the vanilla env would produce,
which is the entire point of turning this on.
"""

import types

import numpy as np


def _degree(node_links):
    return (node_links >= 0).sum(axis=1)


def _sample_at_degree(env, px, py, min_dist, max_dist, min_degree):
    nodes, links = env.city.road_nodes, env.city.node_links
    if nodes is None or len(nodes) == 0:
        return None
    candidates = nodes[_degree(links) >= min_degree]
    if len(candidates) == 0:
        return None
    d = np.hypot(candidates[:, 0] - px, candidates[:, 1] - py)
    mask = (d >= min_dist) & (d <= max_dist)
    if not mask.any():
        return None
    valid = candidates[mask]
    idx = int(env.city.rng.integers(0, len(valid)))
    return float(valid[idx, 0]), float(valid[idx, 1])


def _sample_targets_avoiding_dead_ends(self):
    """Bound onto a `CarNavEnv` instance in place of its own
    `_sample_targets`; same chaining structure (each waypoint drawn within
    `target_min_dist`/`target_max_dist` of the previous one), same
    attributes set (`self.targets`), so nothing downstream (`_observe`,
    `_dist_to_target`, `step`'s `target_idx` advance) needs to know this
    happened. `_targets_at_intersection` records, per waypoint, whether a
    genuine intersection (degree >= 3) was found -- read by
    `carnav_dreamer.env.CarNav` for the `log/target_at_intersection` metric,
    ignored by everything else.
    """
    cfg = self.cfg
    self.targets = []
    self._targets_at_intersection = []
    px, py = self.car.x, self.car.y
    for _ in range(cfg.n_targets):
        pt = _sample_at_degree(self, px, py, cfg.target_min_dist, cfg.target_max_dist, 3)
        at_intersection = pt is not None
        if pt is None:
            pt = _sample_at_degree(self, px, py, cfg.target_min_dist, cfg.target_max_dist, 2)
        if pt is None:
            pt = self.city.sample_point_near(px, py, cfg.target_min_dist, cfg.target_max_dist)
        if pt is None:
            pt = self.city.sample_road_point()
        self.targets.append(pt)
        self._targets_at_intersection.append(at_intersection)
        px, py = pt


def patch_intersection_targets(env):
    """Rebind `env._sample_targets` to the dead-end-avoiding version, on this
    instance only. Idempotent -- patching twice is harmless."""
    env._sample_targets = types.MethodType(_sample_targets_avoiding_dead_ends, env)
    return env
