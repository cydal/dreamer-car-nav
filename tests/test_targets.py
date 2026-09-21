"""`patch_intersection_targets`: waypoints avoid dead ends.

Unlike `test_shaping.py`, this one is easiest to check against the *real*
procedurally generated city rather than a hand-built graph -- degree is a
simple, unambiguous property to verify directly on `env.city.node_links`,
and running it over many seeds is the only way to be confident the fallback
ladder is rarely needed rather than doing all the work.
"""

import numpy as np
import pytest

import carnav
from carnav_dreamer.env import CarNav
from carnav_dreamer.presets import PRESETS
from carnav_dreamer.targets import patch_intersection_targets

SEEDS = range(2000, 2040)     # 40 seeds, distinct from every other test's range


def _degree(env):
    return (env.city.node_links >= 0).sum(axis=1)


def _nearest_node_degree(env, x, y):
    nodes = env.city.road_nodes
    idx = int(np.argmin(np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)))
    return int(_degree(env)[idx])


def test_patched_targets_are_at_or_near_an_intersection():
    """Over many seeds, every waypoint's nearest graph node should have
    degree >= 3 (a genuine intersection) -- the fallback ladder exists for
    maps sparse enough to need it, so allow a small miss rate rather than
    demanding 100%."""
    total, at_intersection = 0, 0
    for seed in SEEDS:
        env = carnav.make(width=48, height=48, seed=seed, **PRESETS['plain'])
        patch_intersection_targets(env)
        env.reset(seed=seed)
        for tx, ty in env.targets:
            total += 1
            at_intersection += _nearest_node_degree(env, tx, ty) >= 3
    rate = at_intersection / total
    assert rate > 0.9, f'only {at_intersection}/{total} waypoints landed at an intersection'


def test_unpatched_targets_sometimes_land_at_dead_ends():
    """Sanity check on the premise itself: without the patch, over the same
    seeds, at least one waypoint should land on a degree-1 (dead-end) node --
    otherwise the whole fix would be solving a problem that doesn't occur."""
    saw_a_dead_end = False
    for seed in SEEDS:
        env = carnav.make(width=48, height=48, seed=seed, **PRESETS['plain'])
        env.reset(seed=seed)
        for tx, ty in env.targets:
            if _nearest_node_degree(env, tx, ty) <= 1:
                saw_a_dead_end = True
    assert saw_a_dead_end, 'no dead-end waypoint in 40 seeds -- widen SEEDS or re-check the premise'


def test_never_fails_to_produce_a_full_set_of_targets():
    for seed in SEEDS:
        env = carnav.make(width=48, height=48, seed=seed, **PRESETS['plain'])
        patch_intersection_targets(env)
        env.reset(seed=seed)
        assert len(env.targets) == env.cfg.n_targets
        assert len(env._targets_at_intersection) == env.cfg.n_targets


def test_reproducible_given_the_same_seed():
    """Same seed -> same episode, still -- patching changes *which* episode a
    seed produces, not whether it's reproducible."""
    def targets_for(seed):
        env = carnav.make(width=48, height=48, seed=seed, **PRESETS['plain'])
        patch_intersection_targets(env)
        env.reset(seed=seed)
        return list(env.targets)
    a, b = targets_for(4242), targets_for(4242)
    assert a == b


def test_patching_changes_the_episode_relative_to_vanilla():
    """The whole point: a patched env draws a *different* target sequence
    than the vanilla sampler would, for the same seed (extremely unlikely to
    coincide by chance if the patch is doing anything at all)."""
    seed = 4242
    vanilla = carnav.make(width=48, height=48, seed=seed, **PRESETS['plain'])
    vanilla.reset(seed=seed)
    patched = carnav.make(width=48, height=48, seed=seed, **PRESETS['plain'])
    patch_intersection_targets(patched)
    patched.reset(seed=seed)
    assert list(vanilla.targets) != list(patched.targets)


def test_idempotent():
    env = carnav.make(width=48, height=48, seed=1, **PRESETS['plain'])
    patch_intersection_targets(env)
    patch_intersection_targets(env)   # must not raise, or double-wrap, or change behaviour
    env.reset(seed=1)
    assert len(env.targets) == env.cfg.n_targets


# ---------------------------------------------------- wired into the adapter

def test_wired_into_carnav_via_intersection_targets_flag():
    actions = np.random.default_rng(0).uniform(-1, 1, size=(60, 3)).astype(np.float32)

    def drive(**kw):
        env = CarNav('plain', seed=4242, width=48, height=48, **kw)
        out = [env.step({'action': np.zeros(3, np.float32), 'reset': True})]
        for a in actions:
            out.append(env.step({'action': a, 'reset': False}))
        return out

    vanilla = drive(intersection_targets=False)
    patched = drive(intersection_targets=True)
    assert 'log/target_at_intersection' in vanilla[0]
    assert patched[0]['log/target_at_intersection'] > vanilla[0]['log/target_at_intersection']
    # nav block (waypoint bearings/distances) must differ: different targets
    assert not np.array_equal(vanilla[1]['nav'], patched[1]['nav'])


def test_composes_with_path_shaping():
    """Both flags together must not raise -- targets patched on the raw env,
    shaping wrapped around it, same instance either way."""
    env = CarNav('plain', seed=1, width=48, height=48,
                 intersection_targets=True, path_shaping=True,
                 max_episode_steps=30)
    obs = env.step({'action': np.zeros(3, np.float32), 'reset': True})
    for _ in range(30):
        obs = env.step({'action': np.array([0.5, 0, 0.1], np.float32), 'reset': False})
        if obs['is_last']:
            break
    assert np.isfinite(obs['reward'])
