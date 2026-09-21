"""`PathDistanceShaper`: substituting road-graph path distance for the env's
straight-line progress term, without touching anything else in the reward.

The synthetic graph below is a square "U" -- car start at one corner, target
at the adjacent corner, but the only road connects them the long way round
two other corners. It's the smallest topology where straight-line distance
and path distance actively *disagree* about which direction is progress,
which is exactly the dead-end/detour case this shaper exists for:

    A(0,0) ---- B(0,10)
      :            |
    [no road]      |
      :            |
    D(10,0) ---- C(10,10)

Moving A -> B increases straight-line distance to D (10.0 -> 14.14) but
decreases path distance (30 -> 20): euclidean progress would be *negative*
for the one move that is actually correct.
"""

import numpy as np
import pytest

import carnav
from carnav_dreamer.env import CarNav
from carnav_dreamer.presets import PRESETS
from carnav_dreamer.shaping import PathDistanceShaper
from wrappers import RewardOverrideWrapper

A, B, C, D = (0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)
NODES = np.array([A, B, C, D])
# DIRS: 0=+x, 1=+y, 2=-x, 3=-y
LINKS = np.array([
    [-1, 1, -1, -1],   # A: +y -> B
    [2, -1, -1, 0],    # B: +x -> C, -y -> A
    [-1, -1, 1, 3],    # C: -x -> B, -y -> D
    [-1, 2, -1, -1],   # D: +y -> C
], dtype=np.int32)


class FakeCity:
  def __init__(self, nodes=NODES, links=LINKS):
    self.road_nodes = nodes
    self.node_links = links


class FakeCfg:
  progress_weight = 1.0


class FakeEnv:
  """Just enough surface for PathDistanceShaper: .city, .target_idx,
  .targets, .cfg.progress_weight."""
  def __init__(self):
    self.city = FakeCity()
    self.target_idx = 0
    self.targets = [D]
    self.cfg = FakeCfg()


def info_at(x, y, progress=0.0):
  return {'x': x, 'y': y, 'reward_components': {'progress': progress}}


def test_path_distance_beats_euclidean_at_the_disagreeing_point():
  env = FakeEnv()
  shaper = PathDistanceShaper(env)
  shaper._dist_field = shaper._distance_field(env)   # build without going through __call__
  assert shaper._path_dist(env, *A) == pytest.approx(30.0)
  assert shaper._path_dist(env, *B) == pytest.approx(20.0)
  assert shaper._path_dist(env, *C) == pytest.approx(10.0)
  assert shaper._path_dist(env, *D) == pytest.approx(0.0)
  # And it disagrees with euclidean in exactly the direction the module
  # docstring claims: A -> B increases euclidean distance, decreases path.
  euclid_a, euclid_b = np.hypot(*np.subtract(D, A)), np.hypot(*np.subtract(D, B))
  assert euclid_b > euclid_a               # euclidean says "worse"
  assert shaper._path_dist(env, *B) < shaper._path_dist(env, *A)  # path says "better"


def test_a_to_b_is_rewarded_not_punished():
  """The actual reward substitution: moving A -> B should come out net
  positive under the shaper despite the env's own progress term being
  negative for that exact move."""
  env = FakeEnv()
  shaper = PathDistanceShaper(env)
  shaper(None, None, reward=-0.1, terminated=False, truncated=False,
         info=info_at(*A, progress=0.0))          # first call: no prior point yet
  old_progress = -4.142                            # (10.0 - 14.142) * weight, the env's own term
  shaped = shaper(None, None, reward=-0.1 + old_progress, terminated=False,
                   truncated=False, info=info_at(*B, progress=old_progress))
  # reward - old_progress + path_progress = -0.1 + (30 - 20)*1.0
  assert shaped == pytest.approx(-0.1 + 10.0)
  assert shaped > 0                                # net positive: the move IS progress


def test_only_the_progress_component_changes():
  """Every other reward component (time, crash, waypoint bonus, ...) must
  pass through byte-for-byte -- the shaper only ever touches `progress`."""
  env = FakeEnv()
  shaper = PathDistanceShaper(env)
  shaper(None, None, 0.0, False, False, info_at(*A))
  other_terms = -0.1 - 100.0 + 50.0                 # time - crash - red_light, say
  reward = other_terms + 3.7                        # + some env progress value
  shaped = shaper(None, None, reward, False, False,
                   info_at(*B, progress=3.7))
  path_progress = 30.0 - 20.0                       # path_dist(A) - path_dist(B)
  assert shaped == pytest.approx(other_terms + path_progress)


def test_first_lookup_after_a_rebuild_charges_nothing():
  """No prior point to compare against right after construction, after a
  target change, or after a map regen -- must not fabricate a phantom jump."""
  env = FakeEnv()
  shaper = PathDistanceShaper(env)
  shaped = shaper(None, None, reward=-0.1, terminated=False, truncated=False,
                   info=info_at(*A, progress=0.0))
  assert shaped == pytest.approx(-0.1)              # -0.1 - 0.0 + 0.0


def test_rebuilds_when_target_changes():
  env = FakeEnv()
  shaper = PathDistanceShaper(env)
  shaper(None, None, 0.0, False, False, info_at(*A))
  env.target_idx = 0                                # same target: no rebuild, distances unchanged
  d1 = shaper._path_dist(env, *B)
  env.targets = [D, A]
  env.target_idx = 1                                # target is now A itself
  shaper(None, None, 0.0, False, False, info_at(*B))
  d2 = shaper._path_dist(env, *B)
  assert d1 == pytest.approx(20.0)                  # B to D (the original target): unchanged
  assert d2 == pytest.approx(10.0)                  # B to A (the new target): one edge, B-A


def test_rebuilds_when_the_map_regenerates():
  """`env.city` is the *same* object across a reset when randomize_map=True
  (only its arrays are reassigned by `city.generate()`); the shaper must key
  off array identity, not city identity."""
  env = FakeEnv()
  shaper = PathDistanceShaper(env)
  shaper(None, None, 0.0, False, False, info_at(*A))
  assert shaper._path_dist(env, *B) == pytest.approx(20.0)
  # Simulate city.generate(): same city object, brand-new node/link arrays
  # with a different topology (B and D now directly connected -- one edge,
  # cost = their straight-line separation, 14.142, not the *previous*
  # topology's 20; the point is that it changed at all, i.e. was rebuilt).
  env.city.road_nodes = np.array([A, B, C, D])
  env.city.node_links = np.array([
      [-1, 1, -1, -1], [2, -1, -1, 3], [-1, -1, 1, -1], [-1, -1, -1, 1],
  ], dtype=np.int32)
  shaper(None, None, 0.0, False, False, info_at(*A))
  assert shaper._path_dist(env, *B) == pytest.approx(np.hypot(*np.subtract(D, B)))


def test_no_more_targets_disables_shaping_without_crashing():
  env = FakeEnv()
  env.target_idx = 5   # past the end of a 1-element targets list
  shaper = PathDistanceShaper(env)
  shaped = shaper(None, None, reward=-0.1, terminated=False, truncated=False,
                   info=info_at(*A, progress=0.0))
  assert shaped == pytest.approx(-0.1)   # progress substituted with 0, nothing else changes


# ------------------------------------------------------- against the real env

def test_wired_into_carnav_via_path_shaping_flag():
  """End to end: carnav_dreamer.env.CarNav(path_shaping=True) actually
  changes obs['reward'] relative to the same seed/actions with it off, and
  never crashes over a real episode."""
  actions = np.random.default_rng(0).uniform(-1, 1, size=(80, 3)).astype(np.float32)

  def drive(**kw):
    env = CarNav('plain', seed=42, width=48, height=48, **kw)
    out = [env.step({'action': np.zeros(3, np.float32), 'reset': True})]
    for a in actions:
      out.append(env.step({'action': a, 'reset': False}))
    return out

  unshaped = drive(path_shaping=False)
  shaped = drive(path_shaping=True)
  # Same episode (same seed, same actions), so everything except reward must
  # be identical -- path_shaping must never change dynamics, obs, or endings.
  for u, s in zip(unshaped, shaped):
    for k in u:
      if k == 'reward':
        continue
      np.testing.assert_array_equal(u[k], s[k], err_msg=k)
  rewards_differ = any(
      abs(u['reward'] - s['reward']) > 1e-6 for u, s in zip(unshaped, shaped))
  assert rewards_differ, 'path_shaping=True produced identical rewards to False'


def test_scripts_evaluate_style_env_ignores_path_shaping_kwarg():
  """scripts/evaluate.py strips adapter-only kwargs (path_shaping among them)
  before calling carnav.make directly -- carnav.make itself must not know
  about it, confirming the flag lives entirely in our adapter."""
  with pytest.raises(TypeError):
    carnav.make(path_shaping=True, **PRESETS['plain'])
