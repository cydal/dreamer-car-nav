"""Task presets: the part of the DreamerV3 task string after `carnav_`.

Each maps to keyword arguments for `carnav.make`. They mirror the four rows of
the baseline table in rl-env3d's INTEGRATION.md ("What to beat"), so a trained
policy's number can be put next to the scripted controller's without any
config archaeology:

    task            dims   scripted baseline (300 episodes, seeds 5000-5299)
    carnav_plain     46    +222 mean reward, 1.90/3 waypoints, 44.0% full route
    carnav_lights    53    +179, 1.86/3, 42.7%
    carnav_traffic   66    +137, 1.42/3, 26.7%
    carnav_full      73    +129, 1.58/3, 31.3%

The env keeps the map, spawn and waypoints identical across these four for a
given seed (independent RNG streams per subsystem), so differences between
rows are differences between tasks, not between maps.
"""

PRESETS = {
    "plain": dict(traffic=False, traffic_lights=False),
    "lights": dict(traffic=False, traffic_lights=True),
    "traffic": dict(traffic=True, traffic_lights=False),
    "full": dict(traffic=True, traffic_lights=True),
}
