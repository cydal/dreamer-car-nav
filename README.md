# dreamer-car-nav

DreamerV3 trained on the [CarNav](https://github.com/cydal/Car-Navigation-Env)
3D car navigation environment, using the reference
[danijar/dreamerv3](https://github.com/danijar/dreamerv3) implementation
(vendored in `dreamerv3/` and `embodied/`, see `UPSTREAM.md`) plus a thin
environment adapter in `carnav_dreamer/`.

Work in progress. Setup:

```bash
conda env create -f environment.yml
conda activate dreamer-carnav
pip install -e "../rl-env3d[gym]"        # sibling checkout of the env
```

On an NVIDIA machine use `pip install -r requirements-cuda.txt` instead of the
last two steps.
