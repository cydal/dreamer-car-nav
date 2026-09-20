"""DreamerV3 on the CarNav environment.

Everything that is *ours* lives in this package. The reference DreamerV3
implementation is fetched, unmodified, into `../dreamerv3` by
`scripts/fetch_upstream.sh` (see UPSTREAM.md); `carnav` and `env` come from
the installed rl-env3d package and are never modified here.

Import-path note: the upstream clone directory is named `dreamerv3/` and sits
in the repo root, the same name as the Python package inside it
(`dreamerv3/dreamerv3/`). When Python runs from the repo root, the clone
directory shadows the real package as an empty namespace package. Putting the
clone root at the front of `sys.path` here makes `import dreamerv3.main` and
`import embodied` resolve correctly whether or not the editable install ran.
"""

import pathlib
import sys

_UPSTREAM = pathlib.Path(__file__).resolve().parent.parent / 'dreamerv3'
if (_UPSTREAM / 'dreamerv3' / '__init__.py').exists():
  if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
