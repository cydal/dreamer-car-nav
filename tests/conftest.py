import pathlib
import sys

# Make `carnav_dreamer` importable when pytest is run from anywhere. Importing
# it also puts the upstream clone (`dreamerv3/`) on the path, see its __init__.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
import carnav_dreamer  # noqa: E402,F401
