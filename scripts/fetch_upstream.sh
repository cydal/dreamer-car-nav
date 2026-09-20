#!/usr/bin/env bash
# Fetch the reference DreamerV3 implementation at a pinned commit into
# ./dreamerv3 (gitignored) and install it into the active Python env with
# no dependencies (its requirements.txt pulls Atari/DMLab/Minecraft; ours are
# in requirements.txt). After this, `import dreamerv3` and `import embodied`
# work from anywhere in the env, and `dreamerv3/` is byte-identical to
# upstream -- every change of ours lives in carnav_dreamer/.
set -euo pipefail

UPSTREAM_URL="https://github.com/danijar/dreamerv3"
UPSTREAM_SHA="e3f02248693a79dc8b0ebd62c93683888ddaccfe"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/dreamerv3"

if [ -d "$DEST/.git" ]; then
  have="$(git -C "$DEST" rev-parse HEAD)"
  if [ "$have" = "$UPSTREAM_SHA" ]; then
    echo "dreamerv3/ already at $UPSTREAM_SHA"
  else
    echo "dreamerv3/ is at $have, expected $UPSTREAM_SHA; refetching"
    git -C "$DEST" fetch --depth 1 origin "$UPSTREAM_SHA"
    git -C "$DEST" checkout -q FETCH_HEAD
  fi
else
  rm -rf "$DEST"
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin "$UPSTREAM_URL"
  git -C "$DEST" fetch -q --depth 1 origin "$UPSTREAM_SHA"
  git -C "$DEST" checkout -q FETCH_HEAD
fi

if git -C "$DEST" status --porcelain | grep -q .; then
  echo "WARNING: dreamerv3/ has local modifications; this project expects it pristine:" >&2
  git -C "$DEST" status --short >&2
fi

# The editable install drops dreamer.egg-info/ into the clone; keep it out of
# the clone's `git status` so the pristine check above stays meaningful.
grep -qx 'dreamer.egg-info/' "$DEST/.git/info/exclude" 2>/dev/null || \
  echo 'dreamer.egg-info/' >> "$DEST/.git/info/exclude"

# compat mode adds the clone root to sys.path via a .pth file, so the real
# `dreamerv3` package wins over the same-named clone directory when Python is
# started from the repo root (carnav_dreamer/__init__.py also guards this).
python -m pip install -q --no-deps -e "$DEST" --config-settings editable_mode=compat
(cd /tmp && python - <<'EOF'
import dreamerv3.main, embodied
print("installed:", dreamerv3.main.__file__)
print("           ", embodied.__file__)
EOF
)
