#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Update project-specific COCO BBOB dimensions
#
# Usage:
#   ./scripts/update_bbob_dimensions.sh 2 3 5 10 20 30 40 50
#
# The script:
#   1. Checks the project's COCO checkout
#   2. Checks the pinned COCO commit
#   3. Restores suite_bbob.c to the pinned COCO version
#   4. Updates the BBOB dimension list
#   5. Regenerates bbob_dimensions.patch
#
# The generated patch is tracked by Git.
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COCO_DIR="$PROJECT_ROOT/.coco"
BBOB_SOURCE="$COCO_DIR/code-experiments/src/suite_bbob.c"
PATCH_FILE="$PROJECT_ROOT/scripts/bbob_dimensions.patch"

COCO_COMMIT="0a7b447fc5e1aeff6aa52fb5dcc37eca4e917219"

error() {
    echo
    echo "ERROR: $1"
    echo
    exit 1
}

if [[ $# -eq 0 ]]; then
    error "No BBOB dimensions supplied.

Usage:
    ./scripts/update_bbob_dimensions.sh 2 3 5 10 20 30 40"
fi

if [[ ! -d "$COCO_DIR/.git" ]]; then
    error "COCO checkout not found:
    $COCO_DIR

Run setup_coco.sh first."
fi

if [[ ! -f "$BBOB_SOURCE" ]]; then
    error "BBOB suite source not found:
    $BBOB_SOURCE"
fi

cd "$COCO_DIR"

CURRENT_COMMIT="$(git rev-parse HEAD)"

if [[ "$CURRENT_COMMIT" != "$COCO_COMMIT" ]]; then
    error "COCO checkout is not at the expected commit.

Expected:
    $COCO_COMMIT

Found:
    $CURRENT_COMMIT"
fi

echo "COCO commit:"
echo "  $CURRENT_COMMIT"

echo
echo "Requested BBOB dimensions:"
printf '  %s\n' "$*"

# Restore the source file to the pinned upstream version.
git checkout -- "$BBOB_SOURCE"

# Construct the C dimension list.
DIMENSIONS="$(IFS=', '; echo "$*")"

python3 - "$BBOB_SOURCE" "$DIMENSIONS" <<'PY'
import re
import sys

path = sys.argv[1]
dimensions = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    source = f.read()

pattern = r"const size_t dimensions\[\] = \{[^}]*\};"

replacement = (
    f"const size_t dimensions[] = {{ {dimensions} }};"
)

source, count = re.subn(pattern, replacement, source, count=1)

if count != 1:
    raise SystemExit(
        "Could not find the BBOB dimension declaration in suite_bbob.c."
    )

with open(path, "w", encoding="utf-8") as f:
    f.write(source)
PY

# Generate the project patch.
git diff -- "$BBOB_SOURCE" > "$PATCH_FILE"

if [[ ! -s "$PATCH_FILE" ]]; then
    error "No changes were generated."
fi

echo
echo "============================================================"
echo "BBOB dimension patch updated"
echo "============================================================"

echo
echo "Patch:"
echo "  $PATCH_FILE"

echo
echo "Resulting dimensions:"
grep "const size_t dimensions" "$BBOB_SOURCE"

echo
echo "Git diff:"
git --no-pager diff -- "$BBOB_SOURCE"