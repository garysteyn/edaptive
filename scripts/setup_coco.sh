#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# COCO setup for edaptive
#
# This script:
#   1. Checks for the project virtual environment
#   2. Installs COCO build dependencies
#   3. Clones COCO into .coco/
#   4. Checks out the pinned COCO commit
#   5. Applies the tracked BBOB patches
#   6. Builds COCO
#   7. Installs cocoex into the project virtual environment
#   8. Verifies the cocoex installation and BBOB suite construction
#
# .coco/ and venv/ should be included in .gitignore.
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR="$PROJECT_ROOT/venv"
COCO_DIR="$PROJECT_ROOT/.coco"

PYTHON="$VENV_DIR/bin/python"

COCO_REPOSITORY="https://github.com/numbbo/coco.git"

# Pin this to the COCO commit used by the project.
COCO_COMMIT="0a7b447fc5e1aeff6aa52fb5dcc37eca4e917219"

DIMENSIONS_PATCH_FILE="$PROJECT_ROOT/scripts/bbob_dimensions.patch"
HIGH_DIMENSIONS_PATCH_FILE="$PROJECT_ROOT/scripts/bbob_support_high_dimensions.patch"

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

error() {
    echo
    echo "ERROR: $1"
    echo
    exit 1
}


# ------------------------------------------------------------
# Check system dependencies
# ------------------------------------------------------------

echo "============================================================"
echo "Checking system dependencies"
echo "============================================================"

command -v git >/dev/null 2>&1 || \
    error "git is not installed."

command -v gcc >/dev/null 2>&1 || \
    error "gcc is not installed."

command -v python3 >/dev/null 2>&1 || \
    error "python3 is not installed."

echo "git:     $(git --version)"
echo "gcc:     $(gcc --version | head -n 1)"
echo "python3: $(python3 --version)"


# ------------------------------------------------------------
# Check patches
# ------------------------------------------------------------

if [[ ! -f "$DIMENSIONS_PATCH_FILE" ]]; then
    error "BBOB dimension patch not found:
    $DIMENSIONS_PATCH_FILE"
fi

if [[ ! -f "$HIGH_DIMENSIONS_PATCH_FILE" ]]; then
    error "BBOB high-dimension support patch not found:
    $HIGH_DIMENSIONS_PATCH_FILE"
fi

# ------------------------------------------------------------
# Check virtual environment
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Checking Python virtual environment"
echo "============================================================"

if [[ ! -x "$PYTHON" ]]; then
    error "Python virtual environment not found:
    $VENV_DIR

Please create the project's venv first."
fi

echo "Using virtual environment:"
echo "  $VENV_DIR"

echo "Python:"
"$PYTHON" --version


# ------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Installing Python dependencies"
echo "============================================================"

"$PYTHON" -m pip install --upgrade \
    pip \
    setuptools \
    wheel

"$PYTHON" -m pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    cython \
    colorama \
    toml


# ------------------------------------------------------------
# Clone COCO
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Setting up COCO"
echo "============================================================"

if [[ ! -d "$COCO_DIR/.git" ]]; then

    echo "Cloning COCO..."

    git clone "$COCO_REPOSITORY" "$COCO_DIR"

else

    echo "COCO repository already exists:"
    echo "  $COCO_DIR"

fi


# ------------------------------------------------------------
# Checkout pinned COCO version
# ------------------------------------------------------------

cd "$COCO_DIR"

if [[ "$COCO_COMMIT" == "YOUR_COCO_COMMIT" ]]; then
    error "COCO_COMMIT has not been set in setup_coco.sh."
fi

echo
echo "Checking out COCO commit:"
echo "  $COCO_COMMIT"

git fetch --all --tags

# Reset the local COCO checkout to the exact pinned commit.
git reset --hard "$COCO_COMMIT"
git clean -fd


# ------------------------------------------------------------
# Apply BBOB patches
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Applying BBOB patches"
echo "============================================================"

if ! git apply --check "$DIMENSIONS_PATCH_FILE" 2>/dev/null; then
    error "Could not apply BBOB dimension patch.

The patch may have been created for a different COCO commit."
fi

git apply "$DIMENSIONS_PATCH_FILE"

echo "BBOB dimension patch applied."

if ! git apply --check "$HIGH_DIMENSIONS_PATCH_FILE" 2>/dev/null; then
    error "Could not apply BBOB high-dimension support patch.

The patch may have been created for a different COCO commit."
fi

git apply "$HIGH_DIMENSIONS_PATCH_FILE"

echo "BBOB high-dimension support patch applied."

# ------------------------------------------------------------
# Build COCO
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Building COCO"
echo "============================================================"

"$PYTHON" scripts/fabricate


# ------------------------------------------------------------
# Install cocoex
# ------------------------------------------------------------

COCO_PYTHON_DIR="$COCO_DIR/code-experiments/build/python"

if [[ ! -d "$COCO_PYTHON_DIR" ]]; then
    error "COCO Python build directory was not created:
    $COCO_PYTHON_DIR"
fi

echo
echo "============================================================"
echo "Installing cocoex"
echo "============================================================"

"$PYTHON" -m pip install "$COCO_PYTHON_DIR"


# ------------------------------------------------------------
# Test cocoex import
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Testing cocoex"
echo "============================================================"

"$PYTHON" - <<'PY'
import cocoex

print("cocoex imported successfully.")

if hasattr(cocoex, "__version__"):
    print("COCO version:", cocoex.__version__)
PY


# ------------------------------------------------------------
# Test BBOB suite construction
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Testing BBOB suite construction"
echo "============================================================"

"$PYTHON" - <<'PY'
import cocoex

suite = cocoex.Suite("bbob", "", "")

print("Successfully constructed the BBOB suite.")
PY


# ------------------------------------------------------------
# Test BBOB problem retrieval
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Testing BBOB problem retrieval"
echo "============================================================"

"$PYTHON" - <<'PY'
import cocoex

suite = cocoex.Suite("bbob", "", "")

problem = suite.get_problem_by_function_dimension_instance(
    1,
    2,
    1,
)

print("Successfully loaded BBOB F1, D2, I1:")
print(problem)
PY

# ------------------------------------------------------------
# Finished
# ------------------------------------------------------------

echo
echo "============================================================"
echo "COCO setup completed successfully"
echo "============================================================"

echo
echo "Python:"
echo "  $PYTHON"

echo
echo "COCO:"
echo "  $COCO_DIR"

echo
echo "Verified:"
echo "  cocoex import"
echo "  BBOB suite construction"
echo "  BBOB F1, D2, I1"

echo