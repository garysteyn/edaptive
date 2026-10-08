#!/usr/bin/env bash
set -euo pipefail

EDAPTIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CALYPSO_REPO="git@github.com:garysteyn/calypso-clean.git"
CALYPSO_DIR="$EDAPTIVE_DIR/calypso"

rm -rf "$CALYPSO_DIR"

echo "Cloning Calypso..."
git clone "$CALYPSO_REPO" "$CALYPSO_DIR"

echo "Applying Calypso patches..."
git -C "$CALYPSO_DIR" apply "$EDAPTIVE_DIR/scripts/calypso-integration.patch"

echo "Installing requirements..."
python -m pip install -r "$EDAPTIVE_DIR/requirements.txt"

echo "Patching TF-Agents..."
python "$EDAPTIVE_DIR/scripts/patch_tf_agents.py"

echo "Calypso successfully prepared."
