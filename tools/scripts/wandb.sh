#!/bin/bash

set -e

if [[ -n "$VIRTUAL_ENV" ]]; then
    ENV_NAME=$(basename "$VIRTUAL_ENV")
elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    ENV_NAME="$CONDA_DEFAULT_ENV"
else
    echo "❌ Not in any virtual environment (virtualenv or conda)."
    exit 1
fi

if [[ "$ENV_NAME" != "simulation" ]]; then
    echo "❌ Environment name is '$ENV_NAME', expected 'simulation'."
    exit 1
fi

WAND_B_DIR="src/utils/external/wandbmon"
if [[ ! -d "$WAND_B_DIR" ]]; then
    echo "❌ Directory '$WAND_B_DIR' does not exist."
    exit 1
fi

echo "✅ Installing wandbmon from $WAND_B_DIR ..."
cd "$WAND_B_DIR"
pip install -e .
echo "✅ wandbmon installed successfully."

