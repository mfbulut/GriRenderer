#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
slangc "$SCRIPT_DIR/shader.slang" -target spirv -matrix-layout-column-major -entry vs_main -entry fs_main -o "$SCRIPT_DIR/shader.spv"
echo "Shader compiled successfully to shader.spv"
