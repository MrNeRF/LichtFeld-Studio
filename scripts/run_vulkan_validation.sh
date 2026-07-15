#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/build/LichtFeld-Studio"
layer_path="$repo_root/../Vulkan-ValidationLayers/build/layers"
fatal=0

usage() {
    cat <<'EOF'
Usage: scripts/run_vulkan_validation.sh [options] [-- app-arguments]

Options:
  --binary PATH      LichtFeld Studio executable
  --layer-path PATH  Source-built Vulkan-ValidationLayers layer directory
  --fatal            Abort on the first validation error
  -h, --help         Show this help
EOF
}

while (($#)); do
    case "$1" in
    --binary)
        binary="${2:?--binary requires a path}"
        shift 2
        ;;
    --layer-path)
        layer_path="${2:?--layer-path requires a path}"
        shift 2
        ;;
    --fatal)
        fatal=1
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if [[ ! -x "$binary" ]]; then
    echo "LichtFeld Studio executable is not available: $binary" >&2
    exit 1
fi
if [[ ! -f "$layer_path/VkLayer_khronos_validation.json" ]]; then
    echo "Source-built Vulkan validation layer is not available: $layer_path" >&2
    exit 1
fi

validation_environment=(
    "VK_LAYER_PATH=$layer_path"
    "VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation"
    "VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT"
    "LFS_VK_VALIDATION=1"
)
if ((fatal)); then
    validation_environment+=("LFS_VK_VALIDATION_FATAL=1")
fi

echo "Validation layer: $layer_path" >&2
exec env "${validation_environment[@]}" "$binary" "$@"
