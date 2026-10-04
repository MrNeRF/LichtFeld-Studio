#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Builds the three macOS backend configurations and checks what each binary
# contains:
#   metal   Metal graphics + Metal tensors; no Vulkan, VMA, volk or MoltenVK.
#   vulkan  Vulkan graphics (MoltenVK) + Vulkan tensors; no Metal backend or viewer.
#   both    Vulkan graphics + Metal and Vulkan tensors, switched at runtime
#           with --tensor-backend=metal|vulkan.
#
# Usage: tools/build_mac_matrix.sh [metal] [vulkan] [both]   (default: all)
# Environment: BUILD_ROOT (default: repo root), JOBS (default: 8),
#              BUILD_TYPE (default: Release), TARGETS (default: app and tests).
# Re-running only rebuilds. Do not run while a LichtFeld instance is running.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$ROOT}"
JOBS="${JOBS:-8}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TARGETS="${TARGETS:-LichtFeld-Studio lichtfeld_tests gpu_program_contracts}"
if [ $# -eq 0 ]; then
    set -- metal vulkan both
fi

if [ "$(uname)" != "Darwin" ]; then
    echo "build_mac_matrix.sh runs on macOS only" >&2
    exit 2
fi
: "${VCPKG_ROOT:?VCPKG_ROOT must point at the vcpkg checkout}"
export VCPKG_ROOT

flags_for() {
    case "$1" in
        metal) echo "-DLFS_GRAPHICS_BACKEND=Metal -DLFS_TENSOR_METAL=ON -DLFS_TENSOR_VULKAN=OFF" ;;
        vulkan) echo "-DLFS_GRAPHICS_BACKEND=Vulkan -DLFS_TENSOR_METAL=OFF -DLFS_TENSOR_VULKAN=ON" ;;
        both) echo "-DLFS_GRAPHICS_BACKEND=Vulkan -DLFS_TENSOR_METAL=ON -DLFS_TENSOR_VULKAN=ON" ;;
        *) return 1 ;;
    esac
}

# Prints matching symbols of the app and its own libraries; the pattern is an
# extended regex over `nm -gU` and `nm -u` output.
symbols() {
    local dir="$1" pattern="$2"
    local files=("$dir/LichtFeld-Studio")
    while IFS= read -r lib; do files+=("$lib"); done < <(find "$dir" -maxdepth 1 -name 'liblfs_*.dylib')
    for file in "${files[@]}"; do
        nm -g "$file" 2>/dev/null | grep -E "$pattern" | sed "s|^|$(basename "$file"): |"
    done
}

linked() {
    local dir="$1" pattern="$2"
    while IFS= read -r file; do
        otool -L "$file" 2>/dev/null | tail -n +2 | grep -iE "$pattern" | sed "s|^|$(basename "$file"): |"
    done < <(find "$dir" -maxdepth 1 \( -name 'LichtFeld-Studio' -o -name '*.dylib' \))
}

# Prints one line per problem; prints nothing when the binaries match the config.
check() {
    local config="$1" dir="$2"
    local vulkan_symbols='[^A-Za-z_](_vk[A-Z]|_vma[A-Z]|_volk)'
    local metal_symbols='MetalTensorReader|MetalViewportRenderer|MetalGraphicsContext'
    case "$config" in
        metal)
            [ -n "$(linked "$dir" 'vulkan|moltenvk')" ] && echo "links Vulkan/MoltenVK"
            [ -n "$(symbols "$dir" "$vulkan_symbols")" ] && echo "has Vulkan/VMA/volk symbols"
            ;;
        vulkan)
            [ -n "$(symbols "$dir" "$metal_symbols")" ] && echo "has Metal backend/viewer symbols"
            [ -z "$(symbols "$dir" "$vulkan_symbols")" ] && echo "lacks Vulkan symbols"
            ;;
        both)
            [ -z "$(symbols "$dir" "$metal_symbols")" ] && echo "lacks Metal symbols"
            [ -z "$(symbols "$dir" "$vulkan_symbols")" ] && echo "lacks Vulkan symbols"
            ;;
    esac
    return 0
}

SUMMARY=""
status=0
for config in "$@"; do
    flags="$(flags_for "$config")" || { echo "unknown config: $config" >&2; exit 2; }
    dir="$BUILD_ROOT/build-mac-$config"
    log="$dir/matrix.log"
    mkdir -p "$dir"
    echo "== $config ($dir)"
    if [ ! -f "$dir/CMakeCache.txt" ]; then
        # shellcheck disable=SC2086
        if ! cmake -S "$ROOT" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBUILD_TESTS=ON \
                $flags >"$log" 2>&1; then
            SUMMARY="$SUMMARY$(printf '%-8s configure failed (see %s)' "$config" "$log")"$'\n'
            status=1
            continue
        fi
    fi
    # shellcheck disable=SC2086
    if ! ninja -C "$dir" -j"$JOBS" $TARGETS >>"$log" 2>&1; then
        SUMMARY="$SUMMARY$(printf '%-8s build failed: %s' "$config" "$(grep -m1 -E 'error:|FAILED' "$log")")"$'\n'
        status=1
        continue
    fi
    problems="$(check "$config" "$dir" | paste -sd ';' -)"
    if [ -z "$problems" ]; then
        SUMMARY="$SUMMARY$(printf '%-8s ok' "$config")"$'\n'
    else
        SUMMARY="$SUMMARY$(printf '%-8s binary check failed: %s' "$config" "$problems")"$'\n'
        status=1
    fi
done

echo
printf '%-8s %s\n' config result
printf '%s' "$SUMMARY"
exit $status
