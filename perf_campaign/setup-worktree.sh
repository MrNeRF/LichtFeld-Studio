#!/usr/bin/env bash
# One-shot setup for a LichtFeld worktree so configure/build/tests just work.
# Usage: ./perf_campaign/setup-worktree.sh [worktree-dir]   (default: pwd)
set -euo pipefail
WT=${1:-$(pwd)}; MAIN=/home/gauss/projects/LichtFeld-Studio
cd "$WT"
git submodule update --init --recursive                       # worktrees don't auto-populate
ln -sfn $MAIN/external/libtorch external/libtorch 2>/dev/null || true   # test oracle (untracked)
export VCPKG_ROOT=${VCPKG_ROOT:-/home/gauss/projects/vcpkg}   # toolchain path derives from this
export PATH=/usr/local/cuda-13.3/bin:$PATH                    # nvcc for fresh configures
# ALWAYS pass ccache flags: sccache races nvcc fatbinary (ISS-005)
echo "Configure with:"
echo "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLFS_CUDA_COMPILER_CACHE=ccache -DENABLE_COMPILER_CACHE=ccache [-DBUILD_TESTS=ON]"
echo "Build with:   ./perf_campaign/build.sh <build-dir>      # 2-slot machine-wide semaphore"
echo "Under systemd-run: pass Environment=PATH=...:VCPKG_ROOT=... explicitly (units have minimal env)"
