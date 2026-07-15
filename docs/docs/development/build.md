---
sidebar_position: 4
---

# Building LichtFeld Studio

The standard developer preset builds the complete application feature set. It
keeps tests out of the normal app graph, compiles for one native CUDA
architecture, uses at most six parallel jobs, and automatically uses
`sccache` or `ccache` when either launcher is installed.

```sh
cmake --preset dev-release
cmake --build --preset dev-release
```

This is a full-featured build: it does not disable USD, FFmpeg, OpenImageIO,
RmlUi, Python, MCP, or GUI support. The compiler cache can be disabled without
changing the feature set:

```sh
cmake -S . -B build/dev-release -DENABLE_COMPILER_CACHE=OFF
cmake --build build/dev-release -j6
```

## Reproducible build measurements

Use the measurement preset for compiler or dependency work. It has the same
application features as the developer preset but disables compiler caching, so
clean and incremental timings describe work performed by the compiler rather
than cache hits.

```sh
cmake --preset measure-release
/usr/bin/time -v cmake --build --preset measure-release
```

Delete `build/measure-release` before measuring clean configure time. Keep
`BUILD_TESTS=OFF`: tests are a separate opt-in graph and are not representative
of the application build.

## Compiler cache behavior

`ENABLE_COMPILER_CACHE` defaults to `ON`. CMake prefers `sccache`, falls back to
`ccache`, and continues without a launcher when neither is installed. On
non-Windows CUDA builds the launcher covers C, C++, and CUDA. On Windows CUDA
remains uncached because the supported nvcc/launcher combinations are less
reliable there.

Inspect the active cache with `sccache --show-stats` or `ccache --show-stats`.
Disable the launcher for compiler diagnostics, cold-build comparisons, or when
investigating a cache-specific failure.

## Parallelism and vcpkg

Use no more than six build jobs on a 31 GiB development machine:

```sh
cmake --build build/dev-release -j6
```

vcpkg runs during configure rather than during the Ninja build. Its automatic
package-build concurrency is capped at six. Override it explicitly only on a
machine with enough memory:

```sh
cmake -S . -B build/dev-release -DLFS_VCPKG_MAX_CONCURRENCY=4
```

An explicit `VCPKG_MAX_CONCURRENCY` environment value remains supported for
standard vcpkg workflows. The project does not set global `MAKEFLAGS`; the
chosen build tool remains responsible for job control.

## Shared dependency downloads

Immutable ONNX Runtime and uv release archives are cached outside disposable
build trees. The default is `$XDG_CACHE_HOME/lichtfeld/downloads` or
`~/.cache/lichtfeld/downloads` on Linux, `%LOCALAPPDATA%/LichtFeldStudio/downloads`
on Windows, and `~/Library/Caches/LichtFeldStudio/downloads` on macOS. Override
it with `-DLFS_DOWNLOAD_CACHE_DIR=PATH` when a CI worker provides its own cache.

Supported archives are verified against their declared SHA-256 before reuse.
Concurrent worktrees share a lock and publish completed downloads atomically,
so a killed or parallel configure cannot expose a partial archive.

## CUDA architecture policy

With CUDA 12.x, local builds retain nvcc's `-arch=native` mode and compile one
architecture (for example `sm_89` on an RTX 4090). A controlled 66-object A/B
was faster with native than with a resolved `89-real`, and `cuobjdump` verified
that the native build contained the intended SASS without a release fan-out.
CUDA 13.1 and newer retain the existing numeric compatibility path. Portable
PTX and release packaging continue to use their explicit architecture policies.

The configure summary prints the selected architecture. Confirm it before a
timed build:

```text
CUDA: native (native)
```

## Tests

Tests remain opt-in and have their own manifest dependency feature:

```sh
cmake -S . -B build/tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build/tests --target lichtfeld_tests -j6
```

This keeps test-only package restore and target generation out of the normal
application loop without changing any shipped feature.
