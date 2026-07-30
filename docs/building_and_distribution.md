# Building and Distribution

## Requirements

- CUDA Toolkit 12.8+
- cuDNN 9 for CUDA 12 (CI installs cuDNN 9.5.0 only when the runner image does not already provide it)
- CMake 3.30+
- vcpkg (`VCPKG_ROOT` environment variable set)
- GCC 14+ (Linux) or Visual Studio 2022 v17.10+ (Windows)

Windows builds require the **C++ Clang Compiler for Windows** Visual Studio
Installer individual component in addition to the regular C++ desktop workload.
It provides `clang-cl`, which is used only to build libplacebo (an unconditional
dependency of the standard build) with a Microsoft-compatible ABI. The port
locates `clang-cl.exe` on `PATH` or under `VSINSTALLDIR`/`VCINSTALLDIR` (for
example from the x64 Native Tools Command Prompt). **MSBuild support for LLVM
(clang-cl) toolset** is optional and only needed if you want the LLVM toolset
selectable in Visual Studio IDE projects; the vcpkg port itself does not use
MSBuild. The rest of LichtFeld Studio continues to use the configured Visual
Studio/MSVC toolchain.

On Windows, set `CUDNN_ROOT_DIR` to the cuDNN version root so the build can copy
the CUDA-versioned cuDNN runtime DLLs next to the executable and into portable
installs:

```bat
set CUDNN_ROOT_DIR=C:\Program Files\NVIDIA\CUDNN\v9.24
```

For unusual layouts, pass `-DLFS_CUDNN_BIN_DIR=...` directly to the cuDNN DLL
directory, for example `...\bin\<cuda-version>\x64`.

## Linux Prerequisites

On Linux, LichtFeld Studio requires SDL3 to be built with at least one windowing backend (`x11` or `wayland`).
SDL3's vcpkg port can otherwise build "successfully" and produce a binary that fails at startup with `No available video device`.

Debian/Ubuntu packages used by CI:

```bash
sudo apt install \
  git curl unzip cmake gcc-14 g++-14 ccache ninja-build zip tar pkg-config python3 python3-dev \
  libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev \
  libwayland-dev libxkbcommon-dev libegl-dev libdecor-0-dev libibus-1.0-dev libdbus-1-dev \
  libsystemd-dev nasm autoconf autoconf-archive automake libtool
```

The configure step now fails early if neither a usable X11 stack nor a usable Wayland stack is present.
If you intentionally want a headless or experimental build, pass `-DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF`.

## Build Options

### 1. Native Build (Development)

Builds for your GPU only. Fastest compile time.

```bash
cmake -B build
cmake --build build -j 16
./build/LichtFeld-Studio --help

# Example training run
./build/LichtFeld-Studio -d /path/to/data -o /path/to/output
```

### 2. Portable Build (Distribution)

Creates a self-contained package that works on any machine with an NVIDIA driver.

```bash
cmake -B build -DBUILD_PORTABLE=ON
cmake --build build -j 16
cmake --install build --prefix ./dist

./dist/bin/run_lichtfeld.sh --help

# Example training run
./dist/bin/run_lichtfeld.sh -d /path/to/data -o /path/to/output
```

## What's the Difference?

| | Native Build | Portable Build |
|---|---|---|
| **Output** | `build/LichtFeld-Studio` (66 MB) | `dist/` folder (518 MB) |
| **Target needs CUDA** | Yes | No |
| **Target needs vcpkg** | Yes | No |
| **Self-contained** | No | Yes |
| **Use case** | Development | End-user distribution |

## Distribution Contents

```
dist/
├── bin/
│   ├── LichtFeld-Studio
│   └── run_lichtfeld.sh    # Use this to launch
├── lib/                    # Bundled CUDA & runtime libs
├── share/LichtFeld-Studio/ # Shaders, icons, fonts
└── LICENSE                 # GPL-3.0
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PORTABLE` | OFF | Create self-contained distribution |
| `BUILD_CUDA_PTX_ONLY` | OFF | PTX-only build (auto-enabled by PORTABLE) |
| `BUILD_CUDA_MIN_SM` | 75 | Minimum GPU (75=Turing, 80=Ampere, 89=Ada) |
| `BUILD_TESTS` | OFF | Build test suite |
| `LFS_ENFORCE_LINUX_GUI_BACKENDS` | ON | Linux only. Fail configure if SDL3 would be built without both X11 and Wayland |

ONNX Runtime is consumed as a pinned prebuilt GPU SDK on x64 Windows and Linux
instead of being built by vcpkg. The default SDK is controlled by
`LFS_ONNXRUNTIME_VERSION`; set `LFS_ONNXRUNTIME_ROOT` to an unpacked ONNX Runtime
SDK to use a local or custom build. Set `LFS_ONNXRUNTIME_USE_PREBUILT=OFF` to
fall back to a package-provided `onnxruntime` CMake config.

## Preprocess Model Downloads

The `preprocess` subcommand downloads the default MoGe-2 ONNX model on first
use when `--model` is not provided. The cached model and every downloaded
temporary file are SHA-256 verified on Windows and Linux before ONNX Runtime can
load them. A hash mismatch deletes the untrusted temporary file, rejects the
cached model, and exits with an error. Use `preprocess --download-only` to
preload and verify the cache, or `--no-download` to require an already verified
cache entry.

## Troubleshooting

**"CUDA driver version is insufficient"** - Update NVIDIA driver.

**"no kernel image is available"** - GPU is older than `BUILD_CUDA_MIN_SM`. Rebuild with lower value.

**Missing libraries on target** - Use `run_lichtfeld.sh` (Linux) or ensure DLLs are with .exe (Windows).

**"SDL3 was found, but the resolved SDL build does not expose an X11 or Wayland video backend"** - A stale vcpkg SDL3 artifact is being reused. Remove SDL3 from the local vcpkg install/build cache, then reconfigure with binary-cache bypass enabled:

```bash
lfs="$(pwd)"
triplet="${VCPKG_TARGET_TRIPLET:-x64-linux}"
cd "$VCPKG_ROOT"
./vcpkg remove "sdl3:$triplet" --recurse \
  --x-install-root="$lfs/build/vcpkg_installed" \
  --x-packages-root="$VCPKG_ROOT/packages" \
  --x-buildtrees-root="$VCPKG_ROOT/buildtrees"

cd "$lfs"
VCPKG_BINARY_SOURCES='clear;default,write' cmake -B build -G Ninja --fresh
cmake --build build -j"$(nproc)"
```
