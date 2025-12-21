# Building and Distribution

## Requirements

- CUDA Toolkit 12.8+
- CMake 3.30+
- vcpkg (`VCPKG_ROOT` environment variable set)
- GCC 14+ (Linux) or Visual Studio 2022 v17.10+ (Windows)

## Build Options

### 1. Native Build (Development)

Builds for your GPU only. Fastest compile time.

```bash
cmake -B build
cmake --build build -j 16
./build/LichtFeld-Studio -d data/test
```

### 2. Portable Build (Distribution)

Creates a self-contained package that works on any machine with an NVIDIA driver.

```bash
cmake -B build -DBUILD_PORTABLE=ON
cmake --build build -j 16
cmake --install build --prefix ./dist

./dist/bin/run_lichtfeld.sh -d data/test
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

## Troubleshooting

**"CUDA driver version is insufficient"** - Update NVIDIA driver.

**"no kernel image is available"** - GPU is older than `BUILD_CUDA_MIN_SM`. Rebuild with lower value.

**Missing libraries on target** - Use `run_lichtfeld.sh` (Linux) or ensure DLLs are with .exe (Windows).
