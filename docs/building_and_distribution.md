# Building and Distribution Guide

This guide explains how to build LichtFeld Studio for different use cases: local development, specific GPU architectures, and portable distribution.

## Requirements

- **CUDA Toolkit 12.8+** (required for GCC 14 / C++23 support)
- **GCC 14+** (for C++23 features)
- **CMake 3.30+**
- **vcpkg** (set `VCPKG_ROOT` environment variable)

## Build Types

### 1. Native Build (Default)

Builds for your local GPU only. This is the fastest to compile and produces the smallest binary.

```bash
cmake -B build
cmake --build build -j 16
```

The build system auto-detects your GPU architecture using `nvidia-smi`. The resulting binary will only run on GPUs with the same compute capability.

**Output:** `build/LichtFeld-Studio`

### 2. Fat Binary Build (Multiple GPU Architectures)

Builds for multiple GPU architectures without bundling runtime libraries. Use this when you want to run on different GPUs but the target system has CUDA installed.

```bash
cmake -B build -DBUILD_CUDA_ALL_SM=ON
cmake --build build -j 16
```

By default, this targets SM 86+ (RTX 30 series and newer). To include older architectures:

```bash
cmake -B build -DBUILD_CUDA_ALL_SM=ON -DBUILD_CUDA_ALL_MIN_SM=75
cmake --build build -j 16
```

| MIN_SM | Supported GPUs |
|--------|----------------|
| 75 | RTX 20 series (Turing) and newer |
| 80 | RTX 30 series (Ampere) and newer |
| 86 | RTX 30 series (GA102/GA104) and newer |
| 89 | RTX 40 series (Ada Lovelace) and newer |
| 90 | H100 (Hopper) and newer |

**Note:** Lower MIN_SM values significantly increase compile time.

### 3. Portable Build (Full Distribution)

Creates a self-contained package with fat CUDA binaries AND all required runtime libraries bundled. The resulting folder can be copied to another machine without needing CUDA installed (only the NVIDIA driver is required).

```bash
cmake -B build -DBUILD_PORTABLE=ON
cmake --build build -j 16
```

This automatically enables `BUILD_CUDA_ALL_SM=ON`.

## Distribution

### Linux Portable Build

After building with `-DBUILD_PORTABLE=ON`, the `build/` directory contains:

```
build/
├── LichtFeld-Studio      # Main executable
├── run_lichtfeld.sh      # Launcher script (use this!)
└── lib/                  # Bundled libraries
    ├── libcudart.so*
    ├── libcublas.so*
    ├── libcublasLt.so*
    ├── libcusolver.so*
    ├── libcusparse.so*
    ├── libnvrtc.so*
    └── libnvJitLink.so*
```

**To distribute:**

1. Copy the entire `build/` folder (or just the files listed above)
2. On the target machine, run using the launcher script:
   ```bash
   ./run_lichtfeld.sh -d /path/to/dataset --headless
   ```

The launcher script sets `LD_LIBRARY_PATH` to find the bundled libraries.

**Target machine requirements:**
- NVIDIA GPU with driver installed (no CUDA toolkit needed)
- GPU compute capability >= BUILD_CUDA_ALL_MIN_SM (default: 86)
- Linux with compatible glibc

### Windows Portable Build

After building with `-DBUILD_PORTABLE=ON`, the `build/Release/` (or `build/`) directory contains:

```
build/
├── LichtFeld-Studio.exe  # Main executable
├── *.dll                 # All runtime DLLs (CUDA, vcpkg dependencies)
```

**To distribute:**

1. Copy the entire build folder
2. Run `LichtFeld-Studio.exe` directly (DLLs are found automatically)

### Creating a Minimal Distribution Package

For Linux, you can create a minimal distribution:

```bash
# After portable build
mkdir -p dist
cp build/LichtFeld-Studio dist/
cp build/run_lichtfeld.sh dist/
cp -r build/lib dist/

# Optional: create archive
tar -czvf lichtfeld-studio-linux-x64.tar.gz dist/
```

For Windows:

```powershell
# After portable build
mkdir dist
copy build\LichtFeld-Studio.exe dist\
copy build\*.dll dist\

# Optional: create archive
Compress-Archive -Path dist\* -DestinationPath lichtfeld-studio-win-x64.zip
```

## Build Options Summary

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PORTABLE` | OFF | Bundle all libraries for distribution |
| `BUILD_CUDA_ALL_SM` | OFF | Build for multiple GPU architectures |
| `BUILD_CUDA_ALL_MIN_SM` | 86 | Minimum SM when BUILD_CUDA_ALL_SM=ON |
| `CMAKE_BUILD_TYPE` | Release | Build type (Release/Debug/RelWithDebInfo) |
| `BUILD_TESTS` | OFF | Build test suite |

## Troubleshooting

### "CUDA driver version is insufficient"
The target machine's NVIDIA driver is too old. Update the driver.

### "no kernel image is available for execution"
The binary wasn't compiled for the target GPU's architecture. Rebuild with a lower `BUILD_CUDA_ALL_MIN_SM` or use `BUILD_CUDA_ALL_SM=ON`.

### Missing library errors on target machine
Make sure you're using `run_lichtfeld.sh` (Linux) or that all DLLs are in the same directory as the executable (Windows).
