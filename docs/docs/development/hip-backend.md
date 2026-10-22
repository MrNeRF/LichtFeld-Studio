# Native HIP backend

HIP is an optional Tensor and training backend on Windows and Linux. It does not
require CUDA, Torch, or PyTorch. Rendering uses the separately selected rendering
backend. Enabling HIP does not change the defaults of other backends.

## Developer build

Use a complete ROCm development SDK and the application's normal dependencies:

```sh
cmake -S . -B build-hip -G Ninja \
  -DLFS_ENABLE_CUDA=OFF \
  -DLFS_HIP_RUNTIME=ON -DLFS_TENSOR_HIP=ON -DLFS_HIP_TRAINING_RUNTIME=ON \
  -DLFS_HIP_ROOT=/absolute/path/to/sdk -DLFS_HIP_ARCHITECTURE=gfx1150
cmake --build build-hip --target LichtFeld-Studio
```

CUDA can also remain enabled in the same build. The HIP code is compiled with
the SDK's compiler independently of the application's C++ and CUDA compilers.
Use separate build directories for different target/profile configurations.

## Portable builds

`LFS_HIP_PORTABLE=ON` includes the declared wave32 targets in the generic runtime
and training modules. `LFS_HIP_ARCHITECTURES` can explicitly restrict the target
list, for example `-DLFS_HIP_ARCHITECTURES="gfx1103;gfx1150"`. An explicit list
takes precedence over the portable default. Unsupported targets fail configure.

The generic modules contain one device image per requested target. They are not
an emulation layer. The HIP runtime selects the image for the actual GPU.
HIPRTC uses the selected device's architecture in both its compiler options and
expression cache identity, independently of the process's current device.

Multiarchitecture builds use conservative FastGS kernels. With
`LFS_HIP_ARCH_PROFILES=ON`, a build containing `gfx1150` additionally includes a
separate validated `gfx1150` training profile. Single-target developer builds
retain their compile-time profile choice. `LFS_HIP_VALIDATED_TUNING=OFF` disables
the specialized kernel options, including cached options inherited from an
earlier configuration. Common stable sort and workspace improvements remain
available in the conservative path.

## Packaging and module selection

The `lfs_hip_package` target stages the native modules in `hip/` beside the
executable, with architecture-specific training modules under
`hip/profiles/<architecture>/`. SDK runtime dependencies are staged beside the
executable. The same layout is included by `cmake --install`.

The application prefers this adjacent package, then its developer build path.
Training selects a matching architecture profile when available. A profile
which cannot load or has incompatible metadata falls back to the generic
module and logs a CLI warning. The failed profile is not retried on every
operation. Fatal device errors after GPU work begins are not silently replayed.

`LFS_HIP_DISABLE_ARCH_PROFILES=1` selects the generic module in a package with
separate profiles. It cannot change the kernels in a single-profile developer
binary. `LFS_HIP_TRAINING_MODULE` is an explicit diagnostic override: a failed
override reports an error rather than silently choosing another file.

## Support boundaries

The trainer currently requires wave32. Portable builds cover RDNA target IDs
declared in `cmake/HipArchitectures.cmake`; GCN/CDNA wave64 targets are rejected.
Compiling a target is not a claim of runtime or numerical validation on that
hardware. GPU/driver availability remains subject to the selected SDK and OS.
The tested device profile is `gfx1150`; other targets use conservative kernels.

## Verification

`cmake/hip-native` builds only native modules for cross-target compilation checks
without the host application. It requires the SDK, Slang compiler, GLM headers,
Python, and the same source checkout. No GPU is required to compile it.

`tests/hip/profile_policy` verifies compile-time profile selection without an
SDK. `tests/hip/module_policy` verifies module paths and target matching without
a GPU. Native numerical and application contracts require a compatible GPU.

Four dedicated GitHub workflows cover Windows/Linux, HIP-only/CUDA+HIP, each in
Release and Debug. They are exclusively `workflow_dispatch`; existing workflows
are unchanged. These jobs verify builds and CPU policy checks, not GPU quality.
Benchmark measurements and development reports are kept outside source commits.
