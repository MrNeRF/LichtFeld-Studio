# Native HIP training

This directory implements the optional native HIP training modules and their
application adapters. It introduces no Torch/PyTorch dependency and can be built
without CUDA. Tensor and rendering device selection remain independent.

For SDK prerequisites, build presets, architecture profiles, packaging, support
boundaries and validation commands, see the maintained
[HIP backend guide](../../../docs/docs/development/hip-backend.md).

## Build integration

Application builds opt in with `LFS_HIP_RUNTIME`, `LFS_TENSOR_HIP` and
`LFS_HIP_TRAINING_RUNTIME`. Configure `LFS_HIP_ROOT` and the required architecture
or portable target list. The selected SDK compiler builds HIP modules separately
from the application's C++ and CUDA compilers. Python preparation scripts generate
private kernel sources without modifying the original backend sources.

## Native module boundary

`training_api.h` defines C-compatible descriptors and opaque handles. Host adapters
validate the module ABI before using it. Allocation and release remain inside
the owning module; C++ containers and native HIP/CUDA headers do not cross this
boundary. Calls on a session must be serialized except for its atomic cancellation
operation. Join callers and successfully destroy sessions before unloading their
module; failed destruction does not authorize unloading live GPU code.

Generic modules retain the configured architecture coverage. Optional matching
profiles specialize validated devices and fall back to generic modules with a
CLI warning if profile loading or metadata validation fails. GPU work already
submitted is not silently replayed after a fatal device failure.

## Tests

See the [test entry points](../../../docs/docs/development/hip-backend.md#cpu-and-gpu-test-entry-points)
and [standalone contract instructions](../../../tests/hip/README.md). Compile-only
coverage is distinct from GPU execution. Numerical, lifecycle, checkpoint/resume,
shared-storage and cross-device publication contracts require compatible hardware.
Development journals, raw measurements and experimental artifacts are maintained
outside the final source diff.
