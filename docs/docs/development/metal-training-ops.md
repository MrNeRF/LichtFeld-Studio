# Metal training ops

The Metal trainer fills the Metal slots of the training ops table (`training_ops(GpuBackend::Metal)`, see [training ops parity](training-ops-parity.md)). Each family behaves as its CUDA table entry: the CUDA `*_cuda.cpp` files and the kernels they launch are the specification.

## Layout

| Piece | Where |
| --- | --- |
| Family accessors | `src/training/ops/metal_families.hpp` |
| Table | `src/training/ops/table_metal.cpp` |
| Host side of a family | `src/training/ops/<family>_metal.cpp` |
| Kernels | `src/training/kernels/metal/<family>.metal`, listed in `LFS_TRAINING_METAL_KERNELS` |
| Shared kernel code | `common.metal`, `joint_adam.metal` (Adam moment codec, mean-step and screen-share helpers) |

The kernel files are concatenated in list order into one MSL library, compiled at first launch. Shared files come first; a family file may use anything listed before it.

## Launching

Trainer code never calls Metal. `ops/metal_kernels.hpp` wraps `core::GpuKernelModule`:

- `metal::address(tensor)` gives a tensor's device address for a parameter block, or 0 for an absent tensor.
- `metal::launch(name, params, {&uses...}, groups, width)`, `launch_items` (one thread per item) and `launch_2d`.
- List every tensor the kernel reads or writes in `uses`. That orders the launch after earlier work on those tensors and makes later tensor work wait for it.

Every kernel takes `constant Params& p [[buffer(0)]]`, a struct of `device T*` pointers and scalars. The C++ struct must match the MSL layout: `uint64_t` for pointers, `metal::Float3` and `metal::Float4` for `float3` and `float4` (16 bytes each), `uint32_t` for flags. Parameter blocks are limited to 1 KiB.

Tensor library ops (`core::Tensor`) run on the same timeline and may carry glue work such as sorts, scans, compaction and reductions, where the CUDA side uses CUB.

## MSL pitfalls

- `thread`, `device`, `constant` and `threadgroup` are keywords.
- Function constants arrive as `uint`; declare them `constant uint k [[function_constant(i)]]`.
- Structs cannot have `static constexpr` data members; use a static function.
- `threadgroup` arrays are declared in kernel scope and passed to helpers.
- There is no `log1p` or `expm1`; `joint_adam.metal` has compensated versions.
- There is no float atomic max; `atomic_max_float` compares and swaps the bits.
- The library builds with fast math, as the CUDA kernels do with `-use_fast_math`. Use `precise::` where a kernel depends on exact results.
- CUDA `roundf` rounds halves away from zero, as MSL `round` does; MSL `rint` rounds them to even.
- Fast math lets the compiler fold `isfinite` and `isnan` to constants; test the exponent bits instead.
- MSL has no `double`. Where a CUDA reduction accumulates in double, sum floats in a fixed order and expect float-tolerance agreement.
