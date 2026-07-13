# Assertion policy

LichtFeld uses two assertion levels:

- `LFS_ASSERT` and `LFS_ASSERT_MSG` are always-on public-boundary contracts.
  They throw `std::runtime_error` in Debug and Release, before invalid dtype,
  shape, device, state, or indices reach an implementation kernel.
- `LFS_DEBUG_ASSERT` is for invariants inside already-validated hot loops and
  CUDA kernels. It is an ordinary `assert` in debug builds and compiles to no
  code when `NDEBUG` is defined.

Do not use `LFS_DEBUG_ASSERT` as the only check for caller-controlled data.
Validate once with an always-on assertion at the call boundary, then use debug
assertions for redundant per-element bounds checks inside the implementation.
