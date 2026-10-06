<!-- SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
     SPDX-License-Identifier: GPL-3.0-or-later -->

# HiGS median-depth GPU regression

`vulkan_depth_contracts` dispatches the production `macro_compose.spv` and
`macro_compose_overlays.spv` with synthetic projected splats and known half-color
partials. It isolates the compose-stage bug: a true transmittance just above
0.5 rounds to exactly 0.5 in FP16, so deriving depth from the color state can
miss the next batch's median. Expected depths are explicit, independent of any
other renderer. The first regression expects 6; the pre-fix shader returns
FAR_DEPTH (1e10).

The 32 cases exercise:

- 1,025 sources across the production 1,024-source batch boundary;
- transmittance above, below and exactly at the inclusive 0.5 threshold;
- no median when total opacity stays below 50%;
- Studio and Portal alpha profiles, plain and overlay shader variants;
- a one-pixel-wide edge tile, with a contributing source loaded by a lane whose
  own output pixel is outside the image;
- FP32 state continuation across compose dispatches and partial-pool reuse;
- sparse exact depth at frustum samples, including continuation across dispatches;
- unchanged color composition when exact depth is requested, output guards,
  and resetting the same buffers for an empty frame.

The continuation fixture shortens the first window to one batch to test the
persisted state without allocating a full production-sized wave pool. The
overlay variant uses neutral overlay settings; active editor overlays and the
full projection/raster path are covered separately by viewer integration tests.
This test does not create a window, use Metal tensors, initialize Python, read
preferences, or download datasets. It needs a Vulkan device with viewer shader
features and fails rather than skipping when one is unavailable.

It is enabled by default on macOS test builds. Other GPU runners can opt in
with `-DLFS_BUILD_VULKAN_DEPTH_TESTS=ON`; Windows/Linux CPU-only CI leaves it off.

```sh
cmake --build build-macos-release --target vulkan_depth_contracts
VK_DRIVER_FILES=/path/to/MoltenVK_icd.json ctest --test-dir build-macos-release \
  -R '^VulkanDepthContracts$' --no-tests=error --output-on-failure
```

For a negative control, the executable accepts a directory containing both
shader variants compiled from the pre-fix source with the production Slang
flags. The expected result is a nonzero exit reporting an incorrect median,
not a pipeline/device initialization error. Keep those alternate binaries in
the build directory; do not replace the application's shader outputs.
