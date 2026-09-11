# Pascal source-build compatibility: tested GTX 1060 case

This patch enables a locally tested Windows/GTX 1060 source-build workflow on top of master `da2fb643b6f049b10c4a68eeaf28ae96296b45a1`. It does not change the project's official supported-hardware policy or establish support for every pre-Volta GPU.

## Problems and behavior

SM61 cannot compile/run the WMMA convolution and SH-assignment paths. Guard those device bodies and provide equivalent scalar/SIMT work. Select SIMT below SM70 before dispatch benchmarking so empty WMMA kernels cannot win the benchmark and leave output unwritten. Newer-device implementations remain available.

On the tested GTX 1060, Gaussian display/live training crashed in `nvoglv64.dll` (`0xc0000005`, module32.0.15.7196, offset0xf276d4). Headless training passed. Disabling Vulkan conditional rendering allowed the same GUI workload to complete; re-enabling it with q16 still disabled crashed a separate513-Gaussian fixture. Disable that extension/feature consistently by default on the selected pre-Volta CUDA device, using the existing fallback. `LFS_VK_DISABLE_CONDITIONAL_RENDERING` remains an explicit diagnostic override.

The same device also lacked the Vulkan16-bit capabilities required for q16 SH. When either `shaderFloat16` or `storageBuffer16BitAccess` is unavailable and no explicit `LFS_SH_VALUE_QUANT` is set, select FP32 SH before viewer storage creation. Decode q16 through canonical coefficients, convert ordinary IEEE half numerically, and retain correct exportable ownership/capacity/swizzled layout through saved-project loading and storage rebinding. Headless defaults remain unchanged. The automatic flag is process-global; a future in-process GPU-switching design should revisit that scope.

## Executed validation

Environment: Windows, GTX1060 6GB/SM6.1, driver571.96, CUDA12.8.61, VS2022 17.14.37/MSVC14.44.35207, clang-format19.1.5, native Release with `CMAKE_CUDA_ARCHITECTURES=61`, Python3.12.13. All results below precede a final whitespace-only clang-format pass.

| Check | Observed result |
|---|---|
| Full native Release build and installed runtime | Passed; Python/stdlib imports and CUDA tensor arithmetic passed |
| LPIPS RGB convolution | Six SM61 GPU cases,41344 outputs, exact agreement with independent CPU reference |
| SH assignment | Eight SM61 GPU cases covering padding, seeds, candidate counts and ties matched independent FP32/FMA reference |
| Full LPIPS model | Identical/distinct image cases passed with default and forced-WMMA settings; SM61 correctly selected SIMT |
| Retained SM75 kernel paths | Compiled; no SM75 GPU execution |
| Baseline viewer plus CUDA synchronization | Still crashed |
| Conditional rendering disabled only | Ten GUI training iterations and visible rendering passed |
| Conditional rendering disabled plus FP32 SH | Saved SH2 PLY rendering and600-step GUI refinement passed |
| Conditional rendering re-enabled, FP32 SH retained | Synthetic513-splat SH2 scene reproduced driver access violation |
| Automatic settings, installed build |713-image COLMAP dataset;600 MRNF iterations;2M splat cap; SH2; mask-ignore; refinement296267 to317006 splats; final loss0.132077; saved and reopened |
| q16 saved-project conversion |513 splats and12312 nonzero SH coefficients across quantization blocks; maximum absolute error0.0; visible rendering |
| Review and source hygiene | Independent read-only review had no blocking findings; clang-format and git diff --check passed |

For the short GUI acceptance run, dataset preparation reset the CLI SH schedule. Set the test schedule after loading through the existing Python API:

```python
p = lf.optimization_params()
p.iterations = 600
p.steps_scaler = 1.0
p.max_cap = 2000000
p.sh_degree = 2
p.set('sh_degree_interval', 10)
assert p.get('sh_degree_interval') == 10
```

Verify active SH degree2 in the trained model, not just configured maximum degree. Use an isolated profile and explicit new project save path: importing a dataset may retain the currently open project's save destination. Treat Python tracebacks as failures even if an editor tool's outer response reports success.

## Limits

The full approximately71300-step workload, upstream full test suite, other GPU models and all model formats were not tested. Vulkan validation layers were unavailable. The non-conditional fallback uses16 depth waves rather than the conditional path's64; FP32 SH requires more VRAM than q16. The tests establish a useful compatibility patch for the reproduced machine/workload, not a general driver diagnosis or complete legacy-GPU support claim.

Related reproduction: [upstream issue1681](https://github.com/MrNeRF/LichtFeld-Studio/issues/1681).
