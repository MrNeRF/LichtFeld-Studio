# Export dtype_matches regression report

## Status

- Production fix reviewed and built in `/home/paja/projects/gsc-wt-export`.
- Branch/HEAD at start: `lfs-elite-wt-export` at `6814c6cc8e8591ae3effa47d942743ef487740cc`.
- Build tree: `/home/paja/.cache/lichtfeld/gsc-wt-export-build` (Release, `BUILD_TESTS=ON`, Ninja). T7 was not used for binaries because it is exFAT and cannot represent the symlinks required by ONNX/vcpkg.
- The regression test asserts q16 residency (`shN_raw()` is `Float16`), installs a CUDA Bool deleted mask, enters `Scene::mergeSplatsWithTransforms(..., Clone)`, and saves a PLY.

## Repro evidence

- Exact resident tensor: production q16 SH-rest storage is pad-dropped `Float16` (`uint16` codes) plus `shN_value_bounds()`; it is not the Float32 float4-swizzled buffer assumed by the old merge code.
- The failing accessor is `ptr<float>()`, which enforces the dtype contract at `src/core/tensor/internal/tensor_impl.hpp:1724`.
- At the frozen base, `Scene::mergeSplatsWithTransforms`'s deleted-mask clone path entered `clone_filtered_swizzled` and called `src.shN_raw().ptr<float>()` at `src/core/scene.cpp:2597`; the other frozen raw-float assumptions were at lines 698, 1324, 2717, 2734 (plus the nearby merge/compact sites).
- The added test reproduces the same tensor/mask/merge/export sequence. The pre-fix red execution was established from that frozen source path before the production fix was applied; per the continuation request, it was not reintroduced by reverting the user's fix. The current focused run is green.

## Breaking commit

- `fd1d14ae3` (`WIP(WO-G3): SH-rest 16-bit value quant wiring (default ON)`) introduced the q16 storage/default wiring.
- `f72588394` (`feat(WO-G3): finish SH q16 densify/export/gsplat wiring + storage TDD`) completed the production q16 path. The default remains enabled in `src/core/include/core/sh_value_quant.hpp`.

## Fix summary

- `Scene::mergeSplatsWithTransforms` and the related compact/training-cache merge path now materialize q16/IEEE-f16/non-Float32 SH-rest through public `shN_canonical()`, then construct canonical-layout results so the owning constructor re-swizzles Float32 storage.
- Deleted-mask filtering is performed on canonical Float32 rows with `index_select`, avoiding float gather kernels on q16 codes. The multi-identity path routes each source through the same safe clone helper.
- `SplatData::apply_deleted`, crop/extract, and random-choice paths use the same canonical filtering for non-Float32 SH-rest. Raw Float32 swizzled fast paths remain unchanged.
- Raw q16 borrow/clone fallback transfers `shN_value_bounds()`; ephemeral export/merge paths prefer materialization, so bounds are not silently lost or reinterpreted as IEEE-f16.

## Mip findings

- “Mip” here is the mip-splatting anti-aliasing filter (`mip_filter`), not SOG texture mip levels.
- The default is explicitly `false` in `OptimizationParameters` and rendering request/uniform state.
- Recent branch fix `d7fec0cdc` (“Fix mip toggle”) wires the toggle into Vulkan uniforms; current code also synchronizes training and viewer settings on trainer-ready, training-started, and training-resumed events (`src/visualizer/visualizer_impl.cpp`).
- No failing test, wrong default, or training/viewer desynchronization was found on this branch. No mip code change was made.

## Gate results

- Configure: passed with Release/BUILD_TESTS=ON in `/home/paja/.cache/lichtfeld/gsc-wt-export-build`, using the private dependency cache and `ENABLE_COMPILER_CACHE=OFF`.
- Build: passed `cmake --build /home/paja/.cache/lichtfeld/gsc-wt-export-build --target lfs_core lfs_io lichtfeld_tests -j 16` (`977/977`).
- Focused regression: `ShValueStorageTest.Q16DeletedMaskSceneMergeAndPlyExport` passed; it produced a non-empty temporary PLY and observed 61 visible rows.
- Q16 storage: all 8 `ShValueStorageTest.*` cases passed.
- Exportable storage: all 18 `ExportableStorageTest.*:SplatExportableStorageTest.*` cases passed.
- SOG: `SogFormatTest.*` passed 8 tests; 6 fixture-dependent tests skipped because the optional external SOG/PLY fixtures were absent.
- PLY: `PythonIOTest.PlySaveLoadRoundtrip`, `PlySaveUsesExternalChannelMajorShOrder`, and `PlySaveFiltersExtraAttributesWhenDeletedMaskPresent` passed.
- Full 1.5k train+export was skipped: the direct CUDA regression covers the failing merge/export transition, and the requested focused/neighboring gates already passed; no separate dataset/checkpoint was supplied for a bounded headless run.
- Memcheck was not run; the change routes existing CUDA kernels through safe canonical materialization and does not alter their implementations or add a new export CUDA copy kernel.

## Commits

- `34773826d` — `fix: make q16 SH export merges dtype-safe` (no push).

## Independent re-verify

- Re-ran `ShValueStorageTest.*` (8/8 PASSED) including `Q16DeletedMaskSceneMergeAndPlyExport` against `/home/paja/.cache/lichtfeld/gsc-wt-export-build/tests/lichtfeld_tests` with libtorch from the private deps cache.
