# Collision timing

These opt-in benchmarks read a saved `.licht` collision volume without loading its
triangle or Gaussian payloads. They do not train, regenerate or write the project.

Build the focused tests and native query benchmark:

```sh
cmake --build build-collision --target lichtfeld_collision_overlay_tests bench_collision_queries
```

Export identical occupancy and camera inputs while measuring native rendering:

```sh
LFS_COLLISION_BENCHMARK_PROJECT=/absolute/path/scene.licht \
LFS_COLLISION_BENCHMARK_EXPORT=/tmp/collision-comparison \
build-collision/tests/lichtfeld_collision_overlay_tests --gtest_filter='*ProductionVolumeBenchmark'
```

This writes `.json`, `.gpu.bin` and `.tree.bin` beside the export prefix. The first
collider in the project is used, including its scene transform. Default camera:
eye `(-5,3,-5)`, target origin, vertical FOV 60°. Optional
`LFS_COLLISION_BENCHMARK_VIEW=walk` uses `(0,1,-3)`; `closeup` uses `(-1,1,-1)`.
The benchmark uses 1920×1080, ten warmups and thirty samples. `gpu_median_ms` and
`gpu_p90_ms` exclude upload, compilation and CPU image readback. `median_ms`
includes readback and conversion and must not be compared to upstream GPU time.
`first_frame_ms` includes the first packing/upload, pipeline work and readback;
report it separately from steady-state GPU rendering.
Keep other GPU workloads idle and compare the same GPU and driver.

To measure generation and rendering without repeatedly loading a large Gaussian
model, export an immutable volume from the native snapshot benchmark:

```sh
LFS_COLLISION_BENCHMARK_VOLUME=/tmp/collision-volume.json \
build-collision/tests/bench_collision_generation snapshot.f32 0.05 auto
LFS_COLLISION_BENCHMARK_VOLUME=/tmp/collision-volume.json \
build-collision/tests/lichtfeld_collision_overlay_tests --gtest_filter='*ProductionVolumeBenchmark'
```

Volume serialization is outside the reported generation interval. This JSON path
uses identity coordinates: the snapshot must already be in the intended viewing
frame. The project path above retains the collider's saved world transform.

Run SuperSplat's pinned, unmodified WGSL against those exact inputs:

```sh
npm ci --prefix tests/benchmarks/collision
node tests/benchmarks/collision/supersplat-overlay.mjs /tmp/collision-comparison.json 30
```

The script fetches `voxel-debug-overlay.ts` from viewer revision
`34c137ab1c010b99a1c199b68c36210990331262` and checks its SHA-256. For offline use,
set `SUPERSPLAT_OVERLAY_SOURCE` to that exact downloaded file. It converts the
native tree losslessly into upstream's compact child-mask format; empty edge
padding completes 4³ bricks. Node Dawn 0.6.2 uses Vulkan and rejects software
fallback/timestamp limitations explicitly. The results report adapter identity,
compute time, composite time and combined GPU time. Both passes use RGBA32F color
and depth32F attachments. Upstream retains its RGBA8 intermediate, original step
limits and edge-shading formula. This is a rendering-kernel comparison, not a
benchmark of the complete SuperSplat application.

Sharp native shading is the default. `LFS_COLLISION_BENCHMARK_FAST=1` explicitly
requests 2×2 hardware fragment shading; the native log reports support. This is a
quality tradeoff and must be reported separately from full-resolution results.

For coverage inspection, `LFS_COLLISION_BENCHMARK_IMAGE=/tmp/native.f32` writes a
CHW float image. Add `"image":"/tmp/upstream.rgba8"` to the exported JSON to save
upstream's interleaved RGBA8 output. Compare nonzero native **RGB**, not alpha:
the offscreen renderer derives alpha from mesh depth, which the overlay does not
write. Different edge-width formulas mean RGB equality is not expected.

Measure native capsule intersection independently:

```sh
build-collision/tests/bench_collision_queries /tmp/collision-comparison
```

This uses 1,000 deterministic near-scene capsules, three warmups and thirty timing
batches. It also exports `.capsules.json` with native boolean results for a query
comparison. These early-exit intersection queries are not equivalent to
SuperSplat's four-pass penetration resolver. Do not compare those two operations
as if they provided the same result.
