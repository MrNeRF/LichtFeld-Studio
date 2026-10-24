# Collision timing

Build the native benchmarks and offscreen Vulkan tests:

```sh
cmake --build build-collision --target bench_collision_generation bench_collision_queries lichtfeld_collision_overlay_tests
```

Generate from a frozen native snapshot (`CollisionGaussian` records: center,
three axis vectors, opacity; 13 float32 values per record):

```sh
LFS_COLLISION_BENCHMARK_VOLUME=/tmp/volume.json \
build-collision/tests/bench_collision_generation snapshot.f32 0.05 auto
```

Optional modes: `none`, `cleanup`, `connected`, `strict`, `auto`. An optional XYZ
seed follows the mode. Parsing and volume serialization are excluded from the
reported generation interval. Keep source, settings, GPU/CPU and driver identical
when comparing revisions; Auto or filtering changes can change the result.

Measure the same saved occupancy without loading Gaussian or mesh payloads:

```sh
LFS_COLLISION_BENCHMARK_VOLUME=/tmp/volume.json \
build-collision/tests/lichtfeld_collision_overlay_tests --gtest_filter='*ProductionVolumeBenchmark'
build-collision/tests/bench_collision_queries /tmp/volume.json
```

Rendering uses identity coordinates, 1920×1080, ten warmups and thirty GPU samples.
Default camera: eye `(-5,3,-5)`, target origin, vertical FOV 60°.
`LFS_COLLISION_BENCHMARK_VIEW=walk` uses `(0,1,-3)`; `closeup` uses `(-1,1,-1)`.
`LFS_COLLISION_BENCHMARK_IMAGE=/tmp/frame.f32` saves the CHW float output.

Report `first_frame_ms` separately: it includes packing/upload, pipeline work and
readback. `gpu_median_ms`/`gpu_p90_ms` measure the overlay pass only, excluding
splat rendering and CPU readback. `median_ms` includes readback. None is complete
application FPS. Use alternating trials and idle other GPU workloads.

Queries use 1,000 deterministic near-origin capsules, three warmups and thirty
batches. These boolean intersection tests are not equivalent to SuperSplat's
penetration resolver. See the feature guide for measured results and limits.
