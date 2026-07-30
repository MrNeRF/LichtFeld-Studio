# Application build performance, July 2026

Marker: `FASTER-BUILD-2026-07-13`

> **Note (2026-07, PR #1395):** Dear ImGui and ImPlot were subsequently removed
> from the application graph. Timings and dependency lists in this report are
> historical snapshots from the build-performance campaign and are not updated
> for the RmlUi-only GUI stack.

This report covers the complete LichtFeld Studio application graph only. Every
timed configure used `Release` and `BUILD_TESTS=OFF`; no shipped application
feature was disabled. USD, FFmpeg, OpenImageIO, RmlUi, Python, MCP, the GUI,
both shader toolchains, and nvImageCodec were present in both the baseline and
the result. Test targets and test timings are intentionally outside this
report.

## Executive result

The comparison below uses Ninja with a hard `-j6` cap on the same machine and
compiler toolchain. "Compiler cold" means the compiler cache is disabled; it
does not mean that vcpkg binary archives or OS filesystem caches were purged.
The default developer configuration enables the compiler cache, so a second
full-featured clean tree is also reported after the cache has been populated.

| Full application build | Baseline | Achieved | Change |
| --- | ---: | ---: | ---: |
| Clean configure, cached vcpkg archives | 1:13.42 | 30.71 s | -58.2% |
| Clean compiler-cold build, `-j6` | 7:43.90 | 5:50.71 | -24.4% |
| Clean configure + compiler-cold build | 8:57.32 | 6:21.42 | -29.0% |
| Clean full-featured build, populated default cache | not enabled | 28.01 s | 16.6x vs. baseline cold build |
| Touch `src/app/application.cpp` | 9.56 s | 7.13 s | -25.4% |
| Touch `src/rendering/selection_ops.cu` | 19.12 s | 18.30 s | -4.3% |
| Touch `src/core/include/core/tensor.hpp` | 4:47.77 | 3:27.18 | -28.0% |

The first three rows are the apples-to-apples headline. The populated-cache
row is a developer-loop result, not a substitute for the compiler-cold number.
All incremental rows include relinking and post-build work required by the
normal application target.

Two successful clean builds were discarded before the final run. They took
15:47.97 and 16:40.37 while unrelated checkouts were compiling, COLMAP was
using six threads, a GPU training process was running, and the host accumulated
330,138 and 547,893 major faults. Those are machine-contention observations,
not project regressions. A later 6:36.05 run was also rejected: two final
rendering CUDA invocations transiently failed native-GPU discovery and nvcc
silently emitted `sm_52` SASS plus PTX for those objects. The accepted result
ran after the unrelated data pipeline had finished, with no other compiler or
dependency-retrieval process. It obtained 590% CPU, incurred two major faults,
reported zero process swaps, and emitted no native-discovery warning.

## Measurement setup

| Item | Value |
| --- | --- |
| Baseline revision | `f2ecbb4b7`, before this branch's build changes |
| CPU / memory | Intel Core i9-13900KF, 31 GiB RAM |
| GPU | NVIDIA RTX 4090, compute capability 8.9, 24 GiB VRAM |
| C++ compiler | GCC 14.1.0 |
| CUDA compiler | nvcc 12.8.93 |
| CMake / generator | CMake 4.0, Ninja 1.10.1 |
| Linker | GNU ld 2.38; neither mold nor lld was installed |
| Configure | `cmake -S . -B BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DENABLE_COMPILER_CACHE=OFF` |
| Build | `cmake --build BUILD -j6` |
| Incremental protocol | successful clean build, `touch` exactly one named source/header, build the normal default target |

The baseline build tree was 6.7 GiB. `ClangBuildAnalyzer` was not installed and
the available Clang 14 was too old for this C++23/CUDA build, so the primary TU
data comes from Ninja's log and commands, GCC `-ftime-report`, dependency files,
and direct command timing. A representative app TU spent 3.46 of 7.39 seconds
(47%) in parsing, 2.12 seconds in template instantiation, and 2.81 seconds in
optimization/generation.

## Baseline cost profile

### Targets

The table sums compile-edge elapsed time. Parallel edges overlap, so these
figures rank work but do not add up to wall time.

| Rank | Target or component | Aggregate compile time |
| ---: | --- | ---: |
| 1 | `lfs_visualizer` | 597.751 s |
| 2 | `lfs_core` | 236.210 s |
| 3 | Python extension bindings | 211.908 s |
| 4 | `lfs_io` | 203.271 s |
| 5 | `lfs_training` C++ | 166.600 s |
| 6 | tensor CUDA kernels | 166.425 s |
| 7 | gsplat CUDA backend | 156.411 s |
| 8 | core CUDA | 151.121 s |
| 9 | main application TUs | 143.325 s |
| 10 | training CUDA kernels | 122.677 s |
| 11 | MCP shared library | 64.314 s |
| 12 | rendering | 60.306 s |
| 13 | I/O CUDA | 54.813 s |
| 14 | Zep | 47.764 s |
| 15 | FastGS | 47.198 s |
| 16 | OpenMesh | 46.287 s |
| 17 | Python utilities | 28.490 s |
| 18 | nvImageCodec objects | 27.901 s |
| 19 | Vulkan rasterizer | 19.860 s |
| 20 | nanobind runtime | 19.084 s |

### Thirty slowest translation units

| Rank | Translation unit | Time |
| ---: | --- | ---: |
| 1 | gsplat `RasterizeToPixelsFromWorld3DGSBwd.cu` | 83.124 s |
| 2 | `src/app/mcp_gui_tools.cpp` | 52.625 s |
| 3 | `tensor_ops.cu` | 49.496 s |
| 4 | gsplat `RasterizeToPixelsFromWorld3DGSFwd.cu` | 36.650 s |
| 5 | `mcmc_kernels.cu` | 28.368 s |
| 6 | I/O `kmeans.cu` | 26.952 s |
| 7 | I/O `morton_encoding.cu` | 26.652 s |
| 8 | `ssim.cu` | 23.856 s |
| 9 | core `kdtree_kmeans.cu` | 22.425 s |
| 10 | `src/python/lfs/py_mesh.cpp` | 22.096 s |
| 11 | core `selection_ops.cu` | 21.955 s |
| 12 | `ppisp_controller.cu` | 21.885 s |
| 13 | `ppisp_controller_pool.cu` | 21.459 s |
| 14 | core `kmeans_new.cu` | 20.301 s |
| 15 | core `morton_encoding_new.cu` | 19.770 s |
| 16 | `ssim_reduction.cu` | 19.723 s |
| 17 | `cuda_vulkan_interop.cu` | 19.595 s |
| 18 | `tensor_masking_ops.cu` | 19.079 s |
| 19 | `src/app/mcp_runtime_tools.cpp` | 19.079 s |
| 20 | `trainer.cpp` | 18.921 s |
| 21 | `lanczos_resize.cu` | 18.541 s |
| 22 | `src/app/mcp_ui_registry_tools.cpp` | 18.405 s |
| 23 | rendering `selection_ops.cu` | 18.225 s |
| 24 | `tensor_strided_ops.cu` | 18.135 s |
| 25 | `undistort.cu` | 18.125 s |
| 26 | `tensor_warp_reduce.cu` | 18.053 s |
| 27 | core `tensor_debug.cu` | 18.033 s |
| 28 | `tensor_fused_pointwise.cu` | 17.781 s |
| 29 | FastGS `backward.cu` | 17.641 s |
| 30 | `src/python/lfs/py_ui.cpp` | 17.010 s |

### Header exposure

This is weighted transitive exposure: each header is charged the elapsed time
of every compile edge whose dependency file contains it. It is useful for
ranking fan-out, but it is not the header's isolated parse time.

| Rank | Header family | Consumer edges | Weighted exposure |
| ---: | --- | ---: | ---: |
| 1 | `core/export.hpp` | 376 | 2449 s |
| 2 | `core/source_site.hpp` | 328 | 2361 s |
| 3 | `core/cuda_error.hpp` | 254 | 2076 s |
| 4 | `core/cuda_safe_format.hpp` | 254 | 2076 s |
| 5 | `core/failure_report.hpp` | 253 | 2039 s |
| 6 | `core/assert.hpp` | 250 | 2031 s |
| 7 | `core/logger.hpp` | 260 | 1942 s |
| 8 | tensor functors | 224 | 1828 s |
| 9 | CUDA stream headers | 235 | 1806 s |
| 10 | tensor operation headers | 222 | 1802 s |
| 11 | `core/path_utils.hpp` | 234 | 1747 s |
| 12 | lazy tensor configuration / IR | about 218 | about 1740 s |
| 13 | `core/tensor.hpp` | 201 | 1585 s |
| 14 | tensor serialization | 201 | 1585 s |
| 15 | nlohmann JSON | 112 | 913.627 s |
| 16 | RmlUi | 66 | 494.177 s |
| 17 | ImGui | 51 | 421.104 s |
| 18 | nanobind | 55 | 275.522 s |
| 19 | TBB | 12 | 102.271 s |
| 20 | OpenMesh | 39 | 96.690 s |

OpenImageIO appeared in only five implementation TUs, but its imported static
target leaked `FMT_HEADER_ONLY` through 115 unrelated TUs. That flag leakage,
not direct OIIO headers, was its principal application compile cost.

### Links and artifacts

| Artifact / edge | Baseline |
| --- | ---: |
| Main executable link | 3.723 s |
| MCP shared-library link | 1.992 s |
| Python extension link | 1.711 s |
| Core shared-library link | 1.307 s |
| Python runtime link | 0.684 s |
| Visualizer archive | 0.332 s |
| Main executable | 177,453,992 bytes |
| Python extension | 61,769,296 bytes |
| Core shared library | 71,685,488 bytes |
| Visualizer archive | 35,744,226 bytes |
| Build tree | 6.7 GiB |

### CUDA

The baseline contained 66 application-owned CUDA objects and one nvImageCodec
vendor object. `cuobjdump` found `sm_89` in all 67; the application-owned
developer build was one native architecture, not a release fan-out.
The retained `-arch=native` policy produces one SASS architecture. Resolving
to plain CMake `89` generated both `sm_89` SASS and redundant `compute_89` PTX;
a direct compile of `selection_ops.cu` took 19.90 seconds versus 17.17 seconds
with native. A controlled rebuild of all 66 application-owned CUDA objects
also favored native: 4:58.04 versus 5:17.25 for `89-real`. Native therefore
remains the measured developer policy rather than changing architecture syntax
for its own sake.

The rejected 6:36.05 run demonstrates that nvcc's native mode can silently
fall back if GPU enumeration fails during an individual invocation. The final
run was accepted only after `cuobjdump` verified all 66 application objects.
A future low-risk hardening step is a post-build architecture audit or a
targeted fail-fast check; globally promoting every nvcc warning to an error is
too broad, and resolving to `89-real` lost the controlled A/B above.

Release `-use_fast_math` remains on the targets that already used it; removing
it would change numerical behavior. Device `-lineinfo` remains Debug-only.
`--ptxas-options=-v` was removed from ordinary builds: it produced very large
logs and assembly timing noise without affecting generated code. The backward
gsplat kernel is the dominant CUDA TU. nvcc `--threads` is explicitly ignored
for a single architecture. `--split-compile` can add optimization threads, but
Ninja 1.10 cannot lend job tokens to nvcc, so enabling it under `-j6` would
violate the six-worker memory cap. Splitting the source is a higher-risk
refactor and was not done here.

### vcpkg

With binary archives populated, the baseline restored 97 packages in 7.3
seconds. The final app manifest resolves 86 packages. ONNX Runtime and `uv`
release archives now live in a shared, locked, checksum-verified download cache,
so a fresh build tree does not repeat those network transfers.

For scale only, existing local vcpkg logs show these uncontrolled first-build
times: USD 913.1 s, FFmpeg 197.7 s, OpenImageIO 174.6 s, OpenColorIO 126.3 s,
Python 84.9 s, Assimp 80.0 s, OpenEXR 56.4 s, RmlUi 46.3 s, glslang 43.4 s,
libarchive 23.8 s, TBB 18.1 s, ZeroMQ 12.4 s, ImGui 11.2 s, WebP 6.3 s, and
Slang 5.3 s. These are historical local observations, not a controlled sum and
not part of the clean application headline.

## Dependency audit

"Compile cost" is the best measured local signal available: a direct TU,
weighted header exposure, target aggregate, package first-build observation,
or "not separable" where static linkage makes an honest isolation impossible.

| Dependency | Used by / evidence | Essential | Measured application-build signal | Verdict | Risk / expected saving |
| --- | --- | --- | ---: | --- | --- |
| RmlUi | 66 C++ consumers; retained GUI, overlays, Python panels | yes | 494.177 s weighted header exposure; 46.3 s historical package build | keep | Required; no replacement considered. |
| USD | `formats/usd.cpp`, USD loader and plugin resources | yes | 913.1 s historical first package build; prebuilt libraries at app compile time | keep | Required format support. Removing or gating would be a feature regression. |
| FFmpeg | video encoder/player/frame extractor and export UI | yes | 197.7 s historical package build; app links static codec stack | keep | Required. Static archive extraction constrains visualizer unity grouping. |
| OpenImageIO | `image_io.cpp`, preprocess, GUI image/environment paths; broad CPU image/HDR/high-bit/resample APIs | yes | five direct TUs / 40.391 s exposure; imported-target flag leaked to 115 TUs | keep, isolate | No lighter library covers the used formats, precision, metadata, and resampling APIs. Wrapper removes the transitive compile poison without losing formats. |
| Assimp | `mesh_loader.cpp`: OBJ, FBX, glTF/GLB, STL, DAE, 3DS, mesh and PLY plus materials/textures | yes | one 7.3 s TU; 80.0 s historical package build; 314 MiB installed package | keep | tinyply/OpenMesh do not cover the supported scene/material formats. A pruned-importer overlay needs fixtures before adoption. |
| ImGui | SDL input/backend, gizmos, immediate Python panels and legacy overlays | yes today | 51 consumers / 421.104 s weighted exposure; 11.2 s historical package build | keep; migrate deliberately | Load-bearing, not vestigial. Removal is a feature migration, not a dependency cleanup. |
| ImPlot | training/performance plotting | yes today | 3 consumers / 42.510 s weighted exposure | keep with ImGui path | Small direct surface; remove only with the corresponding UI migration. |
| Boost.Preprocessor | nvImageCodec `static_switch` implementation | yes in current vendor | one consumer / 11.231 s exposure | keep | Rewriting vendored dispatch has negligible app payoff. |
| Boost.Regex | no source, binding, MCP, plugin, or script use | no | pulled an otherwise unused Boost leaf set | remove | Implemented; no behavior risk. |
| libarchive | SOG ZIP and NuRec USDZ ZIP only | yes for compatibility | two TUs / 18.578 s exposure; 23.8 s historical package build | keep ZIP codecs only | Deflate plus BZip2, LZMA, and Zstandard ZIP entry methods remain; unrelated outer-filter, crypto, and XAR defaults are removed. |
| libdeflate | RAD raw deflate plus zlib/gzip-compatible paths | yes | implementation-local; not separable | keep | Best existing compatible deflate performance. |
| zlib / gzip | NuRec gzip, SPZ gzip, RAD fallback, dependency closure | yes | transitive/prebuilt | keep | File-format compatibility. zlib-ng may improve runtime but needs ABI and corpus validation. |
| zstd | transitive today; benchmarked as future format codec | not for current formats | 697.9 MB/s compress and 1868.3 MB/s decompress at level 1 on real splat data | propose versioned format | Clear runtime win, but silently changing RAD/SOG/SPZ bytes would break compatibility. |
| LZ4 | no direct application call site | no | 1656.8 MB/s compression but 0.979 ratio on corpus | remove transitive leaf | Removed with libarchive feature pruning; poor ratio for this data. |
| cppzmq / ZeroMQ | TCP REP/PUB server, HWM and timeout semantics in three TUs | yes today | four header consumers / 29.207 s exposure; 12.4 s historical package build | keep | `httplib` is HTTP, not PUB/SUB. Plain sockets would recreate protocol and queue semantics for little build benefit. |
| httplib | MCP HTTP server, LLM client and image attachments | yes | vendored header in two main TUs | keep | Distinct from ZeroMQ; already header-only and scoped. |
| libwebp | SOG image payload encode/decode | yes | one consumer / 9.403 s exposure; 6.3 s historical package build | keep | Required existing SOG bytes and quality behavior. |
| stb | mesh textures, fallback image paths, frame writes, MCP captures | yes | implementation-local | keep scoped | Overlaps OIIO by format name, not role: tiny embedded utility paths avoid pulling OIIO into headers. |
| nvImageCodec | GPU JPEG/JPEG2000 decode and extensions | yes | 27.901 s object aggregate | keep | Unique GPU path. Recommend an upstream SASS-only native-build knob; do not fork more vendor logic for a sub-second wall saving. |
| Freetype | RmlUi and ImGui font rendering | yes | one direct consumer / 15.358 s exposure plus prebuilt dependency | keep | Shared required font engine. |
| libvterm | embedded terminal widget | yes | small vendored C target; not in top 20 | keep | Replacing loses terminal semantics for negligible build gain. |
| indicators | converter, preprocess, training progress UI | yes | large single header previously leaked through `progress.hpp` | keep behind pimpl | Implemented pimpl makes it private to implementation TUs. |
| nanoflann | nearest-neighbour operations in `splat_data.cpp` | yes | one implementation TU | keep | Header-only and narrowly scoped. |
| tinyply | native PLY/transforms parsing | yes | two format implementations plus one implementation TU | keep | Assimp is not a replacement for the optimized native PLY path. |
| SPZ | existing compressed splat format | yes | three small vendored sources | keep | Format compatibility. |
| OpenMesh | mesh representation, algorithms, Python mesh API | yes | 39 header consumers / 96.690 s exposure; 46.287 s target aggregate | keep, contain | Load-bearing algorithms and bindings. Continue moving it out of public headers where APIs allow. |
| TBB | RAD/PLY/Colmap/core parallel algorithms | yes | 12 consumers / 102.271 s exposure; 18.1 s historical package build | keep | Replacing working parallel algorithms is high risk and unlikely to reduce full build materially. |
| nativefiledialog-extended | platform dialogs; custom GTK path on Linux | yes | one local implementation path; platform package on Windows/macOS | keep | Platform behavior is the value; already scoped. |
| args | command-line parser in `argument_parser.cpp` | yes | one implementation TU | keep | Small header-only use; replacement gives little full-build return. |
| Python / nanobind | embedded runtime, plugins, full Python API | yes | bindings 211.908 s + nanobind runtime 19.084 s | keep, contain | Public leakage is constrained to Python targets; tested unity batches did not improve the full cold build. |
| nlohmann JSON | project formats, settings, MCP, TCP, Python-facing metadata | yes | 112 consumers / 913.627 s weighted exposure | keep, prefer `json_fwd` in APIs | Only native JSON stack. Highest remaining header-hygiene opportunity. |
| glslang | radix/raster shaders and runtime mesh-to-splat/viewport compilation | yes | 43.4 s historical package build | keep for now | Not duplicate dead weight: call sites are active. Migrate all affected shaders before consolidation. |
| Slang | primary Vulkan shader set and compiler | yes | 5.3 s historical package build | keep | Active and complementary to remaining glslang paths. |
| curl | preprocess model download | yes | one implementation TU; prebuilt library | keep | TLS, redirects and robust transfer behavior; no compile poison. |
| ONNX Runtime | preprocess depth/normal inference, including CUDA-provider fallback | yes | one implementation TU; prebuilt SDK extraction dominated part of clean-tree configure | keep, cache archive | Unique inference runtime. The verified shared archive removes repeated downloads without changing inference behavior. |
| OpenSSL | preprocess model SHA-256 and HTTPS in MCP/LLM clients | yes | implementation-local in preprocess and MCP targets; prebuilt libraries | keep | Replacing it would duplicate crypto/TLS machinery and would not reduce broad header exposure. |
| Zep | embedded Python editor | yes | 47.764 s aggregate compile time | keep | A real GUI feature, not an incidental text widget; isolate further only behind the existing visualizer boundary. |
| tree-sitter + Python grammar | dirty-buffer Python parsing for the embedded editor | yes | 15 small vendored C compile edges; below the top-20 targets | keep | Narrow, compiled C surface and unique syntax behavior. |
| lunasvg | RmlUi SVG document plugin | yes | prebuilt RmlUi feature closure; no broad direct includes | keep | Required by the retained SVG-capable RmlUi GUI. |
| OpenMP | CPU tensor, splat and CUDA-host parallel loops | yes | private compile/link flags on core targets; no public header leakage | keep private | Removing it is a runtime regression and gives no meaningful dependency-build saving. |
| DLPack | zero-copy Python tensor exchange; also private to nvImageCodec vendor code | yes | one Python binding implementation plus a header-only vendor interface | keep scoped | Standard interoperability surface; negligible independent build cost. |
| OIIO codec/color closure | OpenEXR, TIFF, PNG, JPEG and OpenColorIO reached through OIIO dispatch | yes | prebuilt transitive packages; OpenEXR 56.4 s and OpenColorIO 126.3 s historical first builds | keep with OIIO | Not duplicate application decoders that can be dropped independently; removing leaves holes in the proven OIIO format/precision set. |
| CUDA Toolkit | tensor/training/rendering kernels and CUDA/Vulkan interop | yes | 66 application CUDA objects plus one vendor CUDA object; dominant clean-build work | keep one native arch | Core compute platform. Native already emits one `sm_89` image on this developer GPU. |
| SDL3 | window/input/platform backend | yes | broad runtime link; prebuilt headers | keep | Core platform layer. |
| Vulkan / volk / VMA | renderer, loader and allocation | yes | Vulkan rasterizer 19.860 s plus visualizer consumers | keep | Core renderer. |
| glm | math types throughout public APIs | yes | broad header exposure, not separable | keep | Replacing is a product-wide rewrite. |
| spdlog / fmt | logging and formatting | yes | broad but compiled spdlog target | keep; stop accidental header-only mode | OIIO wrapper removes the measured `FMT_HEADER_ONLY` leak. |

### OpenImageIO decision evidence

Only five implementation TUs include OIIO. `core/image_io.cpp` uses
`ImageInput`, `ImageOutput`, memory-backed JPEG input, typed UINT8/UINT16/FLOAT
reads, channel conversion, TIFF metadata, and `ImageBufAlgo::resample` for the
central image API. `preprocess.cpp` accepts JPEG, PNG, BMP, TIFF, and WebP at
8-, 16-, and float precision, resamples float inference inputs, and writes
typed PNG output. `raster_rendering_engine.cpp` and
`vulkan_environment_pass.cpp` load float/HDR pixels, while `gui_manager.cpp`
loads arbitrary plugin thumbnails through OIIO's format dispatch. stb and WebP
cover narrower embedded paths, but neither replaces this format, precision,
metadata, memory-I/O, and high-quality resampling set. OIIO therefore stays;
only its imported-target compile-definition leakage is removed.

### Compression benchmark

The corpus was the first 256 MiB of the real 710 MiB `splat_64400.ply`; every
result was decompressed and byte-validated. Throughput is MiB/s.

| Codec / level | Ratio | Output MiB | Compress | Decompress |
| --- | ---: | ---: | ---: | ---: |
| libdeflate 6 | 0.903 | 231.112 | 128.570 | 711.586 |
| zlib 6 | 0.903 | 231.254 | 45.457 | 282.405 |
| zstd 1 | 0.899 | 230.018 | 697.891 | 1868.266 |
| zstd 3 | 0.900 | 230.298 | 307.201 | 1668.841 |
| LZ4 default | 0.979 | 250.597 | 1656.774 | 9555.288 |

Recommendation: use zstd for a new explicitly versioned splat/checkpoint stream
where readers can negotiate the codec. It is materially faster than deflate at
slightly better ratio. Keep libdeflate for current RAD bytes, libarchive ZIP for
SOG/USDZ containers, gzip for SPZ/NuRec compatibility, and WebP for SOG image
payloads. LZ4 is attractive for transient caches but gives away too much ratio
on this corpus.

## Implemented changes

1. **Compiler caching by default.** CMake auto-detects `sccache`, then `ccache`,
   for C/C++/CUDA on non-Windows builds and has an explicit opt-out for cold
   measurements. Windows keeps CUDA launcher caching disabled. The cache
   changes no feature set. In an isolated cache, the same app C++ command went
   from a 6.53 s miss to a 0.02 s hit; the same CUDA command went from an
   18.27 s miss to a 1.53 s hit. sccache reported the expected C++, CUDA,
   device-code, cubin and PTX hits with no cache errors. The exact final source
   tree rebuilt the complete application from clean outputs in 28.01 s with
   775/775 cacheable artifacts hit and zero misses.
2. **Shared verified downloads.** Fresh build trees reuse ONNX Runtime and uv
   release archives from a platform cache. SHA-256 verification, cross-process
   locks, partial files, and atomic rename preserve supply-chain and concurrent
   configure safety. This changes download work, not the feature graph.
3. **OpenImageIO target isolation.** `lfs_oiio` forwards OIIO's link closure and
   required include metadata without propagating its private header-only fmt
   mode. Exactly the five real OIIO TUs receive the required definition.
   Isolated measurements improved `mcp_gui_tools.cpp` from 52.625 to 42.026 s
   (20.1%) and `py_ui.cpp` from 17.010 to 15.521 s (8.8%).
4. **Header hygiene.** Tensor serialization bodies and filesystem machinery
   moved from the umbrella header to one compiled TU; public API and on-disk
   format are unchanged. Training progress uses a pimpl so `indicators.hpp`
   no longer reaches training consumers. Two Vulkan pass headers use the
   tensor forward-declaration surface instead of the full tensor header.
5. **Quieter CUDA compiles.** Routine `--ptxas-options=-v` was removed. Existing
   fast-math and debug-line policies were preserved.
6. **Smaller dependency graph.** Unused Boost.Regex and its leaf Boost ports
   were removed. libarchive retains all supported ZIP entry codecs while
   unrelated default features are disabled. The optional test graph owns its
   package feature, so a normal app configure resolves 86 rather than 97
   packages. No application dependency was hidden behind an off-by-default
   feature.
7. **Job control and presets.** vcpkg package builds auto-cap at six without
   imposing global `MAKEFLAGS`. `dev-release` and `measure-release` presets
   make the full-featured cached and compiler-cold loops explicit; both build
   presets hard-code six jobs.
8. **Developer documentation.** `docs/docs/development/build.md` documents the
   full build, cache, concurrency and CUDA policy. The flags guide justifies
   every remaining runtime variable and distinguishes CMake choices from
   persisted settings.

### Experiments not retained

| Experiment | Result | Decision |
| --- | --- | --- |
| Broad precompiled headers | `mcp_gui_tools.cpp` 42.026 -> 70.759 s; `py_ui.cpp` 15.521 -> 18.643 s; app incremental 9.56 -> 10.87 s; tree 6.7 -> 7.6 GiB | remove |
| Explicit unity groups | 265 fewer compiler invocations, but accepted cold-run compiler CPU was unchanged (2555.46 -> 2555.15 s) and wall time moved only 7:43.90 -> 7:38.24 amid CUDA variance | remove; no clean-build win to justify wider edit invalidation |
| Broad visualizer unity, batch 4 | anonymous-namespace collisions and GTK/X11 `None` macro leakage into RmlUi | remove |
| Visualizer terminal/video/scene unity group | made an FFmpeg archive member enter the Python shared-module link and fail non-PIC relocation | remove |
| Numeric CUDA `89` / `89-real` | plain `89` emitted redundant PTX; controlled all-CUDA `89-real` build took 5:17.25 versus native 4:58.04 | retain native |
| mold / lld | neither installed | document, do not pretend a result |
| Thin LTO | no evidence that longer compile/optimization time helps the default developer loop | do not enable |
| Replacing OpenImageIO | no candidate covered all actual formats, precision, metadata and resampling APIs | do not regress features |
| Replacing ZeroMQ with HTTP/plain sockets | protocols and queue semantics differ | do not rewrite for an unmeasured win |
| Reduced-feature developer preset | would make a different application graph and invite misleading comparisons | do not add; both presets retain the full app |

## Environment-variable audit

The prior branch had already reduced the environment surface from 47 to 25.
This pass found 15 application runtime variables, five standalone MCP-bridge
variables, and five test/benchmark data selectors. The application variables
are limited to debugger/validation switches, startup hardware budgets, source
hot reload, automation isolation paths/endpoints, and LSP discovery. None is a
static compilation choice masquerading as runtime policy, and deleting one
would remove a supported debugger, hardware, or automation workflow. Therefore
no further runtime variable was deleted merely to hit a numeric target.

Build choices now live in CMake (`ENABLE_COMPILER_CACHE`, the shared download
cache, developer instrumentation options, and the vcpkg job cap); user-facing
product behavior remains in settings/parameters. The full
per-variable justification is in `docs/docs/development/flags.md`.

## Resulting artifacts and final profile

| Artifact / property | Baseline | Achieved |
| --- | ---: | ---: |
| Main executable | 177,453,992 B | 176,977,224 B |
| Python extension | 61,769,296 B | 61,343,280 B |
| Core shared library | 71,685,488 B | 71,695,008 B |
| Visualizer archive | 35,744,226 B | 35,744,502 B |
| Build tree | 6.7 GiB | 6.5 GiB |
| Application CUDA objects with only `sm_89` SASS | 66 / 66 | 66 / 66 |
| nvImageCodec vendor CUDA object | `sm_89` SASS + PTX | `sm_89` SASS + PTX |
| TUs receiving `FMT_HEADER_ONLY` | 115 accidental + 5 intended | 5 intended |

### Final target compile profile

As in the baseline table, aggregate edge time ranks work but is not additive
wall time because six compile edges can overlap.

| Rank | Target | Compile edges | Aggregate compile time |
| ---: | --- | ---: | ---: |
| 1 | `lfs_visualizer` | 142 | 481.627 s |
| 2 | `lfs_core` | 53 | 161.350 s |
| 3 | `lfs_py` | 45 | 153.591 s |
| 4 | `lfs_io` | 28 | 139.119 s |
| 5 | `lfs_tensor_kernels` | 10 | 136.615 s |
| 6 | `lfs_core_cuda` | 10 | 125.146 s |
| 7 | `lfs_training` | 24 | 125.025 s |
| 8 | `gsplat_backend_lfs` | 14 | 121.426 s |
| 9 | `lfs_training_kernels` | 20 | 88.864 s |
| 10 | `LichtFeld-Studio` | 8 | 83.710 s |
| 11 | `lfs_rendering_tensor` | 9 | 47.445 s |
| 12 | `lfs_mcp` | 7 | 46.076 s |
| 13 | `Zep` | 31 | 42.959 s |
| 14 | `lfs_io_cuda` | 3 | 42.598 s |
| 15 | `OpenMeshCore` | 25 | 39.853 s |
| 16 | `fastlfs_backend` | 5 | 33.961 s |
| 17 | `nvimgcodec_obj` | 28 | 24.170 s |
| 18 | `lfs_python_utils` | 6 | 16.661 s |
| 19 | `lfs_vulkan_rasterizer` | 5 | 15.047 s |
| 20 | `edge_compute_backend` | 2 | 14.794 s |

### Final thirty slowest translation units

| Rank | Translation unit | Time |
| ---: | --- | ---: |
| 1 | `src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu` | 58.625 s |
| 2 | `src/core/tensor/tensor_ops.cu` | 40.925 s |
| 3 | `src/app/mcp_gui_tools.cpp` | 34.678 s |
| 4 | `src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSFwd.cu` | 30.454 s |
| 5 | `src/training/kernels/mcmc_kernels.cu` | 21.548 s |
| 6 | `src/io/cuda/morton_encoding.cu` | 21.183 s |
| 7 | `src/io/cuda/kmeans.cu` | 20.242 s |
| 8 | `src/core/cuda/kernels/kdtree_kmeans.cu` | 20.022 s |
| 9 | `src/training/kernels/ssim.cu` | 17.713 s |
| 10 | `src/core/cuda/kernels/morton_encoding_new.cu` | 17.490 s |
| 11 | `src/core/cuda/kernels/selection_ops.cu` | 17.273 s |
| 12 | `src/core/cuda/kernels/kmeans_new.cu` | 17.271 s |
| 13 | `src/core/cuda/lanczos_resize/lanczos_resize.cu` | 15.851 s |
| 14 | `src/python/lfs/py_mesh.cpp` | 15.778 s |
| 15 | `src/core/tensor/tensor_masking_ops.cu` | 15.775 s |
| 16 | `src/training/components/ppisp_controller_pool.cu` | 15.467 s |
| 17 | `src/training/components/ppisp_controller.cu` | 15.301 s |
| 18 | `src/rendering/selection_ops.cu` | 15.027 s |
| 19 | `src/core/tensor/tensor_warp_reduce.cu` | 14.620 s |
| 20 | `src/core/tensor/tensor_strided_ops.cu` | 14.423 s |
| 21 | `src/core/tensor/tensor_broadcast_ops.cu` | 14.403 s |
| 22 | `src/rendering/cuda_vulkan_interop.cu` | 14.306 s |
| 23 | `external/OpenMesh/src/OpenMesh/Core/Utils/PropertyCreator.cc` | 14.174 s |
| 24 | `src/training/kernels/ssim_reduction.cu` | 14.043 s |
| 25 | `src/core/tensor/tensor_fused_pointwise.cu` | 14.008 s |
| 26 | `src/core/cuda/undistort/undistort.cu` | 13.943 s |
| 27 | `src/training/rasterization/fastgs/rasterization/src/backward.cu` | 13.815 s |
| 28 | `src/core/cuda/tensor_debug.cu` | 13.502 s |
| 29 | `src/training/trainer.cpp` | 13.293 s |
| 30 | `src/visualizer/scene/scene_manager.cpp` | 13.245 s |

### Final header exposure

| Rank | Header | Consumer edges | Weighted exposure |
| ---: | --- | ---: | ---: |
| 1 | `core/export.hpp` | 377 | 1842.059 s |
| 2 | `core/source_site.hpp` | 329 | 1776.406 s |
| 3 | `core/cuda_error.hpp` | 254 | 1558.315 s |
| 4 | `core/cuda_safe_format.hpp` | 254 | 1558.315 s |
| 5 | `core/failure_report.hpp` | 254 | 1540.264 s |
| 6 | `core/assert.hpp` | 251 | 1534.280 s |
| 7 | `core/logger.hpp` | 260 | 1456.895 s |
| 8 | tensor functors | 224 | 1373.254 s |
| 9 | tensor operation headers | 222 | 1353.179 s |
| 10 | CUDA stream headers | 235 | 1348.587 s |
| 11 | `core/tensor_fwd.hpp` | 218 | 1305.450 s |
| 12 | lazy tensor configuration | 218 | 1301.511 s |
| 13 | tensor implementation | 217 | 1301.449 s |
| 14 | lazy executor | 217 | 1301.449 s |
| 15 | lazy IR | 217 | 1301.449 s |
| 16 | tensor expressions | 217 | 1301.449 s |
| 17 | tensor expression implementation | 217 | 1301.449 s |
| 18 | tensor serialization declarations | 201 | 1182.731 s |
| 19 | `core/tensor.hpp` | 200 | 1181.231 s |
| 20 | GLM core detail headers | 241 | 1112.584 s |

The serialization declaration header remains visible for source compatibility,
including its historical complete file-stream types, but its implementation,
filesystem conversion, validation loops, and vector machinery compile once in
`tensor_serialization.cpp`. The real umbrella-header rebuild fell by 80.59 s;
weighted exposure also fell from 1585 s to 1183 s.

### Final link profile

| Artifact / edge | Baseline | Achieved |
| --- | ---: | ---: |
| Main executable | 3.723 s | 2.092 s |
| MCP shared library | 1.992 s | 0.902 s |
| Python extension | 1.711 s | 1.241 s |
| Core shared library | 1.307 s | 0.831 s |
| Python runtime | 0.684 s | 0.168 s |
| Visualizer archive | 0.332 s | 0.171 s |
| RmlUi bridge | not isolated | 0.098 s |

## Recommended next work, ranked

1. **Migrate remaining glslang call sites to Slang, then remove glslang**
   (medium/high win for first dependency builds, medium risk). Both compilers
   are active today; consolidation must start with shader-output fixtures and
   runtime parity, not manifest deletion.
2. **Create a versioned zstd stream for new native splat/checkpoint data**
   (large runtime win, medium compatibility risk). Keep old readers/writers and
   validate representative scenes before changing defaults.
3. **Reduce nlohmann JSON fan-out with `json_fwd` and implementation-owned
   adapters** (medium compile win, low/medium source churn). It is the largest
   remaining third-party header exposure.
4. **Split or explicitly instantiate the hottest tensor/CUDA templates**
   (medium potential win, medium/high risk). Start with the measured
   `tensor_ops.cu` and backward gsplat kernels; retain only changes that improve
   both clean and header rebuilds.
5. **Prune Assimp importers in a vcpkg overlay after format fixtures exist**
   (large first-dependency and package-size win, medium risk). Preserve every
   currently advertised format/material path; do not replace it with a
   PLY/OBJ-only loader.
6. **Add an upstreamable nvImageCodec native-SASS-only option** (small build
   win, low risk once upstreamed). Its current helper deliberately adds PTX for
   forward compatibility even in a machine-native app build.
7. **Install and benchmark mold** (likely link-loop win, low risk on Linux).
   Link time is not the dominant clean cost, and no unmeasured linker default
   was committed.
8. **Continue the ImGui-to-RmlUi product migration** (eventual dependency
   simplification, high feature risk). ImGui and ImPlot remain load-bearing;
   this is not a build-only removal.

## Validation

| Gate | Result |
| --- | --- |
| Full-featured compiler-cold `Release`, `BUILD_TESTS=OFF`, `-j6` | pass; 5:50.71 build, 6:21.42 including clean configure |
| CUDA architecture inspection | pass; 66/66 application objects are `sm_89` SASS-only, with no native fallback; the one nvImageCodec vendor object retains `sm_89` SASS + PTX |
| Requested cross-round C++ filter | pass; 197/197 |
| Tensor serialization regression suite | pass; 25/25 |
| `pytest -q tests/python/test_async_plugin_loading.py` | pass; 10/10 |
| GUI launch, PLY load, and render | pass; MCP scene resource reported one visible 3,000,000-Gaussian node and `render.capture_window` returned a valid 640x360 composited PNG |
| 1,000-iteration headless bicycle smoke | pass; exit 0, resume checkpoint and `splat_1000.ply` written |

The headless command used `--quiet` and its terminal stream was captured. The
existing quiet option suppresses logger output but not the progress bar, so the
captured stream was 2075 bytes rather than empty. This build-system work does
not change that CLI behavior.
