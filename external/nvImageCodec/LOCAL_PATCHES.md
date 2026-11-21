# Local patches vs upstream nvImageCodec

This tree is vendored under `external/nvImageCodec`. Record LichtFeld-only deltas here so upstream syncs do not
silently drop them.

- **Upstream:** NVIDIA nvImageCodec [v0.10.0](https://github.com/NVIDIA/nvImageCodec/releases/tag/v0.10.0)
  (tag commit `77a7d15b`).
- **Kept:** the `nvjpeg` and `nvjpeg2k` extensions, `extensions/utils`, the core library (`src/`, `include/`) and the
  CMake files they need. Not kept: tests, Python, Docker, docs, examples and the other extensions.
- **Added by LichtFeld:** `external/NVTX` (NVTX v3 headers), the pre-generated dynlink stubs
  `src/dynlink/dynlink_{cuda,nvjpeg,nvjpeg2k}_gen.cpp`, and this file. (`include/nvimgcodec_version.h` is written
  into the source tree by the configure step and ignored by `.gitignore`; it is not committed.)
- **Formatting:** every C/C++ file is formatted with the repository's clang-format and the `.clang-format` in this
  directory. Upstream uses a different style, so compare against upstream after formatting it the same way, or
  token by token.

## How to update

1. Check out the old and new NVIDIA tags. Format both with this directory's `.clang-format`.
2. Three-way merge each kept file: base = old tag (formatted), ours = this tree, theirs = new tag (formatted).
   Bring in new upstream files the kept files include or the kept CMake files list.
3. Re-check every entry below against the new code. Reconfigure, so `include/nvimgcodec_version.h` is rewritten
   from `include/nvimgcodec_version.h.in`.
4. Dynlink stubs: list the CUDA driver, nvJPEG and nvJPEG2000 functions the new code calls and make sure each is in
   the matching `dynlink_*_gen.cpp`. Extend the stubs by hand in the same pattern, or regenerate them with
   `tools/stub_generator`.
5. Build and run the codec tests (`NvCodecImageLoader*`, `NvCodecSentinelValidator*`, `PipelinedImageLoader*`, the
   nightly JPEG 2000 groups, `test_nvcodec_jpeg2k_chw_layout`, `test_image_codec_exports`).

## Fixes NVIDIA still needs (still present in 0.10.0)

### Hardware JPEG batch status

- **File:** `extensions/nvjpeg/hw_decoder.cpp` (`chooseDecodePath`, legacy batched decode)
- **PR:** #1924
- **Why:** The legacy path makes one `nvjpegDecodeBatched` call for the whole batch and then marks every image
  successful, so a member nvJPEG did not write is reported as decoded (corrupted images on GPUs with hardware JPEG
  engines). LichtFeld routes every multi-image batch to the BatchSingle path, which checks each member, keeps the
  legacy path only for a single image (single-engine GPUs, `force_legacy_batch`), and fails closed if the legacy path
  ever receives more than one image. The success status is set only after post-processing succeeds.
- **Re-check:** `chooseDecodePath` and the status handling after `nvjpegDecodeBatched`. Drop this patch once NVIDIA
  reports per-image results there.

### nvJPEG with partial allocators

- **File:** `extensions/nvjpeg/hw_decoder.cpp` (`SharedContext` constructor)
- **PR:** #3575
- **Why:** `nvjpegCreateExV2` requires both allocators; upstream calls it when only one is set. LichtFeld passes the
  pinned allocator without a device allocator when it has no decode pool. Call V2 only with both allocators,
  otherwise `nvjpegCreateEx` with the defaults.
- **Re-check:** every `nvjpegCreateExV2` call.

### Non-ASCII paths on Windows

- **Files:** `src/std_file_io_stream.cpp`
- **PR:** #650
- **Why:** Upstream opens UTF-8 paths with `std::fopen`, `stat` and `std::ifstream`, which use the ANSI code page on
  Windows. LichtFeld converts to UTF-16 and uses `_wfopen`, `_wstat64` and a wide `std::ifstream`.
- **Re-check:** any new file opening in `src/` (e.g. `mmaped_file_io_stream.cpp`).

## LichtFeld changes

### nvJPEG encoder params reused across encodes

- **File:** `extensions/nvjpeg/cuda_encoder.cpp`
- **PR:** #2394
- **Why:** Destroying encoder params calls `cudaFree`, which waits for the whole device. The per-thread resources keep
  one `nvjpegEncoderParams_t` per stream and recreate it only when a sample stops setting an override (encoding,
  sampling factors) that the previous one set. Destroyed in `~PerThreadResources` after the 0.10.0 process-exit guard.
- **Re-check:** the encode function's params handling and `PerThreadResources`.

### Decoder speed and VRAM

- **Files:** `extensions/nvjpeg/cuda_decoder.cpp`, `extensions/nvjpeg2k/cuda_decoder.cpp`
- **PRs:** #1588, #2460
- **Why:** nvJPEG device buffers grow to the largest image and stay that size; a small shared set (2) leased per
  GPU stage replaces one buffer per CPU thread. Also fixes two upstream slips: the pinned padding is read when a
  pinned allocator is set (upstream tests the device allocator), and a missing pinned allocator resets the pinned
  padding (upstream resets the device padding).
- **Re-check:** `Decoder` construction/destruction and the GPU stage of `decode()`; the padding lines in both decoders.

### Portable builds and plugin loading

- **Files:** `src/plugin_framework.cpp`, `src/CMakeLists.txt` (Windows output directories),
  `extensions/nvjpeg/CMakeLists.txt`, `extensions/nvjpeg2k/CMakeLists.txt`
- **PR:** #563
- **Why:** Extensions are found in `<libdir>/extensions` first, then the installed layout. Windows output directories
  are flattened for multi-config generators; `--exclude-libs` only on Unix; `$ORIGIN` RPATH for the nvJPEG2000
  extension.
- **Re-check:** `BuildDefaultExtensionsPathFromSoDir` and the extension CMake files.

### No Boost

- **File:** `src/imgproc/static_switch.h`
- **PR:** #1953
- **Why:** The image-processing conversion kernel uses only a bounded variadic for-each and parenthesis removal from
  Boost.Preprocessor. Those are replaced by standard C++ variadic macros supporting up to 16 list entries. The
  `TYPE_SWITCH`, `VALUE_SWITCH` and `BOOL_SWITCH` interfaces and documented argument shapes are unchanged.
- **Re-check:** if upstream changes its switch macro interfaces.

### Build integration

- **Files:** `CMakeLists.txt`, `cmake/Dependencies.cmake`, `cmake/CUDA_utils.cmake`, `src/dynlink/CMakeLists.txt`,
  `extensions/CMakeLists.txt`, `src/CMakeLists.txt`, `.clang-format`, `.gitignore`, `Acknowledgements.txt`
- **PRs:** #550, #555, #1393, #1591, #2274
- **Why:**
  - Tests, samples, Python and the extensions LichtFeld does not use default to OFF; CPack removed.
  - `Dependencies.cmake`: nvJPEG2000 headers and runtime come from NVIDIA's official redistributable
    **0.9.0.43** for Windows and Linux (#1393; upstream 0.10.0 fetches Linux 0.12.0.58 only); `NVJPEG2K_REDIST_ROOT`
    for copying the runtime next to the executable; `FetchContent_MakeAvailable` and
    `DOWNLOAD_EXTRACT_TIMESTAMP FALSE` (#2274); TIFF, zlib, zstd, libjpeg-turbo, OpenCV and Boost lookups removed.
  - `CUDA_utils.cmake`: compiler flag checks use `try_compile` with the project's CUDA compiler, so `cl.exe` need
    not be on `PATH` on Windows (#555).
  - `src/dynlink/CMakeLists.txt`: use the pre-generated stubs (no Python/clang at build time).
  - `src/CMakeLists.txt`: templates referenced relative to this directory (`CMAKE_CURRENT_SOURCE_DIR/..`, not
    `CMAKE_SOURCE_DIR`, which is LichtFeld's root), including 0.10.0's new `DispatcherConfig.cmake.in`; licence files
    installed under `share/nvImageCodec`; `nvcv_adapter.hpp` not installed (not vendored).
  - `.gitignore` and `Acknowledgements.txt`: entries for removed parts and trailing whitespace trimmed.
- **Re-check:** upstream's top-level options, the nvJPEG2000 redistributable version and whether the code needs a newer
  one (0.10.0 needs nothing beyond 0.9.0.43's API), new `${CMAKE_SOURCE_DIR}` uses, the install rules.

### Parsers trimmed to the formats used

- **Files:** `src/CMakeLists.txt`, `src/parsers/parsers_ext_module.cpp`
- **PRs:** #550, #1591
- **Why:** Only the JPEG and JPEG 2000 parsers are built and registered (also their file-extension registration, new
  in 0.10.0). The BMP, PNG, PNM, TIFF and WebP parser sources are not vendored.
- **Re-check:** the parser list and the extension module's register/unregister pairs.

### Small header fixes

- **Files:** `src/iimage.h` (`#include <memory>`), `src/library_loader.h` (`#include <stdexcept>`),
  `src/imgproc/geom/vec.h` (the compound-assignment operator macro split over two lines)
- **PR:** #550
- **Why:** The headers use `std::shared_ptr` / `std::runtime_error` without including them (a build fix from the
  first vendoring). The reason for the `vec.h` line split is not recorded in #550; it is carried as is.
- **Re-check:** drop each one once upstream includes what it uses.

### Formatting only

- `src/imgproc/device_guard.h` differs from upstream only by clang-format 18 vs 19 output
  (`(intptr_t) - 1` vs `(intptr_t)-1`).
