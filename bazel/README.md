# Building LichtFeld Studio with Bazel

The Bazel build is an alternative to CMake + vcpkg. It builds the complete
application: `LichtFeld-Studio.exe`, the LichtFeld DLLs, the `lichtfeld` Python
module, the shaders and all runtime files, assembled into the same directory
layout as `cmake --install` of a portable build.

Platform: **Windows x64 with MSVC** (Linux is not ported yet). Tests are not
ported yet either.

## Requirements

- [Bazelisk](https://github.com/bazelbuild/bazelisk); it runs the Bazel version
  of [`.bazelversion`](../.bazelversion) (9.3.2).
- Visual Studio 2026 with the C++ workload and a Windows 10/11 SDK (tested
  with MSVC 14.51). Bazel finds it automatically.
- An NVIDIA driver that supports CUDA 13.5.

Nothing else: CUDA 13.5, the Vulkan SDK (for the loader import library), Slang,
Python 3.12 and all libraries are downloaded by Bazel. No vcpkg, no CUDA
toolkit installation, no system Python.

## Usage

Run all commands in the `LichtFeld-Studio` directory.

```powershell
bazelisk run //:LichtFeld-Studio                    # build and start the application
bazelisk run //:LichtFeld-Studio -- --help          # with command line arguments
bazelisk build //:LichtFeld-Studio                  # only build
bazelisk build //...                                # build everything
```

`//:LichtFeld-Studio` is the application directory
`<bazel-bin>/LichtFeld-Studio` (`bazelisk info bazel-bin`; the build does not
create `bazel-*` symlinks). It is self-contained and can be copied:

```
bin/                     LichtFeld-Studio.exe, LichtFeld/CUDA/SDL3/RmlUi/Slang DLLs,
                         python.exe + python312.dll (embedded Python), uv.exe
Lib/, DLLs/              Python 3.12 standard library and extension modules
lib/python/              lichtfeld.pyd, lfs_plugins, lichtfeld_sessions, type stubs
share/LichtFeld-Studio/  assets, locales, HTML viewer, Vulkan shaders, licenses
extensions/              nvImageCodec extensions (nvJPEG, nvJPEG2000)
```

The Microsoft Visual C++ runtime is not bundled (CMake's install does); it is
installed with Visual Studio and with the VC++ redistributable.

### Configurations

| Option | Meaning |
| --- | --- |
| (default) | Optimized build (`--compilation_mode=opt`, CMake `Release`) |
| `--config=dbg` | Debug build (CMake `Debug`) |
| `--@rules_cuda//cuda:archs=sm_86,sm_89` | CUDA architectures; default `sm_89`. `LFS_MIN_SM` (the startup GPU check) follows the lowest one. |
| `--//bazel/shaders:debug_info` | Shaders with debug info (CMake `LFS_ENABLE_VULKAN_SHADER_DEBUG_INFO`) |
| `--//bazel/shaders:spirv_validation` | Validate all SPIR-V with `spirv-val` |

Put personal options into `user.bazelrc` (not versioned), e.g.
`build --@rules_cuda//cuda:archs=sm_86`.

## Structure

| Path | Content |
| --- | --- |
| [`MODULE.bazel`](../MODULE.bazel) | Dependencies: toolchains (rules_cc, rules_cuda, rules_vulkan, rules_python, nanobind_bazel) and the libraries, newest versions from the [Bazel Central Registry](https://registry.bazel.build) |
| [`.bazelrc`](../.bazelrc) | Global compiler flags (as in `CMakeLists.txt`), FFmpeg/curl features, module flags |
| `src/*/BUILD.bazel` | One package per CMake module; a `headers` target each plus the libraries |
| [`bazel/lfs.bzl`](lfs.bzl) | `lfs_cc_library`, `lfs_cuda_library`, `lfs_dll` (Windows DLLs like CMake `SHARED` libraries) |
| [`bazel/bundle.bzl`](bundle.bzl) | The application directory (`//:LichtFeld-Studio`) |
| [`bazel/registry`](registry) | BCR-format registry for libraries that are not (yet) in the BCR |
| [`bazel/patches`](patches) | Patches for BCR modules |
| [`bazel/compat`](compat) | Forwarding headers: LichtFeld's include paths of formerly vendored libraries → BCR modules |
| [`bazel/deps`](deps) | Header-only views, FFmpeg, Vulkan loader, `SDL3.dll` |
| [`bazel/shaders`](shaders) | Slang/GLSL shader compilation (CMake `CompileShaders.cmake`, `GenerateUserShaderIR.cmake`) |
| [`bazel/gen`](gen) | `config.h`, `git_version.h`, `lfs_core_abi_stamp.h` |
| [`bazel/windows`](windows) | Windows resources: version info of all binaries, application icon |
| [`bazel/repositories.bzl`](repositories.bzl) | `@lfs_external`: the vendored code in `external/` (nvImageCodec, spz, tinyusdz, tree-sitter, Zep, indicators) |

### Binaries (CMake target → Bazel target)

| CMake | Bazel |
| --- | --- |
| `LichtFeld-Studio` | `//src/app:LichtFeld-Studio` |
| `lfs_core`, `lfs_logger`, `lfs_event_bridge` | `//src/core:lfs_core.dll`, `:lfs_logger.dll`, `:lfs_event_bridge.dll` |
| `lfs_diagnostics` | `//src/diagnostics:lfs_diagnostics.dll` |
| `lfs_visualizer` | `//src/visualizer:lfs_visualizer.dll` |
| `lfs_mcp` | `//src/mcp:lfs_mcp.dll` |
| `lfs_python_runtime` (incl. `lfs_python_utils`) | `//src/python:lfs_python_runtime.dll` |
| `lfs_py` (`lichtfeld.pyd`) | `//src/python:lichtfeld` |
| `nvimgcodec`, `nvjpeg_ext`, `nvjpeg2k_ext` | `@lfs_external//nvImageCodec:nvimgcodec.dll`, `:extensions` |

Static libraries keep their CMake names (`//src/training:lfs_training`,
`//src/io:lfs_io`, ...). Dependents use `<name>` of a DLL; `lfs_dll` links its
import library and puts the DLL into the runfiles.

### Libraries from the local registry

`glslang`, `shaderc`, `libplacebo` (patched for MSVC), `rmlui` (with `lunasvg`,
`plutovg`), `openxr_loader`, `nativefiledialog_extended`, `libvterm` (Neovim's
fork, the revision of LichtFeld's submodule), `vulkan_memory_allocator`,
`taywee_args`. After editing a module, update the hashes and restart Bazel:

```powershell
python bazel/tools/registry_integrity.py <module>
bazelisk shutdown
```

## Differences to the CMake build

- **Dependency versions.** All libraries are the newest BCR releases rather
  than the vcpkg baseline or the vendored copies (e.g. nanobind 3.1, cpp-httplib
  0.46, OpenSSL 4.0, OpenMesh 11.0, FFmpeg 9.0). Three source changes adapt
  LichtFeld to them and still build with the old versions:
  `nb::detail::dtype_traits` (`src/python/lfs/py_tensor.cpp`), cpp-httplib's
  session verifier (`core/tls_distrust.hpp`, used by `menu_bar.cpp` and
  `llm_client.cpp`) and OpenMesh's circulator handle type
  (`src/python/lfs/openmesh/circulator.hpp`).
- **Static libraries.** vcpkg builds DLLs; here only libraries whose state must
  be shared across binaries are DLLs (SDL3, RmlUi, Python, CUDA runtime,
  nvImageCodec, LichtFeld's own DLLs). Consequence: the single `spdlog::error`
  call in `lfs_event_bridge` logs to the console instead of the log file.
- **FFmpeg** has NVENC and the `*_cuvid` decoders; the unused NVDEC hwaccels are
  not built. **curl** keeps Schannel (Windows certificate store) as default TLS
  backend, like vcpkg's curl.
- **Python stubs** are the committed ones (`src/python/stubs`); they are not
  regenerated with `nanobind.stubgen`. `THIRD-PARTY-LICENSES.md` (generated from
  vcpkg) is not part of the bundle.
- **Development fallbacks** with absolute source paths (`PROJECT_ROOT_PATH`,
  `VISUALIZER_*_PATH`, `LFS_PYTHON_EXECUTABLE`) are not compiled in; the
  application uses the bundle's resources and Python.
