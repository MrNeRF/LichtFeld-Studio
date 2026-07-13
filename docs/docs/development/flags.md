---
sidebar_position: 5
---

# Developer flags and diagnostics

Project-controlled environment variables use the `LFS_` prefix. C++ code reads
them through `core/environment.hpp`; Python plugin code uses
`lfs_plugins.environment`. Boolean values accept `1/0`, `true/false`,
`yes/no`, and `on/off`, case-insensitively. Invalid values use the documented
default.

Product behavior belongs in command-line options, persisted settings, or the
training parameter system. The environment surface below is limited to
debugger/validation controls, startup-time resource and hardware policy, and
automation overrides.

## CMake developer options

All diagnostic build options are declared in `cmake/DeveloperOptions.cmake`.
Release defaults keep instrumentation out of shipped binaries.

| Option | Release default | Debug / RelWithDebInfo default | Effect |
| --- | --- | --- | --- |
| `LFS_ENABLE_VULKAN_VALIDATION` | `OFF` | `ON` | Sets the runtime default for requesting `VK_LAYER_KHRONOS_validation`; `LFS_VK_VALIDATION` can still override it. |
| `LFS_ENABLE_VULKAN_SHADER_DEBUG_INFO` | `OFF` | `ON` | Adds source-level debug information to Slang and GLSL SPIR-V. |
| `LFS_ENABLE_CUDA_DEVICE_DEBUG` | `OFF` | `ON` in Debug, `OFF` in RelWithDebInfo | Adds CUDA device debug code (`-G`) to Debug CUDA compilation. |
| `LFS_ENABLE_CUDA_FAILURE_INJECTION` | `BUILD_TESTS` | `BUILD_TESTS` | Compiles the gsplat allocation-failure injection API. Production builds without tests have no injection branch or state. |
| `LFS_ENABLE_ALLOCATION_PROFILING` | `OFF` | `OFF` | Captures tensor allocation stacks; enable only for focused allocator investigations. |

Multi-config generators default the configuration-dependent options to `OFF`;
enable the required option explicitly when configuring them. Source-tree Python
and RmlUI imports are controlled by `LFS_DEV_IMPORT_SOURCE_PYTHON` and
`LFS_DEV_IMPORT_SOURCE_RESOURCES`; both default to `ON` in non-portable builds.

## Application runtime variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `LFS_LOG_LEVEL` | `info` | Startup log level. `--verbose`, `--quiet`, and `--log-level` override it. |
| `LFS_CUDA_SYNC_DEBUG` | `OFF` | Synchronizes before and after central CUDA checks to attribute asynchronous failures. This is intentionally slow. |
| `LFS_NO_CRASH_HANDLER` | `OFF` | Leaves fatal signals and unhandled exceptions to an attached debugger or sanitizer. |
| `LFS_VK_VALIDATION` | CMake default | Requests Vulkan validation at startup. Set `0` to override a validation-enabled developer build. |
| `LFS_VK_VALIDATION_FATAL` | `OFF` | Aborts through the fatal path on the first error-severity validation callback. It does not enable validation. |
| `LFS_VRAM_RESERVE_MB` | `512` | Memory-pressure headroom, clamped to 128 MiB through one quarter of device VRAM. |
| `LFS_PINNED_CACHE_LIMIT_MB` | `1024` | Pinned-host cache byte budget. |
| `LFS_NVCODEC_DIAGNOSTICS` | `OFF` | Logs nvImageCodec and dependent-library discovery details. |
| `LFS_RML_DEBUGGER` | `OFF` | Starts the RmlUi debugger overlay. |
| `LFS_DEV_HOT_RELOAD` | `ON` | Enables Python, RmlUI, and locale watching when source imports were compiled in. |
| `LFS_PLUGIN_AUTOLOAD` | `ON` | Automation override for scheduling startup plugin loading. Per-plugin `load_on_startup` remains persisted in plugin settings. |
| `LFS_ASSET_MANAGER_DIR` | Platform data directory | Overrides Asset Manager storage for isolated development and tests. |
| `LFS_PLUGIN_REGISTRY_URL` | Built-in registries | Overrides the plugin registry endpoint for development and tests. |
| `LFS_PYTHON_LSP` | Auto-discovered | Overrides the Python language-server executable. |
| `LFS_PYTHON_LSP_WORKSPACE` | `~/.lichtfeld` | Overrides the language-server workspace directory. |

The VRAM and pinned-cache values are startup-time hardware policy, so they must
remain runtime-selectable across GPUs. Endpoint and path overrides support
isolated development and automation. Render, training, overlay, LOD, and plugin
enablement choices remain in their existing settings or parameter owners.

## MCP bridge variables

The standalone `scripts/lichtfeld_mcp_bridge.py` process uses the same prefix:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LFS_EXECUTABLE` | Repo build directories | Explicit app executable. |
| `LFS_MCP_ENDPOINT` | `http://127.0.0.1:45677/mcp` | HTTP endpoint proxied over stdio. |
| `LFS_MCP_START_TIMEOUT_S` | `90` | App startup timeout in seconds. |
| `LFS_MCP_BRIDGE_LOG` | `~/.codex/log/lichtfeld-mcp-bridge.log` | Bridge and launched-app log. |
| `LFS_MCP_BRIDGE_LOCK` | `~/.codex/log/lichtfeld-mcp-bridge.lock` | Cross-process startup lock. |

## Test and benchmark variables

These variables only select optional test data or benchmark modes; application
code does not read them:

- `LFS_LOD_BUILDER_BENCH`
- `LFS_RAD_AUDIT`
- `LFS_RAD_BENCH_FILE`
- `LFS_RAD_VALIDATE_FILE`
- `LFS_TEST_DATA_PATH`

## Source-built Vulkan validation workflow

Configure shader debug information when source locations are needed, then build
normally:

```sh
cmake -S . -B build \
  -DLFS_ENABLE_VULKAN_VALIDATION=ON \
  -DLFS_ENABLE_VULKAN_SHADER_DEBUG_INFO=ON
cmake --build build -j6
```

The launcher requires a source build of Vulkan-ValidationLayers. By default it
uses the sibling checkout at `../Vulkan-ValidationLayers/build/layers` and
enables GPU-assisted plus synchronization validation:

```sh
scripts/run_vulkan_validation.sh -- --view scene.ply
```

Use `--layer-path PATH` for another source build, `--binary PATH` for another
app build, and `--fatal` to abort on the first validation error. Running
`cmake --build build --target validate-gui -j6` builds and launches the default
GUI flow through the same script.

The packaged `1.4.313.0~rc2` validation layer is not accepted as proof for this
codebase: its GPU-assisted push-descriptor behavior is stale and it has missed
violations detected by current upstream builds. Do not suppress either class of
message.
