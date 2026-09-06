/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "app/converter.hpp"
#include "app/gpu_preflight.hpp"
#include "core/abi.hpp"
#include "core/argument_parser.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda_error.hpp"
#include "core/environment.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/session_breadcrumb.hpp"
#include "core/tensor_backend.hpp"
#include "core/user_paths.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "git_version.h"
#include "lfs_core_abi_stamp.h"
#include "preprocessing/preprocess.hpp"
#include "python/plugin_runner.hpp"
#include "python/runner.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <print>
#include <string>
#include <vector>

namespace {
    // Apply CUDA driver-level VRAM-reduction knobs BEFORE the primary context exists.
    // Setting these after cudaFree(nullptr) is too late — the driver has already
    // committed defaults (1 KiB/thread stack reserve × SMs × max-threads = ~192 MiB on
    // a 4090; eager module loading uploads all kernel cubins on first ctx-init).
    void applyCudaContextTuning() {
#ifdef _WIN32
        _putenv_s("CUDA_MODULE_LOADING", "LAZY");
#else
        setenv("CUDA_MODULE_LOADING", "LAZY", /*overwrite=*/0);
#endif
    }

    void publishResolvedUserPaths() {
        // Publish canonical paths for Python plugins; native code calls UserPaths directly.
        const auto paths = lfs::core::UserPaths::resolve();
        if (!paths)
            return;
        const auto publish = [](const char* const name,
                                const std::filesystem::path& path) {
            const auto value = lfs::core::path_to_utf8(path);
            (void)lfs::core::environment::set_value(name, value);
        };
        publish("LFS_RESOLVED_CONFIG_DIR", paths->configDir());
        publish("LFS_RESOLVED_DATA_DIR", paths->dataDir());
        publish("LFS_RESOLVED_CACHE_DIR", paths->cacheDir());
        publish("LFS_RESOLVED_LOG_DIR", paths->logDir());
        publish("LFS_RESOLVED_PLUGIN_DIR", paths->pluginDir());
        publish("LFS_RESOLVED_VENV_DIR", paths->venvDir());
        publish("LFS_RESOLVED_ASSET_LIBRARY_DIR", paths->assetLibraryDir());
    }

    // Every mode that touches CUDA gates here, before the primary context exists: with
    // CUDA_MODULE_LOADING=EAGER pre-set in the environment, context creation itself loads
    // modules the card cannot run, which would beat the check to the crash.
    void preflightGpuOrExit(const bool show_dialog, const bool viewer_only = false) {
        if (!lfs::app::preflightGpu(show_dialog, viewer_only)) {
            lfs::core::teardown_gpu_before_exit();
            lfs::core::flush_and_exit(1);
        }
    }

    // Probe what the CUDA driver allocates during context creation, *attributed to this
    // process* (NVML per-PID, not device-wide cudaMemGetInfo). Each phase is the delta
    // against the previous probe so the sum reconstructs the total context cost.
    void analyzeCudaContextDistribution() {
        auto& p = lfs::diagnostics::VramProfiler::instance();

        // Phase 1: primary context creation. cudaFree(nullptr) is a documented idiom that
        // forces the primary context to exist on device 0.
        cudaFree(nullptr);

        // Shrink the per-thread stack reserve from the 1 KiB default. Our kernels do not
        // recurse and have small frames; 256 B is comfortable. Default reservation is
        // per_thread_stack × num_SMs × max_threads_per_SM = ~192 MiB on a 4090. Driver
        // accepts the request post-context but applies it on the *next* launch — well
        // before any real kernel runs.
        cudaDeviceSetLimit(cudaLimitStackSize, 256);

        // Phase 2: default cudaMallocAsync pool. Query its initial backing reservation.
        std::size_t pool_reserved = 0;
        int device = 0;
        if (cudaGetDevice(&device) == cudaSuccess) {
#if CUDART_VERSION >= 12080
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                std::uint64_t reserved = 0;
                if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) ==
                    cudaSuccess) {
                    pool_reserved = static_cast<std::size_t>(reserved);
                }
            }
#endif
        }
        p.recordCudaPhaseBytes("default_pool", pool_reserved);

        // Phase 3: driver limits the user can query exactly.
        std::size_t printf_fifo = 0;
        std::size_t per_thread_stack = 0;
        std::size_t malloc_heap = 0;
        cudaDeviceGetLimit(&printf_fifo, cudaLimitPrintfFifoSize);
        cudaDeviceGetLimit(&per_thread_stack, cudaLimitStackSize);
        cudaDeviceGetLimit(&malloc_heap, cudaLimitMallocHeapSize);

        // Stack is per-thread; total reservation = stack * max_threads_per_sm * num_sms.
        cudaDeviceProp prop{};
        std::size_t stack_total = 0;
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            stack_total = per_thread_stack *
                          static_cast<std::size_t>(prop.multiProcessorCount) *
                          static_cast<std::size_t>(prop.maxThreadsPerMultiProcessor);
        }
        p.recordCudaPhaseBytes("printf_fifo", printf_fifo);
        p.recordCudaPhaseBytes("stack_reserve", stack_total);
        p.recordCudaPhaseBytes("malloc_heap", malloc_heap);

        // Device-wide baseline is cheap and does not load NVML. The per-process
        // measurements and libcurand probe are completed by the warmup worker.
        p.captureCudaDeviceBaseline();
    }

    int run_mode(lfs::core::args::ParsedArgs args) {
        return std::visit([](auto&& mode) -> int {
            using T = std::decay_t<decltype(mode)>;

            if constexpr (std::is_same_v<T, lfs::core::args::HelpMode>) {
                return 0;
            } else if constexpr (std::is_same_v<T, lfs::core::args::VersionMode>) {
                std::println("LichtFeld Studio {} ({})", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);
                return 0;
            } else if constexpr (std::is_same_v<T, lfs::core::args::WarmupMode>) {
                applyCudaContextTuning();
                preflightGpuOrExit(false);
                analyzeCudaContextDistribution();
                return 0;
            } else if constexpr (std::is_same_v<T, lfs::core::args::ConvertMode>) {
                preflightGpuOrExit(false);
                return lfs::app::run_converter(mode.params);
            } else if constexpr (std::is_same_v<T, lfs::core::args::Mesh2SplatMode>) {
                preflightGpuOrExit(false);
                return lfs::app::run_mesh2splat(mode.params);
            } else if constexpr (std::is_same_v<T, lfs::core::args::PreprocessMode>) {
                preflightGpuOrExit(false);
                return lfs::preprocessing::run_preprocess(mode.params);
            } else if constexpr (std::is_same_v<T, lfs::core::args::PluginMode>) {
                return lfs::python::run_plugin_command(mode);
            } else if constexpr (std::is_same_v<T, lfs::core::args::TrainingMode>) {
                LOG_INFO("LichtFeld Studio");
                LOG_INFO("version {} | tag {}", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);

                // Driver-level tuning must precede *any* CUDA call, including the pre-flight
                // gate and the cudaFree(nullptr) inside analyzeCudaContextDistribution.
                applyCudaContextTuning();

                const bool interactive =
                    !mode.params->optimization.headless && !mode.params->render_path;
                const bool viewer_only = lfs::app::training_params_are_viewer_only(*mode.params);
                preflightGpuOrExit(interactive, viewer_only);

                // Probe and decompose the CUDA driver's context-creation cost only for the
                // GPU app path. CLI-only modes such as --help, convert, preprocess,
                // plugin, and mesh2splat must not create a CUDA primary context just
                // for HUD metrics.
                if (lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA)) {
                    analyzeCudaContextDistribution();
                }
                if (mode.params->optimization.debug_python) {
                    lfs::python::start_debugpy(mode.params->optimization.debug_python_port);
                }

                lfs::app::Application app;
                return app.run(std::move(mode.params));
            }
        },
                          std::move(args));
    }

} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* wide_argv[]) {
    // The parser expects UTF-8. Narrow CRT argv uses the Windows ANSI code page,
    // which can lose characters in paths opened from Explorer or the command line.
    std::vector<std::string> utf8_args;
    utf8_args.reserve(argc);
    for (int i = 0; i < argc; ++i)
        utf8_args.push_back(lfs::core::wstring_to_utf8(wide_argv[i]));

    std::vector<const char*> utf8_argv;
    utf8_argv.reserve(argc + 1);
    for (const auto& arg : utf8_args)
        utf8_argv.push_back(arg.c_str());
    utf8_argv.push_back(nullptr);
    const auto* argv = utf8_argv.data();
#else
int main(int argc, char* argv[]) {
#endif
    const char* const loaded_core_stamp = lfs_core_abi_stamp();
    if (loaded_core_stamp == nullptr || !lfs_core_abi_matches(LFS_CORE_ABI_STAMP)) {
        std::println(stderr,
                     "Fatal: lfs_core ABI mismatch. The application expects '{}' but loaded '{}'. "
                     "Remove stale binaries and rebuild LichtFeld Studio.",
                     LFS_CORE_ABI_STAMP,
                     loaded_core_stamp != nullptr ? loaded_core_stamp : "<null>");
        return 2;
    }

    lfs::core::install_crash_handlers();
    lfs::core::record_session_start();
    lfs::core::initialize_cuda_diagnostics();

    auto result = lfs::core::args::parse_args(argc, argv);
    if (!result) {
        std::println(stderr, "Error: {}", result.error());
        return 1;
    }

    publishResolvedUserPaths();

    return lfs::core::run_with_exception_firewall(
        [&result] { return run_mode(std::move(*result)); });
}
