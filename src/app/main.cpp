/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "app/converter.hpp"
#include "core/abi.hpp"
#include "core/argument_parser.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda_error.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "git_version.h"
#include "gui/gpu_memory_query.hpp"
#include "lfs_core_abi_stamp.h"
#include "preprocessing/preprocess.hpp"
#include "python/plugin_runner.hpp"
#include "python/runner.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <filesystem>
#include <print>

// pxr/base/tf/hashset.h pulls in the deprecated <ext/hash_set> GNU extension.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <pxr/base/plug/registry.h>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace {
    // Per-process GPU memory query. NVML on Linux / DXGI on Windows. Returns 0 if the
    // query fails or no GPU activity is yet attributed to this PID.
    std::size_t process_used_now() {
        return lfs::vis::gui::queryGpuMemory().process_used;
    }

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

    // Probe what the CUDA driver allocates during context creation, *attributed to this
    // process* (NVML per-PID, not device-wide cudaMemGetInfo). Each phase is the delta
    // against the previous probe so the sum reconstructs the total context cost.
    void analyzeCudaContextDistribution() {
        auto& p = lfs::diagnostics::VramProfiler::instance();

        // Phase 0: pre-context. Should be 0 — no CUDA calls have run.
        const std::size_t before_context = process_used_now();

        // Phase 1: primary context creation. cudaFree(nullptr) is a documented idiom that
        // forces the primary context to exist on device 0.
        cudaFree(nullptr);

        // Shrink the per-thread stack reserve from the 1 KiB default. Our kernels do not
        // recurse and have small frames; 256 B is comfortable. Default reservation is
        // per_thread_stack × num_SMs × max_threads_per_SM = ~192 MiB on a 4090. Driver
        // accepts the request post-context but applies it on the *next* launch — well
        // before any real kernel runs.
        cudaDeviceSetLimit(cudaLimitStackSize, 256);
        const std::size_t after_context = process_used_now();
        const std::size_t primary_context =
            after_context > before_context ? after_context - before_context : after_context;
        p.recordCudaPhaseBytes("primary_context", primary_context);

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

        // Phase 4: libcurand context. Create + destroy a generator so the library code
        // is loaded into the process; the residual delta is the library overhead.
        const std::size_t before_curand = process_used_now();
        curandGenerator_t gen = nullptr;
        if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS) {
            curandDestroyGenerator(gen);
        }
        const std::size_t after_curand = process_used_now();
        const std::size_t curand_load =
            after_curand > before_curand ? after_curand - before_curand : 0;
        p.recordCudaPhaseBytes("curand_load", curand_load);

        // The per-PID baseline = current NVML reading. Used as the breakdown's anchor
        // so cuda.context.residual = baseline − Σphases.
        p.setCudaContextBaselineBytes(process_used_now());

        // Device-wide (cudaMemGetInfo) baseline captured at the same point, so the
        // later kernel-warmup delta is measured against a matching metric instead of
        // the NVML per-PID anchor.
        p.captureCudaDeviceBaseline();
    }

    // Register OpenUSD plugin resources deployed beside the executable.
    // On Windows: <exe_dir>/usd/ — keeps relative LibraryPaths correct.
    // On Linux:   <exe_dir>/../lib/usd/ — conventional layout.
    // Must be called before any USD API usage (stage creation, schema lookup).
    // The env-var approach (PXR_PLUGINPATH_NAME) does not work reliably on
    // Windows because USD DLLs may initialise before main() runs.
    void configure_usd_plugins() {
        std::filesystem::path exe_dir;
        try {
            exe_dir = lfs::core::getExecutableDir();
        } catch (...) {
            return;
        }

        std::error_code ec;

        // On Windows, plugins sit next to the exe at <exe_dir>/usd/ so that
        // relative LibraryPath entries (e.g. "../../usd_ar.dll") resolve to
        // the DLL copies that are already loaded by Windows at startup.
        // On Linux they follow the conventional <exe_dir>/../lib/usd/ layout.
#ifdef _WIN32
        auto usd_dir = exe_dir / "usd";
#else
        auto usd_dir = exe_dir / ".." / "lib" / "usd";
#endif
        usd_dir = std::filesystem::canonical(usd_dir, ec);
        if (ec || !std::filesystem::is_directory(usd_dir, ec)) {
            LOG_ERROR("[USD] plugin directory not found ({})",
                      ec ? ec.message() : "not a directory");
            return;
        }

        const std::string path_utf8 = lfs::core::path_to_utf8(usd_dir);

        // Also set the env var for any code that reads it directly.
#ifdef _WIN32
        _putenv_s("PXR_PLUGINPATH_NAME", path_utf8.c_str());
#else
        setenv("PXR_PLUGINPATH_NAME", path_utf8.c_str(), /*overwrite=*/0);
#endif

        // Programmatically register plugins so the Plug system finds them
        // regardless of compiled-in search paths or env var timing.
        pxr::PlugRegistry::GetInstance().RegisterPlugins(path_utf8);
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
                analyzeCudaContextDistribution();
                return 0;
            } else if constexpr (std::is_same_v<T, lfs::core::args::ConvertMode>) {
                configure_usd_plugins();
                return lfs::app::run_converter(mode.params);
            } else if constexpr (std::is_same_v<T, lfs::core::args::Mesh2SplatMode>) {
                return lfs::app::run_mesh2splat(mode.params);
            } else if constexpr (std::is_same_v<T, lfs::core::args::PreprocessMode>) {
                return lfs::preprocessing::run_preprocess(mode.params);
            } else if constexpr (std::is_same_v<T, lfs::core::args::PluginMode>) {
                return lfs::python::run_plugin_command(mode);
            } else if constexpr (std::is_same_v<T, lfs::core::args::TrainingMode>) {
                LOG_INFO("LichtFeld Studio");
                LOG_INFO("version {} | tag {}", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);

                // Driver-level tuning must precede *any* CUDA call, including the
                // cudaFree(nullptr) inside analyzeCudaContextDistribution.
                applyCudaContextTuning();

                // Probe and decompose the CUDA driver's context-creation cost only for the
                // GPU app path. CLI-only modes such as --help, convert, preprocess,
                // plugin, and mesh2splat must not create a CUDA primary context just
                // for HUD metrics.
                analyzeCudaContextDistribution();
                configure_usd_plugins();

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

int main(int argc, char* argv[]) {
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
    lfs::core::initialize_cuda_diagnostics();

    auto result = lfs::core::args::parse_args(argc, argv);
    if (!result) {
        std::println(stderr, "Error: {}", result.error());
        return 1;
    }

    return lfs::core::run_with_exception_firewall(
        [&result] { return run_mode(std::move(*result)); });
}
