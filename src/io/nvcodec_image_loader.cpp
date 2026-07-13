/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/nvcodec_image_loader.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "cuda/image_format_kernels.cuh"
#include "diagnostics/vram_profiler.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h> // For CUcontext, cuCtxGetCurrent, cuCtxSetCurrent
#include <cuda_runtime.h>
#include <fstream>
#include <mutex>
#include <nvimgcodec.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#endif

namespace lfs::io {

    namespace {
        // Convert nvimgcodec status to string
        const char* nvimgcodec_status_to_string(nvimgcodecStatus_t status) {
            switch (status) {
            case NVIMGCODEC_STATUS_SUCCESS: return "SUCCESS";
            case NVIMGCODEC_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
            case NVIMGCODEC_STATUS_INVALID_PARAMETER: return "INVALID_PARAMETER";
            case NVIMGCODEC_STATUS_BAD_CODESTREAM: return "BAD_CODESTREAM";
            case NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED: return "CODESTREAM_UNSUPPORTED";
            case NVIMGCODEC_STATUS_ALLOCATOR_FAILURE: return "ALLOCATOR_FAILURE";
            case NVIMGCODEC_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
            case NVIMGCODEC_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
            case NVIMGCODEC_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
            case NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED: return "IMPLEMENTATION_UNSUPPORTED";
            case NVIMGCODEC_STATUS_MISSED_DEPENDENCIES: return "MISSED_DEPENDENCIES";
            case NVIMGCODEC_STATUS_EXTENSION_NOT_INITIALIZED: return "EXTENSION_NOT_INITIALIZED";
            case NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER: return "EXTENSION_INVALID_PARAMETER";
            case NVIMGCODEC_STATUS_EXTENSION_BAD_CODE_STREAM: return "EXTENSION_BAD_CODE_STREAM";
            case NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED: return "EXTENSION_CODESTREAM_UNSUPPORTED";
            case NVIMGCODEC_STATUS_EXTENSION_ALLOCATOR_FAILURE: return "EXTENSION_ALLOCATOR_FAILURE";
            case NVIMGCODEC_STATUS_EXTENSION_ARCH_MISMATCH: return "EXTENSION_ARCH_MISMATCH";
            case NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR: return "EXTENSION_INTERNAL_ERROR";
            case NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED: return "EXTENSION_IMPLEMENTATION_NOT_SUPPORTED";
            case NVIMGCODEC_STATUS_EXTENSION_INCOMPLETE_BITSTREAM: return "EXTENSION_INCOMPLETE_BITSTREAM";
            case NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED: return "EXTENSION_EXECUTION_FAILED";
            case NVIMGCODEC_STATUS_EXTENSION_CUDA_CALL_ERROR: return "EXTENSION_CUDA_CALL_ERROR";
            default: return "UNKNOWN_STATUS";
            }
        }

        // Convert processing status to string
        const char* processing_status_to_string(nvimgcodecProcessingStatus_t status) {
            switch (status) {
            case NVIMGCODEC_PROCESSING_STATUS_SUCCESS: return "SUCCESS";
            case NVIMGCODEC_PROCESSING_STATUS_FAIL: return "FAIL";
            case NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED: return "IMAGE_CORRUPTED";
            case NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED: return "CODEC_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED: return "BACKEND_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED: return "CODESTREAM_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED: return "ENCODING_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED: return "RESOLUTION_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED: return "SAMPLING_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED: return "COLOR_SPEC_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED: return "ORIENTATION_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED: return "ROI_UNSUPPORTED";
            case NVIMGCODEC_PROCESSING_STATUS_UNKNOWN: return "UNKNOWN";
            default: return "UNRECOGNIZED_STATUS";
            }
        }

        bool is_fatal_cuda_context_error(const cudaError_t error) noexcept {
            switch (error) {
            case cudaErrorIllegalAddress:
            case cudaErrorLaunchFailure:
            case cudaErrorAssert:
            case cudaErrorIllegalInstruction:
            case cudaErrorMisalignedAddress:
            case cudaErrorInvalidAddressSpace:
            case cudaErrorInvalidPc:
            case cudaErrorLaunchTimeout:
            case cudaErrorECCUncorrectable:
                return true;
            default:
                return false;
            }
        }

        bool cuda_context_poisoned_for_nvcodec_teardown() noexcept {
            const cudaError_t sync_status = cudaDeviceSynchronize();
            if (sync_status == cudaSuccess) {
                return false;
            }
            if (!is_fatal_cuda_context_error(sync_status)) {
                return false;
            }

            LOG_WARN("[NvCodecImageLoader] Skipping nvImageCodec shutdown because CUDA context is poisoned: {}",
                     cudaGetErrorString(sync_status));
            return true;
        }

        // Log comprehensive GPU and driver information
        void log_gpu_diagnostics() {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);

            if (err != cudaSuccess) {
                LOG_ERROR("[nvImageCodec Diagnostics] cudaGetDeviceCount failed: {} ({})",
                          cudaGetErrorString(err), static_cast<int>(err));
                return;
            }

            LOG_INFO("[nvImageCodec Diagnostics] CUDA device count: {}", device_count);

            for (int i = 0; i < device_count; ++i) {
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                    LOG_INFO("[nvImageCodec Diagnostics] GPU {}: {} (SM {}.{}, {} MB, compute capability {}.{})",
                             i, prop.name, prop.major, prop.minor,
                             static_cast<int>(prop.totalGlobalMem / (1024 * 1024)),
                             prop.major, prop.minor);

                    // Log NVJPEG hardware decode capability (SM 3.0+ required, SM 6.0+ for HW decode)
                    if (prop.major < 3) {
                        LOG_WARN("[nvImageCodec Diagnostics] GPU {} compute capability {}.{} is below minimum (3.0) for nvJPEG",
                                 i, prop.major, prop.minor);
                    } else if (prop.major < 6) {
                        LOG_INFO("[nvImageCodec Diagnostics] GPU {} compute capability {}.{} - nvJPEG will use hybrid/CPU decoding (HW decode requires 6.0+)",
                                 i, prop.major, prop.minor);
                    } else {
                        LOG_INFO("[nvImageCodec Diagnostics] GPU {} compute capability {}.{} - nvJPEG hardware decode should be available",
                                 i, prop.major, prop.minor);
                    }
                }
            }

            // Log CUDA driver version
            int driver_version = 0;
            if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
                LOG_INFO("[nvImageCodec Diagnostics] CUDA driver version: {}.{}",
                         driver_version / 1000, (driver_version % 1000) / 10);
            }

            // Log CUDA runtime version
            int runtime_version = 0;
            if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
                LOG_INFO("[nvImageCodec Diagnostics] CUDA runtime version: {}.{}",
                         runtime_version / 1000, (runtime_version % 1000) / 10);
            }
        }

        // Log extension directory and file discovery
        void log_extension_diagnostics(const std::filesystem::path& ext_dir) {
            if (ext_dir.empty()) {
                LOG_WARN("[nvImageCodec Diagnostics] Extensions directory: NOT FOUND (using builtin modules only)");
                return;
            }

            LOG_INFO("[nvImageCodec Diagnostics] Extensions directory: {}", lfs::core::path_to_utf8(ext_dir));

            if (!std::filesystem::exists(ext_dir)) {
                LOG_ERROR("[nvImageCodec Diagnostics] Extensions directory does not exist!");
                return;
            }

            // List extension files
            std::vector<std::string> dll_files;
            std::error_code ec;

            for (std::filesystem::directory_iterator it(ext_dir, ec), end; !ec && it != end; it.increment(ec)) {
                const auto& path = it->path();
                std::string ext = path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

#ifdef _WIN32
                if (ext == ".dll") {
                    dll_files.push_back(lfs::core::path_to_utf8(path.filename()));
                }
#else
                if (ext == ".so") {
                    dll_files.push_back(lfs::core::path_to_utf8(path.filename()));
                }
#endif
            }

            if (ec) {
                LOG_ERROR("[nvImageCodec Diagnostics] Failed to iterate extensions dir: {}", ec.message());
            }

            if (dll_files.empty()) {
#ifdef _WIN32
                LOG_WARN("[nvImageCodec Diagnostics] No .dll extension files found in {}", lfs::core::path_to_utf8(ext_dir));
#else
                LOG_WARN("[nvImageCodec Diagnostics] No .so extension files found in {}", lfs::core::path_to_utf8(ext_dir));
#endif
            } else {
                LOG_INFO("[nvImageCodec Diagnostics] Found {} extension files:", dll_files.size());
                for (const auto& f : dll_files) {
                    LOG_INFO("[nvImageCodec Diagnostics]   - {}", f);
                }
            }

#ifdef _WIN32
            // Test DLL loading with all dependencies
            constexpr DWORD ERROR_MOD_NOT_FOUND_CODE = 126;
            constexpr DWORD ERROR_BAD_EXE_FORMAT_CODE = 193;
            constexpr DWORD ERROR_PROC_NOT_FOUND_CODE = 127;
            constexpr DWORD ERROR_SXS_CANT_GEN_ACTCTX_CODE = 14001;

            for (const auto& f : dll_files) {
                const auto full_path = ext_dir / f;
                const HMODULE hModule = LoadLibraryExW(full_path.wstring().c_str(), nullptr, 0);
                if (hModule) {
                    LOG_DEBUG("[nvImageCodec] {} loaded OK", f);
                    FreeLibrary(hModule);
                } else {
                    const DWORD err = GetLastError();
                    LOG_ERROR("[nvImageCodec] {} load failed: error {}", f, err);
                    if (err == ERROR_MOD_NOT_FOUND_CODE) {
                        LOG_ERROR("[nvImageCodec] Missing dependency (likely nvjpeg64_12.dll)");
                    } else if (err == ERROR_BAD_EXE_FORMAT_CODE) {
                        LOG_ERROR("[nvImageCodec] 32/64-bit mismatch");
                    } else if (err == ERROR_SXS_CANT_GEN_ACTCTX_CODE) {
                        LOG_ERROR("[nvImageCodec] Missing VC++ Redistributable");
                    } else if (err == ERROR_PROC_NOT_FOUND_CODE) {
                        LOG_ERROR("[nvImageCodec] DLL version mismatch");
                    }
                }
            }
#endif
        }

        bool should_run_nvimgcodec_diagnostics() {
            const char* flag = std::getenv("LFS_NVCODEC_DIAGNOSTICS");
            if (!flag || !*flag) {
                return false;
            }
            return std::strcmp(flag, "0") != 0;
        }

        bool check_nvimgcodec_availability_fast() {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0) {
                return false;
            }

            const auto extensions_dir = lfs::core::getExtensionsDir();
            std::string extensions_path_str;
            const char* extensions_path_ptr = nullptr;

            if (!extensions_dir.empty() && std::filesystem::exists(extensions_dir)) {
                extensions_path_str = lfs::core::path_to_utf8(extensions_dir);
                extensions_path_ptr = extensions_path_str.c_str();
            }

            nvimgcodecInstance_t test_instance = nullptr;
            const nvimgcodecInstanceCreateInfo_t create_info{
                NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                sizeof(nvimgcodecInstanceCreateInfo_t),
                nullptr,
                1, // load_builtin_modules
                extensions_path_ptr ? 1 : 0,
                extensions_path_ptr,
                0,
                nullptr,
                0,
                0};

            const auto status = nvimgcodecInstanceCreate(&test_instance, &create_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS || !test_instance) {
                return false;
            }

            nvimgcodecInstanceDestroy(test_instance);
            return true;
        }

        // Comprehensive availability check with diagnostics
        bool check_nvimgcodec_availability_with_diagnostics() {
            LOG_INFO("[nvImageCodec Diagnostics] === Starting nvImageCodec availability check ===");

            // Step 1: Check CUDA
            log_gpu_diagnostics();

            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0) {
                LOG_ERROR("[nvImageCodec Diagnostics] CUDA not available: {} (device_count={})",
                          cudaGetErrorString(err), device_count);
                LOG_INFO("[nvImageCodec Diagnostics] === nvImageCodec UNAVAILABLE (no CUDA) ===");
                return false;
            }

            // Step 2: Check extensions directory
            auto extensions_dir = lfs::core::getExtensionsDir();
            log_extension_diagnostics(extensions_dir);

            // Step 3: Try to create an nvImageCodec instance
            LOG_INFO("[nvImageCodec Diagnostics] Attempting to create nvImageCodec instance...");

            std::string extensions_path_str;
            const char* extensions_path_ptr = nullptr;

            if (!extensions_dir.empty() && std::filesystem::exists(extensions_dir)) {
                extensions_path_str = lfs::core::path_to_utf8(extensions_dir);
                extensions_path_ptr = extensions_path_str.c_str();
            }

            // First try WITHOUT loading extension modules (builtin only)
            nvimgcodecInstance_t test_instance = nullptr;
            nvimgcodecInstanceCreateInfo_t create_info{
                NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                sizeof(nvimgcodecInstanceCreateInfo_t),
                nullptr,
                1,       // load_builtin_modules
                0,       // load_extension_modules = 0 (builtin only)
                nullptr, // extension_modules_path
                0,       // create_debug_messenger
                nullptr, // debug_messenger_desc
                0,       // message_severity
                0        // message_category
            };

            auto status = nvimgcodecInstanceCreate(&test_instance, &create_info);
            if (status == NVIMGCODEC_STATUS_SUCCESS && test_instance) {
                LOG_INFO("[nvImageCodec Diagnostics] Instance creation with BUILTIN modules: SUCCESS");
                nvimgcodecInstanceDestroy(test_instance);
                test_instance = nullptr;
            } else {
                LOG_ERROR("[nvImageCodec Diagnostics] Instance creation with BUILTIN modules FAILED: {} ({})",
                          nvimgcodec_status_to_string(status), static_cast<int>(status));
            }

            // Now try WITH extension modules
            if (!extensions_dir.empty()) {
                create_info.load_extension_modules = 1;
                create_info.extension_modules_path = extensions_path_ptr;

                status = nvimgcodecInstanceCreate(&test_instance, &create_info);
                if (status == NVIMGCODEC_STATUS_SUCCESS && test_instance) {
                    LOG_INFO("[nvImageCodec Diagnostics] Instance creation with EXTENSION modules ({}): SUCCESS",
                             extensions_path_str);
                    nvimgcodecInstanceDestroy(test_instance);
                    test_instance = nullptr;
                } else {
                    LOG_ERROR("[nvImageCodec Diagnostics] Instance creation with EXTENSION modules FAILED: {} ({})",
                              nvimgcodec_status_to_string(status), static_cast<int>(status));
                }
            }

            // Final test with preferred settings
            create_info.load_extension_modules = 1;
            create_info.extension_modules_path = extensions_path_ptr;

            status = nvimgcodecInstanceCreate(&test_instance, &create_info);
            if (status == NVIMGCODEC_STATUS_SUCCESS && test_instance) {
                // Try to create a decoder to verify full functionality
                nvimgcodecDecoder_t test_decoder = nullptr;
                nvimgcodecExecutionParams_t exec_params{
                    NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
                    sizeof(nvimgcodecExecutionParams_t),
                    nullptr,
                    nullptr, // device_allocator
                    nullptr, // pinned_allocator
                    0,       // max_num_cpu_threads
                    nullptr, // executor
                    0,       // device_id
                    0,       // pre_init
                    0,       // skip_pre_sync
                    0,       // num_backends
                    nullptr  // backends
                };

                auto decoder_status = nvimgcodecDecoderCreate(test_instance, &test_decoder, &exec_params, nullptr);
                if (decoder_status == NVIMGCODEC_STATUS_SUCCESS && test_decoder) {
                    LOG_INFO("[nvImageCodec Diagnostics] Decoder creation: SUCCESS");
                    nvimgcodecDecoderDestroy(test_decoder);
                } else {
                    LOG_WARN("[nvImageCodec Diagnostics] Decoder creation FAILED: {} - decode may not work",
                             nvimgcodec_status_to_string(decoder_status));
                }

                nvimgcodecInstanceDestroy(test_instance);
                LOG_INFO("[nvImageCodec Diagnostics] === nvImageCodec AVAILABLE ===");
                return true;
            }

            LOG_ERROR("[nvImageCodec Diagnostics] === nvImageCodec UNAVAILABLE (instance creation failed) ===");
            return false;
        }

        bool cuda_used_bytes_now(size_t& used_bytes) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (err != cudaSuccess || total_bytes < free_bytes) {
                return false;
            }
            used_bytes = total_bytes - free_bytes;
            return true;
        }

        bool cuda_default_pool_used_now(size_t& used_bytes) {
#if CUDART_VERSION >= 12080
            int device = 0;
            cudaMemPool_t pool = nullptr;
            std::uint64_t pool_used = 0;
            if (cudaGetDevice(&device) == cudaSuccess &&
                cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess &&
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &pool_used) == cudaSuccess) {
                used_bytes = static_cast<size_t>(pool_used);
                return true;
            }
#else
            (void)used_bytes;
#endif
            return false;
        }

        struct CudaUsageSnapshot {
            size_t total_used = 0;
            size_t pool_used = 0;
            bool total_valid = false;
            bool pool_valid = false;
        };

        CudaUsageSnapshot cuda_usage_snapshot_now() {
            CudaUsageSnapshot snapshot;
            snapshot.total_valid = cuda_used_bytes_now(snapshot.total_used);
            snapshot.pool_valid = cuda_default_pool_used_now(snapshot.pool_used);
            return snapshot;
        }

        struct NvCodecVramBytes {
            size_t driver_or_direct_bytes = 0;
            size_t default_pool_bytes = 0;

            [[nodiscard]] size_t total() const {
                return driver_or_direct_bytes + default_pool_bytes;
            }
        };

        std::mutex& nvcodec_vram_mutex() {
            static std::mutex mutex;
            return mutex;
        }

        std::unordered_map<const void*, NvCodecVramBytes>& nvcodec_vram_entries() {
            static std::unordered_map<const void*, NvCodecVramBytes> entries;
            return entries;
        }

        void publish_nvcodec_vram_bytes_locked() {
            NvCodecVramBytes total_bytes;
            for (const auto& [_, bytes] : nvcodec_vram_entries()) {
                total_bytes.driver_or_direct_bytes += bytes.driver_or_direct_bytes;
                total_bytes.default_pool_bytes += bytes.default_pool_bytes;
            }

            auto& profiler = lfs::diagnostics::VramProfiler::instance();
            profiler.clearStaticScope("io.nvimagecodec");
            if (total_bytes.driver_or_direct_bytes > 0) {
                profiler.recordStaticBytes("io.nvimagecodec",
                                           "driver_or_direct",
                                           total_bytes.driver_or_direct_bytes,
                                           lfs::diagnostics::VramAllocationMethod::External);
            }
            if (total_bytes.default_pool_bytes > 0) {
                profiler.recordStaticBytes("io.nvimagecodec",
                                           "default_pool",
                                           total_bytes.default_pool_bytes,
                                           lfs::diagnostics::VramAllocationMethod::Async);
            }
        }

        void register_nvcodec_vram(const void* owner, const NvCodecVramBytes bytes) {
            if (!owner || bytes.total() == 0) {
                return;
            }

            std::lock_guard<std::mutex> lock(nvcodec_vram_mutex());
            nvcodec_vram_entries()[owner] = bytes;
            publish_nvcodec_vram_bytes_locked();
        }

        void unregister_nvcodec_vram(const void* owner) {
            if (!owner) {
                return;
            }

            std::lock_guard<std::mutex> lock(nvcodec_vram_mutex());
            if (nvcodec_vram_entries().erase(owner) == 0) {
                return;
            }
            publish_nvcodec_vram_bytes_locked();
        }

        class NvCodecVramAccount {
        public:
            const void* owner = nullptr;
            mutable std::mutex mutex;
            NvCodecVramBytes baseline_bytes;
            NvCodecVramBytes max_lazy_delta_bytes;
            size_t max_lazy_total_delta_bytes = 0;
            std::atomic<int> probe_budget{32};

            void set_owner(const void* new_owner) {
                std::lock_guard<std::mutex> lock(mutex);
                owner = new_owner;
            }

            bool begin_probe(CudaUsageSnapshot& before) {
                int remaining = probe_budget.load(std::memory_order_relaxed);
                while (remaining > 0) {
                    if (probe_budget.compare_exchange_weak(
                            remaining, remaining - 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                        before = cuda_usage_snapshot_now();
                        return before.total_valid;
                    }
                }
                return false;
            }

            void end_probe(const CudaUsageSnapshot& before) {
                const auto after = cuda_usage_snapshot_now();
                if (!before.total_valid || !after.total_valid ||
                    after.total_used <= before.total_used) {
                    return;
                }
                observe_lazy_delta(delta_bytes(before, after));
            }

            void set_baseline_bytes(const NvCodecVramBytes bytes) {
                const void* publish_owner = nullptr;
                NvCodecVramBytes publish_bytes;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    baseline_bytes = bytes;
                    publish_owner = owner;
                    publish_bytes = total_locked();
                }
                publish(publish_owner, publish_bytes);
            }

            void observe_lazy_delta(const NvCodecVramBytes bytes) {
                const size_t total_bytes = bytes.total();
                if (total_bytes == 0) {
                    return;
                }

                const void* publish_owner = nullptr;
                NvCodecVramBytes publish_bytes;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (total_bytes <= max_lazy_total_delta_bytes) {
                        return;
                    }
                    max_lazy_delta_bytes = bytes;
                    max_lazy_total_delta_bytes = total_bytes;
                    publish_owner = owner;
                    publish_bytes = total_locked();
                }
                publish(publish_owner, publish_bytes);
            }

            void unregister() {
                const void* old_owner = nullptr;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    old_owner = owner;
                    owner = nullptr;
                    baseline_bytes = {};
                    max_lazy_delta_bytes = {};
                    max_lazy_total_delta_bytes = 0;
                }
                unregister_nvcodec_vram(old_owner);
            }

            static NvCodecVramBytes delta_bytes(const CudaUsageSnapshot& before,
                                                const CudaUsageSnapshot& after) {
                NvCodecVramBytes bytes;
                const size_t total_delta = after.total_used - before.total_used;
                size_t pool_delta = 0;
                if (before.pool_valid && after.pool_valid && after.pool_used > before.pool_used) {
                    pool_delta = std::min(after.pool_used - before.pool_used, total_delta);
                }
                bytes.default_pool_bytes = pool_delta;
                bytes.driver_or_direct_bytes = total_delta - pool_delta;
                return bytes;
            }

        private:
            NvCodecVramBytes total_locked() const {
                return NvCodecVramBytes{
                    .driver_or_direct_bytes = baseline_bytes.driver_or_direct_bytes +
                                              max_lazy_delta_bytes.driver_or_direct_bytes,
                    .default_pool_bytes = baseline_bytes.default_pool_bytes +
                                          max_lazy_delta_bytes.default_pool_bytes,
                };
            }

            static void publish(const void* owner, const NvCodecVramBytes bytes) {
                if (!owner) {
                    return;
                }
                if (bytes.total() == 0) {
                    unregister_nvcodec_vram(owner);
                    return;
                }
                register_nvcodec_vram(owner, bytes);
            }
        };

        class ScopedNvCodecVramProbe {
        public:
            explicit ScopedNvCodecVramProbe(NvCodecVramAccount& account)
                : account_(account),
                  active_(account_.begin_probe(before_)) {}

            ~ScopedNvCodecVramProbe() {
                if (active_) {
                    account_.end_probe(before_);
                }
            }

            ScopedNvCodecVramProbe(const ScopedNvCodecVramProbe&) = delete;
            ScopedNvCodecVramProbe& operator=(const ScopedNvCodecVramProbe&) = delete;

        private:
            NvCodecVramAccount& account_;
            CudaUsageSnapshot before_;
            bool active_ = false;
        };

    } // anonymous namespace

    struct NvCodecImageLoader::Impl {
        Impl() {
            vram_account.set_owner(this);
        }

        nvimgcodecInstance_t instance = nullptr;
        std::vector<nvimgcodecDecoder_t> decoder_pool;
        std::vector<bool> decoder_available;
        std::mutex pool_mutex;
        std::condition_variable pool_cv;
        nvimgcodecEncoder_t encoder = nullptr;
        std::mutex encoder_mutex;
        int device_id = 0;
        bool fallback_enabled = true;
        NvCodecVramAccount vram_account;

        size_t acquire_decoder() {
            std::unique_lock<std::mutex> lock(pool_mutex);
            pool_cv.wait(lock, [this] {
                return std::find(decoder_available.begin(), decoder_available.end(), true) != decoder_available.end();
            });
            for (size_t i = 0; i < decoder_available.size(); ++i) {
                if (decoder_available[i]) {
                    decoder_available[i] = false;
                    return i;
                }
            }
            return 0;
        }

        void release_decoder(const size_t idx) {
            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                decoder_available[idx] = true;
            }
            pool_cv.notify_one();
        }

        ~Impl() noexcept {
            try {
                CUcontext current_ctx = nullptr;
                const CUresult ctx_result = cuCtxGetCurrent(&current_ctx);
                if (ctx_result == CUDA_SUCCESS && current_ctx != nullptr) {
                    if (cuda_context_poisoned_for_nvcodec_teardown()) {
                        encoder = nullptr;
                        decoder_pool.clear();
                        decoder_available.clear();
                        instance = nullptr;
                        vram_account.unregister();
                        return;
                    }

                    if (encoder) {
                        nvimgcodecEncoderDestroy(encoder);
                        encoder = nullptr;
                    }

                    for (auto& decoder : decoder_pool) {
                        if (decoder) {
                            nvimgcodecDecoderDestroy(decoder);
                            decoder = nullptr;
                        }
                    }
                    decoder_pool.clear();
                    decoder_available.clear();

                    if (instance) {
                        nvimgcodecInstanceDestroy(instance);
                        instance = nullptr;
                    }
                }
            } catch (const std::exception& e) {
                LOG_WARN("[NvCodecImageLoader] Ignoring nvImageCodec shutdown error: {}", e.what());
            } catch (...) {
                LOG_WARN("[NvCodecImageLoader] Ignoring unknown nvImageCodec shutdown error");
            }

            vram_account.unregister();
        }
    };

    namespace {
        constexpr size_t DECODER_POOL_SIZE = 8;
        constexpr int LANCZOS_KERNEL_SIZE = 2;
    } // namespace

    NvCodecImageLoader::NvCodecImageLoader(const Options& options)
        : impl_(std::make_unique<Impl>()) {

        LOG_INFO("[NvCodecImageLoader] Initializing: device={}, pool={}, fallback={}",
                 options.device_id, options.decoder_pool_size, options.enable_fallback);

        impl_->device_id = options.device_id;
        impl_->fallback_enabled = options.enable_fallback;
        const cudaError_t set_device_status = cudaSetDevice(impl_->device_id);
        const auto init_vram_before =
            set_device_status == cudaSuccess ? cuda_usage_snapshot_now() : CudaUsageSnapshot{};

        const auto extensions_dir = lfs::core::getExtensionsDir();
        std::string extensions_path_str;
        const char* extensions_path_ptr = nullptr;

        if (!extensions_dir.empty() && std::filesystem::exists(extensions_dir)) {
            extensions_path_str = lfs::core::path_to_utf8(extensions_dir);
            extensions_path_ptr = extensions_path_str.c_str();
            LOG_DEBUG("[NvCodecImageLoader] Extensions: {}", extensions_path_str);
        }

        const nvimgcodecInstanceCreateInfo_t create_info{
            NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            sizeof(nvimgcodecInstanceCreateInfo_t),
            nullptr, 1, 1, extensions_path_ptr, 0, nullptr, 0, 0};

        auto status = nvimgcodecInstanceCreate(&impl_->instance, &create_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("nvImageCodec init failed: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        const size_t pool_size = options.decoder_pool_size > 0 ? options.decoder_pool_size : DECODER_POOL_SIZE;
        impl_->decoder_pool.resize(pool_size);
        impl_->decoder_available.resize(pool_size, true);

        const nvimgcodecExecutionParams_t exec_params{
            NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
            sizeof(nvimgcodecExecutionParams_t),
            nullptr, nullptr, nullptr,
            options.max_num_cpu_threads, nullptr, options.device_id, 0, 0, 0, nullptr};

        for (size_t i = 0; i < pool_size; ++i) {
            status = nvimgcodecDecoderCreate(impl_->instance, &impl_->decoder_pool[i], &exec_params, nullptr);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                throw std::runtime_error("Decoder creation failed: " +
                                         std::string(nvimgcodec_status_to_string(status)));
            }
        }

        LOG_INFO("[NvCodecImageLoader] {} decoders ready", pool_size);

        status = nvimgcodecEncoderCreate(impl_->instance, &impl_->encoder, &exec_params, nullptr);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            LOG_WARN("[NvCodecImageLoader] Encoder unavailable: {}", nvimgcodec_status_to_string(status));
            impl_->encoder = nullptr;
        }

        const auto init_vram_after = cuda_usage_snapshot_now();
        if (init_vram_before.total_valid && init_vram_after.total_valid &&
            init_vram_after.total_used > init_vram_before.total_used) {
            const auto baseline_bytes =
                NvCodecVramAccount::delta_bytes(init_vram_before, init_vram_after);
            impl_->vram_account.set_baseline_bytes(baseline_bytes);
            LOG_INFO("[NvCodecImageLoader] Accounted nvImageCodec init VRAM: {:.1f} MiB",
                     static_cast<double>(baseline_bytes.total()) / (1024.0 * 1024.0));
        }
    }

    NvCodecImageLoader::~NvCodecImageLoader() = default;

    bool NvCodecImageLoader::is_available() {
        static std::once_flag once;
        static bool available = false;
        std::call_once(once, [] {
            if (should_run_nvimgcodec_diagnostics()) {
                available = check_nvimgcodec_availability_with_diagnostics();
            } else {
                available = check_nvimgcodec_availability_fast();
            }
        });
        return available;
    }

    std::vector<uint8_t> NvCodecImageLoader::read_file(const std::filesystem::path& path) {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file)) {
            throw std::runtime_error("Failed to open file: " + lfs::core::path_to_utf8(path));
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file: " + lfs::core::path_to_utf8(path));
        }

        return buffer;
    }

    lfs::core::Tensor NvCodecImageLoader::load_image_gpu(
        const std::filesystem::path& path,
        int resize_factor,
        int max_width,
        void* cuda_stream,
        DecodeFormat format,
        bool output_uint8) {

        LOG_DEBUG("NvCodecImageLoader: Loading {}", lfs::core::path_to_utf8(path));

        const auto file_data = read_file(path);

        // PNG magic: 0x89 'P' 'N' 'G', WebP magic: 'RIFF'....'WEBP'
        if (file_data.size() >= 4 &&
            file_data[0] == 0x89 && file_data[1] == 0x50 &&
            file_data[2] == 0x4E && file_data[3] == 0x47) {
            throw std::runtime_error("PNG not supported by nvImageCodec");
        }
        if (file_data.size() >= 12 &&
            file_data[0] == 'R' && file_data[1] == 'I' &&
            file_data[2] == 'F' && file_data[3] == 'F' &&
            file_data[8] == 'W' && file_data[9] == 'E' &&
            file_data[10] == 'B' && file_data[11] == 'P') {
            throw std::runtime_error("WebP not supported by nvImageCodec");
        }

        return load_image_from_memory_gpu(file_data, resize_factor, max_width, cuda_stream, format, output_uint8);
    }

    lfs::core::Tensor NvCodecImageLoader::load_image_from_memory_gpu(
        const std::vector<uint8_t>& jpeg_data,
        int resize_factor,
        [[maybe_unused]] int max_width,
        void* cuda_stream,
        DecodeFormat format,
        bool output_uint8) {

        const bool is_grayscale = (format == DecodeFormat::Grayscale);
        const int num_channels = is_grayscale ? 1 : 3;

        const size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];

        // Auto-release decoder on scope exit
        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        nvimgcodecCodeStream_t code_stream;
        auto status = nvimgcodecCodeStreamCreateFromHostMem(
            impl_->instance, &code_stream, jpeg_data.data(), jpeg_data.size());
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create code stream from memory");
        }

        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.struct_next = nullptr;

        status = nvimgcodecCodeStreamGetImageInfo(code_stream, &image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(code_stream);
            LOG_ERROR("[NvCodecImageLoader] GetImageInfo failed: {} ({} bytes)",
                      nvimgcodec_status_to_string(status), jpeg_data.size());
            throw std::runtime_error("Failed to get image info: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        const int src_width = image_info.plane_info[0].width;
        const int src_height = image_info.plane_info[0].height;

        int target_width = src_width;
        int target_height = src_height;
        if (resize_factor > 1) {
            target_width /= resize_factor;
            target_height /= resize_factor;
        }
        if (max_width > 0 && (target_width > max_width || target_height > max_width)) {
            if (target_width > target_height) {
                target_height = std::max(1, max_width * target_height / target_width);
                target_width = max_width;
            } else {
                target_width = std::max(1, max_width * target_width / target_height);
                target_height = max_width;
            }
        }
        const bool needs_resize = (target_width != src_width || target_height != src_height);

        const auto src_sample_type = image_info.plane_info[0].sample_type;
        if (src_sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 &&
            src_sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("Unsupported source sample type: " +
                                     std::to_string(static_cast<int>(src_sample_type)));
        }

        // Grayscale consumers (masks, depth) stay 8-bit; nvImageCodec converts on decode.
        const bool decode_u16 = !is_grayscale &&
                                src_sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
        const auto tensor_datatype = decode_u16 ? lfs::core::DataType::Float16
                                                : lfs::core::DataType::UInt8;
        const size_t bytes_per_sample = decode_u16 ? 2 : 1;

        LOG_DEBUG("Image info: {}x{} -> {}x{} (resize_factor={}, max_width={})",
                  src_width, src_height, target_width, target_height, resize_factor, max_width);

        cudaSetDevice(impl_->device_id);

        using namespace lfs::core;
        Tensor image_tensor_aux;
        if (is_grayscale) {
            image_tensor_aux = Tensor::empty(
                TensorShape({static_cast<size_t>(src_height), static_cast<size_t>(src_width)}),
                Device::CUDA,
                DataType::UInt8);
        } else {
            image_tensor_aux = Tensor::empty(
                TensorShape({static_cast<size_t>(src_height), static_cast<size_t>(src_width), 3}),
                Device::CUDA,
                tensor_datatype);
        }

        void* const gpu_image_buffer = image_tensor_aux.data_ptr();
        const size_t decoded_size = static_cast<size_t>(src_width) * src_height * num_channels * bytes_per_sample;

        nvimgcodecImage_t nv_image;
        nvimgcodecImageInfo_t output_info{};
        output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_info.struct_next = nullptr;

        if (is_grayscale) {
            output_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
            output_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        } else {
            output_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
            output_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        }
        output_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        output_info.num_planes = 1;
        output_info.plane_info[0].height = src_height;
        output_info.plane_info[0].width = src_width;
        output_info.plane_info[0].row_stride = src_width * num_channels * bytes_per_sample;
        output_info.plane_info[0].num_channels = num_channels;
        output_info.plane_info[0].sample_type = decode_u16 ? NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16
                                                           : NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;

        output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        output_info.buffer = gpu_image_buffer;
        output_info.buffer_size = decoded_size;
        output_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        bool decode_success = false;
        {
            ScopedNvCodecVramProbe vram_probe(impl_->vram_account);
            status = nvimgcodecImageCreate(impl_->instance, &nv_image, &output_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                nvimgcodecCodeStreamDestroy(code_stream);
                throw std::runtime_error("Failed to create image descriptor");
            }

            nvimgcodecDecodeParams_t decode_params{};
            decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
            decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
            decode_params.struct_next = nullptr;
            decode_params.apply_exif_orientation = 0;

            nvimgcodecFuture_t decode_future;
            status = nvimgcodecDecoderDecode(
                decoder,
                &code_stream,
                &nv_image,
                1,
                &decode_params,
                &decode_future);

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                nvimgcodecImageDestroy(nv_image);
                nvimgcodecCodeStreamDestroy(code_stream);
                throw std::runtime_error("Failed to decode image from memory");
            }

            status = nvimgcodecFutureWaitForAll(decode_future);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                nvimgcodecFutureDestroy(decode_future);
                nvimgcodecImageDestroy(nv_image);
                nvimgcodecCodeStreamDestroy(code_stream);
                throw std::runtime_error("Failed to wait for decode completion");
            }

            size_t status_size = 1;
            nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);

            decode_success = (decode_status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS);

            nvimgcodecFutureDestroy(decode_future);
            nvimgcodecImageDestroy(nv_image);
            nvimgcodecCodeStreamDestroy(code_stream);
        }

        if (!decode_success) {
            const char* status_str = processing_status_to_string(decode_status);
            LOG_ERROR("[NvCodecImageLoader] Decode failed: {} ({}x{}, {} bytes)",
                      status_str, src_width, src_height, jpeg_data.size());
            throw std::runtime_error(std::string("Decode failed: ") + status_str);
        }

        // Lanczos reads uint8 or float32; widen uint16 to normalized float first.
        std::optional<Tensor> u16_as_float;
        if (decode_u16 && needs_resize) {
            u16_as_float.emplace(Tensor::empty(
                TensorShape({static_cast<size_t>(src_height), static_cast<size_t>(src_width),
                             static_cast<size_t>(num_channels)}),
                Device::CUDA,
                DataType::Float32));

            cuda::launch_uint16_hwc_to_float32_hwc(
                reinterpret_cast<const uint16_t*>(image_tensor_aux.data_ptr()),
                u16_as_float->ptr<float>(),
                src_height, src_width, num_channels, static_cast<cudaStream_t>(cuda_stream));
        }

        const Tensor& resize_input_image = u16_as_float ? *u16_as_float : image_tensor_aux;

        Tensor output_tensor;
        if (needs_resize) {
            if (is_grayscale) {
                output_tensor = lanczos_resize_grayscale(resize_input_image, target_height, target_width,
                                                         LANCZOS_KERNEL_SIZE, static_cast<cudaStream_t>(cuda_stream));
            } else {
                output_tensor = lanczos_resize(resize_input_image, target_height, target_width,
                                               LANCZOS_KERNEL_SIZE, static_cast<cudaStream_t>(cuda_stream));
                if (output_uint8) {
                    auto output_uint8_tensor = Tensor::empty(output_tensor.shape(), Device::CUDA, DataType::UInt8);
                    cuda::launch_float32_chw_to_uint8_chw(
                        output_tensor.ptr<float>(),
                        output_uint8_tensor.ptr<uint8_t>(),
                        output_tensor.shape()[1],
                        output_tensor.shape()[2],
                        output_tensor.shape()[0],
                        static_cast<cudaStream_t>(cuda_stream));
                    output_tensor = std::move(output_uint8_tensor);
                }
            }
        } else {
            if (is_grayscale) {
                const auto shape = image_tensor_aux.shape();
                const size_t H = shape[0], W = shape[1];
                output_tensor = Tensor::empty(TensorShape({H, W}), Device::CUDA, DataType::Float32);
                cuda::launch_uint8_hw_to_float32_hw(
                    reinterpret_cast<const uint8_t*>(image_tensor_aux.data_ptr()),
                    reinterpret_cast<float*>(output_tensor.data_ptr()),
                    H, W, static_cast<cudaStream_t>(cuda_stream));
            } else {
                const auto shape = image_tensor_aux.shape();
                const size_t H = shape[0], W = shape[1], C = shape[2];
                if (output_uint8) {
                    output_tensor = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::UInt8);
                    if (decode_u16) {
                        cuda::launch_uint16_hwc_to_uint8_chw(
                            reinterpret_cast<const uint16_t*>(image_tensor_aux.data_ptr()),
                            reinterpret_cast<uint8_t*>(output_tensor.data_ptr()),
                            H, W, C, static_cast<cudaStream_t>(cuda_stream));
                    } else {
                        cuda::launch_uint8_hwc_to_uint8_chw(
                            reinterpret_cast<const uint8_t*>(image_tensor_aux.data_ptr()),
                            reinterpret_cast<uint8_t*>(output_tensor.data_ptr()),
                            H, W, C, static_cast<cudaStream_t>(cuda_stream));
                    }
                } else {
                    output_tensor = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);
                    if (decode_u16) {
                        cuda::launch_uint16_hwc_to_float32_chw(
                            reinterpret_cast<const uint16_t*>(image_tensor_aux.data_ptr()),
                            reinterpret_cast<float*>(output_tensor.data_ptr()),
                            H, W, C, static_cast<cudaStream_t>(cuda_stream));
                    } else {
                        cuda::launch_uint8_hwc_to_float32_chw(
                            reinterpret_cast<const uint8_t*>(image_tensor_aux.data_ptr()),
                            reinterpret_cast<float*>(output_tensor.data_ptr()),
                            H, W, C, static_cast<cudaStream_t>(cuda_stream));
                    }
                }
            }
        }

        // Materialize before handoff. With a caller stream, sync only it — a
        // device-wide sync would couple image availability to in-flight
        // training kernels on other streams.
        if (cuda_stream) {
            if (const cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream));
                err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
            }
        } else if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
        }
        image_tensor_aux = Tensor();

        return output_tensor;
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::load_images_batch_gpu(
        const std::vector<std::filesystem::path>& paths,
        const int resize_factor,
        const int max_width) {

        std::vector<lfs::core::Tensor> results;
        results.reserve(paths.size());
        for (const auto& path : paths) {
            results.push_back(load_image_gpu(path, resize_factor, max_width));
        }
        return results;
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::batch_decode_from_memory(
        const std::vector<std::vector<uint8_t>>& jpeg_blobs,
        void* cuda_stream) {

        using namespace lfs::core;

        if (jpeg_blobs.empty()) {
            return {};
        }

        const size_t batch_size = jpeg_blobs.size();
        LOG_DEBUG("[NvCodecImageLoader] Batch decoding {} images", batch_size);

        const size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];

        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        CUcontext saved_context = nullptr;
        cuCtxGetCurrent(&saved_context);
        cudaSetDevice(impl_->device_id);

        std::vector<nvimgcodecCodeStream_t> code_streams(batch_size);
        std::vector<nvimgcodecImageInfo_t> image_infos(batch_size);
        std::vector<Tensor> uint8_tensors(batch_size);
        std::vector<nvimgcodecImage_t> nv_images(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            auto status = nvimgcodecCodeStreamCreateFromHostMem(
                impl_->instance, &code_streams[i],
                jpeg_blobs[i].data(), jpeg_blobs[i].size());

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (size_t j = 0; j < i; ++j) {
                    nvimgcodecCodeStreamDestroy(code_streams[j]);
                }
                throw std::runtime_error("Failed to create code stream for image " + std::to_string(i));
            }

            image_infos[i].struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            image_infos[i].struct_size = sizeof(nvimgcodecImageInfo_t);
            image_infos[i].struct_next = nullptr;

            status = nvimgcodecCodeStreamGetImageInfo(code_streams[i], &image_infos[i]);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (size_t j = 0; j <= i; ++j) {
                    nvimgcodecCodeStreamDestroy(code_streams[j]);
                }
                throw std::runtime_error("Failed to get image info for image " + std::to_string(i));
            }
        }

        for (size_t i = 0; i < batch_size; ++i) {
            const int width = image_infos[i].plane_info[0].width;
            const int height = image_infos[i].plane_info[0].height;
            uint8_tensors[i] = Tensor::empty(
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), 3}),
                Device::CUDA,
                DataType::UInt8);
        }

        std::vector<nvimgcodecProcessingStatus_t> decode_statuses(batch_size);
        {
            ScopedNvCodecVramProbe vram_probe(impl_->vram_account);
            for (size_t i = 0; i < batch_size; ++i) {
                const int width = image_infos[i].plane_info[0].width;
                const int height = image_infos[i].plane_info[0].height;
                nvimgcodecImageInfo_t output_info{};
                output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
                output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
                output_info.struct_next = nullptr;
                output_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
                output_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                output_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
                output_info.num_planes = 1;
                output_info.plane_info[0].height = height;
                output_info.plane_info[0].width = width;
                output_info.plane_info[0].row_stride = width * 3;
                output_info.plane_info[0].num_channels = 3;
                output_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
                output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                output_info.buffer = uint8_tensors[i].data_ptr();
                output_info.buffer_size = static_cast<size_t>(height) * width * 3;
                output_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

                const auto status = nvimgcodecImageCreate(impl_->instance, &nv_images[i], &output_info);
                if (status != NVIMGCODEC_STATUS_SUCCESS) {
                    for (size_t j = 0; j < i; ++j) {
                        nvimgcodecImageDestroy(nv_images[j]);
                    }
                    for (size_t j = 0; j < batch_size; ++j) {
                        nvimgcodecCodeStreamDestroy(code_streams[j]);
                    }
                    throw std::runtime_error("Failed to create image descriptor for image " + std::to_string(i));
                }
            }

            nvimgcodecDecodeParams_t decode_params{};
            decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
            decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
            decode_params.struct_next = nullptr;
            decode_params.apply_exif_orientation = 0;

            nvimgcodecFuture_t decode_future;
            auto status = nvimgcodecDecoderDecode(
                decoder,
                code_streams.data(),
                nv_images.data(),
                static_cast<int>(batch_size),
                &decode_params,
                &decode_future);

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (size_t i = 0; i < batch_size; ++i) {
                    nvimgcodecImageDestroy(nv_images[i]);
                    nvimgcodecCodeStreamDestroy(code_streams[i]);
                }
                throw std::runtime_error("Batch decode failed");
            }

            status = nvimgcodecFutureWaitForAll(decode_future);

            size_t status_size = batch_size;
            nvimgcodecFutureGetProcessingStatus(decode_future, decode_statuses.data(), &status_size);

            nvimgcodecFutureDestroy(decode_future);
            for (size_t i = 0; i < batch_size; ++i) {
                nvimgcodecImageDestroy(nv_images[i]);
                nvimgcodecCodeStreamDestroy(code_streams[i]);
            }
        }

        std::vector<Tensor> results;
        results.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            if (decode_statuses[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                LOG_WARN("[NvCodecImageLoader] Batch image {} failed: {}",
                         i, processing_status_to_string(decode_statuses[i]));
                results.push_back(Tensor());
                uint8_tensors[i] = Tensor();
                continue;
            }

            const auto shape = uint8_tensors[i].shape();
            const size_t H = shape[0], W = shape[1], C = shape[2];
            auto output = Tensor::zeros(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);

            cuda::launch_uint8_hwc_to_float32_chw(
                reinterpret_cast<const uint8_t*>(uint8_tensors[i].data_ptr()),
                reinterpret_cast<float*>(output.data_ptr()),
                H, W, C, nullptr);

            results.push_back(std::move(output));
        }

        cudaDeviceSynchronize();
        uint8_tensors.clear();

        if (saved_context) {
            cuCtxSetCurrent(saved_context);
        }

        LOG_DEBUG("[NvCodecImageLoader] Batch decode complete: {}", batch_size);
        return results;
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::batch_decode_from_spans(
        const std::vector<std::pair<const uint8_t*, size_t>>& jpeg_spans,
        void* cuda_stream) {

        using namespace lfs::core;

        if (jpeg_spans.empty()) {
            return {};
        }

        const size_t batch_size = jpeg_spans.size();
        LOG_DEBUG("[NvCodecImageLoader] Batch decoding {} spans", batch_size);

        const size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];

        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        CUcontext saved_context = nullptr;
        cuCtxGetCurrent(&saved_context);
        cudaSetDevice(impl_->device_id);

        std::vector<nvimgcodecCodeStream_t> code_streams(batch_size);
        std::vector<nvimgcodecImageInfo_t> image_infos(batch_size);
        std::vector<Tensor> uint8_tensors(batch_size);
        std::vector<nvimgcodecImage_t> nv_images(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            auto status = nvimgcodecCodeStreamCreateFromHostMem(
                impl_->instance, &code_streams[i],
                jpeg_spans[i].first, jpeg_spans[i].second);

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (size_t j = 0; j < i; ++j) {
                    nvimgcodecCodeStreamDestroy(code_streams[j]);
                }
                throw std::runtime_error("Failed to create code stream for image " + std::to_string(i));
            }

            image_infos[i].struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            image_infos[i].struct_size = sizeof(nvimgcodecImageInfo_t);
            image_infos[i].struct_next = nullptr;

            status = nvimgcodecCodeStreamGetImageInfo(code_streams[i], &image_infos[i]);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (size_t j = 0; j <= i; ++j) {
                    nvimgcodecCodeStreamDestroy(code_streams[j]);
                }
                throw std::runtime_error("Failed to get image info for image " + std::to_string(i));
            }
        }

        for (size_t i = 0; i < batch_size; ++i) {
            const int width = image_infos[i].plane_info[0].width;
            const int height = image_infos[i].plane_info[0].height;
            uint8_tensors[i] = Tensor::zeros(
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), 3}),
                Device::CUDA,
                DataType::UInt8);
        }

        std::vector<nvimgcodecProcessingStatus_t> decode_statuses(batch_size);
        {
            ScopedNvCodecVramProbe vram_probe(impl_->vram_account);
            for (size_t i = 0; i < batch_size; ++i) {
                const int width = image_infos[i].plane_info[0].width;
                const int height = image_infos[i].plane_info[0].height;
                nvimgcodecImageInfo_t out_info{};
                out_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
                out_info.struct_size = sizeof(nvimgcodecImageInfo_t);
                out_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
                out_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                out_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
                out_info.num_planes = 1;
                out_info.plane_info[0].width = width;
                out_info.plane_info[0].height = height;
                out_info.plane_info[0].num_channels = 3;
                out_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
                out_info.plane_info[0].precision = 0;
                out_info.plane_info[0].row_stride = width * 3;
                out_info.buffer = uint8_tensors[i].data_ptr();
                out_info.buffer_size = uint8_tensors[i].bytes();
                out_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                out_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

                const auto status = nvimgcodecImageCreate(impl_->instance, &nv_images[i], &out_info);
                if (status != NVIMGCODEC_STATUS_SUCCESS) {
                    for (size_t j = 0; j < i; ++j) {
                        nvimgcodecImageDestroy(nv_images[j]);
                    }
                    for (size_t j = 0; j < batch_size; ++j) {
                        nvimgcodecCodeStreamDestroy(code_streams[j]);
                    }
                    throw std::runtime_error("Failed to create image descriptor for image " + std::to_string(i));
                }
            }

            nvimgcodecDecodeParams_t decode_params{};
            decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
            decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
            decode_params.struct_next = nullptr;
            decode_params.apply_exif_orientation = 0;

            nvimgcodecFuture_t decode_future;
            auto status = nvimgcodecDecoderDecode(
                decoder,
                code_streams.data(),
                nv_images.data(),
                static_cast<int>(batch_size),
                &decode_params,
                &decode_future);

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (size_t i = 0; i < batch_size; ++i) {
                    nvimgcodecImageDestroy(nv_images[i]);
                    nvimgcodecCodeStreamDestroy(code_streams[i]);
                }
                throw std::runtime_error("Batch decode failed");
            }

            status = nvimgcodecFutureWaitForAll(decode_future);

            size_t status_size = batch_size;
            nvimgcodecFutureGetProcessingStatus(decode_future, decode_statuses.data(), &status_size);

            nvimgcodecFutureDestroy(decode_future);
            for (size_t i = 0; i < batch_size; ++i) {
                nvimgcodecImageDestroy(nv_images[i]);
                nvimgcodecCodeStreamDestroy(code_streams[i]);
            }
        }

        std::vector<Tensor> results;
        results.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            if (decode_statuses[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                LOG_WARN("[NvCodecImageLoader] Batch image {} failed", i);
                results.push_back(Tensor());
                uint8_tensors[i] = Tensor();
                continue;
            }

            const auto shape = uint8_tensors[i].shape();
            const size_t H = shape[0], W = shape[1], C = shape[2];
            auto output = Tensor::zeros(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);

            cuda::launch_uint8_hwc_to_float32_chw(
                reinterpret_cast<const uint8_t*>(uint8_tensors[i].data_ptr()),
                reinterpret_cast<float*>(output.data_ptr()),
                H, W, C, nullptr);

            results.push_back(std::move(output));
        }

        cudaDeviceSynchronize();
        uint8_tensors.clear();

        if (saved_context) {
            cuCtxSetCurrent(saved_context);
        }
        return results;
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_to_jpeg(
        const lfs::core::Tensor& image,
        const int quality,
        void* cuda_stream) {

        using namespace lfs::core;

        if (!impl_->encoder) {
            throw std::runtime_error("JPEG encoder not available");
        }

        std::lock_guard<std::mutex> lock(impl_->encoder_mutex);

        const auto& shape = image.shape();
        if (shape.rank() != 3) {
            throw std::runtime_error("Expected 3D tensor, got " + std::to_string(shape.rank()) + "D");
        }

        const bool is_hwc = (shape[2] == 3);
        const bool is_chw = !is_hwc && (shape[0] == 3 && shape[1] > 3 && shape[2] > 3);
        const int height = static_cast<int>(is_chw ? shape[1] : shape[0]);
        const int width = static_cast<int>(is_chw ? shape[2] : shape[1]);

        Tensor hwc_uint8;
        if (is_chw) {
            auto permuted = image.permute({1, 2, 0}).contiguous();
            hwc_uint8 = (permuted.dtype() == DataType::Float32)
                            ? (permuted * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8)
                            : permuted.to(DataType::UInt8);
        } else {
            hwc_uint8 = (image.dtype() == DataType::Float32)
                            ? (image * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8)
                            : image.to(DataType::UInt8);
        }

        if (hwc_uint8.device() != Device::CUDA) {
            hwc_uint8 = hwc_uint8.to(Device::CUDA);
        }
        hwc_uint8 = hwc_uint8.contiguous();

        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        image_info.num_planes = 1;
        image_info.plane_info[0].height = height;
        image_info.plane_info[0].width = width;
        image_info.plane_info[0].row_stride = width * 3;
        image_info.plane_info[0].num_channels = 3;
        image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        image_info.buffer = hwc_uint8.data_ptr();
        image_info.buffer_size = height * width * 3;
        image_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

        nvimgcodecImage_t nv_image;
        auto status = nvimgcodecImageCreate(impl_->instance, &nv_image, &image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create image for encoding: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        std::vector<uint8_t> output_buffer;

        nvimgcodecImageInfo_t output_info{};
        output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        std::snprintf(output_info.codec_name, sizeof(output_info.codec_name), "%s", "jpeg");

        nvimgcodecCodeStream_t code_stream;
        status = nvimgcodecCodeStreamCreateToHostMem(
            impl_->instance, &code_stream, &output_buffer,
            [](void* ctx, size_t req_size) -> unsigned char* {
                auto* vec = static_cast<std::vector<uint8_t>*>(ctx);
                vec->resize(req_size);
                return vec->data();
            },
            &output_info);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("Failed to create output code stream");
        }

        nvimgcodecEncodeParams_t encode_params{};
        encode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
        encode_params.struct_size = sizeof(nvimgcodecEncodeParams_t);
        encode_params.quality_value = static_cast<float>(quality);
        encode_params.quality_type = NVIMGCODEC_QUALITY_TYPE_QUALITY;

        nvimgcodecFuture_t encode_future;
        status = nvimgcodecEncoderEncode(
            impl_->encoder, &nv_image, &code_stream, 1, &encode_params, &encode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(code_stream);
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("Encode failed");
        }

        nvimgcodecFutureWaitForAll(encode_future);

        nvimgcodecProcessingStatus_t encode_status;
        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status, &status_size);
        nvimgcodecFutureDestroy(encode_future);
        nvimgcodecCodeStreamDestroy(code_stream);
        nvimgcodecImageDestroy(nv_image);

        if (encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            throw std::runtime_error("JPEG encoding failed: " +
                                     std::string(processing_status_to_string(encode_status)));
        }
        return output_buffer;
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_to_jpeg2k(
        const lfs::core::Tensor& image,
        void* cuda_stream,
        bool high_throughput) {

        using namespace lfs::core;

        if (!impl_->encoder) {
            throw std::runtime_error("JPEG2000 encoder not available");
        }

        std::lock_guard<std::mutex> lock(impl_->encoder_mutex);

        const auto& shape = image.shape();
        if (shape.rank() != 3) {
            throw std::runtime_error("Expected 3D tensor, got " + std::to_string(shape.rank()) + "D");
        }

        if (image.dtype() != DataType::Float32 || image.device() != Device::CUDA) {
            throw std::runtime_error("encode_to_jpeg2k expects a Float32 CUDA tensor");
        }

        const bool is_hwc = (shape[2] == 3);
        const bool is_chw = !is_hwc && (shape[0] == 3 && shape[1] > 3 && shape[2] > 3);
        const uint32_t height = static_cast<uint32_t>(is_chw ? shape[1] : shape[0]);
        const uint32_t width = static_cast<uint32_t>(is_chw ? shape[2] : shape[1]);
        const uint32_t channels = 3;

        // Scale float [0,1] -> uint16 [0, 65535] in HWC layout.
        // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
        const Tensor hwc_float = is_chw ? image.permute({1, 2, 0}).contiguous()
                                        : image.contiguous();
        Tensor hwc_uint16 = Tensor::empty(
            TensorShape({height, width, channels}), Device::CUDA, DataType::Float16);
        cuda::launch_float32_hwc_to_uint16_hwc(
            hwc_float.ptr<float>(),
            reinterpret_cast<uint16_t*>(hwc_uint16.data_ptr()),
            height, width, channels, static_cast<cudaStream_t>(cuda_stream));

        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        image_info.num_planes = 1;
        image_info.plane_info[0].height = height;
        image_info.plane_info[0].width = width;
        image_info.plane_info[0].num_channels = 3;
        image_info.plane_info[0].row_stride = static_cast<size_t>(width) * 3 * sizeof(uint16_t);
        image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        image_info.buffer = hwc_uint16.data_ptr();
        image_info.buffer_size = static_cast<size_t>(height) * width * 3 * sizeof(uint16_t);
        image_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

        nvimgcodecImage_t nv_image;
        auto status = nvimgcodecImageCreate(impl_->instance, &nv_image, &image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create image for encoding: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        std::vector<uint8_t> output_buffer;

        nvimgcodecImageInfo_t output_info{};
        output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        std::snprintf(output_info.codec_name, sizeof(output_info.codec_name), "%s", "jpeg2k");
        output_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        output_info.num_planes = 1;

        nvimgcodecCodeStream_t code_stream;
        status = nvimgcodecCodeStreamCreateToHostMem(
            impl_->instance, &code_stream, &output_buffer,
            [](void* ctx, size_t req_size) -> unsigned char* {
                auto* vec = static_cast<std::vector<uint8_t>*>(ctx);
                vec->resize(req_size);
                return vec->data();
            },
            &output_info);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("Failed to create output code stream for JPEG2000");
        }

        // Lossless JPEG 2000: reversible DWT 5/3
        nvimgcodecJpeg2kEncodeParams_t j2k_params{};
        j2k_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
        j2k_params.struct_size = sizeof(nvimgcodecJpeg2kEncodeParams_t);
        j2k_params.struct_next = nullptr;
        j2k_params.stream_type = NVIMGCODEC_JPEG2K_STREAM_J2K;
        j2k_params.prog_order = NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL;
        j2k_params.num_resolutions = 6;
        j2k_params.code_block_w = 64;
        j2k_params.code_block_h = 64;
        j2k_params.ht = high_throughput ? 1 : 0;

        nvimgcodecEncodeParams_t encode_params{};
        encode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
        encode_params.struct_size = sizeof(nvimgcodecEncodeParams_t);
        encode_params.struct_next = &j2k_params;
        encode_params.quality_value = 0.0f; // unused in lossless mode
        encode_params.quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;

        nvimgcodecFuture_t encode_future;
        status = nvimgcodecEncoderEncode(
            impl_->encoder, &nv_image, &code_stream, 1, &encode_params, &encode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(code_stream);
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("JPEG2000 encode launch failed");
        }

        nvimgcodecFutureWaitForAll(encode_future);

        nvimgcodecProcessingStatus_t encode_status;
        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status, &status_size);
        nvimgcodecFutureDestroy(encode_future);
        nvimgcodecCodeStreamDestroy(code_stream);
        nvimgcodecImageDestroy(nv_image);

        if (encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            throw std::runtime_error("JPEG2000 encoding failed: " +
                                     std::string(processing_status_to_string(encode_status)));
        }

        return output_buffer;
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_grayscale_to_jpeg2k(
        const lfs::core::Tensor& image,
        void* cuda_stream,
        bool high_throughput) {

        using namespace lfs::core;

        if (!impl_->encoder) {
            throw std::runtime_error("JPEG2000 encoder not available");
        }

        std::lock_guard<std::mutex> lock(impl_->encoder_mutex);

        const auto& shape = image.shape();
        if (shape.rank() != 2) {
            throw std::runtime_error("Expected 2D tensor, got " + std::to_string(shape.rank()) + "D");
        }
        if (image.dtype() != DataType::Float32 || image.device() != Device::CUDA) {
            throw std::runtime_error("encode_grayscale_to_jpeg2k expects a Float32 CUDA tensor");
        }

        const uint32_t height = static_cast<uint32_t>(shape[0]);
        const uint32_t width = static_cast<uint32_t>(shape[1]);
        constexpr uint32_t channels = 1;

        const Tensor hw_float = image.contiguous();
        Tensor hw_uint16 = Tensor::empty(
            TensorShape({height, width}), Device::CUDA, DataType::Float16);
        cuda::launch_float32_hwc_to_uint16_hwc(
            hw_float.ptr<float>(),
            reinterpret_cast<uint16_t*>(hw_uint16.data_ptr()),
            height, width, channels, static_cast<cudaStream_t>(cuda_stream));

        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
        image_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        image_info.num_planes = 1;
        image_info.plane_info[0].height = height;
        image_info.plane_info[0].width = width;
        image_info.plane_info[0].row_stride = static_cast<size_t>(width) * sizeof(uint16_t);
        image_info.plane_info[0].num_channels = 1;
        image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        image_info.plane_info[0].precision = 16;
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        image_info.buffer = hw_uint16.data_ptr();
        image_info.buffer_size = static_cast<size_t>(height) * width * sizeof(uint16_t);
        image_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

        nvimgcodecImage_t nv_image;
        auto status = nvimgcodecImageCreate(impl_->instance, &nv_image, &image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create image for grayscale JPEG2000 encoding: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        std::vector<uint8_t> output_buffer;

        nvimgcodecImageInfo_t output_info{};
        output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        std::snprintf(output_info.codec_name, sizeof(output_info.codec_name), "%s", "jpeg2k");
        output_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
        output_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        output_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        output_info.num_planes = 1;

        nvimgcodecCodeStream_t code_stream;
        status = nvimgcodecCodeStreamCreateToHostMem(
            impl_->instance, &code_stream, &output_buffer,
            [](void* ctx, size_t req_size) -> unsigned char* {
                auto* vec = static_cast<std::vector<uint8_t>*>(ctx);
                vec->resize(req_size);
                return vec->data();
            },
            &output_info);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("Failed to create output code stream for grayscale JPEG2000");
        }

        nvimgcodecJpeg2kEncodeParams_t j2k_params{};
        j2k_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
        j2k_params.struct_size = sizeof(nvimgcodecJpeg2kEncodeParams_t);
        j2k_params.struct_next = nullptr;
        j2k_params.stream_type = NVIMGCODEC_JPEG2K_STREAM_J2K;
        j2k_params.prog_order = NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL;
        j2k_params.num_resolutions = 6;
        j2k_params.code_block_w = 64;
        j2k_params.code_block_h = 64;
        j2k_params.ht = high_throughput ? 1 : 0;

        nvimgcodecEncodeParams_t encode_params{};
        encode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
        encode_params.struct_size = sizeof(nvimgcodecEncodeParams_t);
        encode_params.struct_next = &j2k_params;
        encode_params.quality_value = 0.0f;
        encode_params.quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;

        nvimgcodecFuture_t encode_future;
        status = nvimgcodecEncoderEncode(
            impl_->encoder, &nv_image, &code_stream, 1, &encode_params, &encode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(code_stream);
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("Grayscale JPEG2000 encode launch failed");
        }

        nvimgcodecFutureWaitForAll(encode_future);

        nvimgcodecProcessingStatus_t encode_status;
        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status, &status_size);
        nvimgcodecFutureDestroy(encode_future);
        nvimgcodecCodeStreamDestroy(code_stream);
        nvimgcodecImageDestroy(nv_image);

        if (encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            throw std::runtime_error("Grayscale JPEG2000 encoding failed: " +
                                     std::string(processing_status_to_string(encode_status)));
        }

        return output_buffer;
    }

    lfs::core::Tensor NvCodecImageLoader::decode_jpeg2k_16bit_from_memory_gpu(
        const std::vector<uint8_t>& jpeg2k_data,
        void* cuda_stream,
        const bool synchronize) {

        using namespace lfs::core;

        const size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];

        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        nvimgcodecCodeStream_t code_stream;
        auto status = nvimgcodecCodeStreamCreateFromHostMem(
            impl_->instance, &code_stream, jpeg2k_data.data(), jpeg2k_data.size());
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create JPEG2000 code stream from memory: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.struct_next = nullptr;

        status = nvimgcodecCodeStreamGetImageInfo(code_stream, &image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("Failed to get JPEG2000 image info: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        if (std::string(image_info.codec_name) != "jpeg2k") {
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("decode_jpeg2k_16bit_from_memory_gpu expects JPEG2000 input");
        }

        const auto sample_type = image_info.plane_info[0].sample_type;
        if (sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("Expected UINT16 JPEG2000 input, got sample type " +
                                     std::to_string(static_cast<int>(sample_type)));
        }

        const size_t num_components =
            std::max(static_cast<size_t>(image_info.num_planes),
                     static_cast<size_t>(image_info.plane_info[0].num_channels));
        if (num_components != 1 && num_components != 3) {
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("Expected grayscale or RGB JPEG2000 input, got " +
                                     std::to_string(num_components) + " components");
        }

        const size_t height = image_info.plane_info[0].height;
        const size_t width = image_info.plane_info[0].width;
        const bool is_grayscale = num_components == 1;

        cudaSetDevice(impl_->device_id);

        Tensor uint16_tensor = Tensor::empty(
            is_grayscale ? TensorShape({height, width})
                         : TensorShape({height, width, num_components}),
            Device::CUDA,
            DataType::Float16);
        if (cuda_stream) {
            uint16_tensor.set_stream(static_cast<cudaStream_t>(cuda_stream));
        }

        nvimgcodecImageInfo_t output_info{};
        output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_info.struct_next = nullptr;
        output_info.sample_format = is_grayscale ? NVIMGCODEC_SAMPLEFORMAT_P_Y
                                                 : NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_info.color_spec = is_grayscale ? NVIMGCODEC_COLORSPEC_GRAY
                                              : NVIMGCODEC_COLORSPEC_SRGB;
        output_info.chroma_subsampling = is_grayscale ? NVIMGCODEC_SAMPLING_GRAY
                                                      : NVIMGCODEC_SAMPLING_444;
        output_info.num_planes = 1;
        output_info.plane_info[0].height = height;
        output_info.plane_info[0].width = width;
        output_info.plane_info[0].row_stride =
            width * num_components * sizeof(uint16_t);
        output_info.plane_info[0].num_channels = num_components;
        output_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        output_info.plane_info[0].precision = 16;
        output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        output_info.buffer = uint16_tensor.data_ptr();
        output_info.buffer_size = height * width * num_components * sizeof(uint16_t);
        output_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

        nvimgcodecImage_t nv_image;
        {
            ScopedNvCodecVramProbe vram_probe(impl_->vram_account);
            status = nvimgcodecImageCreate(impl_->instance, &nv_image, &output_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                nvimgcodecCodeStreamDestroy(code_stream);
                throw std::runtime_error("Failed to create image descriptor for JPEG2000 decode: " +
                                         std::string(nvimgcodec_status_to_string(status)));
            }

            nvimgcodecDecodeParams_t decode_params{};
            decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
            decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
            decode_params.struct_next = nullptr;
            decode_params.apply_exif_orientation = 0;

            nvimgcodecFuture_t decode_future;
            status = nvimgcodecDecoderDecode(
                decoder,
                &code_stream,
                &nv_image,
                1,
                &decode_params,
                &decode_future);

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                nvimgcodecImageDestroy(nv_image);
                nvimgcodecCodeStreamDestroy(code_stream);
                throw std::runtime_error("JPEG2000 decode launch failed: " +
                                         std::string(nvimgcodec_status_to_string(status)));
            }

            status = nvimgcodecFutureWaitForAll(decode_future);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                nvimgcodecFutureDestroy(decode_future);
                nvimgcodecImageDestroy(nv_image);
                nvimgcodecCodeStreamDestroy(code_stream);
                throw std::runtime_error("Failed to wait for JPEG2000 decode completion: " +
                                         std::string(nvimgcodec_status_to_string(status)));
            }

            nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
            size_t status_size = 1;
            nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
            nvimgcodecFutureDestroy(decode_future);
            nvimgcodecImageDestroy(nv_image);
            nvimgcodecCodeStreamDestroy(code_stream);

            if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                throw std::runtime_error("JPEG2000 decoding failed: " +
                                         std::string(processing_status_to_string(decode_status)));
            }
        }

        Tensor output_tensor = Tensor::empty(
            is_grayscale ? TensorShape({height, width})
                         : TensorShape({height, width, num_components}),
            Device::CUDA,
            DataType::Float32);
        if (cuda_stream) {
            output_tensor.set_stream(static_cast<cudaStream_t>(cuda_stream));
        }
        cuda::launch_uint16_hwc_to_float32_hwc(
            reinterpret_cast<const uint16_t*>(uint16_tensor.data_ptr()),
            output_tensor.ptr<float>(),
            height, width, num_components, static_cast<cudaStream_t>(cuda_stream));

        if (synchronize && cuda_stream) {
            if (const cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream));
                err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
            }
        } else if (synchronize) {
            if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
            }
        }

        return output_tensor;
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::decode_jpeg2k_16bit_batch_from_spans(
        const std::vector<std::pair<const uint8_t*, size_t>>& jpeg2k_spans,
        void* cuda_stream,
        const bool synchronize) {

        using namespace lfs::core;

        if (jpeg2k_spans.empty()) {
            return {};
        }

        const size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];
        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        const size_t batch_size = jpeg2k_spans.size();
        std::vector<nvimgcodecCodeStream_t> code_streams(batch_size, nullptr);
        std::vector<nvimgcodecImage_t> nv_images(batch_size, nullptr);
        std::vector<Tensor> uint16_tensors(batch_size);
        std::vector<size_t> heights(batch_size);
        std::vector<size_t> widths(batch_size);
        std::vector<size_t> components(batch_size);
        nvimgcodecFuture_t decode_future = nullptr;

        const auto cleanup = [&] {
            if (decode_future) {
                nvimgcodecFutureDestroy(decode_future);
                decode_future = nullptr;
            }
            for (size_t i = 0; i < batch_size; ++i) {
                if (nv_images[i]) {
                    nvimgcodecImageDestroy(nv_images[i]);
                    nv_images[i] = nullptr;
                }
                if (code_streams[i]) {
                    nvimgcodecCodeStreamDestroy(code_streams[i]);
                    code_streams[i] = nullptr;
                }
            }
        };

        std::vector<nvimgcodecProcessingStatus_t> decode_statuses(
            batch_size, NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);

        try {
            cudaSetDevice(impl_->device_id);
            for (size_t i = 0; i < batch_size; ++i) {
                const auto [data, size] = jpeg2k_spans[i];
                if (!data || size == 0) {
                    throw std::runtime_error("Empty JPEG2000 batch input");
                }

                auto status = nvimgcodecCodeStreamCreateFromHostMem(
                    impl_->instance, &code_streams[i], data, size);
                if (status != NVIMGCODEC_STATUS_SUCCESS) {
                    throw std::runtime_error("Failed to create JPEG2000 batch code stream");
                }

                nvimgcodecImageInfo_t image_info{};
                image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
                image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
                status = nvimgcodecCodeStreamGetImageInfo(code_streams[i], &image_info);
                if (status != NVIMGCODEC_STATUS_SUCCESS ||
                    std::string(image_info.codec_name) != "jpeg2k") {
                    throw std::runtime_error("Invalid JPEG2000 batch input");
                }
                if (image_info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
                    throw std::runtime_error("JPEG2000 batch input is not UINT16");
                }

                components[i] = std::max(
                    static_cast<size_t>(image_info.num_planes),
                    static_cast<size_t>(image_info.plane_info[0].num_channels));
                if (components[i] != 1 && components[i] != 3) {
                    throw std::runtime_error("JPEG2000 batch input is not grayscale or RGB");
                }
                heights[i] = image_info.plane_info[0].height;
                widths[i] = image_info.plane_info[0].width;

                uint16_tensors[i] = Tensor::empty(
                    components[i] == 1
                        ? TensorShape({heights[i], widths[i]})
                        : TensorShape({heights[i], widths[i], components[i]}),
                    Device::CUDA,
                    DataType::Float16);
                if (cuda_stream) {
                    uint16_tensors[i].set_stream(static_cast<cudaStream_t>(cuda_stream));
                }

                nvimgcodecImageInfo_t output_info{};
                output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
                output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
                output_info.sample_format = components[i] == 1
                                                ? NVIMGCODEC_SAMPLEFORMAT_P_Y
                                                : NVIMGCODEC_SAMPLEFORMAT_I_RGB;
                output_info.color_spec = components[i] == 1
                                             ? NVIMGCODEC_COLORSPEC_GRAY
                                             : NVIMGCODEC_COLORSPEC_SRGB;
                output_info.chroma_subsampling = components[i] == 1
                                                     ? NVIMGCODEC_SAMPLING_GRAY
                                                     : NVIMGCODEC_SAMPLING_444;
                output_info.num_planes = 1;
                output_info.plane_info[0].height = heights[i];
                output_info.plane_info[0].width = widths[i];
                output_info.plane_info[0].row_stride =
                    widths[i] * components[i] * sizeof(uint16_t);
                output_info.plane_info[0].num_channels = components[i];
                output_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
                output_info.plane_info[0].precision = 16;
                output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                output_info.buffer = uint16_tensors[i].data_ptr();
                output_info.buffer_size = uint16_tensors[i].bytes();
                output_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

                status = nvimgcodecImageCreate(impl_->instance, &nv_images[i], &output_info);
                if (status != NVIMGCODEC_STATUS_SUCCESS) {
                    throw std::runtime_error("Failed to create JPEG2000 batch image descriptor");
                }
            }

            nvimgcodecDecodeParams_t decode_params{};
            decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
            decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
            decode_params.apply_exif_orientation = 0;

            {
                ScopedNvCodecVramProbe vram_probe(impl_->vram_account);
                const auto status = nvimgcodecDecoderDecode(
                    decoder,
                    code_streams.data(),
                    nv_images.data(),
                    static_cast<int>(batch_size),
                    &decode_params,
                    &decode_future);
                if (status != NVIMGCODEC_STATUS_SUCCESS) {
                    throw std::runtime_error("JPEG2000 batch decode launch failed");
                }
                if (nvimgcodecFutureWaitForAll(decode_future) != NVIMGCODEC_STATUS_SUCCESS) {
                    throw std::runtime_error("Failed to wait for JPEG2000 batch decode");
                }
                size_t status_size = batch_size;
                nvimgcodecFutureGetProcessingStatus(
                    decode_future, decode_statuses.data(), &status_size);
            }
            cleanup();

            std::vector<Tensor> outputs;
            outputs.reserve(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                if (decode_statuses[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    throw std::runtime_error("JPEG2000 batch item failed: " +
                                             std::string(processing_status_to_string(decode_statuses[i])));
                }

                auto output = Tensor::empty(
                    components[i] == 1
                        ? TensorShape({heights[i], widths[i]})
                        : TensorShape({heights[i], widths[i], components[i]}),
                    Device::CUDA,
                    DataType::Float32);
                if (cuda_stream) {
                    output.set_stream(static_cast<cudaStream_t>(cuda_stream));
                }
                cuda::launch_uint16_hwc_to_float32_hwc(
                    reinterpret_cast<const uint16_t*>(uint16_tensors[i].data_ptr()),
                    output.ptr<float>(),
                    heights[i], widths[i], components[i],
                    static_cast<cudaStream_t>(cuda_stream));
                outputs.push_back(std::move(output));
            }

            if (synchronize && cuda_stream) {
                if (const cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream));
                    err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
                }
            } else if (synchronize) {
                if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(err));
                }
            }
            return outputs;
        } catch (...) {
            cleanup();
            throw;
        }
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_grayscale_to_jpeg(
        const lfs::core::Tensor& image,
        const int quality,
        void* cuda_stream) {

        using namespace lfs::core;

        if (!impl_->encoder) {
            throw std::runtime_error("JPEG encoder not available");
        }

        std::lock_guard<std::mutex> lock(impl_->encoder_mutex);

        const auto& shape = image.shape();
        if (shape.rank() != 2) {
            throw std::runtime_error("Expected 2D tensor for grayscale, got " +
                                     std::to_string(shape.rank()) + "D");
        }

        const int height = static_cast<int>(shape[0]);
        const int width = static_cast<int>(shape[1]);

        // Convert to uint8 on GPU
        Tensor hw_uint8;
        if (image.dtype() == DataType::Float32) {
            hw_uint8 = (image * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8);
        } else {
            hw_uint8 = image.to(DataType::UInt8);
        }

        if (hw_uint8.device() != Device::CUDA) {
            hw_uint8 = hw_uint8.to(Device::CUDA);
        }
        hw_uint8 = hw_uint8.contiguous();

        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
        image_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        image_info.num_planes = 1;
        image_info.plane_info[0].height = height;
        image_info.plane_info[0].width = width;
        image_info.plane_info[0].row_stride = width;
        image_info.plane_info[0].num_channels = 1;
        image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        image_info.buffer = hw_uint8.data_ptr();
        image_info.buffer_size = height * width;
        image_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

        nvimgcodecImage_t nv_image;
        auto status = nvimgcodecImageCreate(impl_->instance, &nv_image, &image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create grayscale image for encoding: " +
                                     std::string(nvimgcodec_status_to_string(status)));
        }

        std::vector<uint8_t> output_buffer;

        nvimgcodecImageInfo_t output_info{};
        output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        std::snprintf(output_info.codec_name, sizeof(output_info.codec_name), "%s", "jpeg");
        output_info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;

        nvimgcodecCodeStream_t code_stream;
        status = nvimgcodecCodeStreamCreateToHostMem(
            impl_->instance, &code_stream, &output_buffer,
            [](void* ctx, size_t req_size) -> unsigned char* {
                auto* vec = static_cast<std::vector<uint8_t>*>(ctx);
                vec->resize(req_size);
                return vec->data();
            },
            &output_info);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("Failed to create output code stream for grayscale");
        }

        nvimgcodecEncodeParams_t encode_params{};
        encode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
        encode_params.struct_size = sizeof(nvimgcodecEncodeParams_t);
        encode_params.quality_value = static_cast<float>(quality);
        encode_params.quality_type = NVIMGCODEC_QUALITY_TYPE_QUALITY;

        nvimgcodecFuture_t encode_future;
        status = nvimgcodecEncoderEncode(
            impl_->encoder, &nv_image, &code_stream, 1, &encode_params, &encode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(code_stream);
            nvimgcodecImageDestroy(nv_image);
            throw std::runtime_error("Grayscale encode failed to start");
        }

        nvimgcodecFutureWaitForAll(encode_future);

        nvimgcodecProcessingStatus_t encode_status;
        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(encode_future, &encode_status, &status_size);
        nvimgcodecFutureDestroy(encode_future);
        nvimgcodecCodeStreamDestroy(code_stream);
        nvimgcodecImageDestroy(nv_image);

        if (encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            throw std::runtime_error("Grayscale JPEG encoding failed: " +
                                     std::string(processing_status_to_string(encode_status)));
        }

        LOG_DEBUG("Encoded grayscale JPEG: {}x{} -> {} bytes", width, height, output_buffer.size());
        return output_buffer;
    }

    std::vector<std::vector<uint8_t>> NvCodecImageLoader::encode_batch_rgb_to_jpeg(
        const std::vector<void*>& gpu_ptrs,
        const int width,
        const int height,
        const int quality,
        void* cuda_stream) {

        if (gpu_ptrs.empty()) {
            return {};
        }

        if (!impl_->encoder) {
            throw std::runtime_error("JPEG encoder not available");
        }

        std::lock_guard<std::mutex> lock(impl_->encoder_mutex);

        const int batch_size = static_cast<int>(gpu_ptrs.size());
        const size_t frame_size = static_cast<size_t>(width) * height * 3;

        std::vector<nvimgcodecImage_t> nv_images(batch_size);
        std::vector<nvimgcodecCodeStream_t> code_streams(batch_size);
        std::vector<std::vector<uint8_t>> output_buffers(batch_size);

        for (int i = 0; i < batch_size; i++) {
            nvimgcodecImageInfo_t image_info{};
            image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
            image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
            image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
            image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
            image_info.num_planes = 1;
            image_info.plane_info[0].height = height;
            image_info.plane_info[0].width = width;
            image_info.plane_info[0].row_stride = width * 3;
            image_info.plane_info[0].num_channels = 3;
            image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            image_info.buffer = gpu_ptrs[i];
            image_info.buffer_size = frame_size;
            image_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

            auto status = nvimgcodecImageCreate(impl_->instance, &nv_images[i], &image_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (int j = 0; j < i; j++) {
                    nvimgcodecImageDestroy(nv_images[j]);
                }
                throw std::runtime_error("Failed to create image for batch encoding");
            }

            nvimgcodecImageInfo_t output_info{};
            output_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
            output_info.struct_size = sizeof(nvimgcodecImageInfo_t);
            std::snprintf(output_info.codec_name, sizeof(output_info.codec_name), "%s", "jpeg");

            status = nvimgcodecCodeStreamCreateToHostMem(
                impl_->instance, &code_streams[i], &output_buffers[i],
                [](void* ctx, size_t req_size) -> unsigned char* {
                    auto* vec = static_cast<std::vector<uint8_t>*>(ctx);
                    vec->resize(req_size);
                    return vec->data();
                },
                &output_info);

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                for (int j = 0; j <= i; j++) {
                    nvimgcodecImageDestroy(nv_images[j]);
                    if (j < i)
                        nvimgcodecCodeStreamDestroy(code_streams[j]);
                }
                throw std::runtime_error("Failed to create output code stream for batch encoding");
            }
        }

        nvimgcodecEncodeParams_t encode_params{};
        encode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS;
        encode_params.struct_size = sizeof(nvimgcodecEncodeParams_t);
        encode_params.quality_value = static_cast<float>(quality);
        encode_params.quality_type = NVIMGCODEC_QUALITY_TYPE_QUALITY;

        nvimgcodecFuture_t encode_future;
        auto status = nvimgcodecEncoderEncode(
            impl_->encoder, nv_images.data(), code_streams.data(),
            batch_size, &encode_params, &encode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            for (int i = 0; i < batch_size; i++) {
                nvimgcodecCodeStreamDestroy(code_streams[i]);
                nvimgcodecImageDestroy(nv_images[i]);
            }
            throw std::runtime_error("Batch encode failed to start");
        }

        nvimgcodecFutureWaitForAll(encode_future);

        std::vector<nvimgcodecProcessingStatus_t> statuses(batch_size);
        size_t status_size = batch_size;
        nvimgcodecFutureGetProcessingStatus(encode_future, statuses.data(), &status_size);
        nvimgcodecFutureDestroy(encode_future);

        for (int i = 0; i < batch_size; i++) {
            nvimgcodecCodeStreamDestroy(code_streams[i]);
            nvimgcodecImageDestroy(nv_images[i]);
        }

        int failed_count = 0;
        for (int i = 0; i < batch_size; i++) {
            if (statuses[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                LOG_ERROR("Batch encode frame {} failed: {}", i,
                          processing_status_to_string(statuses[i]));
                output_buffers[i].clear();
                failed_count++;
            }
        }

        if (failed_count > 0) {
            LOG_WARN("Batch encode: {} of {} frames failed", failed_count, batch_size);
        }

        return output_buffers;
    }

} // namespace lfs::io
