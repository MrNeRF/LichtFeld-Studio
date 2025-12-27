/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/nvcodec_image_loader.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "cuda/image_format_kernels.cuh"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <cuda.h> // For CUcontext, cuCtxGetCurrent, cuCtxSetCurrent
#include <cuda_runtime.h>
#include <fstream>
#include <nvimgcodec.h>
#include <sstream>
#include <stdexcept>

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

            LOG_INFO("[nvImageCodec Diagnostics] Extensions directory: {}", ext_dir.string());

            if (!std::filesystem::exists(ext_dir)) {
                LOG_ERROR("[nvImageCodec Diagnostics] Extensions directory does not exist!");
                return;
            }

            // List extension files
            std::vector<std::string> extension_files;
            std::vector<std::string> dll_files;
            std::error_code ec;

            for (const auto& entry : std::filesystem::directory_iterator(ext_dir, ec)) {
                if (ec) {
                    LOG_ERROR("[nvImageCodec Diagnostics] Failed to iterate extensions dir: {}", ec.message());
                    break;
                }
                const auto& path = entry.path();
                std::string ext = path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

#ifdef _WIN32
                if (ext == ".dll") {
                    dll_files.push_back(path.filename().string());
                }
#else
                if (ext == ".so") {
                    dll_files.push_back(path.filename().string());
                }
#endif
            }

            if (dll_files.empty()) {
#ifdef _WIN32
                LOG_WARN("[nvImageCodec Diagnostics] No .dll extension files found in {}", ext_dir.string());
#else
                LOG_WARN("[nvImageCodec Diagnostics] No .so extension files found in {}", ext_dir.string());
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
                extensions_path_str = extensions_dir.string();
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

    } // anonymous namespace

    struct NvCodecImageLoader::Impl {
        nvimgcodecInstance_t instance = nullptr;
        std::vector<nvimgcodecDecoder_t> decoder_pool;
        std::vector<bool> decoder_available;
        std::mutex pool_mutex;
        std::condition_variable pool_cv;
        nvimgcodecEncoder_t encoder = nullptr;
        std::mutex encoder_mutex;
        int device_id = 0;
        bool fallback_enabled = true;

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

        ~Impl() {
            // Check if CUDA context is still valid before cleanup
            CUcontext current_ctx = nullptr;
            CUresult ctx_result = cuCtxGetCurrent(&current_ctx);

            // Only cleanup if CUDA runtime is still initialized and context is valid
            if (ctx_result == CUDA_SUCCESS && current_ctx != nullptr) {
                // Destroy encoder first (if created)
                if (encoder) {
                    nvimgcodecEncoderDestroy(encoder);
                    encoder = nullptr;
                }

                // Destroy all decoders in the pool
                for (auto& decoder : decoder_pool) {
                    if (decoder) {
                        nvimgcodecDecoderDestroy(decoder);
                        decoder = nullptr;
                    }
                }
                decoder_pool.clear();
                decoder_available.clear();

                // Destroy the instance last
                if (instance) {
                    nvimgcodecInstanceDestroy(instance);
                    instance = nullptr;
                }
            }
            // If CUDA context is invalid, skip cleanup to avoid crashes
            // This can happen during program shutdown when CUDA runtime is already torn down
        }
    };

    namespace {
        constexpr size_t DEFAULT_POOL_SIZE = 8;
        constexpr int DEFAULT_LANCZOS_KERNEL = 2;
    } // namespace

    NvCodecImageLoader::NvCodecImageLoader(const Options& options)
        : impl_(std::make_unique<Impl>()) {

        LOG_INFO("[NvCodecImageLoader] Initializing: device={}, pool={}, fallback={}",
                 options.device_id, options.decoder_pool_size, options.enable_fallback);

        impl_->device_id = options.device_id;
        impl_->fallback_enabled = options.enable_fallback;

        const auto extensions_dir = lfs::core::getExtensionsDir();
        std::string extensions_path_str;
        const char* extensions_path_ptr = nullptr;

        if (!extensions_dir.empty() && std::filesystem::exists(extensions_dir)) {
            extensions_path_str = extensions_dir.string();
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

        const size_t pool_size = options.decoder_pool_size > 0 ? options.decoder_pool_size : DEFAULT_POOL_SIZE;
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
    }

    NvCodecImageLoader::~NvCodecImageLoader() = default;

    bool NvCodecImageLoader::is_available() {
        // Run comprehensive diagnostics - this will log all relevant info
        return check_nvimgcodec_availability_with_diagnostics();
    }

    std::vector<uint8_t> NvCodecImageLoader::read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + path.string());
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file: " + path.string());
        }

        return buffer;
    }

    lfs::core::Tensor NvCodecImageLoader::load_image_gpu(
        const std::filesystem::path& path,
        int resize_factor,
        int max_width,
        void* cuda_stream) {

        LOG_DEBUG("NvCodecImageLoader: Loading {}", path.string());

        // Read file into memory first
        auto file_data = read_file(path);

        // Check magic bytes to determine actual format (not extension - files can be mislabeled)
        // PNG magic: 0x89 0x50 0x4E 0x47 (‰PNG)
        // WebP magic: RIFF....WEBP
        // JPEG magic: 0xFF 0xD8 0xFF
        if (file_data.size() >= 8) {
            // Check for PNG magic
            if (file_data[0] == 0x89 && file_data[1] == 0x50 &&
                file_data[2] == 0x4E && file_data[3] == 0x47) {
                throw std::runtime_error("PNG format detected (by magic bytes) - use CPU fallback");
            }
            // Check for WebP magic (RIFF....WEBP)
            if (file_data[0] == 'R' && file_data[1] == 'I' &&
                file_data[2] == 'F' && file_data[3] == 'F' &&
                file_data.size() >= 12 &&
                file_data[8] == 'W' && file_data[9] == 'E' &&
                file_data[10] == 'B' && file_data[11] == 'P') {
                throw std::runtime_error("WebP format detected (by magic bytes) - use CPU fallback");
            }
        }

        return load_image_from_memory_gpu(file_data, resize_factor, max_width, cuda_stream);
    }

    lfs::core::Tensor NvCodecImageLoader::load_image_from_memory_gpu(
        const std::vector<uint8_t>& jpeg_data,
        int resize_factor,
        [[maybe_unused]] int max_width,
        void* cuda_stream) {

        // Decode at full resolution, then resize with Lanczos if needed

        // Acquire a decoder from the pool
        size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];

        // Auto-release decoder on scope exit
        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        // Create code stream from memory
        nvimgcodecCodeStream_t code_stream;
        auto status = nvimgcodecCodeStreamCreateFromHostMem(
            impl_->instance, &code_stream, jpeg_data.data(), jpeg_data.size());
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create code stream from memory");
        }

        // Get image info (dimensions, format)
        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.struct_next = nullptr;

        status = nvimgcodecCodeStreamGetImageInfo(code_stream, &image_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(code_stream);

            // Error for debugging
            const char* error_desc = "unknown";
            switch (status) {
            case NVIMGCODEC_STATUS_INVALID_PARAMETER: error_desc = "invalid parameter"; break;
            case NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED: error_desc = "unsupported codestream format"; break;
            case NVIMGCODEC_STATUS_BAD_CODESTREAM: error_desc = "corrupted/bad codestream"; break;
            default: error_desc = "unknown error"; break;
            }

            // Check if this might be a non-JPEG format by magic bytes
            const char* format_hint = "unknown";
            if (jpeg_data.size() >= 4) {
                if (jpeg_data[0] == 0x89 && jpeg_data[1] == 0x50 &&
                    jpeg_data[2] == 0x4E && jpeg_data[3] == 0x47) {
                    format_hint = "PNG (not supported by GPU decoder)";
                } else if (jpeg_data[0] == 'R' && jpeg_data[1] == 'I' &&
                           jpeg_data[2] == 'F' && jpeg_data[3] == 'F') {
                    format_hint = "WebP (not supported by GPU decoder)";
                } else if (jpeg_data[0] == 0xFF && jpeg_data[1] == 0xD8) {
                    format_hint = "JPEG";
                }
            }
            LOG_ERROR("Failed to decode image blob ({} bytes, format: {}): {} (status={})",
                      jpeg_data.size(), format_hint, error_desc, static_cast<int>(status));
            throw std::runtime_error(std::string("Failed to decode image from memory: ") + error_desc);
        }

        int src_width = image_info.plane_info[0].width;
        int src_height = image_info.plane_info[0].height;

        // Calculate target dimensions based on resize_factor and max_width
        int target_width = src_width;
        int target_height = src_height;
        if (resize_factor > 1) {
            target_width /= resize_factor;
            target_height /= resize_factor;
        }
        if (max_width > 0 && (target_width > max_width || target_height > max_width)) {
            const float scale = (target_width > target_height)
                                    ? static_cast<float>(max_width) / target_width
                                    : static_cast<float>(max_width) / target_height;
            target_width = static_cast<int>(target_width * scale);
            target_height = static_cast<int>(target_height * scale);
        }
        const bool needs_resize = (target_width != src_width || target_height != src_height);

        LOG_DEBUG("Image info: {}x{} -> {}x{} (resize_factor={}, max_width={})",
                  src_width, src_height, target_width, target_height, resize_factor, max_width);

        // Save/restore CUDA context for thread safety
        CUcontext saved_context = nullptr;
        cuCtxGetCurrent(&saved_context);
        cudaSetDevice(0);

        // Decode at full resolution
        using namespace lfs::core;
        auto uint8_tensor = Tensor::empty(
            TensorShape({static_cast<size_t>(src_height), static_cast<size_t>(src_width), 3}),
            Device::CUDA,
            DataType::UInt8);

        void* gpu_uint8_buffer = uint8_tensor.data_ptr();
        size_t decoded_size = src_width * src_height * 3;

        // Create nvImageCodec image descriptor for GPU buffer
        nvimgcodecImage_t nv_image;
        nvimgcodecImageInfo_t output_info = image_info;
        output_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB; // Interleaved RGB
        output_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;

        // Single plane with 3 channels
        output_info.num_planes = 1;
        output_info.plane_info[0].height = src_height;
        output_info.plane_info[0].width = src_width;
        output_info.plane_info[0].row_stride = src_width * 3;
        output_info.plane_info[0].num_channels = 3;
        output_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;

        output_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        output_info.buffer = gpu_uint8_buffer;
        output_info.buffer_size = decoded_size;
        output_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

        status = nvimgcodecImageCreate(impl_->instance, &nv_image, &output_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            // Note: gpu_uint8_buffer is managed by uint8_tensor, don't manually free it
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("Failed to create image descriptor");
        }

        // Decode using the acquired decoder
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;

        nvimgcodecFuture_t decode_future;
        status = nvimgcodecDecoderDecode(
            decoder, // Use decoder from pool
            &code_stream,
            &nv_image,
            1, // batch_size
            &decode_params,
            &decode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            // Note: gpu_uint8_buffer is managed by uint8_tensor, don't manually free it
            nvimgcodecImageDestroy(nv_image);
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("Failed to decode image from memory");
        }

        // Wait for decode to complete (this waits for nvImageCodec's internal operations)
        status = nvimgcodecFutureWaitForAll(decode_future);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecFutureDestroy(decode_future);
            nvimgcodecImageDestroy(nv_image);
            nvimgcodecCodeStreamDestroy(code_stream);
            throw std::runtime_error("Failed to wait for decode completion");
        }

        // Get processing status (for single image decode)
        nvimgcodecProcessingStatus_t decode_status;
        size_t status_size = 1; // We're decoding 1 image
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);

        bool decode_success = (decode_status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS);

        // Cleanup nvImageCodec resources (must be done after waiting)
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(nv_image);
        nvimgcodecCodeStreamDestroy(code_stream);

        if (!decode_success) {
            // uint8_tensor will be automatically freed when it goes out of scope
            const char* status_str = processing_status_to_string(decode_status);

            // Log comprehensive decode failure information
            LOG_ERROR("[NvCodecImageLoader] Decode FAILED with status: {} ({})",
                      status_str, static_cast<int>(decode_status));

            // Provide context-specific help for common issues
            if (decode_status == NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED) {
                LOG_ERROR("[NvCodecImageLoader] CODEC_UNSUPPORTED: No suitable decoder backend found");
                LOG_ERROR("[NvCodecImageLoader]   This usually means:");
                LOG_ERROR("[NvCodecImageLoader]   1. nvJPEG extension not loaded (check extensions directory)");
                LOG_ERROR("[NvCodecImageLoader]   2. Extension DLLs have missing dependencies");
                LOG_ERROR("[NvCodecImageLoader]   3. GPU doesn't support required features");
                LOG_ERROR("[NvCodecImageLoader]   Image info: {}x{}, {} bytes compressed",
                          src_width, src_height, jpeg_data.size());
            } else if (decode_status == NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED) {
                LOG_ERROR("[NvCodecImageLoader] BACKEND_UNSUPPORTED: Backend cannot handle this image format");
                LOG_ERROR("[NvCodecImageLoader]   Check if image is valid JPEG (magic: 0x{:02X}{:02X})",
                          jpeg_data.size() >= 2 ? jpeg_data[0] : 0,
                          jpeg_data.size() >= 2 ? jpeg_data[1] : 0);
            } else if (decode_status == NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED) {
                LOG_ERROR("[NvCodecImageLoader] IMAGE_CORRUPTED: Image data appears to be corrupt");
                LOG_ERROR("[NvCodecImageLoader]   Size: {} bytes, first bytes: {:02X} {:02X} {:02X} {:02X}",
                          jpeg_data.size(),
                          jpeg_data.size() >= 1 ? jpeg_data[0] : 0,
                          jpeg_data.size() >= 2 ? jpeg_data[1] : 0,
                          jpeg_data.size() >= 3 ? jpeg_data[2] : 0,
                          jpeg_data.size() >= 4 ? jpeg_data[3] : 0);
            } else if (decode_status == NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED) {
                LOG_ERROR("[NvCodecImageLoader] CODESTREAM_UNSUPPORTED: Image encoding not supported");
                LOG_ERROR("[NvCodecImageLoader]   The JPEG may use an unsupported color space or encoding mode");
            }

            throw std::runtime_error(std::string("Decode failed: ") + status_str);
        }

        Tensor output_tensor;
        if (needs_resize) {
            // lanczos_resize handles conversion internally
            output_tensor = lanczos_resize(uint8_tensor, target_height, target_width,
                                           DEFAULT_LANCZOS_KERNEL, static_cast<cudaStream_t>(cuda_stream));
        } else {
            // Fused uint8 HWC -> float32 CHW normalized (single kernel!)
            auto shape = uint8_tensor.shape();
            size_t H = shape[0], W = shape[1], C = shape[2];
            output_tensor = Tensor::zeros(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);
            cuda::launch_uint8_hwc_to_float32_chw(
                reinterpret_cast<const uint8_t*>(uint8_tensor.data_ptr()),
                reinterpret_cast<float*>(output_tensor.data_ptr()),
                H, W, C, static_cast<cudaStream_t>(cuda_stream));
        }

        // Free uint8 tensor immediately - no longer needed
        uint8_tensor = Tensor();

        if (saved_context) {
            cuCtxSetCurrent(saved_context);
        }
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

        // Acquire a decoder from the pool
        size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];

        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        // Save/restore CUDA context
        CUcontext saved_context = nullptr;
        cuCtxGetCurrent(&saved_context);
        cudaSetDevice(impl_->device_id);

        // Create code streams for all images
        std::vector<nvimgcodecCodeStream_t> code_streams(batch_size);
        std::vector<nvimgcodecImageInfo_t> image_infos(batch_size);
        std::vector<Tensor> uint8_tensors(batch_size);
        std::vector<nvimgcodecImage_t> nv_images(batch_size);

        // Phase 1: Create code streams and get image info
        for (size_t i = 0; i < batch_size; ++i) {
            auto status = nvimgcodecCodeStreamCreateFromHostMem(
                impl_->instance, &code_streams[i],
                jpeg_blobs[i].data(), jpeg_blobs[i].size());

            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                // Cleanup already created streams
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

        // Phase 2: Allocate GPU buffers and create nvimgcodec images
        for (size_t i = 0; i < batch_size; ++i) {
            int width = image_infos[i].plane_info[0].width;
            int height = image_infos[i].plane_info[0].height;

            uint8_tensors[i] = Tensor::empty(
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), 3}),
                Device::CUDA,
                DataType::UInt8);

            nvimgcodecImageInfo_t output_info = image_infos[i];
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
            output_info.buffer_size = height * width * 3;
            output_info.cuda_stream = static_cast<cudaStream_t>(cuda_stream);

            auto status = nvimgcodecImageCreate(impl_->instance, &nv_images[i], &output_info);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                // Cleanup
                for (size_t j = 0; j < i; ++j) {
                    nvimgcodecImageDestroy(nv_images[j]);
                }
                for (size_t j = 0; j < batch_size; ++j) {
                    nvimgcodecCodeStreamDestroy(code_streams[j]);
                }
                throw std::runtime_error("Failed to create image descriptor for image " + std::to_string(i));
            }
        }

        // Phase 3: Batch decode all images in one call!
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;

        nvimgcodecFuture_t decode_future;
        auto status = nvimgcodecDecoderDecode(
            decoder,
            code_streams.data(),
            nv_images.data(),
            batch_size, // All images in one call!
            &decode_params,
            &decode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            for (size_t i = 0; i < batch_size; ++i) {
                nvimgcodecImageDestroy(nv_images[i]);
                nvimgcodecCodeStreamDestroy(code_streams[i]);
            }
            throw std::runtime_error("Batch decode failed");
        }

        // Wait for decode to complete
        status = nvimgcodecFutureWaitForAll(decode_future);

        // Get processing status for all images
        std::vector<nvimgcodecProcessingStatus_t> decode_statuses(batch_size);
        size_t status_size = batch_size;
        nvimgcodecFutureGetProcessingStatus(decode_future, decode_statuses.data(), &status_size);

        // Cleanup nvimgcodec resources
        nvimgcodecFutureDestroy(decode_future);
        for (size_t i = 0; i < batch_size; ++i) {
            nvimgcodecImageDestroy(nv_images[i]);
            nvimgcodecCodeStreamDestroy(code_streams[i]);
        }

        // Phase 4: Convert to output format using FUSED kernel
        // This avoids 2-3 intermediate GPU allocations per image!
        std::vector<Tensor> results;
        results.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            if (decode_statuses[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                LOG_WARN("[NvCodecImageLoader] Batch decode image {} failed: {}",
                         i, processing_status_to_string(decode_statuses[i]));
                results.push_back(Tensor());
                uint8_tensors[i] = Tensor(); // Free immediately
                continue;
            }

            // Fused uint8 HWC -> float32 CHW normalized (single kernel, no intermediates!)
            auto shape = uint8_tensors[i].shape();
            size_t H = shape[0], W = shape[1], C = shape[2];

            // Allocate output directly in final format
            auto output = Tensor::zeros(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);

            // Launch fused kernel
            cuda::launch_uint8_hwc_to_float32_chw(
                reinterpret_cast<const uint8_t*>(uint8_tensors[i].data_ptr()),
                reinterpret_cast<float*>(output.data_ptr()),
                H, W, C, nullptr);

            // CRITICAL: Free uint8 tensor immediately after use to reduce peak VRAM!
            uint8_tensors[i] = Tensor();

            results.push_back(std::move(output));
        }

        // Ensure all conversion kernels complete before returning tensors
        cudaDeviceSynchronize();

        if (saved_context) {
            cuCtxSetCurrent(saved_context);
        }

        LOG_DEBUG("[NvCodecImageLoader] Batch decode complete: {} images", batch_size);
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
        LOG_DEBUG("[NvCodecImageLoader] Batch decoding {} images from spans (zero-copy)", batch_size);

        // Acquire a decoder from the pool
        size_t decoder_idx = impl_->acquire_decoder();
        nvimgcodecDecoder_t decoder = impl_->decoder_pool[decoder_idx];

        struct DecoderGuard {
            NvCodecImageLoader::Impl* impl;
            size_t idx;
            ~DecoderGuard() { impl->release_decoder(idx); }
        } guard{impl_.get(), decoder_idx};

        // Save/restore CUDA context
        CUcontext saved_context = nullptr;
        cuCtxGetCurrent(&saved_context);
        cudaSetDevice(impl_->device_id);

        // Create code streams for all images
        std::vector<nvimgcodecCodeStream_t> code_streams(batch_size);
        std::vector<nvimgcodecImageInfo_t> image_infos(batch_size);
        std::vector<Tensor> uint8_tensors(batch_size);
        std::vector<nvimgcodecImage_t> nv_images(batch_size);

        // Phase 1: Create code streams and get image info
        for (size_t i = 0; i < batch_size; ++i) {
            auto status = nvimgcodecCodeStreamCreateFromHostMem(
                impl_->instance, &code_streams[i],
                jpeg_spans[i].first, jpeg_spans[i].second); // Use span data directly

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

        // Phase 2: Allocate output tensors and create image descriptors
        for (size_t i = 0; i < batch_size; ++i) {
            int width = image_infos[i].plane_info[0].width;
            int height = image_infos[i].plane_info[0].height;

            // Allocate output tensor on GPU
            uint8_tensors[i] = Tensor::zeros(
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), 3}),
                Device::CUDA,
                DataType::UInt8);

            // Create nvimgcodec image descriptor pointing to our tensor
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

            auto status = nvimgcodecImageCreate(impl_->instance, &nv_images[i], &out_info);
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

        // Phase 3: Batch decode
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;

        nvimgcodecFuture_t decode_future;
        auto status = nvimgcodecDecoderDecode(
            decoder,
            code_streams.data(),
            nv_images.data(),
            batch_size,
            &decode_params,
            &decode_future);

        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            for (size_t i = 0; i < batch_size; ++i) {
                nvimgcodecImageDestroy(nv_images[i]);
                nvimgcodecCodeStreamDestroy(code_streams[i]);
            }
            throw std::runtime_error("Batch decode failed");
        }

        // Wait for decode
        status = nvimgcodecFutureWaitForAll(decode_future);

        std::vector<nvimgcodecProcessingStatus_t> decode_statuses(batch_size);
        size_t status_size = batch_size;
        nvimgcodecFutureGetProcessingStatus(decode_future, decode_statuses.data(), &status_size);

        // Cleanup
        nvimgcodecFutureDestroy(decode_future);
        for (size_t i = 0; i < batch_size; ++i) {
            nvimgcodecImageDestroy(nv_images[i]);
            nvimgcodecCodeStreamDestroy(code_streams[i]);
        }

        // Phase 4: Convert to output format using FUSED kernel
        std::vector<Tensor> results;
        results.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            if (decode_statuses[i] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                LOG_WARN("[NvCodecImageLoader] Batch decode image {} failed", i);
                results.push_back(Tensor());
                uint8_tensors[i] = Tensor(); // Free immediately
                continue;
            }

            // Fused uint8 HWC -> float32 CHW normalized (single kernel!)
            auto shape = uint8_tensors[i].shape();
            size_t H = shape[0], W = shape[1], C = shape[2];
            auto output = Tensor::zeros(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);
            cuda::launch_uint8_hwc_to_float32_chw(
                reinterpret_cast<const uint8_t*>(uint8_tensors[i].data_ptr()),
                reinterpret_cast<float*>(output.data_ptr()),
                H, W, C, nullptr);

            // CRITICAL: Free uint8 tensor immediately to reduce peak VRAM!
            uint8_tensors[i] = Tensor();

            results.push_back(std::move(output));
        }

        // Ensure all conversion kernels complete before returning tensors
        cudaDeviceSynchronize();

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

        // Detect CHW vs HWC format:
        // - Prioritize HWC when shape[2]==3 (channels last is standard for images)
        // - Fall back to CHW when shape[0]==3 and dimensions are clearly spatial
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
        encode_params.quality_type = NVIMGCODEC_QUALITY_TYPE_DEFAULT;

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

} // namespace lfs::io
