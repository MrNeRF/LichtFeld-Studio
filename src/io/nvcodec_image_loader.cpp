/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/nvcodec_image_loader.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <condition_variable>
#include <cuda.h> // For CUcontext, cuCtxGetCurrent, cuCtxSetCurrent
#include <cuda_runtime.h>
#include <fstream>
#include <nvimgcodec.h>
#include <stdexcept>

namespace lfs::io {

    // Internal implementation (PIMPL pattern to hide nvImageCodec types)
    struct NvCodecImageLoader::Impl {
        nvimgcodecInstance_t instance = nullptr;
        std::vector<nvimgcodecDecoder_t> decoder_pool; // Pool of decoders (one per thread)
        std::vector<bool> decoder_available;           // Track which decoders are available
        std::mutex pool_mutex;                         // Protect pool access
        std::condition_variable pool_cv;               // Wait for available decoder
        int device_id = 0;
        bool fallback_enabled = true;

        // Acquire a decoder from the pool (blocks until one is available)
        size_t acquire_decoder() {
            std::unique_lock<std::mutex> lock(pool_mutex);
            pool_cv.wait(lock, [this] {
                return std::find(decoder_available.begin(), decoder_available.end(), true) != decoder_available.end();
            });

            // Find first available decoder
            for (size_t i = 0; i < decoder_available.size(); ++i) {
                if (decoder_available[i]) {
                    decoder_available[i] = false;
                    return i;
                }
            }
            return 0; // Should never reach here
        }

        // Release a decoder back to the pool
        void release_decoder(size_t idx) {
            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                decoder_available[idx] = true;
            }
            pool_cv.notify_one();
        }

        ~Impl() {
            // Clean up nvImageCodec resources
            for (auto decoder : decoder_pool) {
                if (decoder) {
                    nvimgcodecDecoderDestroy(decoder);
                }
            }
            if (instance) {
                nvimgcodecInstanceDestroy(instance);
            }
        }
    };

    NvCodecImageLoader::NvCodecImageLoader(const Options& options)
        : impl_(std::make_unique<Impl>()) {

        impl_->device_id = options.device_id;
        impl_->fallback_enabled = options.enable_fallback;

        // Get extensions path for portable builds
        auto extensions_dir = lfs::core::getExtensionsDir();
        std::string extensions_path_str;
        const char* extensions_path_ptr = nullptr;

        if (!extensions_dir.empty() && std::filesystem::exists(extensions_dir)) {
            extensions_path_str = extensions_dir.string();
            extensions_path_ptr = extensions_path_str.c_str();
            LOG_DEBUG("nvImageCodec extensions path: {}", extensions_path_str);
        }

        // Create nvImageCodec instance
        nvimgcodecInstanceCreateInfo_t create_info{
            NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            sizeof(nvimgcodecInstanceCreateInfo_t),
            nullptr,
            1,                    // load_builtin_modules
            1,                    // load_extension_modules
            extensions_path_ptr,  // extension_modules_path - set for portable builds
            0,                    // create_debug_messenger
            nullptr,              // debug_messenger_desc
            0,                    // message_severity
            0                     // message_category
        };

        auto status = nvimgcodecInstanceCreate(&impl_->instance, &create_info);
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create nvImageCodec instance");
        }

        // Create pool of decoders (one per worker thread)
        // Use pool_size based on expected worker threads (default: 16)
        size_t pool_size = options.decoder_pool_size > 0 ? options.decoder_pool_size : 16;
        impl_->decoder_pool.resize(pool_size);
        impl_->decoder_available.resize(pool_size, true);

        // Use default execution params (will use GPU backend if available via nvjpeg extension)
        nvimgcodecExecutionParams_t exec_params{
            NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS,
            sizeof(nvimgcodecExecutionParams_t),
            nullptr,
            nullptr,                     // device_allocator
            nullptr,                     // pinned_allocator
            options.max_num_cpu_threads, // max_num_cpu_threads
            nullptr,                     // executor
            options.device_id,           // device_id
            0,                           // pre_init
            0,                           // skip_pre_sync
            0,                           // num_backends
            nullptr                      // backends
        };

        for (size_t i = 0; i < pool_size; ++i) {
            status = nvimgcodecDecoderCreate(impl_->instance, &impl_->decoder_pool[i], &exec_params, nullptr);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create nvImageCodec decoder " + std::to_string(i));
            }
        }

        LOG_INFO("NvCodecImageLoader initialized with {} decoders (GPU backend preferred, CPU fallback enabled)", pool_size);
    }

    NvCodecImageLoader::~NvCodecImageLoader() = default;

    bool NvCodecImageLoader::is_available() {
        // Check if CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            return false;
        }

        // Try to create a minimal instance
        nvimgcodecInstance_t test_instance = nullptr;
        nvimgcodecInstanceCreateInfo_t create_info{
            NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            sizeof(nvimgcodecInstanceCreateInfo_t),
            nullptr,
            1,       // load_builtin_modules
            0,       // load_extension_modules
            nullptr, // extension_modules_path
            0,       // create_debug_messenger
            nullptr, // debug_messenger_desc
            0,       // message_severity
            0        // message_category
        };

        auto status = nvimgcodecInstanceCreate(&test_instance, &create_info);
        if (status == NVIMGCODEC_STATUS_SUCCESS && test_instance) {
            nvimgcodecInstanceDestroy(test_instance);
            return true;
        }
        return false;
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

        // Check file extension - nvImageCodec has PNG parser but no PNG decoder
        // Only JPEG, JPEG2000, BMP, TIFF, PNM have decoders
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".webp") {
            throw std::runtime_error("PNG/WebP not supported - no decoder available, use CPU fallback");
        }

        // Read file into memory and decode
        auto file_data = read_file(path);
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

            LOG_ERROR("Failed to get image info from JPEG blob ({} bytes): {} (status={})",
                      jpeg_data.size(), error_desc, static_cast<int>(status));
            throw std::runtime_error(std::string("Failed to get image info from memory: ") + error_desc);
        }

        int src_width = image_info.plane_info[0].width;
        int src_height = image_info.plane_info[0].height;

        // Calculate target dimensions based on resize_factor
        int target_width = src_width;
        int target_height = src_height;
        if (resize_factor > 1) {
            target_width /= resize_factor;
            target_height /= resize_factor;
        }

        LOG_DEBUG("Image info: {}x{} → {}x{} (resize factor {})",
                  src_width, src_height, target_width, target_height, resize_factor);

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
            const char* status_str = decode_status == NVIMGCODEC_PROCESSING_STATUS_FAIL ? "FAIL" : decode_status == NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED      ? "IMAGE_CORRUPTED"
                                                                                               : decode_status == NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED      ? "CODEC_UNSUPPORTED"
                                                                                               : decode_status == NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED    ? "BACKEND_UNSUPPORTED"
                                                                                               : decode_status == NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED ? "CODESTREAM_UNSUPPORTED"
                                                                                                                                                                      : "UNKNOWN";
            throw std::runtime_error(std::string("Decode failed: ") + status_str);
        }

        // Apply Lanczos resize if needed
        Tensor output_tensor;

        if (resize_factor > 1) {
            // Apply Lanczos GPU resize: [H,W,3] uint8 → [C,H,W] float32
            LOG_DEBUG("Applying Lanczos resize: {}x{} → {}x{}",
                      src_width, src_height, target_width, target_height);

            output_tensor = lanczos_resize(
                uint8_tensor,
                target_height,
                target_width,
                2, // kernel_size=2 (good balance of quality and speed)
                static_cast<cudaStream_t>(cuda_stream));

            LOG_DEBUG("Successfully decoded+resized image to GPU: {}x{} → {}x{}",
                      src_width, src_height, target_width, target_height);
        } else {
            // No resize needed - convert [H,W,3] uint8 to [C,H,W] float32
            output_tensor = uint8_tensor.to(DataType::Float32) / 255.0f;
            output_tensor = output_tensor.permute({2, 0, 1}).contiguous(); // [H,W,3] → [3,H,W]

            LOG_DEBUG("Successfully decoded image to GPU: {}x{} (no resize)",
                      src_width, src_height);
        }

        // Restore CUDA context
        if (saved_context != nullptr) {
            cuCtxSetCurrent(saved_context);
        }

        return output_tensor;
    }

    lfs::core::Tensor NvCodecImageLoader::resize_if_needed(
        lfs::core::Tensor input,
        int width,
        int height,
        int resize_factor,
        int max_width) {

        // Calculate target dimensions
        int target_width = width;
        int target_height = height;

        if (resize_factor > 1) {
            target_width /= resize_factor;
            target_height /= resize_factor;
        }

        if (max_width > 0 && (width > max_width || height > max_width)) {
            float scale;
            if (width > height) {
                scale = static_cast<float>(max_width) / width;
            } else {
                scale = static_cast<float>(max_width) / height;
            }
            target_width = static_cast<int>(width * scale);
            target_height = static_cast<int>(height * scale);
        }

        // If no resize needed, return as-is
        if (target_width == width && target_height == height) {
            return input;
        }

        LOG_DEBUG("Resizing from {}x{} to {}x{}", width, height, target_width, target_height);

        // TODO: Implement GPU-based bilinear resize using CUDA or NPP
        // For now, just return the input (resize will be handled by fallback)
        LOG_WARN("GPU resize not yet implemented, returning original size");
        return input;
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::load_images_batch_gpu(
        const std::vector<std::filesystem::path>& paths,
        int resize_factor,
        int max_width) {

        // Phase 2: TODO - implement batch decoding
        LOG_WARN("Batch loading not yet implemented, falling back to sequential");

        std::vector<lfs::core::Tensor> results;
        results.reserve(paths.size());

        for (const auto& path : paths) {
            results.push_back(load_image_gpu(path, resize_factor, max_width));
        }

        return results;
    }

} // namespace lfs::io
