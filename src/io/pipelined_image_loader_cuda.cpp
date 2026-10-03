/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/undistort/undistort.hpp"
#include "cuda/image_format_kernels.cuh"
#include "pipelined_image_loader_ring.hpp"

#include "core/cuda_error.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/shared_image_ops.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "image_execution_cuda.hpp"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>
#include <stb_image.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <tuple>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <unistd.h>
#endif

namespace lfs::io {

    namespace {

        constexpr int DEFAULT_DECODER_POOL_SIZE = 8;

        // Cold workers scale with sidecar decode demand, but their nvimagecodec
        // Cap concurrent GPU decodes so decode pressure stays
        // at pre-sidecar levels regardless of the cold pool size.
        std::counting_semaphore<>& nvcodec_decode_slots() {
            static std::counting_semaphore<> slots{2};
            return slots;
        }

        class NvcodecSlotGuard {
        public:
            NvcodecSlotGuard() { nvcodec_decode_slots().acquire(); }
            ~NvcodecSlotGuard() { nvcodec_decode_slots().release(); }
            NvcodecSlotGuard(const NvcodecSlotGuard&) = delete;
            NvcodecSlotGuard& operator=(const NvcodecSlotGuard&) = delete;
        };

        std::counting_semaphore<>& undistort_slots() {
            static std::counting_semaphore<> slots{1};
            return slots;
        }

        class UndistortSlotGuard {
        public:
            UndistortSlotGuard() { undistort_slots().acquire(); }
            ~UndistortSlotGuard() { undistort_slots().release(); }
            UndistortSlotGuard(const UndistortSlotGuard&) = delete;
            UndistortSlotGuard& operator=(const UndistortSlotGuard&) = delete;
        };

        [[nodiscard]] size_t tensor_reserved_bytes(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid()) {
                return 0;
            }

            if (tensor.capacity() == 0 || tensor.ndim() == 0) {
                return tensor.bytes();
            }

            size_t row_elems = 1;
            if (tensor.ndim() > 1) {
                for (size_t dim = 1; dim < tensor.ndim(); ++dim) {
                    row_elems *= tensor.shape()[dim];
                }
            }

            return tensor.capacity() * row_elems * lfs::core::dtype_size(tensor.dtype());
        }

        void subtract_clamped(std::atomic<size_t>& value, const size_t amount) {
            if (amount == 0) {
                return;
            }

            auto current = value.load(std::memory_order_relaxed);
            while (current > 0) {
                const size_t next = current > amount ? current - amount : 0;
                if (value.compare_exchange_weak(
                        current, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                    return;
                }
            }
        }

        [[nodiscard]] std::string legacy_message_from(const lfs::Error& error) {
            return error.user_message().empty() ? std::string(error.detail())
                                                : std::string(error.user_message());
        }

        struct NvCodecLoaderCacheEntry {
            std::shared_ptr<NvCodecImageLoader> instance;
            size_t owner_count = 0;
        };

        [[nodiscard]] size_t normalize_nvcodec_pool_size(size_t decoder_pool_size) {
            return decoder_pool_size > 0 ? decoder_pool_size : DEFAULT_DECODER_POOL_SIZE;
        }

        std::filesystem::path get_temp_folder() {
#ifdef _WIN32
            const char* temp = std::getenv("TEMP");
            if (!temp)
                temp = std::getenv("TMP");
            return temp ? std::filesystem::path(temp) : std::filesystem::path("C:/Temp");
#else
            return std::filesystem::path("/tmp");
#endif
        }

        [[nodiscard]] std::uint64_t current_process_id() {
#ifdef _WIN32
            return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
            return static_cast<std::uint64_t>(getpid());
#endif
        }

        [[nodiscard]] bool process_is_alive(const std::uint64_t pid) {
#ifdef _WIN32
            HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
            if (!handle)
                return false;
            DWORD exit_code = 0;
            const bool alive = GetExitCodeProcess(handle, &exit_code) && exit_code == STILL_ACTIVE;
            CloseHandle(handle);
            return alive;
#else
            return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#endif
        }

        [[nodiscard]] std::filesystem::path run_spill_base() {
            return get_temp_folder() / "LichtFeld" / "pipeline_runs";
        }

        void remove_stale_run_spill_directories() {
            const auto base = run_spill_base();
            std::error_code ec;
            if (!std::filesystem::is_directory(base, ec) || ec)
                return;

            constexpr std::string_view prefix = "ppl_run_";
            for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
                if (ec)
                    break;
                if (!entry.is_directory(ec) || ec) {
                    ec.clear();
                    continue;
                }
                const auto name = entry.path().filename().string();
                if (!name.starts_with(prefix))
                    continue;

                const auto pid_start = prefix.size();
                const auto pid_end = name.find('_', pid_start);
                std::uint64_t pid = 0;
                const auto parse_end = pid_end == std::string::npos ? name.size() : pid_end;
                const auto [ptr, parse_ec] = std::from_chars(
                    name.data() + pid_start, name.data() + parse_end, pid);
                const bool parsed = parse_ec == std::errc{} && ptr == name.data() + parse_end;
                if (parsed && process_is_alive(pid))
                    continue;

                std::error_code remove_ec;
                std::filesystem::remove_all(entry.path(), remove_ec);
                if (remove_ec) {
                    LOG_WARN("[PipelinedImageLoader] Failed to remove stale run spill {}: {}",
                             lfs::core::path_to_utf8(entry.path()), remove_ec.message());
                }
            }
        }

        std::mutex& get_nvcodec_mutex() {
            static std::mutex mtx;
            return mtx;
        }

        std::unordered_map<size_t, NvCodecLoaderCacheEntry>& get_nvcodec_loader_cache() {
            static std::unordered_map<size_t, NvCodecLoaderCacheEntry> instances;
            return instances;
        }

        std::shared_ptr<NvCodecImageLoader> acquire_nvcodec_loader(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            auto& instances = get_nvcodec_loader_cache();
            const size_t requested_pool_size = normalize_nvcodec_pool_size(decoder_pool_size);

            if (auto it = instances.find(requested_pool_size);
                it != instances.end() && it->second.instance) {
                return it->second.instance;
            }

            auto instance = [&requested_pool_size] {
                NvCodecImageLoader::Options opts;
                opts.device_id = 0;
                opts.decoder_pool_size = requested_pool_size;
                opts.enable_fallback = true;
                return std::make_shared<NvCodecImageLoader>(opts);
            }();

            instances[requested_pool_size].instance = instance;
            return instance;
        }

        void retain_nvcodec_loader_cache(size_t decoder_pool_size) {
            std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
            ++get_nvcodec_loader_cache()[normalize_nvcodec_pool_size(decoder_pool_size)].owner_count;
        }

        void release_nvcodec_loader_cache(size_t decoder_pool_size) {
            std::shared_ptr<NvCodecImageLoader> released_instance;

            {
                std::lock_guard<std::mutex> lock(get_nvcodec_mutex());
                auto& instances = get_nvcodec_loader_cache();
                const size_t requested_pool_size = normalize_nvcodec_pool_size(decoder_pool_size);
                const auto it = instances.find(requested_pool_size);
                if (it == instances.end() || it->second.owner_count == 0)
                    return;

                auto& entry = it->second;
                --entry.owner_count;
                if (entry.owner_count == 0) {
                    released_instance = std::move(entry.instance);
                    instances.erase(it);
                }
            }

            // Drop the cache's last reference outside the mutex so teardown does not block other callers.
            released_instance.reset();
        }

        bool is_nvcodec_available() {
            static std::once_flag flag;
            static bool available = false;
            std::call_once(flag, [] { available = NvCodecImageLoader::is_available(); });
            return available;
        }

        [[nodiscard]] bool load_params_need_processing(const LoadParams& params) {
            return params.resize_factor > 1 || params.max_width > 0 || params.undistort != nullptr;
        }

        void convert_float_hwc_to_rgb(
            float*& data, const int width, const int height, int& channels) {
            if (channels == 3)
                return;
            if (channels < 1 || channels > 4)
                throw std::runtime_error("Unsupported image channel count");

            const size_t pixels = static_cast<size_t>(width) * height;
            auto* rgb = static_cast<float*>(std::malloc(pixels * 3 * sizeof(float)));
            if (!rgb) {
                lfs::core::free_image_float(data);
                data = nullptr;
                throw std::bad_alloc();
            }
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                const size_t source = pixel * channels;
                const float red = data[source];
                const float green = channels >= 3 ? data[source + 1] : red;
                const float blue = channels >= 3 ? data[source + 2] : red;
                rgb[pixel * 3] = red;
                rgb[pixel * 3 + 1] = green;
                rgb[pixel * 3 + 2] = blue;
            }
            lfs::core::free_image_float(data);
            data = rgb;
            channels = 3;
        }

        lfs::core::Tensor quantize_rgb_to_u16_grid(
            const lfs::core::Tensor& tensor, const cudaStream_t stream) {
            const size_t channels = tensor.shape()[0];
            const size_t height = tensor.shape()[1];
            const size_t width = tensor.shape()[2];
            auto hwc = tensor.permute({1, 2, 0}).contiguous();
            auto quantized = lfs::core::Tensor::empty(
                {height, width, channels}, lfs::core::Device::GPU,
                lfs::core::DataType::Float16);
            auto restored = lfs::core::Tensor::empty(
                {height, width, channels}, lfs::core::Device::GPU,
                lfs::core::DataType::Float32);
            cuda::launch_float32_hwc_to_uint16_hwc(
                hwc.ptr<float>(), reinterpret_cast<uint16_t*>(quantized.data_ptr()),
                height, width, channels, stream);
            cuda::launch_uint16_hwc_to_float32_hwc(
                reinterpret_cast<const uint16_t*>(quantized.data_ptr()), restored.ptr<float>(),
                height, width, channels, stream);
            return restored.permute({2, 0, 1}).contiguous();
        }

        void apply_requested_undistort(lfs::core::Tensor& tensor, const LoadParams& params) {
            if (!params.undistort)
                return;

            // The slot spans float staging to completion, so concurrent 8K loads never hold more
            // than one set of full-resolution warp buffers.
            const UndistortSlotGuard slot;
            const auto stream = image_execution_stream(params.cuda_stream);
            const lfs::core::CUDAStreamGuard stream_guard(stream);
            tensor.sync_to_stream(stream);
            const bool restore_uint8 = params.output_uint8;
            if (tensor.dtype() == lfs::core::DataType::UInt8) {
                tensor = tensor.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (tensor.dtype() != lfs::core::DataType::Float32) {
                tensor = tensor.to(lfs::core::DataType::Float32);
            }
            tensor = tensor.clamp(0.0f, 1.0f).contiguous();

            const auto scaled = lfs::core::prepare_undistort_params(
                *params.undistort,
                static_cast<int>(tensor.shape()[2]),
                static_cast<int>(tensor.shape()[1]),
                params.resize_factor,
                params.max_width);
            tensor = lfs::core::undistort_image(tensor, scaled, stream);

            if (restore_uint8) {
                auto uint8_tensor = lfs::core::Tensor::empty(
                    tensor.shape(), lfs::core::Device::GPU, lfs::core::DataType::UInt8);
                cuda::launch_float32_chw_to_uint8_chw(
                    tensor.ptr<float>(),
                    uint8_tensor.ptr<uint8_t>(),
                    tensor.shape()[1],
                    tensor.shape()[2],
                    tensor.shape()[0],
                    stream);
                tensor = std::move(uint8_tensor);
            } else {
                tensor = quantize_rgb_to_u16_grid(tensor, stream);
            }
            const cudaError_t status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("undistortion failed: ") + cudaGetErrorString(status));
        }

        lfs::core::Tensor process_mask(lfs::core::Tensor mask, const float threshold) {
            if (!mask.is_valid())
                return {};
            if (mask.dtype() == lfs::core::DataType::UInt8)
                mask = mask.to(lfs::core::DataType::Float32) / 255.0f;
            else if (mask.dtype() == lfs::core::DataType::Bool)
                mask = mask.to(lfs::core::DataType::Float32);
            mask = mask.clamp(0.0f, 1.0f);
            if (threshold > 0.0f)
                mask = mask.ge(threshold).to(lfs::core::DataType::Float32);
            return mask.contiguous();
        }

        [[nodiscard]] bool is_jpeg_file_signature(const std::filesystem::path& path) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file))
                return false;

            std::array<uint8_t, 3> signature{};
            if (!file.read(reinterpret_cast<char*>(signature.data()),
                           static_cast<std::streamsize>(signature.size()))) {
                return false;
            }

            return signature[0] == 0xFF && signature[1] == 0xD8 && signature[2] == 0xFF;
        }

        [[nodiscard]] bool is_regular_file_no_throw(const std::filesystem::path& path) {
            if (path.empty()) {
                return false;
            }
            std::error_code ec;
            const bool exists = std::filesystem::exists(path, ec);
            if (ec || !exists) {
                return false;
            }
            const bool regular = std::filesystem::is_regular_file(path, ec);
            return !ec && regular;
        }

        [[nodiscard]] std::string describe_current_exception(const char* fallback) {
            try {
                throw;
            } catch (const std::exception& e) {
                return e.what();
            } catch (...) {
                return fallback;
            }
        }

        [[nodiscard]] lfs::core::Tensor decode_cached_rgb_tensor(
            const std::shared_ptr<NvCodecImageLoader>& nvcodec,
            const std::shared_ptr<std::vector<uint8_t>>& jpeg_data,
            const LoadParams& params,
            const bool apply_processing) {
            const auto stream = image_execution_stream(params.cuda_stream);
            const lfs::core::CUDAStreamGuard execution_scope(stream);

            auto tensor = nvcodec->load_image_from_memory_gpu(
                *jpeg_data,
                apply_processing && !params.undistort ? params.resize_factor : 1,
                apply_processing && !params.undistort ? params.max_width : 0,
                params.cuda_stream,
                DecodeFormat::RGB,
                params.output_uint8);
            if (!tensor.is_valid() || tensor.numel() == 0)
                return {};

            if (apply_processing)
                apply_requested_undistort(tensor, params);

            return tensor;
        }

        std::tuple<uint8_t*, int, int> load_grayscale_stb(const std::filesystem::path& path) {
            int w, h, c;
            uint8_t* const data = stbi_load(lfs::core::path_to_utf8(path).c_str(), &w, &h, &c, 1);
            return {data, w, h};
        }

        void synchronize_async_upload_before_free(cudaStream_t stream, const char* context) {
            // A null stream means the legacy default stream. Synchronizing it
            // here would impose a device-wide barrier on unrelated callers.
            if (!stream) {
                return;
            }
            if (const cudaError_t err = cudaStreamSynchronize(stream); err != cudaSuccess) {
                throw std::runtime_error(
                    std::string(context) + " upload sync failed: " + cudaGetErrorString(err));
            }
        }

        [[nodiscard]] cudaStream_t queue_stream(const lfs::core::TensorWorkQueue& queue) {
            return static_cast<cudaStream_t>(queue.native_handle());
        }

    } // namespace

    lfs::core::Tensor PipelinedImageLoader::load_image_cuda(
        const std::filesystem::path& path, const LoadParams& params) {
        if (config_.backend != lfs::core::GpuBackend::CUDA) {
            const lfs::core::GpuBackendScope backend(config_.backend);
            lfs::core::TensorUpload upload;
            return decode_portable_rgb(path, params, upload);
        }
        const auto stream = image_execution_stream(params.cuda_stream);
        const lfs::core::CUDAStreamGuard execution_scope(stream);

        const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::CUDA);
        const auto cache_key = make_cache_key(path, params);
        const bool is_original_jpeg = is_jpeg_file_signature(path);
        const bool needs_requested_processing = load_params_need_processing(params);
        auto decode_cached_hit = [&](const std::shared_ptr<std::vector<uint8_t>>& jpeg_data) -> lfs::core::Tensor {
            if (!is_nvcodec_available())
                return {};

            try {
                auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                auto tensor = decode_cached_rgb_tensor(nvcodec, jpeg_data, params, false);
                if (tensor.is_valid() && tensor.numel() > 0)
                    return tensor;
            } catch (...) {}
            return {};
        };

        if (auto jpeg_data = load_cached_jpeg_blob(cache_key)) {
            if (auto tensor = decode_cached_hit(jpeg_data);
                tensor.is_valid() && tensor.numel() > 0) {
                return tensor;
            }
        }

        lfs::core::Tensor decoded;

        if (is_original_jpeg) {
            auto data = std::make_shared<std::vector<uint8_t>>(read_file(path));
            if (!needs_requested_processing) {
                put_in_jpeg_cache(cache_key, data);
            }

            if (is_nvcodec_available()) {
                try {
                    auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                    auto tensor = decode_cached_rgb_tensor(nvcodec, data, params, needs_requested_processing);
                    if (tensor.is_valid() && tensor.numel() > 0)
                        return tensor;
                } catch (...) {
                    LOG_DEBUG("[PipelinedImageLoader] Immediate JPEG decode fallback for {}: {}",
                              lfs::core::path_to_utf8(path),
                              describe_current_exception("non-standard nvImageCodec exception"));
                }
            }
        } else if (!config_.use_16bit_color && !needs_requested_processing) {
            const std::string path_str = lfs::core::path_to_utf8(path);
            int w = 0, h = 0, ch = 0;
            unsigned char* img_data = stbi_load(path_str.c_str(), &w, &h, &ch, 3);
            const bool used_stbi = (img_data != nullptr);
            if (img_data) {
                ch = 3;
            } else {
                auto [decoded_data, ow, oh, oc] = lfs::core::load_image(path, 1, 0);
                if (!decoded_data)
                    throw std::runtime_error("Failed to decode image: " + path_str);
                img_data = decoded_data;
                w = ow;
                h = oh;
                ch = oc;
            }

            const size_t H = static_cast<size_t>(h);
            const size_t W = static_cast<size_t>(w);
            const size_t C = static_cast<size_t>(ch);

            auto cpu_tensor = lfs::core::Tensor::from_blob(
                img_data, lfs::core::TensorShape({H, W, C}),
                lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::GPU);

            gpu_uint8.set_name("io.image.gpu_staging");
            if (used_stbi)
                stbi_image_free(img_data);
            else
                lfs::core::free_image(img_data);

            if (params.output_uint8) {
                decoded = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({C, H, W}),
                    lfs::core::Device::GPU, lfs::core::DataType::UInt8);
                decoded.set_name("io.image.gpu_uint8");
                lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_uint8, decoded, lfs::gpu_ops::ImageConversion::U8HWCToU8CHW, H, W, C, {});
            } else {
                decoded = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({C, H, W}),
                    lfs::core::Device::GPU, lfs::core::DataType::Float32);
                decoded.set_name("io.image.gpu_float");
                lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_uint8, decoded, lfs::gpu_ops::ImageConversion::U8HWCToF32CHW, H, W, C, {});
            }

            if (is_nvcodec_available()) {
                try {
                    auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                    auto jpeg_bytes = nvcodec->encode_to_jpeg(decoded, config_.cache_jpeg_quality, lfs::core::getCurrentCUDAStream());
                    auto jpeg_shared = std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes));
                    put_in_jpeg_cache(cache_key, jpeg_shared);

                    if (needs_requested_processing) {
                        auto tensor = decode_cached_rgb_tensor(nvcodec, jpeg_shared, params, false);
                        if (tensor.is_valid() && tensor.numel() > 0)
                            return tensor;
                    }
                } catch (...) {
                    LOG_DEBUG("[PipelinedImageLoader] Immediate cache write skipped for {}: {}",
                              lfs::core::path_to_utf8(path),
                              describe_current_exception("non-standard nvImageCodec exception"));
                }
            }
        }

        if (!decoded.is_valid() || decoded.numel() == 0 || needs_requested_processing) {
            decoded = decode_file_on_cpu(path, params);
        }

        apply_requested_undistort(decoded, params);
        return decoded;
    }

    lfs::core::Tensor PipelinedImageLoader::decode_file_on_cpu(
        const std::filesystem::path& path,
        const LoadParams& params) const {
        const auto stream = image_execution_stream(params.cuda_stream);
        const lfs::core::CUDAStreamGuard execution_scope(stream);

        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::core::TensorShape;

        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            ++stats_.cpu_decode_calls;
        }

        Tensor decoded;
        Tensor gpu_staging;
        if (params.undistort) {
            auto [img_data, width, height, channels] = lfs::core::load_image_float(path);
            if (!img_data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            convert_float_hwc_to_rgb(img_data, width, height, channels);
            const size_t H = static_cast<size_t>(height);
            const size_t W = static_cast<size_t>(width);
            const size_t C = static_cast<size_t>(channels);
            auto cpu_tensor = Tensor::from_blob(
                img_data, TensorShape({H, W, C}), Device::CPU, DataType::Float32);
            gpu_staging = cpu_tensor.to(Device::GPU, stream);
            synchronize_async_upload_before_free(stream, "image");
            lfs::core::free_image_float(img_data);
            decoded = gpu_staging.permute({2, 0, 1}).contiguous();
        } else if (config_.use_16bit_color) {
            auto [img_data, width, height, channels] = lfs::core::load_image_u16(
                path, params.resize_factor, params.max_width);
            if (!img_data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));

            const size_t H = static_cast<size_t>(height);
            const size_t W = static_cast<size_t>(width);
            const size_t C = static_cast<size_t>(channels);

            // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
            auto cpu_tensor = Tensor::from_blob(
                img_data, TensorShape({H, W, C}), Device::CPU, DataType::Float16);
            gpu_staging = cpu_tensor.to(Device::GPU, stream);

            if (params.output_uint8) {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::GPU, DataType::UInt8);
                lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_staging, decoded, lfs::gpu_ops::ImageConversion::U16HWCToU8CHW, H, W, C, {});
            } else {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::GPU, DataType::Float32);
                lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_staging, decoded, lfs::gpu_ops::ImageConversion::U16HWCToF32CHW, H, W, C, {});
            }
            synchronize_async_upload_before_free(stream, "image");
            lfs::core::free_image(img_data);
        } else {
            auto [img_data, width, height, channels] = lfs::core::load_image(
                path, params.resize_factor, params.max_width);
            if (!img_data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));

            const size_t H = static_cast<size_t>(height);
            const size_t W = static_cast<size_t>(width);
            const size_t C = static_cast<size_t>(channels);

            auto cpu_tensor = Tensor::from_blob(
                img_data, TensorShape({H, W, C}), Device::CPU, DataType::UInt8);
            gpu_staging = cpu_tensor.to(Device::GPU, stream);

            if (params.output_uint8) {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::GPU, DataType::UInt8);
                lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_staging, decoded, lfs::gpu_ops::ImageConversion::U8HWCToU8CHW, H, W, C, {});
            } else {
                decoded = Tensor::empty(TensorShape({C, H, W}), Device::GPU, DataType::Float32);
                lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_staging, decoded, lfs::gpu_ops::ImageConversion::U8HWCToF32CHW, H, W, C, {});
            }
            synchronize_async_upload_before_free(stream, "image");
            lfs::core::free_image(img_data);
        }

        return decoded;
    }

    void PipelinedImageLoader::write_derived_cache(NvCodecImageLoader& nvcodec,
                                                   const lfs::core::Tensor& tensor,
                                                   const std::string& cache_key,
                                                   void* cuda_stream,
                                                   const LoadParams& params) {
        const bool lossless = config_.use_16bit_color || params.undistort;
        if (lossless && !jpeg2k_cache_available_.load(std::memory_order_relaxed))
            return;

        try {
            auto bytes = lossless
                             ? nvcodec.encode_to_jpeg2k(
                                   tensor.dtype() == lfs::core::DataType::UInt8
                                       ? tensor.to(lfs::core::DataType::Float32) / 255.0f
                                       : tensor,
                                   cuda_stream)
                             : nvcodec.encode_to_jpeg(tensor, config_.cache_jpeg_quality, cuda_stream);
            put_in_jpeg_cache(cache_key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
        } catch (...) {
            const auto message = describe_current_exception("non-standard nvImageCodec exception");
            if (lossless && jpeg2k_cache_available_.exchange(false)) {
                LOG_ERROR("[PipelinedImageLoader] JPEG 2000 cache encode unavailable ({}); "
                          "lossless caching is disabled and source images will be decoded again. "
                          "Build with the nvJPEG2000 library to enable it.",
                          message);
            } else {
                LOG_DEBUG("[PipelinedImageLoader] Derived cache write skipped for key {}: {}",
                          cache_key, message);
            }
        }
    }

    void PipelinedImageLoader::write_sidecar_cache(NvCodecImageLoader& nvcodec,
                                                   const lfs::core::Tensor& tensor,
                                                   const PrefetchedImage& item,
                                                   const SidecarCacheFormat kind,
                                                   void* cuda_stream) {
        if (!jpeg2k_cache_available_.load(std::memory_order_relaxed)) {
            return;
        }

        try {
            std::vector<uint8_t> bytes;
            // Sidecars are stored as lossless HT J2K after all resize/undistort/prior
            // transforms. Depth [H,W] is quantized as round(v*65535). Normals
            // [3,H,W] in [-1,1] are mapped to HWC [0,1] as (n+1)/2 before the same
            // u16 quantization; hot decode reverses that map with <= 1/65535 loss.
            if (kind == SidecarCacheFormat::Depth) {
                bytes = nvcodec.encode_grayscale_to_jpeg2k(tensor, cuda_stream);
            } else {
                if (tensor.ndim() != 3 || tensor.shape()[0] != 3) {
                    throw std::runtime_error("Normal sidecar cache expects [3,H,W] tensor");
                }
                const size_t height = tensor.shape()[1];
                const size_t width = tensor.shape()[2];
                auto hwc = lfs::core::Tensor::empty(
                    lfs::core::TensorShape({height, width, size_t{3}}),
                    lfs::core::Device::GPU,
                    lfs::core::DataType::Float32);
                const lfs::core::CUDAStreamGuard execution_scope(static_cast<cudaStream_t>(cuda_stream));
                lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(tensor, hwc, lfs::gpu_ops::ImageConversion::NormalCHWToJ2KHWC, height, width, 3, {});
                bytes = nvcodec.encode_to_jpeg2k(hwc, cuda_stream);
            }
            put_in_jpeg_cache(item.cache_key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
        } catch (...) {
            const auto message = describe_current_exception("non-standard nvImageCodec exception");
            if (jpeg2k_cache_available_.exchange(false)) {
                LOG_ERROR("[PipelinedImageLoader] JPEG 2000 sidecar cache encode unavailable ({}); "
                          "depth/normal training continues without a sidecar cache.",
                          message);
            } else {
                LOG_DEBUG("[PipelinedImageLoader] Sidecar cache write skipped for {}: {}",
                          lfs::core::path_to_utf8(item.path),
                          message);
            }
        }
    }

    lfs::core::Tensor PipelinedImageLoader::decode_cached_sidecar(NvCodecImageLoader& nvcodec,
                                                                  const PrefetchedImage& item,
                                                                  void* cuda_stream) {
        if (!item.jpeg_data) {
            return {};
        }

        auto decoded = nvcodec.decode_jpeg2k_16bit_from_memory_gpu(
            *item.jpeg_data, cuda_stream, false);
        if (item.is_depth) {
            if (!decoded.is_valid() || decoded.ndim() != 2) {
                throw std::runtime_error("Decoded depth sidecar cache is not [H,W]");
            }
            decoded.set_stream(static_cast<cudaStream_t>(cuda_stream));
            return decoded;
        }

        if (!decoded.is_valid() || decoded.ndim() != 3 || decoded.shape()[2] != 3) {
            throw std::runtime_error("Decoded normal sidecar cache is not [H,W,3]");
        }
        const size_t height = decoded.shape()[0];
        const size_t width = decoded.shape()[1];
        auto normal = lfs::core::Tensor::empty(
            lfs::core::TensorShape({size_t{3}, height, width}),
            lfs::core::Device::GPU,
            lfs::core::DataType::Float32);
        const lfs::core::CUDAStreamGuard execution_scope(static_cast<cudaStream_t>(cuda_stream));
        lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(decoded, normal, lfs::gpu_ops::ImageConversion::J2KHWCToNormalCHW, height, width, 3, {});
        normal.set_stream(static_cast<cudaStream_t>(cuda_stream));
        return lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->resize(normal, static_cast<int>(height), static_cast<int>(width), lfs::gpu_ops::Resample::NormalPrior, 2);
    }

    std::optional<lfs::core::TensorFence> PipelinedImageLoader::record_sidecar_ready_event(void* stream) {
        const auto cuda_stream = static_cast<cudaStream_t>(stream);
        if (!cuda_stream) {
            return std::nullopt;
        }
        try {
            lfs::core::TensorFence fence(lfs::core::GpuBackend::CUDA);
            fence.record(cuda_stream);
            return fence;
        } catch (const std::exception& error) {
            LOG_WARN("[PipelinedImageLoader] sidecar fence record failed: {}", error.what());
            if (const cudaError_t sync = cudaStreamSynchronize(cuda_stream); sync != cudaSuccess) {
                LOG_WARN("[PipelinedImageLoader] sidecar stream sync failed: {}",
                         cudaGetErrorString(sync));
            }
            return std::nullopt;
        }
    }

    void PipelinedImageLoader::gpu_batch_decode_thread_func() {
        const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::CUDA);
        LFS_VRAM_SCOPE("io.image_loader");
        // Tensor ops on this thread (decode targets, format conversion) home
        // onto the decode stream so they order with the explicit-stream kernels.
        const lfs::core::CUDAStreamGuard stream_guard(queue_stream(*decode_queue_));
        std::vector<PrefetchedImage> batch;
        batch.reserve(config_.jpeg_batch_size);

        while (running_) {
            batch.clear();
            const auto deadline = std::chrono::steady_clock::now() + config_.batch_collect_timeout;

            try {
                auto first = hot_queue_.try_pop_for(config_.output_wait_timeout);
                if (!first)
                    continue;
                batch.push_back(std::move(*first));
            } catch (const std::runtime_error&) {
                break;
            }

            const size_t batch_limit = std::clamp<size_t>(adaptive_prefetch_target(), 2, 12);
            while (batch.size() < std::min(config_.jpeg_batch_size, batch_limit)) {
                if (auto item = hot_queue_.try_pop()) {
                    batch.push_back(std::move(*item));
                    continue;
                }
                if (batch.size() >= 2)
                    break;
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());
                if (remaining.count() <= 0)
                    break;
                auto item = hot_queue_.try_pop_for(std::min(remaining, std::chrono::milliseconds{1}));
                if (!item)
                    break;
                batch.push_back(std::move(*item));
            }

            if (batch.empty())
                continue;

            const auto decode_begin = std::chrono::steady_clock::now();
            try {
                auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);
                std::vector<bool> decoded_as_pair(batch.size(), false);

                const auto prepare_sidecar = [&](const PrefetchedImage& item,
                                                 lfs::core::Tensor decoded) {
                    if (item.is_depth) {
                        if (!decoded.is_valid() || decoded.ndim() != 2) {
                            throw std::runtime_error("Decoded depth sidecar cache is not [H,W]");
                        }
                        decoded.set_stream(queue_stream(*decode_queue_));
                        return decoded;
                    }
                    if (!decoded.is_valid() || decoded.ndim() != 3 || decoded.shape()[2] != 3) {
                        throw std::runtime_error("Decoded normal sidecar cache is not [H,W,3]");
                    }
                    const size_t height = decoded.shape()[0];
                    const size_t width = decoded.shape()[1];
                    auto normal = lfs::core::Tensor::empty(
                        lfs::core::TensorShape({size_t{3}, height, width}),
                        lfs::core::Device::GPU,
                        lfs::core::DataType::Float32);
                    lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(decoded, normal, lfs::gpu_ops::ImageConversion::J2KHWCToNormalCHW, height, width, 3, {});
                    normal.set_stream(queue_stream(*decode_queue_));
                    return normal;
                };

                const auto complete_sidecar = [&](const PrefetchedImage& item,
                                                  lfs::core::Tensor sidecar) {
                    auto ready_event = record_sidecar_ready_event(queue_stream(*decode_queue_));
                    if (item.is_depth) {
                        try_complete_pair(
                            item.sequence_id,
                            item.loader_generation,
                            std::nullopt,
                            std::nullopt,
                            std::move(sidecar),
                            std::nullopt,
                            std::move(ready_event));
                    } else {
                        try_complete_pair(
                            item.sequence_id,
                            item.loader_generation,
                            std::nullopt,
                            std::nullopt,
                            std::nullopt,
                            std::move(sidecar),
                            std::move(ready_event));
                    }
                };

                std::vector<size_t> rgb_batch_indices;
                rgb_batch_indices.reserve(batch.size());
                std::vector<size_t> rgb_jpeg2k_batch_indices;
                rgb_jpeg2k_batch_indices.reserve(batch.size());
                bool rgb_dtype_initialized = false;
                bool rgb_output_uint8 = false;
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (!batch[i].is_mask && !batch[i].is_depth && !batch[i].is_normal &&
                        !batch[i].needs_processing &&
                        batch[i].jpeg_data) {
                        const auto& bytes = *batch[i].jpeg_data;
                        const bool is_jpeg2k = bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0x4f;
                        if (is_jpeg2k) {
                            rgb_jpeg2k_batch_indices.push_back(i);
                            continue;
                        }
                        if (!is_jpeg_data(bytes))
                            continue;
                        if (!rgb_dtype_initialized) {
                            rgb_dtype_initialized = true;
                            rgb_output_uint8 = batch[i].params.output_uint8;
                        }
                        if (batch[i].params.output_uint8 != rgb_output_uint8)
                            continue;
                        rgb_batch_indices.push_back(i);
                    }
                }
                if (!rgb_batch_indices.empty()) {
                    auto leases = decoded_frame_ring_->acquire_batch(rgb_batch_indices.size());
                    if (leases.empty()) {
                        if (decoded_frame_ring_->cancelled()) {
                            LOG_DEBUG("[PipelinedImageLoader] GPU batch decode cancelled during shutdown");
                            return;
                        }
                        throw std::runtime_error("decoded frame ring cancelled");
                    }
                    rgb_batch_indices.resize(leases.size());
                    std::vector<std::pair<const uint8_t*, size_t>> spans;
                    spans.reserve(rgb_batch_indices.size());
                    std::vector<lfs::core::Tensor*> reusable_outputs;
                    std::vector<lfs::core::Tensor*> reusable_hwc;
                    reusable_outputs.reserve(rgb_batch_indices.size());
                    reusable_hwc.reserve(rgb_batch_indices.size());
                    assert(rgb_batch_indices.size() <= decode_hwc_workspace_.size());
                    for (size_t i = 0; i < rgb_batch_indices.size(); ++i) {
                        const size_t index = rgb_batch_indices[i];
                        spans.emplace_back(batch[index].jpeg_data->data(),
                                           batch[index].jpeg_data->size());
                        reusable_outputs.push_back(&leases[i]->storage);
                        reusable_hwc.push_back(&decode_hwc_workspace_[i]);
                    }
                    auto decoded = nvcodec->decode_jpeg_batch_from_spans(
                        spans, queue_stream(*decode_queue_), rgb_output_uint8, true,
                        &reusable_hwc, &reusable_outputs);
                    if (decoded.size() != rgb_batch_indices.size()) {
                        throw std::runtime_error("JPEG batch decode returned wrong size");
                    }
                    for (size_t j = 0; j < rgb_batch_indices.size(); ++j) {
                        const size_t index = rgb_batch_indices[j];
                        std::shared_ptr<void> lease = leases[j];
                        try_complete_pair(batch[index].sequence_id, batch[index].loader_generation,
                                          std::move(decoded[j]), std::nullopt, std::nullopt,
                                          std::nullopt, std::nullopt, std::move(lease));
                        decoded_as_pair[index] = true;
                    }
                }

                if (!rgb_jpeg2k_batch_indices.empty()) {
                    std::vector<std::pair<const uint8_t*, size_t>> spans;
                    spans.reserve(rgb_jpeg2k_batch_indices.size());
                    for (const size_t index : rgb_jpeg2k_batch_indices) {
                        spans.emplace_back(batch[index].jpeg_data->data(), batch[index].jpeg_data->size());
                    }
                    auto decoded = nvcodec->decode_jpeg2k_16bit_batch_from_spans(
                        spans, queue_stream(*decode_queue_), false);
                    if (decoded.size() != rgb_jpeg2k_batch_indices.size()) {
                        throw std::runtime_error("JPEG2000 RGB batch decode returned wrong size");
                    }
                    for (size_t j = 0; j < rgb_jpeg2k_batch_indices.size(); ++j) {
                        const size_t index = rgb_jpeg2k_batch_indices[j];
                        auto& tensor = decoded[j];
                        if (!tensor.is_valid() || tensor.ndim() != 3 || tensor.shape()[2] != 3) {
                            throw std::runtime_error("Decoded RGB JPEG2000 cache is not [H,W,3]");
                        }
                        tensor = tensor.permute({2, 0, 1}).contiguous();
                        tensor.set_stream(queue_stream(*decode_queue_));
                        if (batch[index].params.output_uint8) {
                            auto uint8_tensor = lfs::core::Tensor::empty(
                                tensor.shape(), lfs::core::Device::GPU, lfs::core::DataType::UInt8);
                            uint8_tensor.set_stream(queue_stream(*decode_queue_));
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(tensor, uint8_tensor, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW, tensor.shape()[1], tensor.shape()[2], tensor.shape()[0], {});
                            tensor = std::move(uint8_tensor);
                        }
                        try_complete_pair(batch[index].sequence_id, batch[index].loader_generation,
                                          std::move(tensor), std::nullopt);
                        decoded_as_pair[index] = true;
                    }
                }

                for (size_t i = 0; i < batch.size(); ++i) {
                    if (decoded_as_pair[i]) {
                        continue;
                    }
                    try {
                        batch[i].params.cuda_stream = queue_stream(*decode_queue_);
                        if (batch[i].is_depth || batch[i].is_normal) {
                            size_t pair_index = batch.size();
                            for (size_t j = i + 1; j < batch.size(); ++j) {
                                if (!decoded_as_pair[j] &&
                                    batch[j].sequence_id == batch[i].sequence_id &&
                                    batch[j].is_depth != batch[i].is_depth &&
                                    (batch[j].is_depth || batch[j].is_normal) &&
                                    batch[i].jpeg_data && batch[j].jpeg_data) {
                                    pair_index = j;
                                    break;
                                }
                            }

                            if (pair_index != batch.size()) {
                                const std::vector<std::pair<const uint8_t*, size_t>> spans{
                                    {batch[i].jpeg_data->data(), batch[i].jpeg_data->size()},
                                    {batch[pair_index].jpeg_data->data(), batch[pair_index].jpeg_data->size()}};
                                auto decoded = nvcodec->decode_jpeg2k_16bit_batch_from_spans(
                                    spans, queue_stream(*decode_queue_), false);
                                if (decoded.size() != 2) {
                                    throw std::runtime_error("JPEG2000 sidecar pair decode returned wrong size");
                                }
                                auto first = prepare_sidecar(batch[i], std::move(decoded[0]));
                                auto second = prepare_sidecar(batch[pair_index], std::move(decoded[1]));
                                complete_sidecar(batch[i], std::move(first));
                                complete_sidecar(batch[pair_index], std::move(second));
                                decoded_as_pair[pair_index] = true;
                            } else {
                                auto sidecar = decode_cached_sidecar(*nvcodec, batch[i], queue_stream(*decode_queue_));
                                if (!sidecar.is_valid() || sidecar.numel() == 0) {
                                    LOG_WARN("[PipelinedImageLoader] GPU sidecar decode failed for {}",
                                             lfs::core::path_to_utf8(batch[i].path));
                                    throw std::runtime_error("Invalid sidecar tensor");
                                }
                                complete_sidecar(batch[i], std::move(sidecar));
                            }
                        } else if (batch[i].is_mask) {
                            lfs::core::Tensor mask_tensor;
                            const auto& bytes = *batch[i].jpeg_data;
                            const bool is_jpeg2k = bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0x4f;
                            if (is_jpeg2k) {
                                auto raw = nvcodec->decode_jpeg2k_16bit_from_memory_gpu(
                                    bytes, queue_stream(*decode_queue_), false, true);
                                // Pipeline mask sidecars are always encoded as UINT8 below.
                                // The decoder returns normalized Float32 for UINT16 streams;
                                // reinterpreting that output as raw u16 would normalize twice.
                                LFS_ASSERT_MSG(
                                    raw.dtype() == lfs::core::DataType::UInt8,
                                    "pipeline JPEG2000 mask cache must contain eight-bit samples");
                                mask_tensor = raw.to(lfs::core::DataType::Float32) / 255.0f;
                            } else {
                                mask_tensor = nvcodec->load_image_from_memory_gpu(
                                    bytes, 1, 0, queue_stream(*decode_queue_), DecodeFormat::Grayscale);
                            }

                            if (!mask_tensor.is_valid() || mask_tensor.numel() == 0) {
                                LOG_WARN("[PipelinedImageLoader] GPU mask decode failed for {}",
                                         lfs::core::path_to_utf8(batch[i].path));
                                throw std::runtime_error("Invalid mask tensor");
                            }

                            if (batch[i].mask_params.invert)
                                mask_tensor = mask_tensor * -1.0f + 1.0f;
                            mask_tensor = process_mask(std::move(mask_tensor), batch[i].mask_params.threshold);
                            try_complete_pair(batch[i].sequence_id, batch[i].loader_generation,
                                              std::nullopt, std::move(mask_tensor));

                        } else {
                            const bool decode_from_base_cache = batch[i].needs_processing;
                            const bool decode_full_resolution =
                                decode_from_base_cache && batch[i].params.undistort;
                            auto tensor = nvcodec->load_image_from_memory_gpu(
                                *batch[i].jpeg_data,
                                decode_from_base_cache && !decode_full_resolution
                                    ? batch[i].params.resize_factor
                                    : 1,
                                decode_from_base_cache && !decode_full_resolution
                                    ? batch[i].params.max_width
                                    : 0,
                                batch[i].params.cuda_stream,
                                DecodeFormat::RGB,
                                batch[i].params.output_uint8);

                            if (!tensor.is_valid() || tensor.numel() == 0) {
                                LOG_WARN("[PipelinedImageLoader] GPU decode failed for {}",
                                         lfs::core::path_to_utf8(batch[i].path));
                                throw std::runtime_error("Invalid tensor");
                            }

                            if (decode_from_base_cache) {
                                apply_requested_undistort(tensor, batch[i].params);
                                write_derived_cache(*nvcodec, tensor, batch[i].cache_key,
                                                    batch[i].params.cuda_stream, batch[i].params);
                            }

                            try_complete_pair(batch[i].sequence_id, batch[i].loader_generation,
                                              std::move(tensor), std::nullopt);
                        }
                    } catch (...) {
                        auto& item = batch[i];
                        const auto message =
                            describe_current_exception("non-standard nvImageCodec exception");
                        if (item.is_cache_hit) {
                            const char* kind = item.is_depth    ? "depth sidecar"
                                               : item.is_normal ? "normal sidecar"
                                               : item.is_mask   ? "mask"
                                                                : "image";
                            LOG_WARN("[PipelinedImageLoader] Cached {} decode failed for {}; "
                                     "evicting cache entry and reprocessing: {}",
                                     kind,
                                     lfs::core::path_to_utf8(item.path),
                                     message);
                            invalidate_cache_entry(item.cache_key);
                        }
                        item.is_cache_hit = false;
                        item.needs_processing = true;
                        if (item.raw_bytes.empty()) {
                            try {
                                item.raw_bytes = read_file(item.path);
                            } catch (...) {
                                const auto failure_message = describe_current_exception(
                                    "failed to read sidecar for decoder fallback");
                                std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                                if (item.is_mask || item.is_depth || item.is_normal) {
                                    const auto kind = item.is_mask    ? SidecarKind::Mask
                                                      : item.is_depth ? SidecarKind::Depth
                                                                      : SidecarKind::Normal;
                                    fail_sidecar_locked(item.sequence_id, item.loader_generation,
                                                        kind, item.path,
                                                        failure_message, lock);
                                } else {
                                    lock.unlock();
                                    publish_image_failure(
                                        item.sequence_id,
                                        item.loader_generation,
                                        item.path,
                                        failure_message);
                                }
                                continue;
                            }
                        }
                        cold_queue_.push(std::move(item));
                    }
                }

                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.gpu_batch_decodes;
                stats_.total_decode_calls += batch.size();

                record_decode_latency(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - decode_begin)
                                          .count() /
                                      std::max<size_t>(1, batch.size()));

            } catch (...) {
                LOG_ERROR("[PipelinedImageLoader] Batch decode error: {}",
                          describe_current_exception("non-standard nvImageCodec exception"));
                for (auto& item : batch) {
                    item.is_cache_hit = false;
                    item.needs_processing = true;
                    if (item.raw_bytes.empty()) {
                        try {
                            item.raw_bytes = read_file(item.path);
                        } catch (...) {
                            const auto failure_message = describe_current_exception(
                                "failed to read sidecar for decoder fallback");
                            std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                            if (item.is_mask || item.is_depth || item.is_normal) {
                                const auto kind = item.is_mask    ? SidecarKind::Mask
                                                  : item.is_depth ? SidecarKind::Depth
                                                                  : SidecarKind::Normal;
                                fail_sidecar_locked(item.sequence_id, item.loader_generation,
                                                    kind, item.path,
                                                    failure_message, lock);
                            } else {
                                lock.unlock();
                                publish_image_failure(
                                    item.sequence_id,
                                    item.loader_generation,
                                    item.path,
                                    failure_message);
                            }
                            continue;
                        }
                    }
                    cold_queue_.push(std::move(item));
                }
                record_decode_latency(std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - decode_begin)
                                          .count() /
                                      std::max<size_t>(1, batch.size()));
            }
        }
    }

    void PipelinedImageLoader::cold_process_thread_func(const size_t worker_index) {
        const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::CUDA);
        const cudaStream_t sidecar_stream =
            queue_stream(*sidecar_queues_.at(worker_index));
        const lfs::core::CUDAStreamGuard execution_scope(sidecar_stream);
        while (running_) {
            PrefetchedImage item;
            try {
                item = cold_queue_.pop();
                item.params.cuda_stream = sidecar_stream;
            } catch (const std::runtime_error&) {
                break;
            }

            try {
                auto nvcodec = acquire_nvcodec_loader(config_.decoder_pool_size);

                if (item.alpha_as_mask) {
                    auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
                        item.path,
                        item.undistort ? 1 : item.params.resize_factor,
                        item.undistort ? 0 : item.params.max_width);

                    if (!img_data || channels != 4)
                        throw std::runtime_error("Failed to load RGBA image");

                    const size_t H = static_cast<size_t>(height);
                    const size_t W = static_cast<size_t>(width);

                    auto cpu_tensor = lfs::core::Tensor::from_blob(
                        img_data, lfs::core::TensorShape({H, W, 4}),
                        lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                    auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::GPU);

                    lfs::core::free_image(img_data);

                    auto rgb = item.params.output_uint8
                                   ? lfs::core::Tensor::empty(
                                         lfs::core::TensorShape({3, H, W}),
                                         lfs::core::Device::GPU, lfs::core::DataType::UInt8)
                                   : lfs::core::Tensor::empty(
                                         lfs::core::TensorShape({3, H, W}),
                                         lfs::core::Device::GPU, lfs::core::DataType::Float32);
                    auto alpha = lfs::core::Tensor::empty(
                        lfs::core::TensorShape({H, W}),
                        lfs::core::Device::GPU, lfs::core::DataType::Float32);

                    lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->rgba_split(gpu_uint8, rgb, alpha);

                    gpu_uint8 = lfs::core::Tensor();

                    if (item.undistort) {
                        const auto scaled = lfs::core::prepare_undistort_params(
                            *item.undistort, static_cast<int>(W), static_cast<int>(H),
                            item.params.resize_factor, item.params.max_width);
                        apply_requested_undistort(rgb, item.params);
                        const UndistortSlotGuard slot;
                        alpha = lfs::core::undistort_mask_area(alpha, scaled, sidecar_stream);
                    }

                    if (is_nvcodec_available()) {
                        try {
                            write_derived_cache(*nvcodec, rgb, item.cache_key, lfs::core::getCurrentCUDAStream(), item.params);

                            const auto alpha_key = make_mask_cache_key(item.path, item.params);
                            auto alpha_jpeg = nvcodec->encode_grayscale_to_jpeg2k(alpha, lfs::core::getCurrentCUDAStream(), true, true);
                            put_in_jpeg_cache(alpha_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(alpha_jpeg)));
                        } catch (...) {
                        }
                    }

                    if (item.alpha_mask_params.invert)
                        lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->mask(alpha, lfs::gpu_ops::MaskTransform::Invert, 0.f);
                    if (item.alpha_mask_params.threshold > 0)
                        lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->mask(alpha, lfs::gpu_ops::MaskTransform::Threshold, item.alpha_mask_params.threshold);
                    alpha = process_mask(std::move(alpha), item.alpha_mask_params.threshold);

                    try_complete_pair(item.sequence_id, item.loader_generation,
                                      std::move(rgb), std::move(alpha));

                } else if (item.is_mask || item.is_depth) {
                    const cudaStream_t aux_stream = sidecar_stream;
                    const lfs::core::CUDAStreamGuard stream_guard(aux_stream);
                    lfs::core::Tensor aux_tensor;
                    bool used_gpu = false;
                    // nvimagecodec and the uint8 path truncate to 8 bits; 16-bit
                    // depth priors must keep their precision.
                    const bool depth_16bit =
                        item.is_depth && stbi_is_16_bit(lfs::core::path_to_utf8(item.path).c_str());

                    // Depth sidecars keep their first-touch 16-bit PNG precision by
                    // decoding with stb before the processed tensor is transcoded.
                    if (is_nvcodec_available() && !item.is_depth) {
                        try {
                            const NvcodecSlotGuard slot;
                            aux_tensor = nvcodec->load_image_gpu(
                                item.path,
                                item.undistort ? 1 : item.params.resize_factor,
                                item.undistort ? 0 : item.params.max_width,
                                aux_stream, DecodeFormat::Grayscale);
                            used_gpu = true;
                        } catch (...) {
                        }
                    }

                    if (!used_gpu) {
                        int src_w = 0;
                        int src_h = 0;
                        lfs::core::Tensor gpu_gray;
                        if (depth_16bit) {
                            int channels = 0;
                            stbi_us* const gray16 = stbi_load_16(
                                lfs::core::path_to_utf8(item.path).c_str(),
                                &src_w, &src_h, &channels, 1);
                            if (!gray16)
                                throw std::runtime_error("Failed to decode 16-bit depth");
                            const lfs::core::TensorShape shape(
                                {static_cast<size_t>(src_h), static_cast<size_t>(src_w)});
                            // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
                            auto cpu_tensor = lfs::core::Tensor::from_blob(
                                gray16, shape, lfs::core::Device::CPU, lfs::core::DataType::Float16);
                            auto gpu_staging = cpu_tensor.to(lfs::core::Device::GPU, aux_stream);
                            synchronize_async_upload_before_free(gpu_staging.stream(), "depth");
                            stbi_image_free(gray16);
                            gpu_gray = lfs::core::Tensor::empty(
                                shape, lfs::core::Device::GPU, lfs::core::DataType::Float32);
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_staging, gpu_gray, lfs::gpu_ops::ImageConversion::U16HWCToF32HWC, static_cast<size_t>(src_h), static_cast<size_t>(src_w), 1, {});
                        } else {
                            const auto [gray_data, w, h] = load_grayscale_stb(item.path);
                            if (!gray_data)
                                throw std::runtime_error(item.is_mask ? "Failed to decode mask" : "Failed to decode depth");
                            src_w = w;
                            src_h = h;
                            auto cpu_tensor = lfs::core::Tensor::empty(
                                lfs::core::TensorShape({static_cast<size_t>(src_h), static_cast<size_t>(src_w)}),
                                lfs::core::Device::CPU,
                                lfs::core::DataType::UInt8);
                            std::memcpy(cpu_tensor.data_ptr(), gray_data, cpu_tensor.bytes());
                            gpu_gray = cpu_tensor.to(lfs::core::Device::GPU, aux_stream);
                            stbi_image_free(gray_data);
                        }

                        const auto [target_w, target_h] = sidecar_target_size(item, src_w, src_h);

                        if (item.is_depth) {
                            if (gpu_gray.dtype() == lfs::core::DataType::UInt8)
                                gpu_gray = gpu_gray.to(lfs::core::DataType::Float32).div(255.0f);
                            aux_tensor = lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->resize(gpu_gray, target_h, target_w, lfs::gpu_ops::Resample::DepthPrior, 2);
                        } else if (target_w != src_w || target_h != src_h) {
                            aux_tensor = lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->resize(gpu_gray, target_h, target_w, lfs::gpu_ops::Resample::LanczosGray, 2);
                        } else if (gpu_gray.dtype() == lfs::core::DataType::Float32) {
                            aux_tensor = std::move(gpu_gray);
                        } else {
                            aux_tensor = lfs::core::Tensor::empty(
                                lfs::core::TensorShape({static_cast<size_t>(target_h), static_cast<size_t>(target_w)}),
                                lfs::core::Device::GPU, lfs::core::DataType::Float32);
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_gray, aux_tensor, lfs::gpu_ops::ImageConversion::U8HWToF32HW, target_h, target_w, 1, {});
                        }
                    }

                    if (!aux_tensor.is_valid() || aux_tensor.ndim() != 2) {
                        throw std::runtime_error(
                            item.is_mask ? "Mask preprocessing produced an invalid tensor"
                                         : "Depth preprocessing produced an invalid tensor");
                    }
                    if (item.is_mask && aux_tensor.dtype() == lfs::core::DataType::UInt8) {
                        aux_tensor = aux_tensor.to(lfs::core::DataType::Float32) / 255.0f;
                    }
                    const size_t H = aux_tensor.shape()[0];
                    const size_t W = aux_tensor.shape()[1];

                    if (item.undistort) {
                        const auto scaled = lfs::core::prepare_undistort_params(
                            *item.undistort,
                            static_cast<int>(W), static_cast<int>(H),
                            item.params.resize_factor,
                            item.params.max_width);
                        const UndistortSlotGuard slot;
                        aux_tensor = (item.is_depth ? lfs::core::undistort_depth_area(aux_tensor, scaled, aux_stream) : lfs::core::undistort_mask_area(aux_tensor, scaled, aux_stream));
                    }

                    if (item.is_mask && is_nvcodec_available()) {
                        try {
                            auto jpeg_bytes = nvcodec->encode_grayscale_to_jpeg2k(aux_tensor, lfs::core::getCurrentCUDAStream(), true, true);
                            put_in_jpeg_cache(item.cache_key,
                                              std::make_shared<std::vector<uint8_t>>(std::move(jpeg_bytes)));
                        } catch (...) {
                            LOG_DEBUG("[PipelinedImageLoader] Mask cache encode skipped for {}: {}",
                                      lfs::core::path_to_utf8(item.path),
                                      describe_current_exception("non-standard nvImageCodec exception"));
                        }
                    }

                    if (item.is_mask) {
                        if (item.mask_params.invert) {
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->mask(aux_tensor, lfs::gpu_ops::MaskTransform::Invert, 0.f);
                        }
                        if (item.mask_params.threshold > 0) {
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->mask(aux_tensor, lfs::gpu_ops::MaskTransform::Threshold, item.mask_params.threshold);
                        }
                        aux_tensor = process_mask(std::move(aux_tensor), item.mask_params.threshold);
                    } else {
                        if (item.aux_target_width > 0 && item.aux_target_height > 0 &&
                            aux_tensor.ndim() == 2 &&
                            (static_cast<int>(aux_tensor.shape()[1]) != item.aux_target_width ||
                             static_cast<int>(aux_tensor.shape()[0]) != item.aux_target_height)) {
                            aux_tensor = lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->resize(aux_tensor, item.aux_target_height, item.aux_target_width, lfs::gpu_ops::Resample::DepthPrior, 2);
                        }
                        aux_tensor = aux_tensor.contiguous();
                    }
                    if (item.is_mask) {
                        try_complete_pair(item.sequence_id, item.loader_generation,
                                          std::nullopt, std::move(aux_tensor));
                    } else {
                        write_sidecar_cache(*nvcodec, aux_tensor, item, SidecarCacheFormat::Depth, sidecar_stream);
                        auto ready_event = record_sidecar_ready_event(sidecar_stream);
                        try_complete_pair(
                            item.sequence_id,
                            item.loader_generation,
                            std::nullopt,
                            std::nullopt,
                            std::move(aux_tensor),
                            std::nullopt,
                            std::move(ready_event));
                    }

                } else if (item.is_normal) {
                    const lfs::core::CUDAStreamGuard stream_guard(sidecar_stream);
                    const std::string path_utf8 = lfs::core::path_to_utf8(item.path);
                    int src_w = 0;
                    int src_h = 0;
                    int channels = 0;
                    // nvimagecodec truncates to 8 bits; 16-bit normal priors
                    // must keep their precision, so both paths decode on CPU.
                    // The v = n*0.5 + 0.5 file encoding (with the sRGB display
                    // transform when the startup probe detected it), Y/Z flip,
                    // and world->camera rotation are all inverted in one GPU
                    // kernel; the loss re-normalizes per pixel, so resampling
                    // shrinkage is fine.
                    const bool normal_16bit = stbi_is_16_bit(path_utf8.c_str());
                    lfs::core::Tensor gpu_staging;
                    if (normal_16bit) {
                        stbi_us* const rgb16 = stbi_load_16(path_utf8.c_str(), &src_w, &src_h, &channels, 3);
                        if (!rgb16)
                            throw std::runtime_error("Failed to decode 16-bit normal map");
                        // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            rgb16,
                            lfs::core::TensorShape(
                                {static_cast<size_t>(src_h), static_cast<size_t>(src_w), 3}),
                            lfs::core::Device::CPU, lfs::core::DataType::Float16);
                        gpu_staging = cpu_tensor.to(lfs::core::Device::GPU, sidecar_stream);
                        synchronize_async_upload_before_free(gpu_staging.stream(), "normal");
                        stbi_image_free(rgb16);
                    } else {
                        stbi_uc* const rgb8 = stbi_load(path_utf8.c_str(), &src_w, &src_h, &channels, 3);
                        if (!rgb8)
                            throw std::runtime_error("Failed to decode normal map");
                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            rgb8,
                            lfs::core::TensorShape(
                                {static_cast<size_t>(src_h), static_cast<size_t>(src_w), 3}),
                            lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                        gpu_staging = cpu_tensor.to(lfs::core::Device::GPU, sidecar_stream);
                        synchronize_async_upload_before_free(gpu_staging.stream(), "normal");
                        stbi_image_free(rgb8);
                    }

                    lfs::gpu_ops::NormalPriorTransform prior_transform;
                    prior_transform.srgb = item.normal_srgb;
                    prior_transform.flip_yz = item.normal_flip_yz;
                    prior_transform.world_to_camera = item.normal_transform_world_to_camera;
                    std::copy(item.normal_world_to_camera.begin(),
                              item.normal_world_to_camera.end(), prior_transform.w2c);

                    auto normal_tensor = lfs::core::Tensor::empty(
                        lfs::core::TensorShape(
                            {3, static_cast<size_t>(src_h), static_cast<size_t>(src_w)}),
                        lfs::core::Device::GPU, lfs::core::DataType::Float32);
                    if (normal_16bit) {
                        lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_staging, normal_tensor, lfs::gpu_ops::ImageConversion::NormalPriorU16, static_cast<size_t>(src_h), static_cast<size_t>(src_w), 3, prior_transform);
                    } else {
                        lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_staging, normal_tensor, lfs::gpu_ops::ImageConversion::NormalPriorU8, static_cast<size_t>(src_h), static_cast<size_t>(src_w), 3, prior_transform);
                    }

                    const auto [target_w, target_h] = sidecar_target_size(item, src_w, src_h);
                    normal_tensor = lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->resize(normal_tensor, target_h, target_w, lfs::gpu_ops::Resample::NormalPrior, 2);

                    if (!normal_tensor.is_valid() || normal_tensor.ndim() != 3 || normal_tensor.shape()[0] != 3) {
                        throw std::runtime_error("Normal preprocessing produced an invalid tensor");
                    }
                    if (item.undistort) {
                        const auto scaled = lfs::core::prepare_undistort_params(
                            *item.undistort,
                            static_cast<int>(normal_tensor.shape()[2]),
                            static_cast<int>(normal_tensor.shape()[1]),
                            item.params.resize_factor,
                            item.params.max_width);
                        const UndistortSlotGuard slot;
                        normal_tensor = lfs::core::undistort_normal_area(normal_tensor, scaled, sidecar_stream);
                    }

                    if (item.aux_target_width > 0 && item.aux_target_height > 0 &&
                        normal_tensor.ndim() == 3 &&
                        (static_cast<int>(normal_tensor.shape()[2]) != item.aux_target_width ||
                         static_cast<int>(normal_tensor.shape()[1]) != item.aux_target_height)) {
                        normal_tensor = lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->resize(normal_tensor, item.aux_target_height, item.aux_target_width, lfs::gpu_ops::Resample::NormalPrior, 2);
                    }
                    normal_tensor = normal_tensor.contiguous();
                    if (!normal_tensor.is_valid() || normal_tensor.ndim() != 3 || normal_tensor.shape()[0] != 3) {
                        throw std::runtime_error("Normal preprocessing produced an invalid tensor");
                    }
                    write_sidecar_cache(*nvcodec, normal_tensor, item, SidecarCacheFormat::Normal, sidecar_stream);
                    auto ready_event = record_sidecar_ready_event(sidecar_stream);

                    try_complete_pair(
                        item.sequence_id,
                        item.loader_generation,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::move(normal_tensor),
                        std::move(ready_event));

                } else {
                    lfs::core::Tensor decoded;
                    bool used_gpu = false;

                    if (is_nvcodec_available() && item.is_original_jpeg) {
                        try {
                            const NvcodecSlotGuard slot;
                            if (item.undistort) {
                                decoded = nvcodec->load_image_from_memory_gpu(
                                    read_file(item.path), 1, 0, sidecar_stream,
                                    DecodeFormat::RGB, item.params.output_uint8);
                            } else {
                                decoded = nvcodec->load_image_gpu(
                                    item.path, item.params.resize_factor, item.params.max_width,
                                    sidecar_stream, DecodeFormat::RGB, item.params.output_uint8);
                            }
                            used_gpu = true;
                        } catch (...) {
                        }
                    }

                    if (!used_gpu) {
                        decoded = decode_file_on_cpu(item.path, item.params);
                    }

                    apply_requested_undistort(decoded, item.params);

                    if (is_nvcodec_available()) {
                        write_derived_cache(*nvcodec, decoded, item.cache_key, sidecar_stream, item.params);
                    }

                    try_complete_pair(item.sequence_id, item.loader_generation,
                                      std::move(decoded), std::nullopt);
                }

            } catch (...) {
                const auto message = describe_current_exception("non-standard image loader exception");
                if (item.alpha_as_mask) {
                    LOG_WARN("[PipelinedImageLoader] Alpha-as-mask failed {}: {} - loading as RGB",
                             lfs::core::path_to_utf8(item.path), message);
                    try {
                        auto [img_data, width, height, channels] = lfs::core::load_image(
                            item.path, item.params.resize_factor, item.params.max_width);
                        if (!img_data)
                            throw std::runtime_error("RGB fallback also failed");

                        const size_t H = static_cast<size_t>(height);
                        const size_t W = static_cast<size_t>(width);
                        const size_t C = static_cast<size_t>(channels);

                        auto cpu_tensor = lfs::core::Tensor::from_blob(
                            img_data, lfs::core::TensorShape({H, W, C}),
                            lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                        auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::GPU);

                        lfs::core::free_image(img_data);

                        auto decoded = item.params.output_uint8
                                           ? lfs::core::Tensor::empty(
                                                 lfs::core::TensorShape({C, H, W}),
                                                 lfs::core::Device::GPU, lfs::core::DataType::UInt8)
                                           : lfs::core::Tensor::empty(
                                                 lfs::core::TensorShape({C, H, W}),
                                                 lfs::core::Device::GPU, lfs::core::DataType::Float32);
                        if (item.params.output_uint8) {
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_uint8, decoded, lfs::gpu_ops::ImageConversion::U8HWCToU8CHW, H, W, C, {});
                        } else {
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(gpu_uint8, decoded, lfs::gpu_ops::ImageConversion::U8HWCToF32CHW, H, W, C, {});
                        }
                        {
                            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                            if (auto it = pending_pairs_.find(item.sequence_id);
                                it != pending_pairs_.end() &&
                                it->second.loader_generation == item.loader_generation) {
                                it->second.mask_failed = true;
                                it->second.mask_expected = false;
                            }
                        }
                        try_complete_pair(item.sequence_id, item.loader_generation,
                                          std::move(decoded), std::nullopt);
                    } catch (...) {
                        const auto fallback_message =
                            describe_current_exception("non-standard RGB fallback exception");
                        LOG_ERROR("[PipelinedImageLoader] RGB fallback also failed {}: {}",
                                  lfs::core::path_to_utf8(item.path),
                                  fallback_message);
                        publish_image_failure(item.sequence_id, item.loader_generation,
                                              item.path, fallback_message);
                    }
                } else if (item.is_mask || item.is_depth || item.is_normal) {
                    LOG_WARN("[PipelinedImageLoader] Cold process {} error {}: {} - continuing without it",
                             item.is_mask ? "mask" : (item.is_depth ? "depth" : "normal"),
                             lfs::core::path_to_utf8(item.path), message);
                    std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                    const auto kind = item.is_mask    ? SidecarKind::Mask
                                      : item.is_depth ? SidecarKind::Depth
                                                      : SidecarKind::Normal;
                    fail_sidecar_locked(item.sequence_id, item.loader_generation,
                                        kind, item.path, message, lock);
                } else {
                    LOG_ERROR("[PipelinedImageLoader] Cold process error {}: {}",
                              lfs::core::path_to_utf8(item.path), message);
                    publish_image_failure(item.sequence_id, item.loader_generation,
                                          item.path, message);
                }
            }
        }
    }

    bool PipelinedImageLoader::decodes_on_gpu(const lfs::core::GpuBackend backend) {
        return backend == lfs::core::GpuBackend::CUDA && lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA);
    }

    bool PipelinedImageLoader::attach_cuda_decode_stage() {
        if (config_.backend != lfs::core::GpuBackend::CUDA)
            return false;
        if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA))
            return false;

        decode_queue_ = std::make_unique<lfs::core::TensorWorkQueue>(lfs::core::GpuBackend::CUDA);
        sidecar_queues_.reserve(config_.cold_process_threads);
        for (size_t i = 0; i < config_.cold_process_threads; ++i)
            sidecar_queues_.push_back(std::make_unique<lfs::core::TensorWorkQueue>(lfs::core::GpuBackend::CUDA));

        decoded_frame_ring_ = std::make_shared<DecodedFrameRing>(
            config_.decode_frame_ring_capacity, &running_);
        decoded_frame_ring_->set_capacity(adaptive_target_ + 2);
        decode_hwc_workspace_.resize(config_.jpeg_batch_size);

        const bool nvcodec = is_nvcodec_available();
        nvcodec_hot_path_ = nvcodec && static_cast<bool>(decode_queue_);
        cuda_immediate_ = &PipelinedImageLoader::load_image_cuda;
        return true;
    }

    void PipelinedImageLoader::start_cuda_decode_workers() {
        if (nvcodec_hot_path_)
            gpu_decode_thread_ = std::thread([this] { gpu_batch_decode_thread_func(); });
        for (size_t i = 0; i < sidecar_queues_.size(); ++i)
            cold_process_threads_.emplace_back([this, i] { cold_process_thread_func(i); });
        if (nvcodec_hot_path_)
            retain_nvcodec_loader_cache(config_.decoder_pool_size);
    }

    void PipelinedImageLoader::release_cuda_decode_stage() {
        if (config_.backend == lfs::core::GpuBackend::CUDA)
            release_nvcodec_loader_cache(config_.decoder_pool_size);
    }

} // namespace lfs::io
