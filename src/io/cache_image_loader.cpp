/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/cache_image_loader.hpp"
#if LFS_HAS_CUDA
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/mask_normalization.hpp"
#include "cuda/image_format_kernels.cuh"
#include "image_execution_cuda.hpp"
#endif
#include "core/event_bridge/localization_manager.hpp"
#include "core/host_metrics.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/shared_image_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"
#include <stb_image.h>
#if LFS_HAS_CUDA
#include "io/cuda/masked_image_fill.hpp"
#include "io/nvcodec_image_loader.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <bit>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>

namespace lfs::io {

    namespace {
    } // namespace

#if LFS_HAS_CUDA
    namespace {
        void synchronize_decode_stream(const cudaStream_t stream) {
            if (const auto status = cudaStreamSynchronize(stream); status != cudaSuccess)
                throw std::runtime_error(std::string("CPU-decoded image upload failed: ") + cudaGetErrorString(status));
        }
    } // namespace
#endif

    std::size_t get_total_physical_memory() {
        if (const auto memory = core::host_metrics::memory())
            return memory->total_bytes;
        return DEFAULT_FALLBACK_MEMORY_GB * BYTES_PER_GB;
    }

    std::size_t get_available_physical_memory() {
        if (const auto memory = core::host_metrics::memory())
            return memory->available_bytes;
        return DEFAULT_FALLBACK_AVAILABLE_GB * BYTES_PER_GB;
    }

    void CacheLoader::update_cache_params(bool use_cpu_memory, int num_expected_images,
                                          float min_cpu_free_GB, float min_cpu_free_memory_ratio,
                                          bool print_cache_status, int print_status_freq_num) {
        use_cpu_memory_ = use_cpu_memory;
        num_expected_images_ = num_expected_images;
        min_cpu_free_GB_ = min_cpu_free_GB;
        min_cpu_free_memory_ratio_ = min_cpu_free_memory_ratio;
        print_cache_status_ = print_cache_status;
        print_status_freq_num_ = print_status_freq_num;
    }

    std::unique_ptr<CacheLoader> CacheLoader::instance_ = nullptr;
    std::once_flag CacheLoader::init_flag_;

    CacheLoader& CacheLoader::getInstance(bool use_cpu_memory) {
        std::call_once(init_flag_, [&]() {
            instance_.reset(new CacheLoader(use_cpu_memory));
        });
        return *instance_;
    }

    CacheLoader& CacheLoader::getInstance() {
        if (!instance_) {
            throw std::runtime_error("CacheLoader not initialized");
        }
        return *instance_;
    }

    CacheLoader::CacheLoader(bool use_cpu_memory)
        : use_cpu_memory_(use_cpu_memory) {
        min_cpu_free_memory_ratio_ = std::clamp(min_cpu_free_memory_ratio_, 0.0f, 1.0f);
    }

    void CacheLoader::reset_cache() {
        clear_cpu_cache();
        cache_mode_ = CacheMode::Undetermined;
        num_expected_images_ = 0;
    }

    void CacheLoader::clear_cpu_cache() {
        {
            std::lock_guard lock(cpu_cache_mutex_);
            cpu_cache_.clear();
        }
        {
            std::lock_guard lock(jpeg_blob_mutex_);
            jpeg_blob_cache_.clear();
        }
    }

    bool CacheLoader::has_sufficient_memory(std::size_t required_bytes) const {
        const std::size_t available = get_available_physical_memory();
        const std::size_t total = get_total_physical_memory();
        const std::size_t min_free_bytes = (std::max)(static_cast<std::size_t>(total * min_cpu_free_memory_ratio_),
                                                      static_cast<std::size_t>(min_cpu_free_GB_ * BYTES_PER_GB));
        return available > required_bytes + min_free_bytes;
    }

    void CacheLoader::evict_until_satisfied() {
        const std::size_t total = get_total_physical_memory();
        const std::size_t min_free_bytes = (std::max)(static_cast<std::size_t>(total * min_cpu_free_memory_ratio_),
                                                      static_cast<std::size_t>(min_cpu_free_GB_ * BYTES_PER_GB));

        while (get_available_physical_memory() <= min_free_bytes) {
            std::lock_guard lock(cpu_cache_mutex_);
            if (cpu_cache_.empty())
                break;

            auto oldest = std::min_element(cpu_cache_.begin(), cpu_cache_.end(),
                                           [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });
            cpu_cache_.erase(oldest);
        }
    }

    void CacheLoader::evict_if_needed(std::size_t required_bytes) {
        while (!cpu_cache_.empty() && !has_sufficient_memory(required_bytes)) {
            auto oldest = std::min_element(cpu_cache_.begin(), cpu_cache_.end(),
                                           [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });
            if (oldest == cpu_cache_.end())
                break;
            cpu_cache_.erase(oldest);
        }
    }

    std::size_t CacheLoader::get_cpu_cache_size() const {
        std::size_t total = 0;
        for (const auto& [key, data] : cpu_cache_) {
            total += data.size_bytes;
        }
        return total;
    }

    std::string CacheLoader::generate_cache_key(
        const std::filesystem::path& path,
        const LoadParams& params,
        const bool include_output_format) const {
        if (params.undistort) {
            constexpr std::uint64_t offset = 14695981039346656037ULL;
            constexpr std::uint64_t prime = 1099511628211ULL;
            std::uint64_t hash = offset;
            const auto mix = [&](const std::uint64_t value) {
                for (int byte = 0; byte < 8; ++byte) {
                    hash ^= (value >> (byte * 8)) & 0xffULL;
                    hash *= prime;
                }
            };
            const auto mix_float = [&](const float value) {
                mix(std::bit_cast<std::uint32_t>(value));
            };
            const auto& undistort = *params.undistort;
            const auto grid = lfs::core::compute_undistort_grid(
                undistort, params.resize_factor, params.max_width);
            mix(4);
            mix(static_cast<std::uint64_t>(params.resize_factor));
            mix(static_cast<std::uint64_t>(params.max_width));
            mix(params.output_uint8 ? 8 : 16);
            mix(static_cast<std::uint64_t>(undistort.src_width));
            mix(static_cast<std::uint64_t>(undistort.src_height));
            mix(static_cast<std::uint64_t>(grid.width));
            mix(static_cast<std::uint64_t>(grid.height));
            mix(static_cast<std::uint64_t>(undistort.model_type));
            mix(static_cast<std::uint64_t>(undistort.num_distortion));
            mix_float(undistort.src_fx);
            mix_float(undistort.src_fy);
            mix_float(undistort.src_cx);
            mix_float(undistort.src_cy);
            mix_float(undistort.dst_fx * grid.scale_x);
            mix_float(undistort.dst_fy * grid.scale_y);
            mix_float(undistort.dst_cx * grid.scale_x);
            mix_float(undistort.dst_cy * grid.scale_y);
            for (int i = 0; i < undistort.num_distortion; ++i)
                mix_float(undistort.distortion[i]);
            std::ostringstream key;
            key << lfs::core::path_to_utf8(path) << ":udr4_" << std::hex << hash;
            if (include_output_format)
                key << (params.output_uint8 ? "_u8" : "_f32");
            if (include_output_format && params.decode_16bit)
                key << "_decode16";
            return params.mask_grayscale ? key.str() + "_mask" : key.str();
        }
        auto key = std::format("{}:rf{}_mw{}", lfs::core::path_to_utf8(path), params.resize_factor, params.max_width);
        if (include_output_format)
            key += params.output_uint8 ? "_u8" : "_f32";
        if (include_output_format && params.decode_16bit)
            key += "_decode16";
        return params.mask_grayscale ? key + "_mask" : key;
    }

    namespace {

#if LFS_HAS_CUDA
        lfs::core::Tensor quantize_rgb_to_u16_grid(
            const lfs::core::Tensor& tensor, const cudaStream_t stream) {
            const lfs::core::CUDAStreamGuard execution_scope(stream);
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
            const auto* image_ops = lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA);
            image_ops->convert(hwc, quantized, lfs::gpu_ops::ImageConversion::F32HWCToU16HWC,
                               height, width, channels, {});
            image_ops->convert(quantized, restored, lfs::gpu_ops::ImageConversion::U16HWCToF32HWC,
                               height, width, channels, {});
            return restored.permute({2, 0, 1}).contiguous();
        }

#endif
        constexpr int LANCZOS_KERNEL_SIZE = 2;

        // Upload completes before the borrowed decoder allocation is released.
        template <typename T>
        lfs::core::Tensor upload_hwc(const T* data, const int width, const int height, const int channels) {
            using namespace lfs::core;
            const auto cpu = Tensor::from_blob(
                const_cast<T*>(data),
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)}),
                Device::CPU, std::is_same_v<T, uint16_t> ? DataType::Float16 : DataType::UInt8);
            auto gpu = cpu.to(Device::GPU);
            gpu.set_name("io.image.gpu_staging");
            return gpu;
        }

        lfs::core::Tensor hwc_to_chw(const lfs::core::Tensor& hwc, const int resize_factor, const int max_width,
                                     const bool output_uint8, const bool exact_mask_alpha = false) {
            using namespace lfs::core;
            const size_t H = hwc.shape()[0], W = hwc.shape()[1], C = hwc.shape()[2];
            const bool sixteen_bit = hwc.dtype() == DataType::Float16;
            // The caller's backend scope decides where the upload landed.
            const auto* ops = shared_image_ops(gpu_backend_of(hwc).value_or(default_gpu_backend()));
            const auto [target_width, target_height] = resized_image_dimensions(
                static_cast<int>(W), static_cast<int>(H), resize_factor, max_width);
            if (exact_mask_alpha && C == 4) {
                Tensor chw = Tensor::empty({C, H, W}, Device::GPU, DataType::Float32);
                const auto conversion = sixteen_bit
                                            ? lfs::gpu_ops::ImageConversion::U16HWCToF32CHWExactMaskAlpha
                                            : lfs::gpu_ops::ImageConversion::U8HWCToF32CHWExactMaskAlpha;
                ops->convert(hwc, chw, conversion, H, W, C, {});
                if (static_cast<size_t>(target_width) != W || static_cast<size_t>(target_height) != H)
                    chw = ops->resize(chw, target_height, target_width,
                                      lfs::gpu_ops::Resample::LanczosFloatCHW, LANCZOS_KERNEL_SIZE);
                if (!output_uint8)
                    return chw;
                auto output = Tensor::empty(chw.shape(), Device::GPU, DataType::UInt8);
                ops->convert(chw, output, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW,
                             chw.shape()[1], chw.shape()[2], C, {});
                return output;
            }
            if (static_cast<size_t>(target_width) != W || static_cast<size_t>(target_height) != H) {
                Tensor source = hwc;
                if (sixteen_bit) {
                    source = Tensor::empty(hwc.shape(), Device::GPU, DataType::Float32);
                    ops->convert(hwc, source, lfs::gpu_ops::ImageConversion::U16HWCToF32HWC, H, W, C, {});
                }
                auto resized = ops->resize(source, target_height, target_width, lfs::gpu_ops::Resample::LanczosRGB, LANCZOS_KERNEL_SIZE);
                if (!output_uint8)
                    return resized;
                auto output = Tensor::empty(resized.shape(), Device::GPU, DataType::UInt8);
                ops->convert(resized, output, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW, target_height, target_width, C, {});
                return output;
            }
            auto output = Tensor::empty({C, H, W}, Device::GPU, output_uint8 ? DataType::UInt8 : DataType::Float32);
            const auto conversion = sixteen_bit
                                        ? (output_uint8 ? lfs::gpu_ops::ImageConversion::U16HWCToU8CHW : lfs::gpu_ops::ImageConversion::U16HWCToF32CHW)
                                        : (output_uint8 ? lfs::gpu_ops::ImageConversion::U8HWCToU8CHW : lfs::gpu_ops::ImageConversion::U8HWCToF32CHW);
            ops->convert(hwc, output, conversion, H, W, C, {});
            return output;
        }

    } // namespace

    const char* masked_image_fill_error() {
        const auto& localization = lfs::event::LocalizationManager::getInstance();
        return localization.hasKey("io.mask_fill_failed") ? LOC("io.mask_fill_failed")
                                                          : "Unable to prepare this image for masked training.";
    }

    lfs::core::Tensor upload_rgb_image(const unsigned char* data, const int width, const int height, const int channels,
                                       const LoadParams& params) {
        assert(data && channels == 3);
#if LFS_HAS_CUDA
        std::optional<lfs::core::CUDAStreamGuard> execution_scope;
        if (lfs::core::TensorExecutionTarget::current().backend() == lfs::core::GpuBackend::CUDA)
            execution_scope.emplace(image_execution_stream(params.cuda_stream));
#endif
        auto staging = upload_hwc(data, width, height, channels);
        if (params.masked_fill) {
#if LFS_HAS_CUDA
            if (lfs::core::gpu_backend_of(staging) != lfs::core::GpuBackend::CUDA)
                throw std::runtime_error(masked_image_fill_error());
            fill_masked_image(staging, *params.masked_fill, staging.stream());
#else
            throw std::runtime_error(masked_image_fill_error());
#endif
        }
        return hwc_to_chw(staging, params.resize_factor, params.max_width, params.output_uint8);
    }

    lfs::core::Tensor upload_rgb_image(const std::uint16_t* data, const int width, const int height, const int channels,
                                       const LoadParams& params) {
        assert(data && channels == 3);
#if LFS_HAS_CUDA
        std::optional<lfs::core::CUDAStreamGuard> execution_scope;
        if (lfs::core::TensorExecutionTarget::current().backend() == lfs::core::GpuBackend::CUDA)
            execution_scope.emplace(image_execution_stream(params.cuda_stream));
#endif
        auto staging = upload_hwc(data, width, height, channels);
        if (params.masked_fill) {
#if LFS_HAS_CUDA
            if (lfs::core::gpu_backend_of(staging) != lfs::core::GpuBackend::CUDA)
                throw std::runtime_error(masked_image_fill_error());
            fill_masked_image(staging, *params.masked_fill, staging.stream());
#else
            throw std::runtime_error(masked_image_fill_error());
#endif
        }
        return hwc_to_chw(staging, params.resize_factor, params.max_width, params.output_uint8);
    }

    lfs::core::Tensor load_rgb_image_cpu_decoded(
        const std::filesystem::path& path, const LoadParams& params, const bool decode_16bit) {
        const auto require_rgb = [&](const void* data, const int channels) {
            if (!data || channels != 3)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
        };
        if (decode_16bit || params.decode_16bit) {
            auto [data, width, height, channels] = lfs::core::load_image_u16(path, 1, 0);
            const std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owned(data, &lfs::core::free_image);
            require_rgb(data, channels);
            return upload_rgb_image(owned.get(), width, height, channels, params);
        }
        auto [data, width, height, channels] = lfs::core::load_image(path, 1, 0);
        const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> owned(data, &lfs::core::free_image);
        require_rgb(data, channels);
        return upload_rgb_image(owned.get(), width, height, channels, params);
    }

    lfs::core::Tensor load_mask_image_cpu_decoded(const std::filesystem::path& path, const LoadParams& params) {
        using namespace lfs::core;
        // Match GPU allocations made inside a loader's explicit backend scope.
        const auto backend = TensorExecutionTarget::current().backend();
        const auto stream = static_cast<cudaStream_t>(params.cuda_stream);
        std::optional<CUDAStreamGuard> execution_scope;
        if (backend == GpuBackend::CUDA)
            execution_scope.emplace(stream);
        const auto* const image_ops = shared_image_ops(backend);
        const auto upload = [&](const Tensor& cpu) {
            return backend == GpuBackend::CUDA ? cpu.to(Device::CUDA, stream)
                                               : cpu.to(Device::GPU);
        };
        const auto normalize_bytes = [&](const Tensor& bytes) {
#if LFS_HAS_CUDA
            if (backend == GpuBackend::CUDA)
                return lfs::core::cuda::normalize_mask_bytes(bytes);
#endif
            if (!image_ops || !image_ops->convert)
                throw std::runtime_error("Mask byte normalization is unavailable for the active tensor backend");
            auto normalized = Tensor::empty(bytes.shape(), Device::GPU, DataType::Float32);
            image_ops->convert(bytes, normalized, gpu_ops::ImageConversion::MaskU8HWToF32HW,
                               bytes.shape()[0], bytes.shape()[1], 1, {});
            return normalized;
        };
        const auto resize_mask = [&](const Tensor& source, const int height, const int width,
                                     const bool normalize_u8) {
#if LFS_HAS_CUDA
            if (backend == GpuBackend::CUDA)
                return lanczos_resize_grayscale(source, height, width, 2, stream,
                                                normalize_u8);
#endif
            if (!image_ops || !image_ops->resize)
                throw std::runtime_error("Mask resizing is unavailable for the active tensor backend");
            const Tensor input = normalize_u8 && source.dtype() == DataType::UInt8
                                     ? normalize_bytes(source)
                                     : source;
            return image_ops->resize(input, height, width,
                                     gpu_ops::Resample::LanczosGray, 2);
        };
        auto extension = lfs::core::path_to_utf8(path.extension());
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (extension == ".tif" || extension == ".tiff") {
            // The pipeline, Camera and source-RGB fill all need the same UInt16
            // source values before resizing onto the existing byte cache grid.
            std::tuple<float*, int, int, int> decoded{nullptr, 0, 0, 0};
            try {
                decoded = load_image_float(path);
            } catch (...) {
                LOG_DEBUG("TIFF mask precision decode unavailable; retrying the regular decoder");
            }
            auto [data, width, height, channels] = decoded;
            const std::unique_ptr<float, decltype(&free_image_float)> pixels(data, &free_image_float);
            if (pixels && width > 0 && height > 0 && channels > 0) {
                const auto cpu = Tensor::from_blob(
                    pixels.get(), TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)}),
                    Device::CPU, DataType::Float32);
                const auto image = upload(cpu);
#if LFS_HAS_CUDA
                if (backend == GpuBackend::CUDA)
                    synchronize_decode_stream(stream);
#endif
                const auto first = image.slice(2, 0, 1).squeeze(2);
                auto source = channels < 3
                                  ? first.contiguous()
                                  : (first * 77.0f + image.slice(2, 1, 2).squeeze(2) * 150.0f +
                                     image.slice(2, 2, 3).squeeze(2) * 29.0f) /
                                        256.0f;
                const auto [target_w, target_h] = resized_image_dimensions(width, height, params.resize_factor, params.max_width);
                if (target_w != width || target_h != height)
                    source = resize_mask(source, target_h, target_w, false);
#if LFS_HAS_CUDA
                if (backend == GpuBackend::CUDA) {
                    synchronize_decode_stream(stream);
                    source.set_stream(getCurrentCUDAStream());
                }
#endif
                return source.unsqueeze(0);
            }
            // An unusual TIFF still gets the pre-existing byte-decoder fallback.
        }
        int width = 0, height = 0, channels = 0;
        const std::unique_ptr<std::FILE, decltype(&std::fclose)> file(lfs::core::open_file(path, "rb"), &std::fclose);
        const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> gray(
            file ? stbi_load_from_file(file.get(), &width, &height, &channels, 1) : nullptr, &stbi_image_free);
        Tensor cpu;
        if (gray) {
            cpu = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU, DataType::UInt8);
            std::memcpy(cpu.data_ptr(), gray.get(), cpu.bytes());
        } else {
            // Preserve codecs accepted by the regular image loader but not stb.
            auto [data, w, h, c] = load_image(path, 1, 0);
            const std::unique_ptr<unsigned char, decltype(&free_image)> rgb(data, &free_image);
            if (!data || c != 3)
                throw std::runtime_error(std::string(lfs::event::LocalizationManager::getInstance().hasKey("io.mask_decode_failed")
                                                         ? LOC("io.mask_decode_failed")
                                                         : "Failed to decode mask") +
                                         ": " + path_to_utf8(path));
            width = w;
            height = h;
            cpu = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU, DataType::UInt8);
            auto* output = cpu.ptr<uint8_t>();
            for (size_t i = 0; i < cpu.numel(); ++i)
                output[i] = static_cast<uint8_t>((77 * data[i * 3] + 150 * data[i * 3 + 1] + 29 * data[i * 3 + 2]) >> 8);
        }
        const auto bytes = upload(cpu);
        const auto [target_w, target_h] = resized_image_dimensions(width, height, params.resize_factor, params.max_width);
        auto mask = target_w == width && target_h == height
                        ? normalize_bytes(bytes)
                        : resize_mask(bytes, target_h, target_w, true);
#if LFS_HAS_CUDA
        if (backend == GpuBackend::CUDA) {
            synchronize_decode_stream(stream);
            mask.set_stream(getCurrentCUDAStream());
        }
#endif
        return mask.unsqueeze(0);
    }

    lfs::core::Tensor load_rgba_image_cpu_decoded(
        const std::filesystem::path& path, const int resize_factor, const int max_width, void* const cuda_stream,
        const bool decode_16bit, const bool exact_mask_alpha, const MaskedImageFill* masked_fill,
        const bool force_masked_fill) {
        using namespace lfs::core;
#if LFS_HAS_CUDA
        std::optional<CUDAStreamGuard> execution_scope;
        if (TensorExecutionTarget::current().backend() == GpuBackend::CUDA)
            execution_scope.emplace(image_execution_stream(cuda_stream));
#endif
        const auto finish = [&](const auto* data, const int width, const int height, const int channels) {
            (void)force_masked_fill;
            if (!data || channels != 4)
                throw std::runtime_error("Failed to decode RGBA image: " + path_to_utf8(path));
            auto staging = upload_hwc(data, width, height, channels);
            if (masked_fill) {
#if LFS_HAS_CUDA
                if (gpu_backend_of(staging) != GpuBackend::CUDA)
                    throw std::runtime_error(masked_image_fill_error());
                fill_masked_image(staging, *masked_fill, static_cast<cudaStream_t>(cuda_stream));
#else
                throw std::runtime_error("Masked image fill requires the CUDA backend");
#endif
            }
            auto result = hwc_to_chw(staging, resize_factor, max_width, false, exact_mask_alpha);
            if (masked_fill) {
#if LFS_HAS_CUDA
                auto result_hwc = result.permute({1, 2, 0}).contiguous();
                fill_masked_image(result_hwc, *masked_fill, static_cast<cudaStream_t>(cuda_stream));
                result = result_hwc.permute({2, 0, 1}).contiguous();
#endif
            }
            return result;
        };
        if (decode_16bit) {
            auto [data, width, height, channels] = load_image_with_alpha_u16(path, 1, 0);
            const std::unique_ptr<uint16_t, decltype(&free_image)> owned(data, &free_image);
            return finish(owned.get(), width, height, channels);
        }
        auto [data, width, height, channels] = load_image_with_alpha(path, 1, 0);
        const std::unique_ptr<unsigned char, decltype(&free_image)> owned(data, &free_image);
        return finish(owned.get(), width, height, channels);
    }

    lfs::core::Tensor CacheLoader::load_cached_image_from_cpu(
        const std::filesystem::path& path, const LoadParams& params) {
        using namespace lfs::core;

        const std::string cache_key = generate_cache_key(path, params, true);

        // Check cache
        {
            std::lock_guard lock(cpu_cache_mutex_);
            if (auto it = cpu_cache_.find(cache_key); it != cpu_cache_.end()) {
                it->second.last_access = std::chrono::steady_clock::now();
                const auto& cached = *it->second.tensor;
                auto pinned = Tensor::empty(cached.shape(), Device::CPU, cached.dtype(), true);
                std::memcpy(pinned.data_ptr(), cached.data_ptr(), cached.bytes());
                return pinned;
            }
        }

        // Check if another thread is loading
        bool is_being_loaded = false;
        {
            std::lock_guard lock(cpu_cache_mutex_);
            is_being_loaded = image_being_loaded_cpu_.contains(cache_key);
            if (!is_being_loaded) {
                image_being_loaded_cpu_.insert(cache_key);
            }
        }

        // Concurrent load - skip caching
        if (is_being_loaded) {
            return (params.mask_grayscale ? load_mask_image_cpu_decoded(path, params) : load_rgb_image_cpu_decoded(path, params)).to(Device::CPU);
        }

        Tensor tensor;
        try {
            tensor = (params.mask_grayscale ? load_mask_image_cpu_decoded(path, params) : load_rgb_image_cpu_decoded(path, params)).to(Device::CPU);
        } catch (...) {
            std::lock_guard lock(cpu_cache_mutex_);
            image_being_loaded_cpu_.erase(cache_key);
            throw;
        }
        const int channels = static_cast<int>(tensor.shape()[0]);
        const int height = static_cast<int>(tensor.shape()[1]);
        const int width = static_cast<int>(tensor.shape()[2]);

        const std::size_t tensor_bytes = tensor.bytes();

        // Cache if memory available
        {
            std::lock_guard lock(cpu_cache_mutex_);
            if (!params.skip_blob_cache && has_sufficient_memory(tensor_bytes)) {
                evict_if_needed(tensor_bytes);
                auto unpinned = Tensor::empty_unpinned(tensor.shape(), tensor.dtype());
                std::memcpy(unpinned.data_ptr(), tensor.data_ptr(), tensor_bytes);

                cpu_cache_[cache_key] = CachedImageData{
                    .tensor = std::make_shared<Tensor>(std::move(unpinned)),
                    .width = width,
                    .height = height,
                    .channels = channels,
                    .size_bytes = tensor_bytes,
                    .last_access = std::chrono::steady_clock::now()};
            }
            image_being_loaded_cpu_.erase(cache_key);
        }

        evict_until_satisfied();
        return tensor;
    }

    void CacheLoader::determine_cache_mode(const std::filesystem::path& path, const LoadParams& params) {
        if (cache_mode_ != CacheMode::Undetermined)
            return;

        std::lock_guard lock(cache_mode_mutex_);
        if (cache_mode_ != CacheMode::Undetermined)
            return;

        if (!use_cpu_memory_) {
            cache_mode_ = CacheMode::NoCache;
            return;
        }

        if (num_expected_images_ <= 0) {
            LOG_ERROR("num_expected_images not set, disabling cache");
            cache_mode_ = CacheMode::NoCache;
            return;
        }

        clear_cpu_cache();
        auto [img_data, width, height, channels] = lfs::core::load_image(path, params.resize_factor, params.max_width);
        lfs::core::free_image(img_data);

        const std::size_t bytes_per_channel = params.output_uint8 ? sizeof(uint8_t) : sizeof(float);
        const std::size_t img_size = static_cast<std::size_t>(width) * height * channels * bytes_per_channel;
        const std::size_t required_bytes = img_size * num_expected_images_;

        if (use_cpu_memory_ && has_sufficient_memory(required_bytes)) {
            LOG_INFO("Cache mode: CPU memory");
            cache_mode_ = CacheMode::CPU_memory;
            return;
        }

        const double required_gb = static_cast<double>(required_bytes) / BYTES_PER_GB;
        const double available_gb = static_cast<double>(get_available_physical_memory()) / BYTES_PER_GB;
        LOG_DEBUG("Required {:.2f}GB, available {:.2f}GB", required_gb, available_gb);

        cache_mode_ = CacheMode::NoCache;
    }

    lfs::core::Tensor CacheLoader::load_cached_image(const std::filesystem::path& path, const LoadParams& params) {
        using namespace lfs::core;

        // Hardware decode and its layout kernels write CUDA storage. Viewer
        // image loading on another tensor backend uses the existing CPU cache.
#if LFS_HAS_CUDA
        if (TensorExecutionTarget::current().backend() == GpuBackend::CUDA) {
            determine_nv_image_codec();
            if (nv_image_codec_available_ == NvImageCodecMode::Available && is_jpeg_format(path)) {
                return load_jpeg_with_hardware_decode(path, params);
            }
        }
#endif

        determine_cache_mode(path, params);

        if (use_cpu_memory_ && cache_mode_ == CacheMode::CPU_memory) {
            print_cache_status();
            return load_cached_image_from_cpu(path, params);
        }

        return params.mask_grayscale ? load_mask_image_cpu_decoded(path, params) : load_rgb_image_cpu_decoded(path, params);
    }

    void CacheLoader::print_cache_status() const {
        if (!print_cache_status_)
            return;

        std::lock_guard lock(counter_mutex_);
        if (++load_counter_ <= print_status_freq_num_)
            return;

        load_counter_ = 0;
        const double total_gb = static_cast<double>(get_total_physical_memory()) / BYTES_PER_GB;
        const double cache_pct = 100.0 * get_cpu_cache_size() / get_total_physical_memory();
        const double jpeg_pct = 100.0 * get_jpeg_blob_cache_size() / get_total_physical_memory();

        LOG_TRACE("Cache: {} images, {} JPEG blobs | {:.1f}GB total | cache {:.1f}% | JPEG {:.1f}%",
                  cpu_cache_.size(), jpeg_blob_cache_.size(), total_gb, cache_pct, jpeg_pct);
    }

    bool CacheLoader::is_jpeg_format(const std::filesystem::path& path) const {
        std::string ext = lfs::core::path_to_utf8(path.extension());
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".jpg" || ext == ".jpeg" || ext == ".jp2";
    }

    std::size_t CacheLoader::get_jpeg_blob_cache_size() const {
        std::size_t total = 0;
        for (const auto& [key, data] : jpeg_blob_cache_) {
            total += data.size_bytes;
        }
        return total;
    }

    void CacheLoader::evict_jpeg_blobs_if_needed(std::size_t required_bytes) {
        while (!jpeg_blob_cache_.empty() && !has_sufficient_memory(required_bytes)) {
            auto oldest = std::min_element(jpeg_blob_cache_.begin(), jpeg_blob_cache_.end(),
                                           [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });
            if (oldest == jpeg_blob_cache_.end())
                break;
            jpeg_blob_cache_.erase(oldest);
        }
    }

#if LFS_HAS_CUDA
    namespace {

        NvCodecImageLoader& get_nvcodec_loader() {
            static std::once_flag init_flag;
            // nvImageCodec can throw during process shutdown after CUDA/nvJPEG teardown.
            // Keep this singleton alive for process lifetime; pipeline loaders still own their normal instances.
            static NvCodecImageLoader* instance = nullptr;

            std::call_once(init_flag, [] {
                NvCodecImageLoader::Options opts;
                opts.device_id = lfs::core::selected_cuda_device();
                opts.decoder_pool_size = DEFAULT_DECODER_POOL_SIZE;
                opts.enable_fallback = true;
                instance = new NvCodecImageLoader(opts);
            });
            return *instance;
        }

        lfs::core::Tensor decode_with_cpu_fallback(const std::filesystem::path& path, const LoadParams& params) {
            return params.mask_grayscale ? load_mask_image_cpu_decoded(path, params) : load_rgb_image_cpu_decoded(path, params);
        }

    } // anonymous namespace

    namespace {
        constexpr int CACHE_JPEG_QUALITY = 100;
    }

    lfs::core::Tensor CacheLoader::load_jpeg_with_hardware_decode(
        const std::filesystem::path& path, const LoadParams& params) {
        const auto stream = image_execution_stream(params.cuda_stream);
        const lfs::core::CUDAStreamGuard execution_scope(stream);

        using namespace lfs::core;

        const std::string cache_key = generate_cache_key(path, params, false);
        std::vector<uint8_t> jpeg_bytes;
        bool from_cache = false;

        {
            std::lock_guard lock(jpeg_blob_mutex_);
            if (auto it = jpeg_blob_cache_.find(cache_key); it != jpeg_blob_cache_.end()) {
                it->second.last_access = std::chrono::steady_clock::now();
                jpeg_bytes = it->second.compressed_data;
                from_cache = true;
            }
        }

        if (!from_cache) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file)) {
                throw std::runtime_error("Failed to open: " + lfs::core::path_to_utf8(path));
            }
            const auto size = file.tellg();
            file.seekg(0, std::ios::beg);
            jpeg_bytes.resize(size);
            if (!file.read(reinterpret_cast<char*>(jpeg_bytes.data()), size)) {
                throw std::runtime_error("Failed to read: " + lfs::core::path_to_utf8(path));
            }
        }

        const bool is_jpeg = jpeg_bytes.size() >= 2 && jpeg_bytes[0] == 0xFF && jpeg_bytes[1] == 0xD8;
        const bool is_jpeg2k = jpeg_bytes.size() >= 2 && jpeg_bytes[0] == 0xFF && jpeg_bytes[1] == 0x4F;

        if (is_jpeg || is_jpeg2k) {
            try {
                auto& nvcodec = get_nvcodec_loader();

                if (params.mask_grayscale) {
                    auto mask = nvcodec.load_image_from_memory_gpu(
                        jpeg_bytes, params.resize_factor, params.max_width, params.cuda_stream, DecodeFormat::Mask);
                    if (!from_cache && !params.skip_blob_cache) {
                        std::lock_guard lock(jpeg_blob_mutex_);
                        if (has_sufficient_memory(jpeg_bytes.size())) {
                            evict_jpeg_blobs_if_needed(jpeg_bytes.size());
                            jpeg_blob_cache_[cache_key] = {
                                .compressed_data = jpeg_bytes,
                                .size_bytes = jpeg_bytes.size(),
                                .last_access = std::chrono::steady_clock::now()};
                        }
                    }
                    return mask.unsqueeze(0);
                }

                if (from_cache) {
                    if (is_jpeg2k) {
                        auto tensor = nvcodec.decode_jpeg2k_16bit_from_memory_gpu(
                            jpeg_bytes, params.cuda_stream);
                        tensor = tensor.permute({2, 0, 1}).contiguous();
                        if (params.output_uint8) {
                            auto uint8_tensor = Tensor::empty(
                                tensor.shape(), Device::GPU, DataType::UInt8);
                            const lfs::core::CUDAStreamGuard execution_scope(
                                static_cast<cudaStream_t>(params.cuda_stream));
                            lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(tensor, uint8_tensor, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW, tensor.shape()[1], tensor.shape()[2], tensor.shape()[0], {});
                            return uint8_tensor;
                        }
                        return tensor;
                    }
                    return nvcodec.load_image_from_memory_gpu(
                        jpeg_bytes, 1, 0, params.cuda_stream, DecodeFormat::RGB, params.output_uint8);
                }

                const bool needs_resize = (params.resize_factor > 1 || params.max_width > 0);
                auto tensor = nvcodec.load_image_from_memory_gpu(
                    jpeg_bytes,
                    params.undistort ? 1 : params.resize_factor,
                    params.undistort ? 0 : params.max_width,
                    params.cuda_stream,
                    DecodeFormat::RGB, params.output_uint8);

                if (params.undistort) {
                    const bool restore_uint8 = params.output_uint8;
                    if (tensor.dtype() == DataType::UInt8) {
                        tensor = tensor.to(DataType::Float32) / 255.0f;
                    }
                    tensor = tensor.clamp(0.0f, 1.0f).contiguous();
                    const auto scaled = lfs::core::prepare_undistort_params(
                        *params.undistort,
                        static_cast<int>(tensor.shape()[2]),
                        static_cast<int>(tensor.shape()[1]),
                        params.resize_factor,
                        params.max_width);
                    tensor = lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->undistort(tensor, scaled, false);
                    if (restore_uint8) {
                        auto uint8_tensor = Tensor::empty(tensor.shape(), Device::GPU, DataType::UInt8);
                        lfs::core::shared_image_ops(lfs::core::GpuBackend::CUDA)->convert(tensor, uint8_tensor, lfs::gpu_ops::ImageConversion::F32CHWToU8CHW, tensor.shape()[1], tensor.shape()[2], tensor.shape()[0], {});
                        tensor = std::move(uint8_tensor);
                    } else {
                        tensor = quantize_rgb_to_u16_grid(
                            tensor, static_cast<cudaStream_t>(params.cuda_stream));
                    }
                }

                if (!params.skip_blob_cache) {
                    bool should_cache = false;
                    {
                        std::lock_guard lock(jpeg_blob_mutex_);
                        should_cache = !jpeg_being_loaded_.contains(cache_key);
                        if (should_cache) {
                            jpeg_being_loaded_.insert(cache_key);
                        }
                    }

                    if (should_cache) {
                        std::vector<uint8_t> cache_bytes;
                        if (params.undistort) {
                            try {
                                const auto encode_tensor = tensor.dtype() == DataType::UInt8
                                                               ? tensor.to(DataType::Float32) / 255.0f
                                                               : tensor;
                                cache_bytes = nvcodec.encode_to_jpeg2k(
                                    encode_tensor, RgbLayout::CHW, params.cuda_stream);
                            } catch (const std::exception& enc_err) {
                                LOG_DEBUG("[CacheLoader] JPEG 2000 encode failed: {}", enc_err.what());
                            } catch (...) {
                                LOG_DEBUG("[CacheLoader] JPEG 2000 encode failed with unknown error");
                            }
                        } else if (needs_resize) {
                            try {
                                cache_bytes = nvcodec.encode_to_jpeg(
                                    tensor, CACHE_JPEG_QUALITY, params.cuda_stream);
                            } catch (const std::exception& enc_err) {
                                LOG_DEBUG("[CacheLoader] JPEG re-encode failed: {}, using original bytes", enc_err.what());
                                cache_bytes = jpeg_bytes;
                            } catch (...) {
                                LOG_DEBUG("[CacheLoader] JPEG re-encode failed with unknown error, using original bytes");
                                cache_bytes = jpeg_bytes;
                            }
                        } else {
                            cache_bytes = jpeg_bytes;
                        }

                        const std::size_t cache_size = cache_bytes.size();
                        std::lock_guard lock(jpeg_blob_mutex_);
                        if (cache_size > 0 && has_sufficient_memory(cache_size)) {
                            evict_jpeg_blobs_if_needed(cache_size);
                            jpeg_blob_cache_[cache_key] = CachedJpegBlob{
                                .compressed_data = std::move(cache_bytes),
                                .size_bytes = cache_size,
                                .last_access = std::chrono::steady_clock::now()};
                            // Only remove from tracking set after successful caching
                            jpeg_being_loaded_.erase(cache_key);
                        } else {
                            // Memory insufficient - remove from tracking to allow retry later
                            jpeg_being_loaded_.erase(cache_key);
                        }
                    }
                }
                return tensor;
            } catch (const std::exception& e) {
                LOG_WARN("[CacheLoader] GPU decode failed, using CPU: {}", e.what());
                return decode_with_cpu_fallback(path, params);
            }
        }

        return decode_with_cpu_fallback(path, params);
    }

    void CacheLoader::determine_nv_image_codec() {
        if (nv_image_codec_available_ != NvImageCodecMode::Undetermined)
            return;

        std::lock_guard lock(nvcodec_mutex_);
        if (nv_image_codec_available_ != NvImageCodecMode::Undetermined)
            return;

        LOG_INFO("[CacheLoader] Checking nvImageCodec availability...");

        // is_available() now runs comprehensive diagnostics and logs detailed info
        bool available = NvCodecImageLoader::is_available();
        nv_image_codec_available_ = available
                                        ? NvImageCodecMode::Available
                                        : NvImageCodecMode::UnAvailable;

        if (available) {
            LOG_INFO("[CacheLoader] nvImageCodec: AVAILABLE - GPU-accelerated JPEG decoding enabled");
        } else {
            LOG_WARN("[CacheLoader] nvImageCodec: UNAVAILABLE - will use CPU fallback for all images");
            LOG_WARN("[CacheLoader] Check diagnostic logs above for details on why nvImageCodec is unavailable");
        }
    }
#endif

} // namespace lfs::io
