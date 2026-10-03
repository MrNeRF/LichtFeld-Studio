/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/cache_image_loader.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "io/nvcodec_image_loader.hpp"

#include <algorithm>
#include <bit>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <type_traits>

#ifdef __linux__
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lfs::io {

    std::size_t get_total_physical_memory() {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.totalram * info.mem_unit;
        }
#elif defined(_WIN32)
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&mem_info)) {
            return mem_info.ullTotalPhys;
        }
#endif
        return DEFAULT_FALLBACK_MEMORY_GB * BYTES_PER_GB;
    }

    std::size_t get_available_physical_memory() {
#ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemAvailable:") == 0) {
                    std::istringstream iss(line);
                    std::string label;
                    std::size_t value_kb;
                    iss >> label >> value_kb;
                    return value_kb * 1024;
                }
            }
        }
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.freeram * info.mem_unit;
        }
#elif defined(_WIN32)
        MEMORYSTATUSEX mem_info;
        mem_info.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&mem_info)) {
            return mem_info.ullAvailPhys;
        }
#endif
        return DEFAULT_FALLBACK_AVAILABLE_GB * BYTES_PER_GB;
    }

    double get_memory_usage_ratio() {
        const std::size_t total = get_total_physical_memory();
        if (total == 0)
            return 1.0;
        const std::size_t available = get_available_physical_memory();
        return 1.0 - (static_cast<double>(available) / static_cast<double>(total));
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
            return key.str();
        }
        auto key = std::format("{}:rf{}_mw{}", lfs::core::path_to_utf8(path), params.resize_factor, params.max_width);
        if (include_output_format)
            key += params.output_uint8 ? "_u8" : "_f32";
        return key;
    }

    namespace {

        lfs::core::Tensor quantize_rgb_to_u16_grid(
            const lfs::core::Tensor& tensor, const cudaStream_t stream) {
            const size_t channels = tensor.shape()[0];
            const size_t height = tensor.shape()[1];
            const size_t width = tensor.shape()[2];
            auto hwc = tensor.permute({1, 2, 0}).contiguous();
            auto quantized = lfs::core::Tensor::empty(
                {height, width, channels}, lfs::core::Device::CUDA,
                lfs::core::DataType::Float16);
            auto restored = lfs::core::Tensor::empty(
                {height, width, channels}, lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
            cuda::launch_float32_hwc_to_uint16_hwc(
                hwc.ptr<float>(), reinterpret_cast<uint16_t*>(quantized.data_ptr()),
                height, width, channels, stream);
            cuda::launch_uint16_hwc_to_float32_hwc(
                reinterpret_cast<const uint16_t*>(quantized.data_ptr()), restored.ptr<float>(),
                height, width, channels, stream);
            return restored.permute({2, 0, 1}).contiguous();
        }

        constexpr int LANCZOS_KERNEL_SIZE = 2;

        void synchronize_decode_stream(const cudaStream_t stream) {
            if (const auto status = cudaStreamSynchronize(stream); status != cudaSuccess)
                throw std::runtime_error(std::string("CPU-decoded image upload failed: ") + cudaGetErrorString(status));
        }

        // Float16 is only a 2-byte container for uint16 samples (no UInt16 dtype).
        template <typename T>
        lfs::core::Tensor upload_hwc(const T* data, const int width, const int height, const int channels,
                                     const cudaStream_t stream) {
            using namespace lfs::core;
            const auto cpu = Tensor::from_blob(
                const_cast<T*>(data),
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)}),
                Device::CPU, std::is_same_v<T, uint16_t> ? DataType::Float16 : DataType::UInt8);
            auto gpu = cpu.to(Device::CUDA, stream);
            gpu.set_name("io.image.gpu_staging");
            return gpu;
        }

        lfs::core::Tensor float_chw_to_uint8(const lfs::core::Tensor& image, const cudaStream_t stream) {
            using namespace lfs::core;
            auto output = Tensor::empty(image.shape(), Device::CUDA, DataType::UInt8);
            output.set_stream(stream);
            image.sync_to_stream(stream);
            cuda::launch_float32_chw_to_uint8_chw(
                image.ptr<float>(), output.ptr<uint8_t>(),
                image.shape()[1], image.shape()[2], image.shape()[0], stream);
            return output;
        }

        lfs::core::Tensor lanczos_to_chw(const lfs::core::Tensor& hwc, const int target_width, const int target_height,
                                         const cudaStream_t stream) {
            hwc.sync_to_stream(stream);
            auto resized = lfs::core::lanczos_resize(hwc, target_height, target_width, LANCZOS_KERNEL_SIZE, stream);
            if (!resized.is_valid())
                throw std::runtime_error("GPU Lanczos resize failed for CPU-decoded image");
            return resized;
        }

        // Runs on the staging tensor's stream when the caller passed none, and synchronizes
        // before returning so every staging buffer outlives the kernels reading it.
        lfs::core::Tensor hwc_to_chw(const lfs::core::Tensor& hwc, const int resize_factor, const int max_width,
                                     const bool output_uint8, const cudaStream_t requested_stream) {
            using namespace lfs::core;
            const size_t H = hwc.shape()[0];
            const size_t W = hwc.shape()[1];
            const size_t C = hwc.shape()[2];
            const bool sixteen_bit = hwc.dtype() == DataType::Float16;
            const auto [target_width, target_height] = resized_image_dimensions(
                static_cast<int>(W), static_cast<int>(H), resize_factor, max_width);
            const cudaStream_t stream = requested_stream ? requested_stream : hwc.stream();
            hwc.sync_to_stream(stream);
            Tensor output;
            if (static_cast<size_t>(target_width) != W || static_cast<size_t>(target_height) != H) {
                Tensor source = hwc;
                if (sixteen_bit) {
                    source = Tensor::empty(hwc.shape(), Device::CUDA, DataType::Float32);
                    source.set_stream(stream);
                    cuda::launch_uint16_hwc_to_float32_hwc(
                        static_cast<const uint16_t*>(hwc.data_ptr()), source.ptr<float>(), H, W, C, stream);
                }
                const auto resized = lanczos_to_chw(source, target_width, target_height, stream);
                output = output_uint8 ? float_chw_to_uint8(resized, stream) : resized;
            } else {
                output = Tensor::empty(TensorShape({C, H, W}), Device::CUDA,
                                       output_uint8 ? DataType::UInt8 : DataType::Float32);
                output.set_stream(stream);
                if (sixteen_bit && output_uint8) {
                    cuda::launch_uint16_hwc_to_uint8_chw(
                        static_cast<const uint16_t*>(hwc.data_ptr()), output.ptr<uint8_t>(), H, W, C, stream);
                } else if (sixteen_bit) {
                    cuda::launch_uint16_hwc_to_float32_chw(
                        static_cast<const uint16_t*>(hwc.data_ptr()), output.ptr<float>(), H, W, C, stream);
                } else if (output_uint8) {
                    cuda::launch_uint8_hwc_to_uint8_chw(hwc.ptr<uint8_t>(), output.ptr<uint8_t>(), H, W, C, stream);
                } else {
                    cuda::launch_uint8_hwc_to_float32_chw(hwc.ptr<uint8_t>(), output.ptr<float>(), H, W, C, stream);
                }
            }
            synchronize_decode_stream(stream);
            output.set_stream(getCurrentCUDAStream());
            return output;
        }

    } // namespace

    lfs::core::Tensor load_rgb_image_cpu_decoded(
        const std::filesystem::path& path, const LoadParams& params, const bool decode_16bit) {
        const auto stream = static_cast<cudaStream_t>(params.cuda_stream);
        const auto finish = [&](const auto* data, const int width, const int height, const int channels) {
            if (!data || channels != 3)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            return hwc_to_chw(upload_hwc(data, width, height, channels, stream),
                              params.resize_factor, params.max_width, params.output_uint8, stream);
        };
        if (decode_16bit) {
            auto [data, width, height, channels] = lfs::core::load_image_u16(path, 1, 0);
            const std::unique_ptr<uint16_t, decltype(&lfs::core::free_image)> owned(data, &lfs::core::free_image);
            return finish(owned.get(), width, height, channels);
        }
        auto [data, width, height, channels] = lfs::core::load_image(path, 1, 0);
        const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> owned(data, &lfs::core::free_image);
        return finish(owned.get(), width, height, channels);
    }

    lfs::core::Tensor load_rgba_image_cpu_decoded(
        const std::filesystem::path& path, const int resize_factor, const int max_width, void* const cuda_stream,
        const bool decode_16bit) {
        using namespace lfs::core;
        const auto stream = static_cast<cudaStream_t>(cuda_stream);
        const auto finish = [&](const auto* data, const int width, const int height, const int channels) {
            if (!data || channels != 4)
                throw std::runtime_error("Failed to decode RGBA image: " + path_to_utf8(path));
            return hwc_to_chw(upload_hwc(data, width, height, channels, stream),
                              resize_factor, max_width, false, stream);
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
            return load_rgb_image_cpu_decoded(path, params).to(Device::CPU);
        }

        Tensor tensor;
        try {
            tensor = load_rgb_image_cpu_decoded(path, params).to(Device::CPU);
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

        determine_nv_image_codec();

        if (nv_image_codec_available_ == NvImageCodecMode::Available && is_jpeg_format(path)) {
            return load_jpeg_with_hardware_decode(path, params);
        }

        determine_cache_mode(path, params);

        if (use_cpu_memory_ && cache_mode_ == CacheMode::CPU_memory) {
            print_cache_status();
            return load_cached_image_from_cpu(path, params);
        }

        return load_rgb_image_cpu_decoded(path, params);
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
        std::string ext = path.extension().string();
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

    namespace {

        NvCodecImageLoader& get_nvcodec_loader() {
            static std::once_flag init_flag;
            // nvImageCodec can throw during process shutdown after CUDA/nvJPEG teardown.
            // Keep this singleton alive for process lifetime; pipeline loaders still own their normal instances.
            static NvCodecImageLoader* instance = nullptr;

            std::call_once(init_flag, [] {
                NvCodecImageLoader::Options opts;
                opts.device_id = 0;
                opts.decoder_pool_size = DEFAULT_DECODER_POOL_SIZE;
                opts.enable_fallback = true;
                instance = new NvCodecImageLoader(opts);
            });
            return *instance;
        }

        lfs::core::Tensor decode_with_cpu_fallback(const std::filesystem::path& path, const LoadParams& params) {
            return load_rgb_image_cpu_decoded(path, params);
        }

    } // anonymous namespace

    namespace {
        constexpr int CACHE_JPEG_QUALITY = 100;
    }

    lfs::core::Tensor CacheLoader::load_jpeg_with_hardware_decode(
        const std::filesystem::path& path, const LoadParams& params) {
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

                if (from_cache) {
                    if (is_jpeg2k) {
                        auto tensor = nvcodec.decode_jpeg2k_16bit_from_memory_gpu(
                            jpeg_bytes, params.cuda_stream);
                        tensor = tensor.permute({2, 0, 1}).contiguous();
                        if (params.output_uint8) {
                            auto uint8_tensor = Tensor::empty(
                                tensor.shape(), Device::CUDA, DataType::UInt8);
                            lfs::io::cuda::launch_float32_chw_to_uint8_chw(
                                tensor.ptr<float>(), uint8_tensor.ptr<uint8_t>(),
                                tensor.shape()[1], tensor.shape()[2], tensor.shape()[0],
                                static_cast<cudaStream_t>(params.cuda_stream));
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
                    tensor = lfs::core::undistort_image(
                        tensor, scaled, static_cast<cudaStream_t>(params.cuda_stream));
                    if (restore_uint8) {
                        auto uint8_tensor = Tensor::empty(tensor.shape(), Device::CUDA, DataType::UInt8);
                        lfs::io::cuda::launch_float32_chw_to_uint8_chw(
                            tensor.ptr<float>(),
                            uint8_tensor.ptr<uint8_t>(),
                            tensor.shape()[1],
                            tensor.shape()[2],
                            tensor.shape()[0],
                            static_cast<cudaStream_t>(params.cuda_stream));
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
                                    encode_tensor, params.cuda_stream);
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

} // namespace lfs::io
