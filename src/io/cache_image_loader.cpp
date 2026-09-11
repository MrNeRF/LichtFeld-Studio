/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/cache_image_loader.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "io/nvcodec_image_loader.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>

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
        auto key = std::format("{}:rf{}_mw{}", lfs::core::path_to_utf8(path), params.resize_factor, params.max_width);
        if (params.undistort)
            key += "_ud";
        if (include_output_format)
            key += params.output_uint8 ? "_u8" : "_f32";
        return key;
    }

    namespace {

        lfs::core::Tensor preprocess_loaded_rgb_image(
            unsigned char* data,
            const int width,
            const int height,
            const int channels,
            const bool output_uint8) {
            using namespace lfs::core;

            auto tensor = Tensor::from_blob(
                data,
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)}),
                Device::CPU,
                DataType::UInt8);

            if (output_uint8) {
                tensor = tensor.permute({2, 0, 1}).contiguous();
            } else {
                tensor = (tensor.to(DataType::Float32) / 255.0f).permute({2, 0, 1}).contiguous();
            }
            free_image(data);
            return tensor;
        }

    } // namespace

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
            auto [img_data, width, height, channels] = load_image(path, params.resize_factor, params.max_width);
            return preprocess_loaded_rgb_image(img_data, width, height, channels, params.output_uint8);
        }

        // Load image
        auto [img_data, width, height, channels] = load_image(path, params.resize_factor, params.max_width);
        if (!img_data) {
            std::lock_guard lock(cpu_cache_mutex_);
            image_being_loaded_cpu_.erase(cache_key);
            throw std::runtime_error("Failed to load: " + lfs::core::path_to_utf8(path));
        }

        auto tensor = preprocess_loaded_rgb_image(img_data, width, height, channels, params.output_uint8);

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

        auto [data, width, height, channels] = load_image(path, params.resize_factor, params.max_width);
        return preprocess_loaded_rgb_image(data, width, height, channels, params.output_uint8);
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
            using namespace lfs::core;

            auto [img_data, width, height, channels] = load_image(path, params.resize_factor, params.max_width);
            if (!img_data) {
                throw std::runtime_error("Failed to load: " + lfs::core::path_to_utf8(path));
            }

            auto cpu_tensor = Tensor::empty_unpinned(
                TensorShape({static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)}),
                DataType::UInt8);
            std::memcpy(cpu_tensor.data_ptr(), img_data, static_cast<size_t>(height) * width * channels);
            free_image(img_data);

            auto gpu_uint8 = cpu_tensor.cuda();
            const auto H = static_cast<size_t>(height);
            const auto W = static_cast<size_t>(width);
            const auto C = static_cast<size_t>(channels);

            if (params.output_uint8) {
                auto output = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::UInt8);
                lfs::io::cuda::launch_uint8_hwc_to_uint8_chw(
                    gpu_uint8.ptr<uint8_t>(), output.ptr<uint8_t>(), H, W, C,
                    static_cast<cudaStream_t>(params.cuda_stream));
                return output;
            }

            auto output = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);
            lfs::io::cuda::launch_uint8_hwc_to_float32_chw(
                gpu_uint8.ptr<uint8_t>(), output.ptr<float>(), H, W, C,
                static_cast<cudaStream_t>(params.cuda_stream));
            return output;
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

        if (is_jpeg) {
            try {
                auto& nvcodec = get_nvcodec_loader();

                if (from_cache) {
                    return nvcodec.load_image_from_memory_gpu(
                        jpeg_bytes, 1, 0, params.cuda_stream, DecodeFormat::RGB, params.output_uint8);
                }

                const bool needs_resize = (params.resize_factor > 1 || params.max_width > 0);
                auto tensor = nvcodec.load_image_from_memory_gpu(
                    jpeg_bytes, params.resize_factor, params.max_width, params.cuda_stream,
                    DecodeFormat::RGB, params.output_uint8);

                if (params.undistort) {
                    const bool restore_uint8 = params.output_uint8 && tensor.dtype() == DataType::UInt8;
                    if (restore_uint8) {
                        tensor = tensor.to(DataType::Float32) / 255.0f;
                    }
                    const auto scaled = lfs::core::scale_undistort_params(
                        *params.undistort,
                        static_cast<int>(tensor.shape()[2]),
                        static_cast<int>(tensor.shape()[1]));
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
                        if (needs_resize || params.undistort) {
                            // Re-encode resized image
                            try {
                                cache_bytes = nvcodec.encode_to_jpeg(tensor, CACHE_JPEG_QUALITY, params.cuda_stream);
                            } catch (const std::exception& enc_err) {
                                LOG_DEBUG("[CacheLoader] JPEG re-encode failed: {}, using original bytes", enc_err.what());
                                cache_bytes = jpeg_bytes; // Fall back to original
                            } catch (...) {
                                LOG_DEBUG("[CacheLoader] JPEG re-encode failed with unknown error, using original bytes");
                                cache_bytes = jpeg_bytes; // Fall back to original
                            }
                        } else {
                            // No resize - cache original bytes directly
                            cache_bytes = jpeg_bytes;
                        }

                        const std::size_t cache_size = cache_bytes.size();
                        std::lock_guard lock(jpeg_blob_mutex_);
                        if (has_sufficient_memory(cache_size)) {
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
