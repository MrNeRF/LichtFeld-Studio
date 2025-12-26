/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    class Tensor;
}

namespace lfs::io {

    // Memory info
    std::size_t get_total_physical_memory();
    std::size_t get_available_physical_memory();
    double get_memory_usage_ratio();

    // Constants
    inline constexpr std::size_t BYTES_PER_GB = 1024ULL * 1024 * 1024;
    inline constexpr std::size_t DEFAULT_FALLBACK_MEMORY_GB = 16;
    inline constexpr std::size_t DEFAULT_FALLBACK_AVAILABLE_GB = 1;
    inline constexpr float DEFAULT_MIN_FREE_MEMORY_RATIO = 0.1f;
    inline constexpr float DEFAULT_MIN_FREE_GB = 1.0f;
    inline constexpr int DEFAULT_PRINT_STATUS_FREQ = 500;
    inline constexpr int DEFAULT_DECODER_POOL_SIZE = 8;
    inline constexpr std::string_view CACHE_PREFIX = "lfs_cache_";

    struct LoadParams {
        int resize_factor = 1;
        int max_width = 0;
        void* cuda_stream = nullptr;
    };

    struct CachedImageData {
        std::shared_ptr<lfs::core::Tensor> tensor;
        int width = 0;
        int height = 0;
        int channels = 0;
        std::size_t size_bytes = 0;
        std::chrono::steady_clock::time_point last_access;
    };

    struct CachedJpegBlob {
        std::vector<uint8_t> compressed_data;
        int original_width = 0;
        int original_height = 0;
        std::size_t size_bytes = 0;
        std::chrono::steady_clock::time_point last_access;
    };

    class CacheLoader {
    public:
        enum class CacheMode { Undetermined,
                               NoCache,
                               CPU_memory,
                               FileSystem };
        enum class NvImageCodecMode { Undetermined,
                                      Available,
                                      UnAvailable };

        CacheLoader(const CacheLoader&) = delete;
        CacheLoader& operator=(const CacheLoader&) = delete;

        static CacheLoader& getInstance(bool use_cpu_memory, bool use_fs_cache);
        static CacheLoader& getInstance();
        static bool hasInstance() { return instance_ != nullptr; }

        [[nodiscard]] lfs::core::Tensor load_cached_image(const std::filesystem::path& path, const LoadParams& params);

        void create_new_cache_folder();
        void reset_cache();
        void clean_cache_folders();
        void clear_cpu_cache();

        void update_cache_params(bool use_cpu_memory, bool use_fs_cache, int num_expected_images,
                                 float min_cpu_free_GB, float min_cpu_free_memory_ratio,
                                 bool print_cache_status, int print_status_freq_num);

        [[nodiscard]] CacheMode get_cache_mode() const { return cache_mode_; }
        void set_num_expected_images(int num_expected_images) { num_expected_images_ = num_expected_images; }

        static std::string to_string(CacheMode mode);

    private:
        CacheLoader(bool use_cpu_memory, bool use_fs_cache);

        [[nodiscard]] lfs::core::Tensor load_cached_image_from_cpu(const std::filesystem::path& path, const LoadParams& params);
        [[nodiscard]] lfs::core::Tensor load_cached_image_from_fs(const std::filesystem::path& path, const LoadParams& params);
        [[nodiscard]] lfs::core::Tensor load_jpeg_with_hardware_decode(const std::filesystem::path& path, const LoadParams& params);

        [[nodiscard]] std::string generate_cache_key(const std::filesystem::path& path, const LoadParams& params) const;
        [[nodiscard]] bool has_sufficient_memory(std::size_t required_bytes) const;
        [[nodiscard]] std::size_t get_cpu_cache_size() const;
        [[nodiscard]] std::size_t get_jpeg_blob_cache_size() const;
        [[nodiscard]] bool is_jpeg_format(const std::filesystem::path& path) const;

        void evict_if_needed(std::size_t required_bytes);
        void evict_until_satisfied();
        void evict_jpeg_blobs_if_needed(std::size_t required_bytes);
        void print_cache_status() const;
        void determine_cache_mode(const std::filesystem::path& path, const LoadParams& params);
        void determine_nv_image_codec();

        // Singleton
        static std::unique_ptr<CacheLoader> instance_;
        static std::once_flag init_flag_;

        // CPU cache
        bool use_cpu_memory_;
        float min_cpu_free_memory_ratio_ = DEFAULT_MIN_FREE_MEMORY_RATIO;
        float min_cpu_free_GB_ = DEFAULT_MIN_FREE_GB;
        std::unordered_map<std::string, CachedImageData> cpu_cache_;
        std::mutex cpu_cache_mutex_;
        std::set<std::string> image_being_loaded_cpu_;

        // JPEG blob cache
        std::unordered_map<std::string, CachedJpegBlob> jpeg_blob_cache_;
        std::mutex jpeg_blob_mutex_;
        std::set<std::string> jpeg_being_loaded_;

        // FS cache
        std::filesystem::path cache_folder_;
        bool use_fs_cache_;
        std::mutex cache_mutex_;
        std::set<std::string> image_being_saved_;

        // Status
        mutable std::mutex counter_mutex_;
        bool print_cache_status_ = true;
        mutable int load_counter_ = 0;
        int print_status_freq_num_ = DEFAULT_PRINT_STATUS_FREQ;

        CacheMode cache_mode_ = CacheMode::Undetermined;
        int num_expected_images_ = 0;
        NvImageCodecMode nv_image_codec_available_ = NvImageCodecMode::Undetermined;
        std::mutex nvcodec_mutex_;
    };

} // namespace lfs::io