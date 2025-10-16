/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <vector>
#include <unordered_map>

namespace gs::system {
    /**
     * @brief Get total physical memory in bytes
     * @return Total physical RAM in bytes, or default value if unavailable
     */
    std::size_t get_total_physical_memory();

    /**
     * @brief Get available physical memory in bytes
     * @return Available physical RAM in bytes, or default value if unavailable
     */
    std::size_t get_available_physical_memory();

    /**
     * @brief Get memory usage percentage (0.0 to 1.0)
     * @return Ratio of used memory to total memory
     */
    double get_memory_usage_ratio();
} // namespace gs::system

namespace gs::loader {
    struct LoadParams {
        int resize_factor;
        int max_width;
    };

    struct CachedImageData {
        std::vector<unsigned char> data;
        int width;
        int height;
        int channels;
        std::size_t size_bytes;
        std::chrono::steady_clock::time_point last_access;
    };

    class CacheLoader {
    public:
        // Delete copy constructor and assignment operator
        CacheLoader(const CacheLoader&) = delete;
        CacheLoader& operator=(const CacheLoader&) = delete;

        // Static method to get the singleton instance
        static CacheLoader& getInstance(
            bool use_cpu_memory,
            bool use_fs_cache) {
            std::call_once(init_flag_, [&]() {
                instance_.reset(new CacheLoader(use_cpu_memory, use_fs_cache));
            });
            return *instance_;
        }

        // Static method to get existing instance (throws if not initialized)
        static CacheLoader& getInstance() {
            if (!instance_) {
                throw std::runtime_error("CacheLoader not initialized. Call getInstance with parameters first.");
            }
            return *instance_;
        }

        // Main method - to be implemented
        [[nodiscard]] std::tuple<unsigned char*, int, int, int> load_cached_image(const std::filesystem::path& path, const LoadParams& params);

        void create_new_cache_folder();
        void clean_cache_folders();
        void clear_cpu_cache();

        void update_cache_params(bool use_cpu_memory, bool use_fs_cache) {
            use_cpu_memory_ = use_cpu_memory;
            use_fs_cache_ = use_fs_cache;
        }

    private:
        [[nodiscard]] std::tuple<unsigned char*, int, int, int> load_cached_image_from_cpu(const std::filesystem::path& path, const LoadParams& params);
        [[nodiscard]] std::tuple<unsigned char*, int, int, int> load_cached_image_from_fs(const std::filesystem::path& path, const LoadParams& params);
        // Private constructor
        CacheLoader(bool use_cpu_memory, bool use_fs_cache);

        // CPU cache params
        bool use_cpu_memory_;
        float min_cpu_free_memory_ratio_ = 0.1f; // make sure at least 10% RAM is free
        std::size_t min_cpu_free_GB_ = 1;        // min GB we want to be free

        // CPU cache storage
        std::unordered_map<std::string, CachedImageData> cpu_cache_;
        std::mutex cpu_cache_mutex_;
        std::set<std::string> image_being_loaded_cpu_;

        // Helper methods
        std::string generate_cache_key(const std::filesystem::path& path, const LoadParams& params) const;
        bool has_sufficient_memory(std::size_t required_bytes) const;
        void evict_if_needed(std::size_t required_bytes);
        void evict_until_statisfied();
        std::size_t get_cpu_cache_size() const;
        void print_cache_status() const;

        // FS cache params
        std::filesystem::path cache_folder_;
        bool use_fs_cache_;

        // Singleton instance and initialization flag
        static std::unique_ptr<CacheLoader> instance_;
        static std::once_flag init_flag_;

        std::mutex cache_mutex_;
        std::set<std::string> image_being_saved_;

        // log/debug members
        mutable std::mutex counter_mutex_;
        const bool print_cache_status_ = true;
        mutable int load_counter_ = 0;
        const int print_status_freq_num_ = 500; // every print_status_freq_num calls for load print cache status

        const std::string LFS_CACHE_PREFIX = "lfs_cache_";
    };
} // namespace gs::loader