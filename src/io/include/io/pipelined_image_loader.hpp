/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace lfs::io {

    namespace config {
        constexpr size_t DEFAULT_BATCH_SIZE = 8;
        constexpr size_t DEFAULT_PREFETCH_COUNT = 8;
        constexpr size_t DEFAULT_OUTPUT_QUEUE_SIZE = 4;
        constexpr size_t DEFAULT_IO_THREADS = 2;
        constexpr size_t DEFAULT_COLD_THREADS = 2;
        constexpr size_t DEFAULT_MAX_CACHE_BYTES = 4ULL * 1024 * 1024 * 1024;
        constexpr float DEFAULT_MIN_FREE_RATIO = 0.2f;
        constexpr int DEFAULT_JPEG_QUALITY = 95;
        constexpr int DEFAULT_BATCH_TIMEOUT_MS = 3;
        constexpr int DEFAULT_OUTPUT_TIMEOUT_MS = 50;
    } // namespace config

    struct PipelinedLoaderConfig {
        size_t jpeg_batch_size = config::DEFAULT_BATCH_SIZE;
        size_t prefetch_count = config::DEFAULT_PREFETCH_COUNT;
        size_t output_queue_size = config::DEFAULT_OUTPUT_QUEUE_SIZE;
        size_t io_threads = config::DEFAULT_IO_THREADS;
        size_t cold_process_threads = config::DEFAULT_COLD_THREADS;
        size_t max_cache_bytes = config::DEFAULT_MAX_CACHE_BYTES;
        float min_free_memory_ratio = config::DEFAULT_MIN_FREE_RATIO;
        bool use_filesystem_cache = true;
        int cache_jpeg_quality = config::DEFAULT_JPEG_QUALITY;
        std::chrono::milliseconds batch_collect_timeout{config::DEFAULT_BATCH_TIMEOUT_MS};
        std::chrono::milliseconds output_wait_timeout{config::DEFAULT_OUTPUT_TIMEOUT_MS};
    };

    struct ImageRequest {
        size_t sequence_id;
        std::filesystem::path path;
        LoadParams params;
    };

    struct ReadyImage {
        size_t sequence_id;
        lfs::core::Tensor tensor;
        cudaStream_t stream = nullptr;
    };

    class PipelinedImageLoader {
    public:
        struct CacheStats {
            size_t jpeg_cache_entries = 0;
            size_t jpeg_cache_bytes = 0;
            size_t hot_path_hits = 0;
            size_t cold_path_misses = 0;
            size_t gpu_batch_decodes = 0;
            size_t total_images_loaded = 0;
            double file_read_time_ms = 0;
            double cache_lookup_time_ms = 0;
            double gpu_decode_time_ms = 0;
            double cold_process_time_ms = 0;
            size_t total_bytes_read = 0;
            size_t total_decode_calls = 0;
        };

        explicit PipelinedImageLoader(PipelinedLoaderConfig config = {});
        ~PipelinedImageLoader();

        PipelinedImageLoader(const PipelinedImageLoader&) = delete;
        PipelinedImageLoader& operator=(const PipelinedImageLoader&) = delete;

        void prefetch(const std::vector<ImageRequest>& requests);
        void prefetch(size_t sequence_id, const std::filesystem::path& path, const LoadParams& params);

        ReadyImage get();
        std::optional<ReadyImage> try_get();
        std::optional<ReadyImage> try_get_for(std::chrono::milliseconds timeout);

        size_t ready_count() const;
        size_t in_flight_count() const;
        void clear();
        void shutdown();
        bool is_running() const { return running_.load(); }
        CacheStats get_stats() const;

    private:
        struct PrefetchedImage {
            size_t sequence_id;
            std::filesystem::path path;
            LoadParams params;
            std::string cache_key;
            std::shared_ptr<std::vector<uint8_t>> jpeg_data;
            std::vector<uint8_t> raw_bytes;
            bool is_cache_hit = false;
            bool is_original_jpeg = false;
            bool needs_processing = false;
        };

        template <typename T>
        class ThreadSafeQueue {
        public:
            void push(T value) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    queue_.push(std::move(value));
                }
                cv_.notify_one();
            }

            T pop() {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
                if (shutdown_ && queue_.empty())
                    throw std::runtime_error("Queue shutdown");
                T value = std::move(queue_.front());
                queue_.pop();
                return value;
            }

            std::optional<T> try_pop() {
                std::lock_guard<std::mutex> lock(mutex_);
                if (queue_.empty())
                    return std::nullopt;
                T value = std::move(queue_.front());
                queue_.pop();
                return value;
            }

            std::optional<T> try_pop_for(std::chrono::milliseconds timeout) {
                std::unique_lock<std::mutex> lock(mutex_);
                if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; })) {
                    return std::nullopt;
                }
                if (queue_.empty())
                    return std::nullopt;
                T value = std::move(queue_.front());
                queue_.pop();
                return value;
            }

            size_t size() const {
                std::lock_guard<std::mutex> lock(mutex_);
                return queue_.size();
            }

            void clear() {
                std::lock_guard<std::mutex> lock(mutex_);
                while (!queue_.empty())
                    queue_.pop();
            }

            void signal_shutdown() {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    shutdown_ = true;
                }
                cv_.notify_all();
            }

        private:
            mutable std::mutex mutex_;
            std::condition_variable cv_;
            std::queue<T> queue_;
            bool shutdown_ = false;
        };

        void prefetch_thread_func();
        void gpu_batch_decode_thread_func();
        void cold_process_thread_func();

        std::string make_cache_key(const std::filesystem::path& path, const LoadParams& params) const;
        std::filesystem::path get_fs_cache_path(const std::string& cache_key) const;
        bool is_jpeg_data(const std::vector<uint8_t>& data) const;
        std::vector<uint8_t> read_file(const std::filesystem::path& path) const;
        void save_to_fs_cache(const std::string& cache_key, const std::vector<uint8_t>& data);

        std::shared_ptr<std::vector<uint8_t>> get_from_jpeg_cache(const std::string& cache_key);
        void put_in_jpeg_cache(const std::string& cache_key, std::shared_ptr<std::vector<uint8_t>> data);
        void put_in_jpeg_cache(const std::string& cache_key, std::vector<uint8_t>&& data);
        void evict_jpeg_cache_if_needed(size_t required_bytes);

        PipelinedLoaderConfig config_;
        std::atomic<bool> running_{false};
        std::vector<std::thread> io_threads_;
        std::thread gpu_decode_thread_;
        std::vector<std::thread> cold_process_threads_;

        ThreadSafeQueue<ImageRequest> prefetch_queue_;
        ThreadSafeQueue<PrefetchedImage> hot_queue_;
        ThreadSafeQueue<PrefetchedImage> cold_queue_;
        ThreadSafeQueue<ReadyImage> output_queue_;

        struct JpegCacheEntry {
            std::shared_ptr<std::vector<uint8_t>> data;
            std::chrono::steady_clock::time_point last_access;
            size_t size_bytes;
        };
        std::unordered_map<std::string, JpegCacheEntry> jpeg_cache_;
        mutable std::mutex jpeg_cache_mutex_;
        std::atomic<size_t> jpeg_cache_bytes_{0};

        std::filesystem::path fs_cache_folder_;
        std::mutex fs_cache_mutex_;
        std::set<std::string> files_being_written_;

        mutable std::mutex stats_mutex_;
        CacheStats stats_;
        std::atomic<size_t> in_flight_{0};
    };

} // namespace lfs::io
