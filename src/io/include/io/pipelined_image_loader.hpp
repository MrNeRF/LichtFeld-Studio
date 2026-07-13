/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct CUstream_st;
using cudaStream_t = CUstream_st*;
struct CUevent_st;

namespace lfs::io {

    class NvCodecImageLoader;

    namespace config {
        constexpr size_t DEFAULT_BATCH_SIZE = 8;
        constexpr size_t DEFAULT_PREFETCH_COUNT = 8;
        constexpr size_t DEFAULT_OUTPUT_QUEUE_SIZE = 4;
        constexpr size_t DEFAULT_IO_THREADS = 2;
        constexpr size_t DEFAULT_COLD_THREADS = 2;
        constexpr size_t DEFAULT_MAX_CACHE_BYTES = 8ULL * 1024 * 1024 * 1024;
        constexpr float DEFAULT_MIN_FREE_RATIO = 0.2f;
        constexpr int DEFAULT_JPEG_QUALITY = 95;
        constexpr int DEFAULT_BATCH_TIMEOUT_MS = 3;
        constexpr int DEFAULT_OUTPUT_TIMEOUT_MS = 50;
    } // namespace config

    struct PipelinedLoaderConfig {
        size_t jpeg_batch_size = config::DEFAULT_BATCH_SIZE;
        size_t prefetch_count = config::DEFAULT_PREFETCH_COUNT;
        size_t output_queue_size = config::DEFAULT_OUTPUT_QUEUE_SIZE;
        size_t decoder_pool_size = config::DEFAULT_BATCH_SIZE;
        size_t io_threads = config::DEFAULT_IO_THREADS;
        size_t cold_process_threads = config::DEFAULT_COLD_THREADS;
        size_t max_cache_bytes = config::DEFAULT_MAX_CACHE_BYTES;
        float min_free_memory_ratio = config::DEFAULT_MIN_FREE_RATIO;
        bool use_filesystem_cache = true;
        int cache_jpeg_quality = config::DEFAULT_JPEG_QUALITY;
        std::chrono::milliseconds batch_collect_timeout{config::DEFAULT_BATCH_TIMEOUT_MS};
        std::chrono::milliseconds output_wait_timeout{config::DEFAULT_OUTPUT_TIMEOUT_MS};
        bool use_16bit_color = false;
    };

    /**
     * @brief Parameters for mask processing
     */
    struct MaskParams {
        bool invert = false;    // Invert mask values (1.0 - mask)
        float threshold = 0.0f; // Binary threshold: >= threshold → 1.0, else → 0.0
    };

    struct ImageRequest {
        size_t sequence_id;
        std::filesystem::path path;
        LoadParams params;
        // Optional mask to load alongside the image
        std::optional<std::filesystem::path> mask_path;
        // Optional depth map to load alongside the image
        std::optional<std::filesystem::path> depth_path;
        // Optional normal map to load alongside the image
        std::optional<std::filesystem::path> normal_path;
        // Convert the normal prior from OpenGL to OpenCV camera convention
        // (dataset-level decision made by the trainer's startup probe)
        bool normal_flip_yz = false;
        // Invert the sRGB display transform before the v = n*0.5+0.5 decode
        bool normal_srgb = false;
        bool normal_transform_world_to_camera = false;
        std::array<float, 9> normal_world_to_camera{};
        int aux_target_width = 0;
        int aux_target_height = 0;
        MaskParams mask_params;
        bool extract_alpha_as_mask = false;
        MaskParams alpha_mask_params;
        const lfs::core::UndistortParams* undistort = nullptr;
    };

    struct ReadyImage {
        size_t sequence_id;
        lfs::core::Tensor tensor;              // Image tensor [C,H,W], float32
        std::optional<lfs::core::Tensor> mask; // Optional mask [H,W], float32
        cudaStream_t stream = nullptr;
        std::optional<lfs::core::Tensor> depth;  // Optional depth [H,W], float32
        std::optional<lfs::core::Tensor> normal; // Optional normals [3,H,W], float32 in [-1,1]
        // Depth and normal record readiness on different worker streams, so
        // each carries its own event; consumers must wait on both.
        CUevent_st* depth_ready_event = nullptr;
        CUevent_st* normal_ready_event = nullptr;
    };

    class LFS_IO_API PipelinedImageLoader {
    public:
        struct GpuMemoryStats {
            size_t output_image_bytes = 0;
            size_t output_mask_bytes = 0;
            size_t output_depth_bytes = 0;
            size_t output_normal_bytes = 0;
            size_t pending_image_bytes = 0;
            size_t pending_mask_bytes = 0;
            size_t pending_depth_bytes = 0;
            size_t pending_normal_bytes = 0;

            [[nodiscard]] size_t total_bytes() const {
                return output_image_bytes + output_mask_bytes + output_depth_bytes + output_normal_bytes +
                       pending_image_bytes + pending_mask_bytes + pending_depth_bytes + pending_normal_bytes;
            }
        };

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
            // Mask loading stats
            size_t masks_loaded = 0;
            size_t mask_cache_hits = 0;
            size_t mask_cache_misses = 0;
            size_t depths_loaded = 0;
            size_t normals_loaded = 0;
            // Pending pairs (for leak detection)
            size_t pending_pairs_count = 0;
            // Queue sizes (for monitoring pipeline state)
            size_t prefetch_queue_size = 0;
            size_t hot_queue_size = 0;
            size_t cold_queue_size = 0;
            size_t output_queue_size = 0;
            size_t output_image_bytes = 0;
            size_t output_mask_bytes = 0;
            size_t output_depth_bytes = 0;
            size_t output_normal_bytes = 0;
            size_t pending_image_bytes = 0;
            size_t pending_mask_bytes = 0;
            size_t pending_depth_bytes = 0;
            size_t pending_normal_bytes = 0;
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

        lfs::core::Tensor load_image_immediate(
            const std::filesystem::path& path, const LoadParams& params);

        size_t ready_count() const;
        size_t in_flight_count() const;
        void clear();
        void shutdown();
        bool is_running() const { return running_.load(); }
        CacheStats get_stats() const;
        GpuMemoryStats get_gpu_memory_stats() const;

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
            // Optional auxiliary image fields
            bool is_mask = false;        // True if this item is a mask (not an image)
            bool is_depth = false;       // True if this item is a depth map (not an image)
            bool is_normal = false;      // True if this item is a normal map (not an image)
            bool normal_flip_yz = false; // OpenGL -> OpenCV convention flip (only used if is_normal)
            bool normal_srgb = false;    // Invert sRGB encoding before decode (only used if is_normal)
            bool normal_transform_world_to_camera = false;
            std::array<float, 9> normal_world_to_camera{};
            int aux_target_width = 0;
            int aux_target_height = 0;
            MaskParams mask_params; // Invert/threshold params (only used if is_mask)
            bool alpha_as_mask = false;
            MaskParams alpha_mask_params;
            const lfs::core::UndistortParams* undistort = nullptr;
        };

        struct CachedJpegHit {
            std::shared_ptr<std::vector<uint8_t>> data;
            bool from_base_key = false;
        };

        // Pairing buffer: wait for the image and requested auxiliary images before output
        struct PendingPair {
            std::optional<lfs::core::Tensor> image;
            std::optional<lfs::core::Tensor> mask;
            std::optional<lfs::core::Tensor> depth;
            std::optional<lfs::core::Tensor> normal;
            cudaStream_t stream = nullptr;
            CUevent_st* depth_ready_event = nullptr;
            CUevent_st* normal_ready_event = nullptr;
            bool mask_expected = false; // True if a mask was requested for this sequence_id
            bool depth_expected = false;
            bool normal_expected = false;
            size_t image_bytes = 0;
            size_t mask_bytes = 0;
            size_t depth_bytes = 0;
            size_t normal_bytes = 0;
        };

        using PendingPairMap = std::unordered_map<size_t, PendingPair>;
        using PendingPairIterator = PendingPairMap::iterator;

        template <typename T>
        class ThreadSafeQueue {
        public:
            explicit ThreadSafeQueue(size_t capacity = 0)
                : capacity_(capacity) {}

            bool push(T value) {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    not_full_cv_.wait(lock, [this] {
                        return shutdown_ || capacity_ == 0 || queue_.size() < capacity_;
                    });
                    if (shutdown_)
                        return false;
                    queue_.push(std::move(value));
                }
                cv_.notify_one();
                return true;
            }

            template <typename OnDrop>
            bool push(T value, OnDrop&& on_drop) {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    not_full_cv_.wait(lock, [this] {
                        return shutdown_ || capacity_ == 0 || queue_.size() < capacity_;
                    });
                    if (shutdown_) {
                        on_drop(value);
                        return false;
                    }
                    queue_.push(std::move(value));
                }
                cv_.notify_one();
                return true;
            }

            T pop() {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
                if (shutdown_ && queue_.empty())
                    throw std::runtime_error("Queue shutdown");
                T value = std::move(queue_.front());
                queue_.pop();
                not_full_cv_.notify_one();
                return value;
            }

            std::optional<T> try_pop() {
                std::lock_guard<std::mutex> lock(mutex_);
                if (queue_.empty())
                    return std::nullopt;
                T value = std::move(queue_.front());
                queue_.pop();
                not_full_cv_.notify_one();
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
                not_full_cv_.notify_one();
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
                not_full_cv_.notify_all();
            }

            void signal_shutdown() {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    shutdown_ = true;
                }
                cv_.notify_all();
                not_full_cv_.notify_all();
            }

        private:
            mutable std::mutex mutex_;
            std::condition_variable cv_;
            std::condition_variable not_full_cv_;
            std::queue<T> queue_;
            size_t capacity_ = 0;
            bool shutdown_ = false;
        };

        void prefetch_thread_func();
        void gpu_batch_decode_thread_func();
        void cold_process_thread_func(size_t worker_index);

        std::string make_cache_key(const std::filesystem::path& path, const LoadParams& params) const;
        std::filesystem::path get_fs_cache_path(const std::string& cache_key) const;
        bool is_jpeg_data(const std::vector<uint8_t>& data) const;
        std::vector<uint8_t> read_file(const std::filesystem::path& path) const;
        void save_to_fs_cache(const std::string& cache_key, const std::vector<uint8_t>& data);
        std::shared_ptr<std::vector<uint8_t>> load_cached_jpeg_blob(const std::string& cache_key);
        std::optional<CachedJpegHit> find_cached_jpeg(const std::string& cache_key,
                                                      const std::string& base_key,
                                                      const std::filesystem::path& source_path);
        lfs::core::Tensor decode_file_on_cpu(const std::filesystem::path& path,
                                             const LoadParams& params) const;
        void write_derived_cache(NvCodecImageLoader& nvcodec,
                                 const lfs::core::Tensor& tensor,
                                 const std::string& cache_key,
                                 void* cuda_stream);

        enum class SidecarKind : uint8_t {
            Depth,
            Normal
        };

        std::string make_sidecar_key(const PrefetchedImage& item, SidecarKind kind) const;
        void write_sidecar_cache(NvCodecImageLoader& nvcodec,
                                 const lfs::core::Tensor& tensor,
                                 const PrefetchedImage& item,
                                 SidecarKind kind,
                                 void* cuda_stream);
        lfs::core::Tensor decode_cached_sidecar(NvCodecImageLoader& nvcodec,
                                                const PrefetchedImage& item,
                                                void* cuda_stream);
        CUevent_st* record_sidecar_ready_event(cudaStream_t stream);
        std::pair<int, int> sidecar_target_size(const PrefetchedImage& item, int src_w, int src_h) const;

        std::shared_ptr<std::vector<uint8_t>> get_from_jpeg_cache(const std::string& cache_key);
        void put_in_jpeg_cache(const std::string& cache_key, std::shared_ptr<std::vector<uint8_t>> data);
        void put_in_jpeg_cache(const std::string& cache_key, std::vector<uint8_t>&& data);
        void invalidate_cache_entry(const std::string& cache_key);
        void evict_jpeg_cache_if_needed(size_t required_bytes);

        // Auxiliary image pairing helpers
        std::string make_mask_cache_key(
            const std::filesystem::path& path,
            const LoadParams& params) const;
        void try_complete_pair(
            size_t sequence_id,
            std::optional<lfs::core::Tensor> image,
            std::optional<lfs::core::Tensor> mask,
            cudaStream_t stream,
            std::optional<lfs::core::Tensor> depth = std::nullopt,
            std::optional<lfs::core::Tensor> normal = std::nullopt,
            CUevent_st* sidecar_ready_event = nullptr);
        void try_push_ready_locked(size_t sequence_id,
                                   PendingPairIterator it,
                                   std::unique_lock<std::mutex>& pending_lock);
        void add_output_ready_bytes(const ReadyImage& ready);
        void release_output_ready_bytes(const ReadyImage& ready);
        void push_output_ready(ReadyImage ready);
        void erase_pending_pair_locked(PendingPairIterator it);
        void destroy_sidecar_ready_event(CUevent_st*& event);
        void clear_output_queue();
        void clear_pending_pairs();
        void reset_pipeline_gpu_bytes();

        PipelinedLoaderConfig config_;
        std::atomic<bool> running_{false};
        std::vector<std::thread> io_threads_;
        std::thread gpu_decode_thread_;
        std::vector<std::thread> cold_process_threads_;

        // Non-blocking stream for the hot GPU decode path, so image decode and
        // H2D work overlap training instead of serializing on the legacy stream.
        // Images are still stream-synced before handoff (materialized on arrival).
        cudaStream_t decode_stream_ = nullptr;
        std::vector<cudaStream_t> sidecar_streams_;

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

        // Cleared on the first failed JPEG 2000 encode (e.g. nvjpeg2k extension not
        // built) so 16-bit runs degrade to uncached decoding with a single error.
        std::atomic<bool> jpeg2k_cache_available_{true};

        mutable std::mutex stats_mutex_;
        CacheStats stats_;
        std::atomic<size_t> in_flight_{0};
        std::atomic<size_t> output_image_bytes_{0};
        std::atomic<size_t> output_mask_bytes_{0};
        std::atomic<size_t> output_depth_bytes_{0};
        std::atomic<size_t> output_normal_bytes_{0};
        std::atomic<size_t> pending_image_bytes_{0};
        std::atomic<size_t> pending_mask_bytes_{0};
        std::atomic<size_t> pending_depth_bytes_{0};
        std::atomic<size_t> pending_normal_bytes_{0};

        // Pairing buffer for image and auxiliary image delivery
        PendingPairMap pending_pairs_;
        mutable std::mutex pending_pairs_mutex_;
    };

} // namespace lfs::io
