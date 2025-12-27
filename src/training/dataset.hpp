/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "io/pipelined_image_loader.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    // ============================================================================
    // Thread-Safe Queue (MPMC)
    // ============================================================================

    /// A basic locked, blocking MPMC queue.
    /// Every push/pop is guarded by a mutex. Condition variable is used
    /// to communicate insertion of new elements for waiting threads.
    template <typename T>
    class ThreadSafeQueue {
    public:
        /// Push a new value to the back and notify one waiting thread
        void push(T value) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(std::move(value));
            }
            cv_.notify_one();
        }

        /// Pop front element, blocking until available or timeout
        /// Returns nullopt on timeout
        std::optional<T> pop(std::optional<std::chrono::milliseconds> timeout = std::nullopt) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (timeout) {
                if (!cv_.wait_for(lock, *timeout, [this] { return !queue_.empty(); })) {
                    return std::nullopt; // Timeout
                }
            } else {
                cv_.wait(lock, [this] { return !queue_.empty(); });
            }

            T value = std::move(queue_.front());
            queue_.pop();
            return value;
        }

        /// Empty the queue and return number of elements cleared
        size_t clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto size = queue_.size();
            while (!queue_.empty()) {
                queue_.pop();
            }
            return size;
        }

        /// Check if empty (not thread-safe by itself, just a snapshot)
        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

    private:
        std::queue<T> queue_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
    };

    // ============================================================================
    // Random Sampler
    // ============================================================================

    /// Random sampler that shuffles indices once and iterates through them
    class RandomSampler {
    public:
        explicit RandomSampler(size_t size) : size_(size),
                                              index_(0) {
            reset();
        }

        /// Reset and shuffle indices
        void reset(std::optional<size_t> new_size = std::nullopt) {
            if (new_size) {
                size_ = *new_size;
            }

            // Generate indices 0...size-1
            indices_.resize(size_);
            for (size_t i = 0; i < size_; ++i) {
                indices_[i] = i;
            }

            // Shuffle using random device
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(indices_.begin(), indices_.end(), gen);

            index_ = 0;
        }

        /// Get next batch of indices
        std::optional<std::vector<size_t>> next(size_t batch_size) {
            if (index_ >= size_) {
                return std::nullopt;
            }

            const size_t end = std::min(index_ + batch_size, size_);
            std::vector<size_t> batch(indices_.begin() + index_, indices_.begin() + end);
            index_ = end;

            return batch;
        }

        size_t size() const { return size_; }

    private:
        size_t size_;
        size_t index_;
        std::vector<size_t> indices_;
    };

    /// Infinite random sampler - automatically resets when exhausted
    class InfiniteRandomSampler : public RandomSampler {
    public:
        explicit InfiniteRandomSampler(size_t size) : RandomSampler(size) {}

        std::optional<std::vector<size_t>> next(size_t batch_size) {
            auto batch = RandomSampler::next(batch_size);
            if (!batch) {
                reset();
                batch = RandomSampler::next(batch_size);
            }
            return batch;
        }
    };

    // ============================================================================
    // Dataset
    // ============================================================================

    /// Camera with loaded image
    struct CameraWithImage {
        lfs::core::Camera* camera;
        lfs::core::Tensor image;
    };

    /// Dataset example type
    struct CameraExample {
        CameraWithImage data;
        lfs::core::Tensor target; // Empty tensor, not used
    };

    /// Camera dataset configuration
    struct DatasetConfig {
        int resize_factor = 1;
        int max_width = 0;
        int test_every = 8;
    };

    /// Camera dataset - loads images from cameras
    class CameraDataset {
    public:
        enum class Split {
            TRAIN,
            VAL,
            ALL
        };

        CameraDataset(std::vector<std::shared_ptr<lfs::core::Camera>> cameras,
                      const DatasetConfig& config,
                      Split split = Split::ALL,
                      std::optional<std::vector<std::string>> included_images = std::nullopt)
            : cameras_(std::move(cameras)),
              config_(config),
              split_(split) {

            // Create indices based on split
            indices_.clear();
            if (included_images.has_value()) {
                for (size_t i = 0; i < cameras_.size(); ++i) {
                    // Simple filename matching without extension
                    auto img_name = cameras_[i]->image_name();
                    // Remove extension
                    auto dot_pos = img_name.find_last_of('.');
                    if (dot_pos != std::string::npos) {
                        img_name = img_name.substr(0, dot_pos);
                    }

                    if (std::find(included_images->begin(), included_images->end(), img_name) !=
                        included_images->end()) {
                        indices_.push_back(i);
                    }
                }
            } else {
                for (size_t i = 0; i < cameras_.size(); ++i) {
                    const bool is_test = (i % config.test_every) == 0;

                    if (split_ == Split::ALL || (split_ == Split::TRAIN && !is_test) ||
                        (split_ == Split::VAL && is_test)) {
                        indices_.push_back(i);
                    }
                }
            }

            LOG_INFO("Dataset created with {} images (split: {})", indices_.size(), static_cast<int>(split_));
        }

        /// Get single example by index
        CameraExample get(size_t index) const {
            if (index >= indices_.size()) {
                throw std::out_of_range("Dataset index out of range");
            }

            const size_t camera_idx = indices_[index];
            auto& cam = cameras_[camera_idx];

            // Load image using the new LibTorch-free Camera
            lfs::core::Tensor image = cam->load_and_get_image(config_.resize_factor, config_.max_width);

            return {
                {cam.get(), std::move(image)},
                lfs::core::Tensor() // Empty target
            };
        }

        /// Get batch of examples by indices
        std::vector<CameraExample> get_batch(const std::vector<size_t>& indices) const {
            std::vector<CameraExample> batch;
            batch.reserve(indices.size());
            for (size_t idx : indices) {
                batch.push_back(get(idx));
            }
            return batch;
        }

        size_t size() const { return indices_.size(); }

        const std::vector<std::shared_ptr<lfs::core::Camera>>& get_cameras() const { return cameras_; }

        Split get_split() const { return split_; }

        size_t get_num_bytes() const {
            if (cameras_.empty()) {
                return 0;
            }
            size_t total_bytes = 0;
            for (const auto& cam : cameras_) {
                total_bytes +=
                    cam->get_num_bytes_from_file(config_.resize_factor, config_.max_width);
            }
            return total_bytes;
        }

        [[nodiscard]] std::optional<lfs::core::Camera*> get_camera_by_filename(
            const std::string& filename) const {
            for (const auto& cam : cameras_) {
                if (cam->image_name() == filename) {
                    return cam.get();
                }
            }
            return std::nullopt;
        }

        void set_resize_factor(int resize_factor) { config_.resize_factor = resize_factor; }
        void set_max_width(int max_width) { config_.max_width = max_width; }

        int get_resize_factor() const { return config_.resize_factor; }
        int get_max_width() const { return config_.max_width; }

        /// Returns fraction of non-JPEG images (0.0 = all JPEG, 1.0 = none)
        [[nodiscard]] float get_non_jpeg_ratio() const {
            if (cameras_.empty())
                return 0.0f;
            const auto count = std::count_if(cameras_.begin(), cameras_.end(), [](const auto& cam) {
                auto ext = cam->image_path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                return ext != ".jpg" && ext != ".jpeg";
            });
            return static_cast<float>(count) / static_cast<float>(cameras_.size());
        }

    private:
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras_;
        DatasetConfig config_;
        Split split_;
        std::vector<size_t> indices_;
    };

    // ============================================================================
    // DataLoader
    // ============================================================================

    /// Options for configuring DataLoader
    struct DataLoaderOptions {
        size_t batch_size = 1;
        size_t num_workers = 0;
        size_t max_jobs = 0; // 0 means 2 * num_workers
        std::optional<std::chrono::milliseconds> timeout = std::nullopt;
        bool enforce_ordering = true;
        bool drop_last = false;
    };

    /// DataLoader - multi-threaded data loading with prefetching
    template <typename Sampler>
    class DataLoader {
    public:
        using BatchType = std::vector<CameraExample>;

        DataLoader(std::shared_ptr<CameraDataset> dataset,
                   Sampler sampler,
                   DataLoaderOptions options)
            : dataset_(dataset),
              sampler_(std::move(sampler)),
              options_(options),
              sequence_number_(0),
              in_flight_jobs_(0),
              shutdown_(false) {

            // Set max_jobs default
            if (options_.max_jobs == 0) {
                options_.max_jobs = std::max<size_t>(2 * options_.num_workers, 2);
            }

            // Start worker threads
            for (size_t i = 0; i < options_.num_workers; ++i) {
                workers_.emplace_back([this] { worker_thread(); });
            }

            // Prefetch initial jobs
            prefetch(options_.max_jobs);
        }

        ~DataLoader() {
            shutdown();
        }

        // Disable copying and moving
        DataLoader(const DataLoader&) = delete;
        DataLoader& operator=(const DataLoader&) = delete;
        DataLoader(DataLoader&&) = delete;
        DataLoader& operator=(DataLoader&&) = delete;

        /// Get next batch (blocking)
        std::optional<BatchType> next() {
            if (options_.num_workers > 0) {
                // Multi-threaded path
                while (auto result = pop_result()) {
                    if (result->exception) {
                        std::rethrow_exception(result->exception);
                    } else if (result->batch) {
                        prefetch(1); // Keep pipeline full
                        return std::move(result->batch);
                    }
                }
            } else {
                // Single-threaded path
                auto indices = sampler_.next(options_.batch_size);
                if (!indices || (indices->size() < options_.batch_size && options_.drop_last)) {
                    return std::nullopt;
                }
                return dataset_->get_batch(*indices);
            }
            return std::nullopt;
        }

        /// Reset the dataloader for new epoch
        void reset() {
            drain();
            sampler_.reset();
            sequence_number_ = 0;
            prefetch(options_.max_jobs);
        }

        /// Shutdown worker threads
        void shutdown() {
            if (shutdown_) {
                return;
            }
            shutdown_ = true;

            drain();

            // Send quit signal to all workers
            for (size_t i = 0; i < options_.num_workers; ++i) {
                Job quit_job;
                quit_job.quit = true;
                quit_job.sequence_number = sequence_number_++;
                job_queue_.push(std::move(quit_job));
            }

            // Join all workers
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        size_t in_flight_jobs() const { return in_flight_jobs_; }

    private:
        struct Job {
            size_t sequence_number = 0;
            std::optional<std::vector<size_t>> indices;
            bool quit = false;
        };

        struct Result {
            size_t sequence_number = 0;
            std::optional<BatchType> batch;
            std::exception_ptr exception;
        };

        /// Worker thread function
        void worker_thread() {
            while (true) {
                auto job_opt = job_queue_.pop();
                if (!job_opt) {
                    continue; // Spurious wakeup
                }

                Job job = std::move(*job_opt);

                if (job.quit) {
                    break;
                }

                try {
                    auto batch = dataset_->get_batch(*job.indices);

                    Result result;
                    result.sequence_number = job.sequence_number;
                    result.batch = std::move(batch);
                    result_queue_.push(std::move(result));
                } catch (...) {
                    Result result;
                    result.sequence_number = job.sequence_number;
                    result.exception = std::current_exception();
                    result_queue_.push(std::move(result));
                }
            }
        }

        /// Prefetch n jobs
        void prefetch(size_t n) {
            for (size_t i = 0; i < n; ++i) {
                auto indices = sampler_.next(options_.batch_size);
                if (!indices || (indices->size() < options_.batch_size && options_.drop_last)) {
                    break;
                }

                Job job;
                job.sequence_number = sequence_number_++;
                job.indices = std::move(indices);

                job_queue_.push(std::move(job));
                ++in_flight_jobs_;
            }
        }

        /// Pop result from queue
        std::optional<Result> pop_result() {
            if (in_flight_jobs_ == 0) {
                return std::nullopt;
            }

            auto result_opt = result_queue_.pop(options_.timeout);
            if (result_opt) {
                --in_flight_jobs_;
                return result_opt;
            }

            return std::nullopt;
        }

        /// Drain all pending jobs
        void drain() {
            // Clear pending jobs
            const size_t cleared = job_queue_.clear();
            in_flight_jobs_ -= cleared;

            // Wait for in-flight jobs to complete
            while (in_flight_jobs_ > 0) {
                pop_result();
            }
        }

        std::shared_ptr<CameraDataset> dataset_;
        Sampler sampler_;
        DataLoaderOptions options_;

        size_t sequence_number_;
        std::atomic<size_t> in_flight_jobs_;

        std::vector<std::thread> workers_;
        ThreadSafeQueue<Job> job_queue_;
        ThreadSafeQueue<Result> result_queue_;

        bool shutdown_;
    };

    // Pipelined DataLoader with GPU batch JPEG decoding
    template <typename Sampler>
    class PipelinedDataLoader {
    public:
        using BatchType = std::vector<CameraExample>;

        PipelinedDataLoader(std::shared_ptr<CameraDataset> dataset,
                            Sampler sampler,
                            lfs::io::PipelinedLoaderConfig config = {})
            : dataset_(dataset),
              sampler_(std::move(sampler)),
              config_(config),
              loader_(std::make_unique<lfs::io::PipelinedImageLoader>(config)),
              shutdown_(false) {

            // Prefetch initial batch
            prefetch_next_batch();
        }

        ~PipelinedDataLoader() {
            shutdown();
        }

        PipelinedDataLoader(const PipelinedDataLoader&) = delete;
        PipelinedDataLoader& operator=(const PipelinedDataLoader&) = delete;

        std::optional<CameraExample> next() {
            if (shutdown_)
                return std::nullopt;
            prefetch_next_batch();

            try {
                auto ready = loader_->get();
                const auto it = sequence_to_camera_.find(ready.sequence_id);
                if (it == sequence_to_camera_.end()) {
                    LOG_ERROR("[PipelinedDataLoader] Unknown sequence_id: {}", ready.sequence_id);
                    return std::nullopt;
                }
                const size_t camera_idx = it->second;
                sequence_to_camera_.erase(it);

                auto& cam = dataset_->get_cameras()[camera_idx];
                const auto shape = ready.tensor.shape();
                cam->set_image_dimensions(static_cast<int>(shape[2]), static_cast<int>(shape[1]));

                return CameraExample{
                    CameraWithImage{cam.get(), std::move(ready.tensor)},
                    lfs::core::Tensor()};
            } catch (const std::exception& e) {
                LOG_ERROR("[PipelinedDataLoader] Error: {}", e.what());
                return std::nullopt;
            }
        }

        void reset() {
            loader_->clear();
            sampler_.reset();
            sequence_to_camera_.clear();
            next_sequence_id_ = 0;
            prefetch_next_batch();
        }

        void shutdown() {
            if (shutdown_)
                return;
            shutdown_ = true;
            loader_->shutdown();
        }

        auto get_stats() const { return loader_->get_stats(); }

    private:
        void prefetch_next_batch() {
            while (loader_->in_flight_count() < config_.prefetch_count) {
                const auto indices = sampler_.next(1);
                if (!indices || indices->empty())
                    break;

                const size_t camera_idx = (*indices)[0];
                auto& cam = dataset_->get_cameras()[camera_idx];
                const size_t seq_id = next_sequence_id_++;
                sequence_to_camera_[seq_id] = camera_idx;

                lfs::io::LoadParams params;
                params.resize_factor = dataset_->get_resize_factor();
                params.max_width = dataset_->get_max_width();
                loader_->prefetch(seq_id, cam->image_path(), params);
            }
        }

        std::shared_ptr<CameraDataset> dataset_;
        Sampler sampler_;
        lfs::io::PipelinedLoaderConfig config_;
        std::unique_ptr<lfs::io::PipelinedImageLoader> loader_;

        std::unordered_map<size_t, size_t> sequence_to_camera_;
        size_t next_sequence_id_ = 0;

        bool shutdown_ = false;
    };

    template <typename SamplerType = RandomSampler>
    inline auto create_dataloader_from_dataset(std::shared_ptr<CameraDataset> dataset,
                                               int num_workers = 4) {
        const size_t dataset_size = dataset->size();
        DataLoaderOptions options;
        options.batch_size = 1;
        options.num_workers = num_workers;
        options.enforce_ordering = false;
        return std::make_unique<DataLoader<SamplerType>>(
            dataset, SamplerType(dataset_size), options);
    }

    inline auto create_infinite_dataloader_from_dataset(std::shared_ptr<CameraDataset> dataset,
                                                        int num_workers = 4) {
        return create_dataloader_from_dataset<InfiniteRandomSampler>(dataset, num_workers);
    }

    template <typename SamplerType = RandomSampler>
    inline auto create_pipelined_dataloader(std::shared_ptr<CameraDataset> dataset,
                                            lfs::io::PipelinedLoaderConfig config = {}) {
        const size_t dataset_size = dataset->size();
        return std::make_unique<PipelinedDataLoader<SamplerType>>(
            dataset, SamplerType(dataset_size), config);
    }

    inline auto create_infinite_pipelined_dataloader(std::shared_ptr<CameraDataset> dataset,
                                                     lfs::io::PipelinedLoaderConfig config = {}) {
        return create_pipelined_dataloader<InfiniteRandomSampler>(dataset, config);
    }

} // namespace lfs::training
