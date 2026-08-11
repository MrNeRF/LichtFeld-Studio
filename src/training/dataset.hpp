/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "io/pipelined_image_loader.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <format>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    // R_w2c premultiplied with the resolved prior-world -> reconstruction-world
    // rotation, so the loader converts prior normals with one matrix.
    inline std::array<float, 9> camera_world_to_camera_normal_matrix(
        const lfs::core::Camera& cam,
        const std::array<float, 9>& prior_world_rotation) {
        std::array<float, 9> r_w2c{};
        auto R_cpu = cam.R().cpu().contiguous();
        auto R_acc = R_cpu.accessor<float, 2>();
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                r_w2c[row * 3 + col] = R_acc(row, col);
            }
        }
        std::array<float, 9> result{};
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                float sum = 0.0f;
                for (size_t k = 0; k < 3; ++k) {
                    sum += r_w2c[row * 3 + k] * prior_world_rotation[k * 3 + col];
                }
                result[row * 3 + col] = sum;
            }
        }
        return result;
    }

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

    /// Random sampler that shuffles indices once and iterates through them
    class RandomSampler {
    public:
        explicit RandomSampler(size_t size, std::optional<std::uint64_t> seed = std::nullopt)
            : size_(size),
              index_(0),
              seed_(seed) {
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

            // Training can recreate a loader when resuming.  An optional seed
            // makes that recreation produce the same camera order; callers
            // that omit it retain the historical random-device behavior.
            std::mt19937 gen;
            if (seed_) {
                const auto epoch_seed = *seed_ + epoch_ * 0x9e3779b97f4a7c15ULL;
                gen.seed(static_cast<std::uint32_t>(epoch_seed ^ (epoch_seed >> 32)));
            } else {
                std::random_device rd;
                gen.seed(rd());
            }
            std::shuffle(indices_.begin(), indices_.end(), gen);

            index_ = 0;
            ++epoch_;
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

    protected:
        size_t current_index() const { return index_; }
        void set_current_index(size_t index) { index_ = index; }

    private:
        size_t size_;
        size_t index_;
        std::vector<size_t> indices_;
        std::optional<std::uint64_t> seed_;
        std::uint64_t epoch_ = 0;
    };

    /// Infinite random sampler - automatically resets when exhausted
    class InfiniteRandomSampler : public RandomSampler {
    public:
        explicit InfiniteRandomSampler(size_t size,
                                       std::optional<std::uint64_t> seed = std::nullopt,
                                       size_t start_offset = 0)
            : RandomSampler(size, seed) {
            skip(start_offset);
        }

        std::optional<std::vector<size_t>> next(size_t batch_size) {
            auto batch = RandomSampler::next(batch_size);
            if (!batch) {
                reset();
                batch = RandomSampler::next(batch_size);
            }
            return batch;
        }

        void skip(size_t count) {
            while (count > 0 && size() > 0) {
                const size_t remaining = size() - current_index();
                if (count < remaining) {
                    set_current_index(current_index() + count);
                    return;
                }
                count -= remaining;
                reset();
            }
        }
    };

    /// Camera with loaded image
    struct CameraWithImage {
        lfs::core::Camera* camera;
        lfs::core::Tensor image;
    };

    /// Dataset example type
    struct CameraExample {
        CameraWithImage data;
        lfs::core::Tensor target;                     // Empty tensor, not used
        std::optional<lfs::core::Tensor> mask = {};   // Optional mask [H,W], float32
        std::optional<lfs::core::Tensor> depth = {};  // Optional depth [H,W], float32
        std::optional<lfs::core::Tensor> normal = {}; // Optional normals [3,H,W], float32 in [-1,1]
        CUevent_st* depth_ready_event = nullptr;
        CUevent_st* normal_ready_event = nullptr;
        // Ring-backed tensors must never outlive this keepalive handle.
        std::shared_ptr<void> decoded_frame_keepalive;
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

        lfs::core::Camera* get_camera(size_t index) const {
            assert(index < indices_.size());
            return cameras_[indices_[index]].get();
        }

        size_t local_to_source(size_t index) const {
            assert(index < indices_.size());
            return indices_[index];
        }

        /// Get single example by index
        CameraExample get(size_t index) const {
            if (index >= indices_.size()) {
                throw std::out_of_range("Dataset index out of range");
            }

            const size_t camera_idx = indices_[index];
            auto& cam = cameras_[camera_idx];

            // Load image using the new LibTorch-free Camera
            lfs::core::Tensor image = cam->load_and_get_image(config_.resize_factor, config_.max_width, true);

            return CameraExample{
                .data = {cam.get(), std::move(image)},
                .target = lfs::core::Tensor(),
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

    /// Configuration for optional mask/depth/normal loading in PipelinedDataLoader
    struct PipelinedAuxiliaryImageConfig {
        bool load_masks = false;         // Whether to load masks alongside images
        bool load_depths = false;        // Whether to load depth maps alongside images
        bool load_normals = false;       // Whether to load normal maps alongside images
        bool normal_flip_yz = false;     // Convert OpenGL-convention normal priors to OpenCV
        bool normal_world_space = false; // Convert world-space normal priors to camera space
        bool normal_srgb = false;        // Normal priors are sRGB-encoded (invert before decode)
        // Prior-world -> reconstruction-world rotation (row-major), identity unless resolved otherwise
        std::array<float, 9> normal_world_rotation{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        std::vector<std::array<float, 9>> normal_world_to_camera_by_source;
        bool invert_masks = false;      // Invert mask values (1.0 - mask)
        float mask_threshold = 0.0f;    // If > 0, values >= threshold become 1.0
        bool use_alpha_as_mask = false; // Extract alpha channel from RGBA as mask
    };

    // Pipelined DataLoader with GPU batch JPEG decoding
    template <typename Sampler>
    class PipelinedDataLoader {
    public:
        using BatchType = std::vector<CameraExample>;

        PipelinedDataLoader(std::shared_ptr<CameraDataset> dataset,
                            Sampler sampler,
                            lfs::io::PipelinedLoaderConfig config = {},
                            PipelinedAuxiliaryImageConfig aux_config = {})
            : dataset_(dataset),
              sampler_(std::move(sampler)),
              config_(config),
              aux_config_(aux_config),
              loader_(std::make_shared<lfs::io::PipelinedImageLoader>(config)),
              shutdown_(false) {

            // Canonicalize every source once for this training run. The loader
            // retains only final encoded blobs in its run-local RAM/spill tier;
            // normal iteration is decode-only consumption of those blobs.
            if (dataset_->size() > 0) {
                std::vector<lfs::io::ImageRequest> run_requests;
                run_requests.reserve(dataset_->size());
                const size_t sequence_base =
                    std::numeric_limits<size_t>::max() - dataset_->size();
                for (size_t local_idx = 0; local_idx < dataset_->size(); ++local_idx) {
                    run_requests.push_back(make_request(
                        dataset_->local_to_source(local_idx), sequence_base + local_idx));
                }
                loader_->canonicalize(run_requests);

                // Canonicalization is a named pre-training boundary.  All
                // decode leases are dead here, so release their ring storage
                // and trim every transient allocator before the first training
                // frame can overlap this import phase.
                loader_->reclaim_idle_decoded_frames();
                lfs::core::Tensor::trim_memory_pool();
                LOG_INFO("[PipelinedDataLoader] canonicalization boundary trim complete");
            }

            // Prefetch initial batch from the now-canonical run cache.
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

                CameraExample example{
                    .data = {cam.get(), std::move(ready.tensor)},
                    .target = lfs::core::Tensor(),
                    .depth_ready_event = ready.depth_ready_event,
                    .normal_ready_event = ready.normal_ready_event,
                };
                example.decoded_frame_keepalive = std::make_shared<
                    std::vector<std::shared_ptr<void>>>(std::move(ready.decoded_frame_leases));
                ready.depth_ready_event = nullptr;
                ready.normal_ready_event = nullptr;

                // Attach mask if present
                if (ready.mask && ready.mask->is_valid()) {
                    example.mask = std::move(*ready.mask);
                } else if (aux_config_.load_masks && cam->has_in_memory_mask()) {
                    // Direct-scene plugins attach masks as in-memory tensors
                    // via Camera::set_mask_tensor — load_and_get_mask returns
                    // the processed-and-cached tensor (skips file I/O).
                    bool segment_and_ignore = aux_config_.mask_threshold <= 0.f;
                    auto m = cam->load_and_get_mask(
                        dataset_->get_resize_factor(),
                        dataset_->get_max_width(),
                        aux_config_.invert_masks,
                        aux_config_.mask_threshold,
                        !segment_and_ignore);
                    if (m.is_valid()) {
                        example.mask = std::move(m);
                    }
                }
                if (ready.depth && ready.depth->is_valid()) {
                    example.depth = std::move(*ready.depth);
                } else if (aux_config_.load_depths && cam->has_depth()) {
                    auto depth = cam->load_and_get_depth(
                        dataset_->get_resize_factor(),
                        dataset_->get_max_width());
                    if (depth.is_valid()) {
                        example.depth = std::move(depth);
                    }
                }
                if (ready.normal && ready.normal->is_valid()) {
                    example.normal = std::move(*ready.normal);
                }

                return example;
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

        void observe_training_iteration(const double train_ms,
                                        const double dl_wait_ms,
                                        const std::size_t iter) {
            loader_->observe_training_iteration(train_ms, dl_wait_ms, iter);
        }

        lfs::io::PipelinedImageLoader* get_loader() const { return loader_.get(); }
        std::shared_ptr<lfs::io::PipelinedImageLoader> get_loader_shared() const { return loader_; }

    private:
        lfs::io::ImageRequest make_request(const size_t camera_idx,
                                           const size_t seq_id) const {
            auto& cam = dataset_->get_cameras()[camera_idx];
            lfs::io::ImageRequest request;
            request.sequence_id = seq_id;
            request.path = cam->image_path();
            request.params.resize_factor = dataset_->get_resize_factor();
            request.params.max_width = dataset_->get_max_width();
            request.params.output_uint8 = !config_.use_16bit_color;
            if (!cam->image_size_loaded() ||
                (dataset_->get_max_width() > 0 &&
                 (cam->image_height() > dataset_->get_max_width() ||
                  cam->image_width() > dataset_->get_max_width()))) {
                cam->load_image_size(dataset_->get_resize_factor(), dataset_->get_max_width());
            }
            request.aux_target_width = cam->image_width();
            request.aux_target_height = cam->image_height();
            if (cam->is_undistort_prepared()) {
                request.undistort = &cam->undistort_params();
                request.params.undistort = request.undistort;
            }

            if (aux_config_.load_masks && cam->has_mask()) {
                request.mask_path = cam->mask_path();
                request.mask_params.invert = aux_config_.invert_masks;
                request.mask_params.threshold = aux_config_.mask_threshold;
            } else if (aux_config_.use_alpha_as_mask && cam->has_alpha()) {
                request.extract_alpha_as_mask = true;
                request.alpha_mask_params.invert = aux_config_.invert_masks;
                request.alpha_mask_params.threshold = aux_config_.mask_threshold;
            }
            if (aux_config_.load_depths && cam->has_depth()) {
                request.depth_path = cam->depth_path();
            }
            if (aux_config_.load_normals && cam->has_normal()) {
                request.normal_path = cam->normal_path();
                request.normal_flip_yz = aux_config_.normal_flip_yz;
                request.normal_srgb = aux_config_.normal_srgb;
                request.normal_transform_world_to_camera = aux_config_.normal_world_space;
                if (aux_config_.normal_world_space) {
                    if (camera_idx < aux_config_.normal_world_to_camera_by_source.size()) {
                        request.normal_world_to_camera =
                            aux_config_.normal_world_to_camera_by_source[camera_idx];
                    } else {
                        request.normal_world_to_camera = camera_world_to_camera_normal_matrix(
                            *cam, aux_config_.normal_world_rotation);
                    }
                }
            }
            return request;
        }

        void prefetch_next_batch() {
            while (loader_->in_flight_count() < loader_->adaptive_prefetch_target()) {
                const auto indices = sampler_.next(1);
                if (!indices || indices->empty())
                    break;

                const size_t local_idx = (*indices)[0];
                const size_t camera_idx = dataset_->local_to_source(local_idx);
                const size_t seq_id = next_sequence_id_++;
                sequence_to_camera_[seq_id] = camera_idx;
                auto request = make_request(camera_idx, seq_id);
                loader_->prefetch({request});
            }
        }

        std::shared_ptr<CameraDataset> dataset_;
        Sampler sampler_;
        lfs::io::PipelinedLoaderConfig config_;
        PipelinedAuxiliaryImageConfig aux_config_;
        std::shared_ptr<lfs::io::PipelinedImageLoader> loader_;

        std::unordered_map<size_t, size_t> sequence_to_camera_;
        size_t next_sequence_id_ = 0;

        bool shutdown_ = false;
    };

    template <typename SamplerType = RandomSampler>
    inline auto create_pipelined_dataloader(std::shared_ptr<CameraDataset> dataset,
                                            lfs::io::PipelinedLoaderConfig config = {},
                                            PipelinedAuxiliaryImageConfig aux_config = {},
                                            std::optional<std::uint64_t> sampler_seed = std::nullopt) {
        const size_t dataset_size = dataset->size();
        return std::make_unique<PipelinedDataLoader<SamplerType>>(
            dataset, SamplerType(dataset_size, sampler_seed), config, aux_config);
    }

    inline auto create_infinite_pipelined_dataloader(std::shared_ptr<CameraDataset> dataset,
                                                     lfs::io::PipelinedLoaderConfig config = {},
                                                     PipelinedAuxiliaryImageConfig aux_config = {},
                                                     std::optional<std::uint64_t> sampler_seed = std::nullopt,
                                                     size_t sampler_offset = 0) {
        const size_t dataset_size = dataset->size();
        return std::make_unique<PipelinedDataLoader<InfiniteRandomSampler>>(
            dataset,
            InfiniteRandomSampler(dataset_size, sampler_seed, sampler_offset),
            config,
            aux_config);
    }

} // namespace lfs::training
