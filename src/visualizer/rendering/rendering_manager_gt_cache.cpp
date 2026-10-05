/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Ground-truth comparison image cache and its loader thread. Backend-neutral:
// shared by the Vulkan and tensor rendering managers.

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"
#include "display_tensors.hpp"
#include "gt_comparison_cache_utils.hpp"
#include "rendering/image_layout.hpp"
#include "rendering_manager.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <utility>

namespace lfs::vis {
    bool RenderingManager::gtRequestMatches(const GTComparisonImageJobRequest& lhs,
                                            const GTComparisonImageJobRequest& rhs) {
        return lhs.camera_uid == rhs.camera_uid &&
               lhs.mode == rhs.mode &&
               lhs.image_path == rhs.image_path &&
               lhs.image_size == rhs.image_size &&
               lhs.undistort_requested == rhs.undistort_requested &&
               lhs.depth_visualization_mode == rhs.depth_visualization_mode &&
               lhs.background_color == rhs.background_color;
    }

    bool RenderingManager::gtCacheEntryMatches(const GTComparisonImageCacheEntry& entry,
                                               const GTComparisonImageJobRequest& request) {
        return entry.camera_uid == request.camera_uid &&
               entry.mode == request.mode &&
               entry.image_path == request.image_path &&
               entry.image_size == request.image_size &&
               entry.undistort_requested == request.undistort_requested &&
               entry.depth_visualization_mode == request.depth_visualization_mode &&
               entry.background_color == request.background_color;
    }

    void RenderingManager::insertGTComparisonImageCacheEntry(
        const GTComparisonImageJobRequest& request,
        std::shared_ptr<lfs::core::Tensor> image,
        std::string error,
        const std::chrono::steady_clock::time_point now) {
        const auto same_key = [&request](const GTComparisonImageCacheEntry& entry) {
            return gtCacheEntryMatches(entry, request);
        };
        const bool image_valid = image && image->is_valid();
        assert(!image_valid || image->device() == lfs::core::Device::CPU);
        const std::size_t image_bytes = image_valid ? image->bytes() : 0;
        auto entry = std::find_if(gt_comparison_image_cache_.begin(),
                                  gt_comparison_image_cache_.end(),
                                  same_key);
        if (entry != gt_comparison_image_cache_.end()) {
            gt_comparison_image_cache_bytes_ -= entry->image && entry->image->is_valid()
                                                    ? entry->image->bytes()
                                                    : 0;
            *entry = {
                .camera_uid = request.camera_uid,
                .mode = request.mode,
                .undistort_requested = request.undistort_requested,
                .image_path = request.image_path,
                .image_size = request.image_size,
                .depth_visualization_mode = request.depth_visualization_mode,
                .background_color = request.background_color,
                .image = std::move(image),
                .error = std::move(error),
                .failure_time = image_valid ? std::chrono::steady_clock::time_point{} : now,
                .last_used = now};
        } else {
            gt_comparison_image_cache_.push_back({.camera_uid = request.camera_uid,
                                                  .mode = request.mode,
                                                  .undistort_requested = request.undistort_requested,
                                                  .image_path = request.image_path,
                                                  .image_size = request.image_size,
                                                  .depth_visualization_mode = request.depth_visualization_mode,
                                                  .background_color = request.background_color,
                                                  .image = std::move(image),
                                                  .error = std::move(error),
                                                  .failure_time = image_valid ? std::chrono::steady_clock::time_point{} : now,
                                                  .last_used = now});
        }
        gt_comparison_image_cache_bytes_ += image_bytes;

        while (gt_comparison_image_cache_.size() > GT_COMPARISON_IMAGE_CACHE_MAX_ENTRIES ||
               gt_comparison_image_cache_bytes_ > GT_COMPARISON_IMAGE_CACHE_MAX_BYTES) {
            const auto lru = std::min_element(
                gt_comparison_image_cache_.begin(),
                gt_comparison_image_cache_.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.last_used < rhs.last_used; });
            gt_comparison_image_cache_bytes_ -= lru->image && lru->image->is_valid()
                                                    ? lru->image->bytes()
                                                    : 0;
            gt_comparison_image_cache_.erase(lru);
        }
    }

    RenderingManager::GTComparisonImageLookup RenderingManager::getOrQueueGTComparisonImage(
        GTComparisonImageJobRequest request) {
        const auto request_matches = [](const GTComparisonImageJobRequest& lhs,
                                        const GTComparisonImageJobRequest& rhs) {
            return gtRequestMatches(lhs, rhs);
        };
        const auto cache_matches = [&request](const GTComparisonImageCacheEntry& entry) {
            return gtCacheEntryMatches(entry, request);
        };

        bool queued = false;
        GTComparisonImageLookup result;
        const auto now = std::chrono::steady_clock::now();
        // Capture the camera identity before `request` can be moved into the
        // pending/active slots; the stale-image fallback below needs it to
        // avoid showing a different camera's reference while reloading.
        const int request_camera_uid = request.camera_uid;
        const std::filesystem::path request_image_path = request.image_path;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            auto cache_entry = std::find_if(gt_comparison_image_cache_.begin(),
                                            gt_comparison_image_cache_.end(),
                                            cache_matches);
            const bool cache_hit = cache_entry != gt_comparison_image_cache_.end();
            if (cache_hit) {
                cache_entry->last_used = now;
            }

            const bool same_as_pending =
                pending_gt_comparison_image_request_ &&
                request_matches(*pending_gt_comparison_image_request_, request);
            const bool same_as_active =
                active_gt_comparison_image_request_ &&
                request_matches(*active_gt_comparison_image_request_, request);
            if (same_as_active && active_gt_comparison_image_is_prefetch_) {
                active_gt_comparison_image_is_prefetch_ = false;
                active_gt_comparison_image_request_->generation =
                    ++gt_comparison_image_request_generation_;
                pending_gt_comparison_image_request_.reset();
            } else if (same_as_active && active_gt_comparison_image_request_->generation !=
                                             gt_comparison_image_request_generation_) {
                active_gt_comparison_image_request_->generation =
                    ++gt_comparison_image_request_generation_;
                pending_gt_comparison_image_request_.reset();
            }

            const bool current_request_pending = pending_gt_comparison_image_request_.has_value();
            const bool current_request_active =
                active_gt_comparison_image_request_ && !active_gt_comparison_image_is_prefetch_ &&
                active_gt_comparison_image_request_->generation ==
                    gt_comparison_image_request_generation_;
            const bool retry_failed =
                cache_hit && (!cache_entry->image || !cache_entry->image->is_valid()) &&
                cache_entry->failure_time.time_since_epoch().count() != 0 &&
                now - cache_entry->failure_time >= GT_COMPARISON_IMAGE_RETRY_COOLDOWN;

            if (cache_hit && !retry_failed) {
                if (current_request_pending || current_request_active) {
                    ++gt_comparison_image_request_generation_;
                    pending_gt_comparison_image_request_.reset();
                }
                result.image = cache_entry->image;
                result.error = cache_entry->error;
                result.status = result.image && result.image->is_valid()
                                    ? GTComparisonImageStatus::Ready
                                    : GTComparisonImageStatus::Failed;
                return result;
            }

            if (retry_failed && (same_as_pending || current_request_active)) {
                result.error = cache_entry->error;
                result.status = GTComparisonImageStatus::Failed;
                return result;
            }

            if (retry_failed) {
                request.generation = ++gt_comparison_image_request_generation_;
                request.queued_at = now;
                pending_gt_comparison_image_request_ = std::move(request);
                result.error = cache_entry->error;
                result.status = GTComparisonImageStatus::Failed;
                queued = true;
            } else if (!same_as_pending && !current_request_active) {
                auto prefetch = std::find_if(prefetch_gt_comparison_image_requests_.begin(),
                                             prefetch_gt_comparison_image_requests_.end(),
                                             [&request_matches, &request](const auto& candidate) {
                                                 return request_matches(candidate, request);
                                             });
                if (prefetch != prefetch_gt_comparison_image_requests_.end()) {
                    request = std::move(*prefetch);
                    prefetch_gt_comparison_image_requests_.erase(prefetch);
                }
                request.generation = ++gt_comparison_image_request_generation_;
                if (request.queued_at.time_since_epoch().count() == 0) {
                    request.queued_at = now;
                }
                pending_gt_comparison_image_request_ = std::move(request);
                queued = true;
            }

            if (!retry_failed) {
                const auto* in_flight =
                    pending_gt_comparison_image_request_ &&
                            (queued ||
                             request_matches(*pending_gt_comparison_image_request_, request))
                        ? &*pending_gt_comparison_image_request_
                    : current_request_active &&
                            request_matches(*active_gt_comparison_image_request_, request)
                        ? &*active_gt_comparison_image_request_
                        : nullptr;
                result.status = GTComparisonImageStatus::Loading;
                if (!cache_hit) {
                    // Prefer the previous image of the SAME camera at any
                    // resolution so a transient re-decode never flashes a
                    // different camera's ground truth while the request is in
                    // flight; otherwise fall back to the most recently used
                    // valid image across the cache.
                    const auto is_same_camera = [request_camera_uid, request_image_path](
                                                    const GTComparisonImageCacheEntry& entry) {
                        return entry.camera_uid == request_camera_uid &&
                               entry.image_path == request_image_path;
                    };
                    const auto stale = std::max_element(
                        gt_comparison_image_cache_.begin(),
                        gt_comparison_image_cache_.end(),
                        [&](const GTComparisonImageCacheEntry& lhs,
                            const GTComparisonImageCacheEntry& rhs) {
                            const auto rank = [&](const GTComparisonImageCacheEntry& entry) {
                                if (!entry.image || !entry.image->is_valid()) {
                                    return 0;
                                }
                                return is_same_camera(entry) ? 2 : 1;
                            };
                            const int lhs_rank = rank(lhs);
                            const int rhs_rank = rank(rhs);
                            if (lhs_rank != rhs_rank) {
                                return lhs_rank < rhs_rank;
                            }
                            return lhs.last_used < rhs.last_used;
                        });
                    if (stale != gt_comparison_image_cache_.end() && stale->image &&
                        stale->image->is_valid()) {
                        result.stale_image = stale->image;
                    }
                }
                result.grace_elapsed = !in_flight ||
                                       in_flight->queued_at.time_since_epoch().count() == 0 ||
                                       now - in_flight->queued_at >= GT_COMPARISON_IMAGE_GRACE_PERIOD;
            }
        }

        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
        return result;
    }

    void RenderingManager::queueGTComparisonImagePrefetch(GTComparisonImageJobRequest request) {
        const auto request_matches = [](const GTComparisonImageJobRequest& lhs,
                                        const GTComparisonImageJobRequest& rhs) {
            return gtRequestMatches(lhs, rhs);
        };
        const auto now = std::chrono::steady_clock::now();
        bool queued = false;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            if (request.image_size.x <= 0 || request.image_size.y <= 0) {
                return;
            }
            const std::size_t request_bytes = gt_comparison_detail::previewBytes(request.image_size);
            // Keep at least the current request and one neighbor resident. A pending
            // current request is not in the cache yet, so reserve its final uint8 size
            // while deciding whether to enqueue this prefetch.
            std::size_t reserved_current_bytes = 0;
            if (pending_gt_comparison_image_request_) {
                const auto& current = *pending_gt_comparison_image_request_;
                reserved_current_bytes = gt_comparison_detail::previewBytes(current.image_size);
            }
            if (!gt_comparison_detail::prefetchFits(gt_comparison_image_cache_bytes_,
                                                    reserved_current_bytes,
                                                    request_bytes,
                                                    GT_COMPARISON_IMAGE_CACHE_MAX_BYTES)) {
                return;
            }
            const auto cache_hit = std::find_if(
                gt_comparison_image_cache_.begin(),
                gt_comparison_image_cache_.end(),
                [&request](const auto& entry) {
                    return gtCacheEntryMatches(entry, request);
                });
            const bool in_cache = cache_hit != gt_comparison_image_cache_.end();
            const bool same_as_pending =
                pending_gt_comparison_image_request_ &&
                request_matches(*pending_gt_comparison_image_request_, request);
            const bool same_as_active =
                active_gt_comparison_image_request_ &&
                request_matches(*active_gt_comparison_image_request_, request);
            const bool same_as_prefetch = std::any_of(
                prefetch_gt_comparison_image_requests_.begin(),
                prefetch_gt_comparison_image_requests_.end(),
                [&request_matches, &request](const auto& candidate) {
                    return request_matches(candidate, request);
                });
            if (!in_cache && !same_as_pending && !same_as_active && !same_as_prefetch) {
                request.queued_at = now;
                if (prefetch_gt_comparison_image_requests_.size() >=
                    GT_COMPARISON_IMAGE_PREFETCH_MAX_ENTRIES) {
                    prefetch_gt_comparison_image_requests_.pop_front();
                }
                prefetch_gt_comparison_image_requests_.push_back(std::move(request));
                queued = true;
            }
        }
        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
    }

    void RenderingManager::invalidateGTComparisonImageCache(ViewRenderState& view) {
        std::lock_guard lock(gt_comparison_image_mutex_);
        ++gt_comparison_image_request_generation_;
        gt_comparison_image_cache_.clear();
        gt_comparison_image_cache_bytes_ = 0;
        pending_gt_comparison_image_request_.reset();
        active_gt_comparison_image_request_.reset();
        active_gt_comparison_image_is_prefetch_ = false;
        prefetch_gt_comparison_image_requests_.clear();
        gt_comparison_cuda_image_.reset();
        gt_comparison_cuda_source_ = nullptr;
        gt_comparison_cuda_generation_ = 0;
        gt_comparison_cuda_camera_uid_ = -1;
        gt_comparison_cuda_size_ = {0, 0};
        gt_comparison_cuda_undistorted_ = false;
        view.split_left_source_ = nullptr;
        view.split_left_source_size_ = {0, 0};
        view.split_left_source_camera_uid_ = -1;
        view.split_left_source_undistorted_ = false;
        view.split_left_image_generation_ = 0;
        view.split_right_source_size_ = {0, 0};
        view.split_right_image_generation_ = 0;
        gt_comparison_loading_placeholder_.reset();
        gt_comparison_failed_placeholder_.reset();
    }

    void RenderingManager::gtComparisonImageWorkerLoop(const std::stop_token stop_token) {
        while (true) {
            GTComparisonImageJobRequest request;
            bool is_prefetch = false;
            {
                std::unique_lock lock(gt_comparison_image_mutex_);
                gt_comparison_image_cv_.wait(lock, stop_token, [this] {
                    return pending_gt_comparison_image_request_.has_value() ||
                           !prefetch_gt_comparison_image_requests_.empty();
                });
                if (stop_token.stop_requested()) {
                    pending_gt_comparison_image_request_.reset();
                    prefetch_gt_comparison_image_requests_.clear();
                    active_gt_comparison_image_request_.reset();
                    active_gt_comparison_image_is_prefetch_ = false;
                    return;
                }

                if (pending_gt_comparison_image_request_) {
                    request = std::move(*pending_gt_comparison_image_request_);
                    pending_gt_comparison_image_request_.reset();
                } else {
                    request = std::move(prefetch_gt_comparison_image_requests_.back());
                    prefetch_gt_comparison_image_requests_.pop_back();
                    is_prefetch = true;
                }
                active_gt_comparison_image_request_ = request;
                active_gt_comparison_image_is_prefetch_ = is_prefetch;
            }

            std::shared_ptr<lfs::core::Tensor> image;
            std::string error;
            try {
                lfs::core::Tensor gt_tensor;
                if (request.mode == GTComparisonMode::RGB) {
                    auto [pixels, width, height, channels] = lfs::core::load_image(
                        request.image_path, -1, request.undistort_requested ? 0 : request.preview_max_dimension);
                    const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> owner(
                        pixels, &lfs::core::free_image);
                    if (pixels && width > 0 && height > 0 && channels > 0) {
                        gt_tensor = lfs::core::Tensor::from_blob(
                                        pixels,
                                        {static_cast<size_t>(height), static_cast<size_t>(width),
                                         static_cast<size_t>(channels)},
                                        lfs::core::Device::CPU, lfs::core::DataType::UInt8)
                                        .permute({2, 0, 1})
                                        .clone();
                    }
                }
                if (request.mode == GTComparisonMode::RGB) {
                    if (gt_tensor.is_valid() && gt_tensor.ndim() == 3) {
                        const auto gt_layout = lfs::rendering::detectImageLayout(gt_tensor);
                        if (gt_layout != lfs::rendering::ImageLayout::Unknown) {
                            const bool undistort_gt =
                                gt_layout == lfs::rendering::ImageLayout::CHW &&
                                request.undistort_requested;
                            if (undistort_gt) {
                                if (gt_tensor.device() != lfs::core::Device::GPU) {
                                    gt_tensor = gt_tensor.to(lfs::core::Device::GPU);
                                }
                                if (gt_tensor.dtype() == lfs::core::DataType::UInt8) {
                                    gt_tensor = gt_tensor.to(lfs::core::DataType::Float32) / 255.0f;
                                }
                                const auto scaled = lfs::core::prepare_undistort_params(
                                    request.undistort_params,
                                    lfs::rendering::imageWidth(gt_tensor, gt_layout),
                                    lfs::rendering::imageHeight(gt_tensor, gt_layout),
                                    1,
                                    request.preview_max_dimension);
                                gt_tensor = lfs::core::undistort_image(
                                    gt_tensor.clamp(0.0f, 1.0f).contiguous(), scaled, nullptr);
                            }
                            gt_tensor = lfs::rendering::flipImageVertical(gt_tensor, gt_layout);
                            // Static GT display images must be decoupled from the CUDA pool
                            // while training can recycle device buffers mid-frame.
                            gt_tensor = gt_tensor.cpu();
                            image = std::make_shared<lfs::core::Tensor>(std::move(gt_tensor));
                            image = resizeChwDisplayTensor(image, request.image_size);
                        }
                    }
                } else if (request.camera) {
                    if (request.mode == GTComparisonMode::Depth) {
                        auto depth = request.camera->load_and_get_depth(
                            -1, request.preview_max_dimension);
                        if (depth.is_valid() && depth.ndim() == 2) {
                            if (request.undistort_requested) {
                                const auto scaled = lfs::core::prepare_undistort_params(
                                    request.undistort_params,
                                    static_cast<int>(depth.shape()[1]),
                                    static_cast<int>(depth.shape()[0]),
                                    1,
                                    request.preview_max_dimension);
                                depth = lfs::core::undistort_depth_area(depth, scaled, nullptr);
                            }
                            image = makeDepthDisplayTensor(
                                depth, request.depth_visualization_mode, request.background_color);
                            image = resizeChwDisplayTensor(image, request.image_size);
                            if (image) {
                                auto flipped = lfs::rendering::flipImageVertical(
                                    *image, lfs::rendering::ImageLayout::CHW);
                                image = std::make_shared<lfs::core::Tensor>(std::move(flipped));
                            }
                        }
                    } else {
                        auto normal = request.camera->load_and_get_normal(
                            -1, request.preview_max_dimension, lfs::core::Camera::NormalPriorDecode{});
                        if (normal.is_valid() && normal.ndim() == 3) {
                            const auto normal_layout = lfs::rendering::detectImageLayout(normal);
                            if (request.undistort_requested &&
                                normal_layout != lfs::rendering::ImageLayout::Unknown) {
                                const auto scaled = lfs::core::prepare_undistort_params(
                                    request.undistort_params,
                                    lfs::rendering::imageWidth(normal, normal_layout),
                                    lfs::rendering::imageHeight(normal, normal_layout),
                                    1,
                                    request.preview_max_dimension);
                                normal = lfs::core::undistort_normal_area(normal, scaled, nullptr);
                            }
                            image = makeNormalDisplayTensor(normal);
                            image = resizeChwDisplayTensor(image, request.image_size);
                            if (image) {
                                auto flipped = lfs::rendering::flipImageVertical(
                                    *image, lfs::rendering::ImageLayout::CHW);
                                image = std::make_shared<lfs::core::Tensor>(std::move(flipped));
                            }
                        }
                    }
                }
                image = gt_comparison_detail::convertDisplayTensorToUInt8(image);
                if (!image || !image->is_valid()) {
                    image.reset();
                    error = request.mode == GTComparisonMode::RGB
                                ? "RGB GT comparison could not load the source image"
                            : request.mode == GTComparisonMode::Depth
                                ? "Depth GT comparison could not load the camera depth map"
                                : "Normal GT comparison could not load the camera normal map";
                }
            } catch (const std::exception& e) {
                image.reset();
                error = std::format("RGB GT comparison image load failed: {}", e.what());
                LOG_WARN("{}", error);
            } catch (...) {
                image.reset();
                error = "RGB GT comparison image load failed with an unknown error";
                LOG_WARN("{}", error);
            }

            bool applied = false;
            {
                std::lock_guard lock(gt_comparison_image_mutex_);
                const auto request_matches = [](const GTComparisonImageJobRequest& lhs,
                                                const GTComparisonImageJobRequest& rhs) {
                    return gtRequestMatches(lhs, rhs);
                };
                const bool active_request_matches =
                    active_gt_comparison_image_request_ &&
                    request_matches(*active_gt_comparison_image_request_, request);
                const bool completed_as_prefetch = active_gt_comparison_image_is_prefetch_;
                const bool current_request_is_valid =
                    active_request_matches && !completed_as_prefetch &&
                    active_gt_comparison_image_request_->generation ==
                        gt_comparison_image_request_generation_;
                if (!stop_token.stop_requested() && active_request_matches &&
                    (completed_as_prefetch || current_request_is_valid)) {
                    insertGTComparisonImageCacheEntry(
                        request, std::move(image), std::move(error), std::chrono::steady_clock::now());
                    if (completed_as_prefetch && pending_gt_comparison_image_request_ &&
                        request_matches(*pending_gt_comparison_image_request_, request)) {
                        pending_gt_comparison_image_request_.reset();
                        applied = true;
                    } else if (!completed_as_prefetch) {
                        applied = true;
                    }
                }
                if (active_request_matches) {
                    active_gt_comparison_image_request_.reset();
                    active_gt_comparison_image_is_prefetch_ = false;
                }
            }

            if (applied) {
                markDirty(DirtyFlag::SPLIT_VIEW, lfs::vis::FrameReason::SettingsChange);
            }
        }
    }

} // namespace lfs::vis
