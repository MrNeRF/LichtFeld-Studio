/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"

#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/path_utils.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"
#include "display_tensors.hpp"
#include "gt_comparison_cache_utils.hpp"
#include "rendering/image_layout.hpp"
#include "visualizer/app_store.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> resizePreview(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const glm::ivec2 target_size) {
            if (!image || !image->is_valid() || image->ndim() != 3 ||
                target_size.x <= 0 || target_size.y <= 0) {
                return {};
            }
            const auto layout = lfs::rendering::detectImageLayout(*image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                return {};
            }

            lfs::core::Tensor source = *image;
            if (source.dtype() == lfs::core::DataType::UInt8) {
                source = source.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (source.dtype() != lfs::core::DataType::Float32) {
                source = source.to(lfs::core::DataType::Float32);
            }
            if (layout == lfs::rendering::ImageLayout::HWC) {
                source = source.permute({2, 0, 1}).contiguous();
            }
            source = source.cpu().contiguous();

            const int source_channels = static_cast<int>(source.size(0));
            const int source_height = static_cast<int>(source.size(1));
            const int source_width = static_cast<int>(source.size(2));
            if (source_channels <= 0 || source_width <= 0 || source_height <= 0) {
                return {};
            }
            if (source_width == target_size.x && source_height == target_size.y &&
                source_channels >= 3) {
                return std::make_shared<lfs::core::Tensor>(std::move(source));
            }

            const int destination_width = target_size.x;
            const int destination_height = target_size.y;
            const std::size_t destination_pixels =
                static_cast<std::size_t>(destination_width) * destination_height;
            std::vector<float> output(3 * destination_pixels, 0.0f);
            const float* const source_data = source.ptr<float>();
            if (!source_data) {
                return {};
            }
            const auto sample = [&](const int channel, const int x, const int y) {
                const int bounded_channel = std::clamp(channel, 0, source_channels - 1);
                return source_data[(static_cast<std::size_t>(bounded_channel) * source_height + y) *
                                       source_width +
                                   x];
            };

            const float scale_x = static_cast<float>(source_width) / destination_width;
            const float scale_y = static_cast<float>(source_height) / destination_height;
            for (int y = 0; y < destination_height; ++y) {
                const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                int y0 = static_cast<int>(std::floor(source_y));
                float wy = source_y - static_cast<float>(y0);
                if (y0 < 0) {
                    y0 = 0;
                    wy = 0.0f;
                }
                int y1 = y0 + 1;
                if (y1 >= source_height) {
                    y1 = y0 = source_height - 1;
                    wy = 0.0f;
                }
                for (int x = 0; x < destination_width; ++x) {
                    const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                    int x0 = static_cast<int>(std::floor(source_x));
                    float wx = source_x - static_cast<float>(x0);
                    if (x0 < 0) {
                        x0 = 0;
                        wx = 0.0f;
                    }
                    int x1 = x0 + 1;
                    if (x1 >= source_width) {
                        x1 = x0 = source_width - 1;
                        wx = 0.0f;
                    }
                    const std::size_t destination_index =
                        static_cast<std::size_t>(y) * destination_width + x;
                    for (int channel = 0; channel < 3; ++channel) {
                        const float top = std::lerp(
                            sample(channel, x0, y0), sample(channel, x1, y0), wx);
                        const float bottom = std::lerp(
                            sample(channel, x0, y1), sample(channel, x1, y1), wx);
                        output[static_cast<std::size_t>(channel) * destination_pixels +
                               destination_index] = std::lerp(top, bottom, wy);
                    }
                }
            }

            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {3, static_cast<std::size_t>(destination_height),
                 static_cast<std::size_t>(destination_width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> resizeUInt8Preview(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const glm::ivec2 target_size) {
            if (!image || !image->is_valid() || image->device() != lfs::core::Device::CPU ||
                image->dtype() != lfs::core::DataType::UInt8 || image->ndim() != 3 ||
                image->size(0) != 3 || !image->is_contiguous() ||
                target_size.x <= 0 || target_size.y <= 0) {
                return {};
            }

            const int source_height = static_cast<int>(image->size(1));
            const int source_width = static_cast<int>(image->size(2));
            if (source_width <= 0 || source_height <= 0 ||
                target_size.x > source_width || target_size.y > source_height) {
                return {};
            }

            auto resized = lfs::core::Tensor::empty(
                {3, static_cast<std::size_t>(target_size.y),
                 static_cast<std::size_t>(target_size.x)},
                lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            const auto* const source = image->ptr<std::uint8_t>();
            auto* const destination = resized.ptr<std::uint8_t>();
            const std::size_t source_pixels =
                static_cast<std::size_t>(source_width) * source_height;
            const std::size_t destination_pixels =
                static_cast<std::size_t>(target_size.x) * target_size.y;
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < target_size.y; ++y) {
                    const int y0 = y * source_height / target_size.y;
                    const int y1 = std::max(
                        y0 + 1, (y + 1) * source_height / target_size.y);
                    for (int x = 0; x < target_size.x; ++x) {
                        const int x0 = x * source_width / target_size.x;
                        const int x1 = std::max(
                            x0 + 1, (x + 1) * source_width / target_size.x);
                        std::uint64_t sum = 0;
                        for (int source_y = y0; source_y < y1; ++source_y) {
                            const auto row =
                                static_cast<std::size_t>(source_y) * source_width;
                            for (int source_x = x0; source_x < x1; ++source_x) {
                                sum += source[static_cast<std::size_t>(channel) *
                                                  source_pixels +
                                              row + source_x];
                            }
                        }
                        const auto samples =
                            static_cast<std::uint64_t>(x1 - x0) *
                            static_cast<std::uint64_t>(y1 - y0);
                        destination[static_cast<std::size_t>(channel) *
                                        destination_pixels +
                                    static_cast<std::size_t>(y) * target_size.x + x] =
                            static_cast<std::uint8_t>((sum + samples / 2) / samples);
                    }
                }
            }

            auto normalized = resized.to(lfs::core::DataType::Float32).div(255.0f);
            return std::make_shared<lfs::core::Tensor>(std::move(normalized));
        }
    } // namespace

    bool RenderingManager::gtRequestMatches(
        const GTComparisonPreviewRequest& lhs,
        const GTComparisonPreviewRequest& rhs) {
        return lhs.cache_epoch == rhs.cache_epoch &&
               lhs.calibration_revision == rhs.calibration_revision &&
               lhs.camera_uid == rhs.camera_uid &&
               lhs.mode == rhs.mode &&
               lhs.image_path == rhs.image_path &&
               lhs.image_size == rhs.image_size &&
               lhs.undistort_requested == rhs.undistort_requested &&
               lhs.depth_visualization_mode == rhs.depth_visualization_mode &&
               lhs.background_color == rhs.background_color;
    }

    bool RenderingManager::gtCacheEntryMatches(
        const GTComparisonImageCacheEntry& entry,
        const GTComparisonPreviewRequest& request) {
        return entry.cache_epoch == request.cache_epoch &&
               entry.calibration_revision == request.calibration_revision &&
               entry.camera_uid == request.camera_uid &&
               entry.mode == request.mode &&
               entry.image_path == request.image_path &&
               entry.image_size == request.image_size &&
               entry.undistort_requested == request.undistort_requested &&
               entry.depth_visualization_mode == request.depth_visualization_mode &&
               entry.background_color == request.background_color;
    }

    void RenderingManager::insertGTComparisonImageCacheEntry(
        const GTComparisonPreviewRequest& request,
        std::shared_ptr<lfs::core::Tensor> image,
        std::string error,
        const std::chrono::steady_clock::time_point now) {
        const bool image_valid = image && image->is_valid();
        assert(!image_valid || image->device() == lfs::core::Device::CPU);
        const std::size_t image_bytes = image_valid ? image->bytes() : 0;
        auto entry = std::find_if(
            gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
            [&request](const auto& candidate) {
                return gtCacheEntryMatches(candidate, request);
            });
        if (entry != gt_comparison_image_cache_.end()) {
            gt_comparison_image_cache_bytes_ -=
                entry->image && entry->image->is_valid() ? entry->image->bytes() : 0;
            *entry = {
                .cache_epoch = request.cache_epoch,
                .calibration_revision = request.calibration_revision,
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
            gt_comparison_image_cache_.push_back({.cache_epoch = request.cache_epoch,
                                                  .calibration_revision = request.calibration_revision,
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
                                                  .last_used = now});
        }
        gt_comparison_image_cache_bytes_ += image_bytes;

        while (gt_comparison_image_cache_.size() > GT_COMPARISON_IMAGE_CACHE_MAX_ENTRIES ||
               gt_comparison_image_cache_bytes_ > GT_COMPARISON_IMAGE_CACHE_MAX_BYTES) {
            auto lru = gt_comparison_image_cache_.end();
            for (auto candidate = gt_comparison_image_cache_.begin();
                 candidate != gt_comparison_image_cache_.end(); ++candidate) {
                if (displayed_gt_comparison_request_ &&
                    gtCacheEntryMatches(*candidate, *displayed_gt_comparison_request_))
                    continue;
                if (lru == gt_comparison_image_cache_.end() || candidate->last_used < lru->last_used)
                    lru = candidate;
            }
            // A foreground image can exceed the speculative budget by itself.
            // It remains visible; speculative work is refused until it fits.
            if (lru == gt_comparison_image_cache_.end())
                break;
            gt_comparison_image_cache_bytes_ -=
                lru->image && lru->image->is_valid() ? lru->image->bytes() : 0;
            gt_comparison_image_cache_.erase(lru);
        }
    }

    RenderingManager::GTComparisonImageLookup
    RenderingManager::getOrQueueGTComparisonImage(
        GTComparisonPreviewRequest request) {
        GTComparisonImageLookup result;
        bool queued = false;
        const auto now = std::chrono::steady_clock::now();
        const detail::GTComparisonSourceKey source_key{
            .camera_uid = request.camera_uid,
            .image_path = request.image_path};
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            request.owner = request.owner == kNoView ? activeViewId() : request.owner;
            request.cache_epoch = gt_comparison_cache_epoch_;
            request.calibration_revision = request.camera ? request.camera->calibration_revision() : request.calibration_revision;
            displayed_gt_comparison_request_ = request;
            auto cache_entry = std::find_if(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&request](const auto& entry) {
                    return gtCacheEntryMatches(entry, request);
                });
            if (cache_entry != gt_comparison_image_cache_.end()) {
                cache_entry->last_used = now;
                const bool failed =
                    !cache_entry->image || !cache_entry->image->is_valid();
                const bool retry_ready =
                    failed &&
                    cache_entry->failure_time.time_since_epoch().count() != 0 &&
                    now - cache_entry->failure_time >=
                        GT_COMPARISON_IMAGE_RETRY_COOLDOWN;
                if (!retry_ready) {
                    result.image = cache_entry->image;
                    result.error = cache_entry->error;
                    result.status = failed ? GTComparisonImageStatus::Failed
                                           : GTComparisonImageStatus::Ready;
                    return result;
                }
                result.status = GTComparisonImageStatus::Failed;
                result.error = cache_entry->error;
            }

            auto* active = active_gt_comparison_worker_request_
                               ? std::get_if<GTComparisonPreviewRequest>(
                                     &*active_gt_comparison_worker_request_)
                               : nullptr;
            const bool same_as_active = active && gtRequestMatches(*active, request);
            if (same_as_active &&
                (active_gt_comparison_image_is_prefetch_ ||
                 active->generation != gt_comparison_preview_request_generation_)) {
                active_gt_comparison_image_is_prefetch_ = false;
                active->generation = ++gt_comparison_preview_request_generation_;
                active->owner = request.owner;
                pending_gt_comparison_image_request_.reset();
            }
            const bool active_current =
                same_as_active && !active_gt_comparison_image_is_prefetch_ &&
                active->generation == gt_comparison_preview_request_generation_;
            const bool pending_same =
                pending_gt_comparison_image_request_ &&
                gtRequestMatches(*pending_gt_comparison_image_request_, request);

            if (!active_current && !pending_same) {
                const auto prefetched = std::find_if(
                    prefetch_gt_comparison_image_requests_.begin(),
                    prefetch_gt_comparison_image_requests_.end(),
                    [&request](const auto& candidate) {
                        return gtRequestMatches(candidate, request);
                    });
                if (prefetched != prefetch_gt_comparison_image_requests_.end()) {
                    const auto owner = request.owner;
                    request = std::move(*prefetched);
                    request.owner = owner;
                    prefetch_gt_comparison_image_requests_.erase(prefetched);
                }
                request.generation = ++gt_comparison_preview_request_generation_;
                if (request.queued_at.time_since_epoch().count() == 0) {
                    request.queued_at = now;
                }
                pending_gt_comparison_image_request_ = request;
                queued = true;
            }

            if (result.status != GTComparisonImageStatus::Failed) {
                result.status = GTComparisonImageStatus::Loading;
            }
            const GTComparisonPreviewRequest* in_flight = nullptr;
            if (pending_gt_comparison_image_request_ &&
                gtRequestMatches(*pending_gt_comparison_image_request_, request)) {
                in_flight = &*pending_gt_comparison_image_request_;
            } else if (active && gtRequestMatches(*active, request)) {
                in_flight = active;
            }

            const auto stale = std::max_element(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&source_key](const auto& lhs, const auto& rhs) {
                    const auto rank = [&source_key](const auto& entry) {
                        if (!entry.image || !entry.image->is_valid()) {
                            return 0;
                        }
                        return entry.camera_uid == source_key.camera_uid &&
                                       entry.image_path == source_key.image_path
                                   ? 2
                                   : 1;
                    };
                    const int lhs_rank = rank(lhs);
                    const int rhs_rank = rank(rhs);
                    return lhs_rank != rhs_rank ? lhs_rank < rhs_rank
                                                : lhs.last_used < rhs.last_used;
                });
            if (stale != gt_comparison_image_cache_.end() && stale->image &&
                stale->image->is_valid()) {
                result.stale_image = stale->image;
            }
            result.grace_elapsed =
                !in_flight || in_flight->queued_at.time_since_epoch().count() == 0 ||
                now - in_flight->queued_at >= GT_COMPARISON_IMAGE_GRACE_PERIOD;
        }
        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
        return result;
    }

    RenderingManager::GTComparisonFullSourceLookup
    RenderingManager::getOrQueueGTComparisonFullSource(
        GTComparisonFullSourceRequest request) {
        GTComparisonFullSourceLookup result;
        std::optional<GTComparisonFullSourceSlot> released_source;
        bool queued = false;
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            request.owner = request.owner == kNoView ? activeViewId() : request.owner;
            if (gt_comparison_full_source_slot_ &&
                gt_comparison_full_source_slot_->owner == request.owner &&
                gt_comparison_full_source_slot_->source_key == request.source_key) {
                auto& slot = *gt_comparison_full_source_slot_;
                result.status = slot.status;
                result.generation = slot.generation;
                result.source = slot.cpu_source;
                result.error = slot.error;
                const bool retry_ready =
                    slot.status == GTComparisonImageStatus::Failed &&
                    slot.failure_time.time_since_epoch().count() != 0 &&
                    now - slot.failure_time >= GT_COMPARISON_IMAGE_RETRY_COOLDOWN;
                if (!retry_ready) {
                    return result;
                }
            }

            request.generation = ++gt_comparison_full_source_generation_;
            request.queued_at = now;
            pending_gt_comparison_full_source_request_ = request;
            prefetch_gt_comparison_image_requests_.clear();
            released_source = std::move(gt_comparison_full_source_slot_);
            gt_comparison_full_source_slot_ = GTComparisonFullSourceSlot{
                .owner = request.owner,
                .status = GTComparisonImageStatus::Loading,
                .source_key = request.source_key,
                .generation = request.generation};
            result.status = GTComparisonImageStatus::Loading;
            result.generation = request.generation;
            queued = true;
        }
        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
        return result;
    }

    void RenderingManager::queueGTComparisonImagePrefetch(
        GTComparisonPreviewRequest request) {
        bool queued = false;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            request.owner = request.owner == kNoView ? activeViewId() : request.owner;
            request.cache_epoch = gt_comparison_cache_epoch_;
            request.calibration_revision = request.camera ? request.camera->calibration_revision() : request.calibration_revision;
            const bool in_cache = std::any_of(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&request](const auto& entry) {
                    return gtCacheEntryMatches(entry, request);
                });
            const auto* active = active_gt_comparison_worker_request_
                                     ? std::get_if<GTComparisonPreviewRequest>(
                                           &*active_gt_comparison_worker_request_)
                                     : nullptr;
            const bool already_queued =
                (pending_gt_comparison_image_request_ &&
                 gtRequestMatches(*pending_gt_comparison_image_request_, request)) ||
                (active && gtRequestMatches(*active, request)) ||
                std::any_of(
                    prefetch_gt_comparison_image_requests_.begin(),
                    prefetch_gt_comparison_image_requests_.end(),
                    [&request](const auto& candidate) {
                        return gtRequestMatches(candidate, request);
                    });
            if (!in_cache && !already_queued &&
                prefetch_gt_comparison_image_requests_.size() < GT_COMPARISON_IMAGE_PREFETCH_MAX_ENTRIES &&
                gtPrefetchFits(request, gt_comparison_detail::previewBytes(request.image_size))) {
                request.queued_at = std::chrono::steady_clock::now();
                prefetch_gt_comparison_image_requests_.push_back(std::move(request));
                queued = true;
            }
        }
        if (queued) {
            gt_comparison_image_cv_.notify_one();
        }
    }

    void RenderingManager::setGTComparisonActualSizeError(ViewRenderState& view, std::string error) {
        if (view.gt_comparison_actual_size_state_.error != error) {
            view.gt_comparison_actual_size_state_.error = std::move(error);
            publish_viewport_toolbar_generation();
        }
    }

    void RenderingManager::invalidateGTComparisonImageCache(ViewRenderState& view) {
        std::optional<GTComparisonFullSourceSlot> released_full_source;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            ++gt_comparison_cache_epoch_;
            displayed_gt_comparison_request_.reset();
            ++gt_comparison_preview_request_generation_;
            ++gt_comparison_full_source_generation_;
            gt_comparison_image_cache_.clear();
            gt_comparison_image_cache_bytes_ = 0;
            pending_gt_comparison_image_request_.reset();
            pending_gt_comparison_full_source_request_.reset();
            prefetch_gt_comparison_image_requests_.clear();
            released_full_source = std::exchange(gt_comparison_full_source_slot_, std::nullopt);
        }
        // This is a global cache/scene epoch change, independent of which
        // viewport happened to request it. Release every per-view source handle.
        const auto reset_view = [&](ViewRenderState& target) {
            setGTComparisonActualSizeError(target, {});
            target.gt_comparison_actual_size_state_.reset();
            clearPublishedGTComparisonActualFrame(target);
            target.split_left_source_ = nullptr;
            target.split_left_source_size_ = {0, 0};
            target.split_left_source_camera_uid_ = -1;
            target.split_left_source_undistorted_ = false;
        };
        {
            std::lock_guard lock(views_mutex_);
            reset_view(view);
            for (auto& [id, target] : view_states_)
                if (target.get() != &view)
                    reset_view(*target);
        }
        gt_comparison_cuda_owner_ = kNoView;
        gt_comparison_cuda_image_.reset();
        gt_comparison_cuda_source_ = nullptr;
        gt_comparison_cuda_generation_ = 0;
        gt_comparison_cuda_camera_uid_ = -1;
        gt_comparison_cuda_size_ = {0, 0};
        gt_comparison_cuda_undistorted_ = false;
        gt_comparison_loading_placeholder_.reset();
        gt_comparison_failed_placeholder_.reset();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::ensureCudaGTViewportImage(ViewRenderState& view,
                                                                                   std::shared_ptr<lfs::core::Tensor> image,
                                                                                   const int camera_uid,
                                                                                   const glm::ivec2 size,
                                                                                   const bool undistorted,
                                                                                   const std::string_view label) {
        if (!image || !image->is_valid()) {
            return {};
        }
        if (image->device() == lfs::core::Device::CPU) {
            updateSplitLeftCpuSourceIdentity(view,
                                             image.get(), size, camera_uid, undistorted);
        }
        if (image->device() == lfs::core::Device::GPU) {
            return image;
        }
        if (gt_comparison_cuda_owner_ == view.id &&
            gt_comparison_cuda_image_ && gt_comparison_cuda_source_ == image.get() &&
            gt_comparison_cuda_generation_ == view.split_left_image_generation_ &&
            gt_comparison_cuda_camera_uid_ == camera_uid && gt_comparison_cuda_size_ == size &&
            gt_comparison_cuda_undistorted_ == undistorted) {
            return gt_comparison_cuda_image_;
        }
        auto cuda_image = image->gpu();
        if (!cuda_image.is_valid() || cuda_image.device() != lfs::core::Device::GPU) {
            LOG_WARN("{} produced a non-GPU tensor; falling back to the external image path", label);
            return {};
        }
        gt_comparison_cuda_owner_ = view.id;
        gt_comparison_cuda_source_ = image.get();
        gt_comparison_cuda_generation_ = view.split_left_image_generation_;
        gt_comparison_cuda_camera_uid_ = camera_uid;
        gt_comparison_cuda_size_ = size;
        gt_comparison_cuda_undistorted_ = undistorted;
        gt_comparison_cuda_image_ =
            std::make_shared<lfs::core::Tensor>(std::move(cuda_image));
        return gt_comparison_cuda_image_;
    }

    void RenderingManager::updateSplitLeftCpuSourceIdentity(ViewRenderState& view,
                                                            const lfs::core::Tensor* source,
                                                            const glm::ivec2 size,
                                                            const int camera_uid,
                                                            const bool undistorted) {
        if (source == view.split_left_source_ && size == view.split_left_source_size_ &&
            camera_uid == view.split_left_source_camera_uid_ &&
            undistorted == view.split_left_source_undistorted_) {
            return;
        }

        view.split_left_source_ = source;
        view.split_left_source_size_ = size;
        view.split_left_source_camera_uid_ = camera_uid;
        view.split_left_source_undistorted_ = undistorted;
        // High bit keeps this counter disjoint from the per-frame
        // view.split_view_image_generation_ space used by the same interop slots.
        view.split_left_image_generation_ =
            (view.split_left_image_generation_ + 1) | SPLIT_LEFT_GENERATION_BIT;
    }

    void RenderingManager::syncGTComparisonViewSettings(ViewRenderState& view, const RenderSettings& settings) {
        const bool rgb_comparison = splitViewUsesGTComparison(settings.split_view_mode) &&
                                    settings.gt_comparison_mode == GTComparisonMode::RGB;
        if (view.rendered_settings) {
            const auto& previous = *view.rendered_settings;
            if (previous.gt_comparison_actual_size != settings.gt_comparison_actual_size ||
                (previous.split_view_mode != settings.split_view_mode ||
                 previous.gt_comparison_mode != settings.gt_comparison_mode)) {
                invalidateGTComparisonActualSizeResources(view, rgb_comparison);
                clearPublishedGTComparisonActualFrame(view);
            }
        }
        if (!splitViewUsesGTComparison(settings.split_view_mode))
            view.gt_comparison_camera_uid_ = -1;
        else if (view.gt_comparison_camera_uid_ < 0)
            view.gt_comparison_camera_uid_ = camera_interaction_service_.currentCameraId();
    }

    void RenderingManager::invalidateGTComparisonActualSizeTile(ViewRenderState& view) {
        view.gt_comparison_actual_size_state_.invalidateTile();
        if (gt_comparison_cuda_owner_ == view.id) {
            gt_comparison_cuda_image_.reset();
            gt_comparison_cuda_source_ = nullptr;
            gt_comparison_cuda_generation_ = 0;
            gt_comparison_cuda_camera_uid_ = -1;
            gt_comparison_cuda_size_ = {0, 0};
            gt_comparison_cuda_undistorted_ = false;
            gt_comparison_cuda_owner_ = kNoView;
        }
        view.split_left_source_ = nullptr;
        view.split_left_source_size_ = {0, 0};
        view.split_left_source_camera_uid_ = -1;
        view.split_left_source_undistorted_ = false;
    }

    void RenderingManager::invalidateGTComparisonActualSizeResources(ViewRenderState& view, const bool preserve_ready_source) {
        std::optional<GTComparisonFullSourceSlot> released_full_source;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            const bool owns_source = gt_comparison_full_source_slot_ &&
                                     gt_comparison_full_source_slot_->owner == view.id;
            const bool keep_source = preserve_ready_source && owns_source &&
                                     gt_comparison_full_source_slot_->status == GTComparisonImageStatus::Ready &&
                                     gt_comparison_full_source_slot_->cpu_source &&
                                     gt_comparison_full_source_slot_->cpu_source->is_valid();
            if (owns_source && !keep_source) {
                ++gt_comparison_full_source_generation_;
                released_full_source = std::exchange(gt_comparison_full_source_slot_, std::nullopt);
            }
            if (pending_gt_comparison_full_source_request_ &&
                pending_gt_comparison_full_source_request_->owner == view.id)
                pending_gt_comparison_full_source_request_.reset();
            std::erase_if(prefetch_gt_comparison_image_requests_,
                          [&](const auto& request) { return request.owner == view.id; });
            if (!preserve_ready_source && displayed_gt_comparison_request_ &&
                displayed_gt_comparison_request_->owner == view.id) {
                ++gt_comparison_preview_request_generation_;
                displayed_gt_comparison_request_.reset();
                pending_gt_comparison_image_request_.reset();
            }
        }
        setGTComparisonActualSizeError(view, {});
        invalidateGTComparisonActualSizeTile(view);
        view.gt_comparison_actual_size_state_.reset();
    }

    void RenderingManager::retryGTComparisonActualSize() {
        auto& view = state();
        const auto settings = getSettings();
        if (!settings.gt_comparison_actual_size || settings.gt_comparison_mode != GTComparisonMode::RGB ||
            !isGTComparisonActive()) {
            return;
        }
        std::optional<GTComparisonFullSourceSlot> released_source;
        {
            std::lock_guard lock(gt_comparison_image_mutex_);
            if (gt_comparison_full_source_slot_ &&
                gt_comparison_full_source_slot_->owner == view.id &&
                gt_comparison_full_source_slot_->status == GTComparisonImageStatus::Failed) {
                released_source = std::exchange(gt_comparison_full_source_slot_, std::nullopt);
            }
        }
        auto& state = view.gt_comparison_actual_size_state_;
        state.tile_failure.reset();
        setGTComparisonActualSizeError(view, {});
        state.requested_at = std::chrono::steady_clock::now();
        state.lookup_timed = false;
        markDirty(DirtyFlag::SPLIT_VIEW, FrameReason::AsyncCompletion);
    }

    void RenderingManager::publishGTComparisonActualFrame(ViewRenderState& view,
                                                          const GTComparisonActualFrame::Snapshot& snapshot) {
        if (!view.gt_comparison_published_actual_frame_) {
            publish_viewport_toolbar_generation();
        }
        view.gt_comparison_published_actual_frame_ = snapshot;
        if (view.gt_comparison_actual_size_state_.source_key != snapshot.source_key ||
            view.gt_comparison_actual_size_state_.source_generation !=
                snapshot.source_generation) {
            return;
        }

        auto& state = view.gt_comparison_actual_size_state_;
        setGTComparisonActualSizeError(view, {});
        if (state.requested_at.time_since_epoch().count() != 0) {
            LOG_PERF("GT 1:1 published camera={} generation={} extent={}x{} cache_hit={} request_ms={:.2f}",
                     snapshot.source_key.camera_uid, snapshot.source_generation,
                     snapshot.full_extent.x, snapshot.full_extent.y, state.source_cache_hit,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - state.requested_at)
                         .count());
            state.requested_at = {};
        }
        state.fit_fallback.reset();
        std::lock_guard lock(gt_comparison_image_mutex_);
        gt_comparison_image_cache_.clear();
        gt_comparison_image_cache_bytes_ = 0;
        prefetch_gt_comparison_image_requests_.clear();
    }

    void RenderingManager::clearPublishedGTComparisonActualFrame(ViewRenderState& view) {
        if (view.gt_comparison_published_actual_frame_) {
            view.gt_comparison_published_actual_frame_.reset();
            publish_viewport_toolbar_generation();
        }
    }

    RenderingManager::GTComparisonActualFrame
    RenderingManager::prepareGTActualFrame(ViewRenderState& view,
                                           const lfs::core::Camera& camera,
                                           const glm::ivec2 physical_viewport) {
        GTComparisonActualFrame frame;
        std::optional<detail::GTComparisonTileKey> attempted_tile_key;
        const auto fail_tile = [&](std::string error) {
            view.gt_comparison_actual_size_state_.invalidateTile();
            if (attempted_tile_key) {
                view.gt_comparison_actual_size_state_.tile_failure =
                    GTComparisonActualSizeState::TileFailure{
                        .key = *attempted_tile_key,
                        .time = std::chrono::steady_clock::now(),
                        .error = error};
            }
            frame.status = GTComparisonImageStatus::Failed;
            frame.tile.reset();
            frame.error = std::move(error);
            setGTComparisonActualSizeError(view, frame.error);
            LOG_WARN("{}", frame.error);
        };
        const detail::GTComparisonSourceKey source_key{
            .camera_uid = camera.uid(),
            .image_path = camera.image_path(),
            .calibration_revision = camera.calibration_revision()};

        if (view.gt_comparison_actual_size_state_.source_key != source_key) {
            const auto requested_at = view.gt_comparison_actual_size_state_.requested_at;
            const auto pending_pan_camera_uid =
                view.gt_comparison_actual_size_state_.pending_pan_camera_uid;
            const auto pending_pan_offset =
                view.gt_comparison_actual_size_state_.pending_pan_offset;
            setGTComparisonActualSizeError(view, {});
            view.gt_comparison_actual_size_state_.reset();
            view.gt_comparison_actual_size_state_.source_key = source_key;
            if (pending_pan_camera_uid == camera.uid()) {
                view.gt_comparison_actual_size_state_.pending_pan_camera_uid =
                    pending_pan_camera_uid;
                view.gt_comparison_actual_size_state_.pending_pan_offset =
                    pending_pan_offset;
            }
            view.gt_comparison_actual_size_state_.requested_at =
                requested_at.time_since_epoch().count() != 0 ? requested_at : std::chrono::steady_clock::now();
            std::lock_guard lock(gt_comparison_image_mutex_);
            ++gt_comparison_preview_request_generation_;
            pending_gt_comparison_image_request_.reset();
            if (active_gt_comparison_image_is_prefetch_) {
                active_gt_comparison_image_is_prefetch_ = false;
            }
            prefetch_gt_comparison_image_requests_.clear();

            const auto fallback = std::max_element(
                gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                [&source_key](const auto& lhs, const auto& rhs) {
                    const auto rank = [&source_key](const auto& entry) {
                        return entry.mode == GTComparisonMode::RGB &&
                                       entry.camera_uid == source_key.camera_uid &&
                                       entry.image_path == source_key.image_path &&
                                       entry.image && entry.image->is_valid()
                                   ? 1
                                   : 0;
                    };
                    const int lhs_rank = rank(lhs);
                    const int rhs_rank = rank(rhs);
                    return lhs_rank != rhs_rank ? lhs_rank < rhs_rank
                                                : lhs.last_used < rhs.last_used;
                });
            if (fallback != gt_comparison_image_cache_.end() &&
                fallback->mode == GTComparisonMode::RGB &&
                fallback->camera_uid == source_key.camera_uid &&
                fallback->image_path == source_key.image_path && fallback->image &&
                fallback->image->is_valid()) {
                view.gt_comparison_actual_size_state_.fit_fallback = fallback->image;
            }

            gt_comparison_image_cache_.remove_if(
                [&source_key](const GTComparisonImageCacheEntry& entry) {
                    return entry.camera_uid != source_key.camera_uid ||
                           entry.image_path != source_key.image_path;
                });
            gt_comparison_image_cache_bytes_ = 0;
            for (const auto& entry : gt_comparison_image_cache_) {
                if (entry.image && entry.image->is_valid()) {
                    gt_comparison_image_cache_bytes_ += entry.image->bytes();
                }
            }
        }

        const auto lookup = getOrQueueGTComparisonFullSource({.owner = view.id, .source_key = source_key});
        frame.status = lookup.status;
        frame.error = lookup.error;
        auto& state = view.gt_comparison_actual_size_state_;
        if (!state.lookup_timed) {
            state.lookup_timed = true;
            state.source_cache_hit = lookup.status == GTComparisonImageStatus::Ready && lookup.source;
            LOG_PERF("GT 1:1 lookup camera={} generation={} cache_hit={} request_ms={:.2f}",
                     source_key.camera_uid, lookup.generation, state.source_cache_hit,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - state.requested_at)
                         .count());
        }
        if (lookup.status == GTComparisonImageStatus::Failed) {
            setGTComparisonActualSizeError(view, lookup.error);
        }
        frame.fallback = state.fit_fallback;
        if (lookup.status != GTComparisonImageStatus::Ready || !lookup.source ||
            !lookup.source->is_valid()) {
            return frame;
        }

        try {
            if (lookup.source->device() != lfs::core::Device::CPU ||
                lookup.source->dtype() != lfs::core::DataType::UInt8 ||
                lookup.source->ndim() != 3 || lookup.source->shape()[0] != 3 ||
                !lookup.source->is_contiguous()) {
                throw std::runtime_error(
                    "full-resolution source is not owned contiguous CPU CHW RGB8");
            }

            const glm::ivec2 source_extent{
                static_cast<int>(lookup.source->shape()[2]),
                static_cast<int>(lookup.source->shape()[1])};
            const bool distorted = camera.has_distortion();
            std::optional<lfs::core::UndistortParams> scaled_undistort;
            glm::ivec2 full_extent = source_extent;
            if (distorted) {
                scaled_undistort = lfs::core::scale_undistort_params(
                    camera.undistort_params(), source_extent.x, source_extent.y);
                full_extent = {
                    scaled_undistort->dst_width, scaled_undistort->dst_height};
            }

            const bool source_changed =
                view.gt_comparison_actual_size_state_.source_generation != lookup.generation ||
                state.cpu_source != lookup.source;
            const bool reset_crop =
                source_changed || view.gt_comparison_actual_size_state_.full_extent != full_extent;
            if (reset_crop || !state.desired_crop_center) {
                state.desired_crop_center = glm::dvec2(full_extent) * 0.5;
                if (state.pending_pan_camera_uid == camera.uid()) {
                    *state.desired_crop_center += glm::dvec2(state.pending_pan_offset);
                }
            }
            const auto crop = detail::cropGTComparisonFromCenter(full_extent, physical_viewport,
                                                                 *state.desired_crop_center);
            if (!crop.valid()) {
                throw std::runtime_error("full-resolution comparison crop is empty");
            }
            if (state.pending_pan_camera_uid == camera.uid()) {
                state.pending_pan_camera_uid.reset();
                state.pending_pan_offset = {0, 0};
            }

            if (source_changed) {
                view.gt_comparison_actual_size_state_.cpu_source = lookup.source;
                view.gt_comparison_actual_size_state_.cuda_source.reset();
                view.gt_comparison_actual_size_state_.source_generation = lookup.generation;
            }
            view.gt_comparison_actual_size_state_.full_extent = full_extent;
            view.gt_comparison_actual_size_state_.framebuffer_extent = physical_viewport;
            view.gt_comparison_actual_size_state_.crop = crop;

            const detail::GTComparisonTileKey tile_key{
                .source_generation = lookup.generation,
                .full_extent = full_extent,
                .framebuffer_extent = physical_viewport,
                .crop = crop,
                .distorted = distorted};
            attempted_tile_key = tile_key;
            const bool needs_tile =
                !view.gt_comparison_actual_size_state_.tile_key ||
                *view.gt_comparison_actual_size_state_.tile_key != tile_key ||
                !view.gt_comparison_actual_size_state_.visible_tile;
            if (needs_tile) {
                const auto now = std::chrono::steady_clock::now();
                if (view.gt_comparison_actual_size_state_.tile_failure &&
                    view.gt_comparison_actual_size_state_.tile_failure->suppresses(
                        tile_key, now, GT_COMPARISON_IMAGE_RETRY_COOLDOWN)) {
                    frame.status = GTComparisonImageStatus::Failed;
                    frame.error =
                        view.gt_comparison_actual_size_state_.tile_failure->error;
                    return frame;
                }
                LOG_TIMER("GT 1:1 tile preparation (CPU and GPU submission)");
                lfs::core::Tensor visible;
                if (scaled_undistort) {
                    if (!view.gt_comparison_actual_size_state_.cuda_source) {
                        LOG_TIMER("GT 1:1 source upload submission");
                        auto cuda_source =
                            lookup.source->to(lfs::core::Device::GPU).contiguous();
                        view.gt_comparison_actual_size_state_.cuda_source =
                            std::make_shared<lfs::core::Tensor>(std::move(cuda_source));
                    }
                    LOG_TIMER("GT 1:1 undistortion submission");
                    visible = lfs::core::undistort_image_region(
                        *view.gt_comparison_actual_size_state_.cuda_source,
                        *scaled_undistort,
                        crop.origin.x,
                        crop.origin.y,
                        crop.extent.x,
                        crop.extent.y,
                        nullptr);
                } else {
                    visible = lookup.source
                                  ->slice(
                                      1,
                                      static_cast<std::size_t>(crop.origin.y),
                                      static_cast<std::size_t>(
                                          crop.origin.y + crop.extent.y))
                                  .slice(
                                      2,
                                      static_cast<std::size_t>(crop.origin.x),
                                      static_cast<std::size_t>(
                                          crop.origin.x + crop.extent.x))
                                  .contiguous();
                }
                view.gt_comparison_actual_size_state_.visible_tile =
                    std::make_shared<lfs::core::Tensor>(std::move(visible));
                view.gt_comparison_actual_size_state_.tile_key = tile_key;
            }
            // Prepare the display upload before Ready so pinhole allocation
            // failures share the native tile recovery path and retry cooldown.
            auto cuda_tile = ensureCudaGTViewportImage(view,
                                                       view.gt_comparison_actual_size_state_.visible_tile,
                                                       camera.uid(),
                                                       crop.extent,
                                                       camera.camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                                           camera.is_undistort_precomputed(),
                                                       "GT comparison ground-truth display");
            if (!cuda_tile) {
                throw std::runtime_error("could not upload the RGB GT comparison 1:1 tile");
            }
            view.gt_comparison_actual_size_state_.tile_failure.reset();

            frame.status = GTComparisonImageStatus::Ready;
            frame.tile = std::move(cuda_tile);
            frame.pixel_region = detail::GTComparisonPixelRegion{
                .origin = crop.origin,
                .full_extent = full_extent,
                .full_intrinsics =
                    scaled_undistort
                        ? std::optional{lfs::rendering::CameraIntrinsics{
                              .focal_x = scaled_undistort->dst_fx,
                              .focal_y = scaled_undistort->dst_fy,
                              .center_x = scaled_undistort->dst_cx,
                              .center_y = scaled_undistort->dst_cy}}
                        : std::nullopt};
            frame.snapshot = GTComparisonActualFrame::Snapshot{
                .source_key = source_key,
                .source_generation = lookup.generation,
                .full_extent = full_extent,
                .framebuffer_extent = physical_viewport,
                .crop = crop};
            frame.content_rect = detail::centeredGTComparisonContentRect(
                physical_viewport, crop.extent);
        } catch (const lfs::core::MemoryAllocationError& error) {
            fail_tile(std::format(
                "Not enough memory to build the RGB GT comparison 1:1 tile "
                "({} bytes requested): {}",
                error.requested_bytes(),
                error.what()));
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): fail_tile records the failure and reports it to the viewport and log.
            fail_tile(std::format(
                "RGB GT comparison 1:1 tile failed: {}", error.what()));
        }
        return frame;
    }

    bool RenderingManager::gtPreviewDecodeValid(const GTComparisonPreviewRequest& request) const {
        return request.cache_epoch == gt_comparison_cache_epoch_ &&
               (!request.camera || request.calibration_revision == request.camera->calibration_revision());
    }

    bool RenderingManager::gtPrefetchFits(const GTComparisonPreviewRequest& request,
                                          const std::size_t bytes, const bool completing) const {
        std::size_t reserved = 0;
        std::size_t entries = gt_comparison_image_cache_.size();
        const auto reserve = [&](const std::size_t amount) {
            if (!gt_comparison_detail::prefetchFits(reserved, gt_comparison_image_cache_bytes_,
                                                    amount, GT_COMPARISON_IMAGE_CACHE_MAX_BYTES))
                return false;
            reserved += amount;
            ++entries;
            return true;
        };
        if (displayed_gt_comparison_request_ &&
            std::none_of(gt_comparison_image_cache_.begin(), gt_comparison_image_cache_.end(),
                         [&](const auto& entry) { return gtCacheEntryMatches(entry, *displayed_gt_comparison_request_); }) &&
            !reserve(gt_comparison_detail::previewBytes(displayed_gt_comparison_request_->image_size)))
            return false;
        for (const auto& pending : prefetch_gt_comparison_image_requests_) {
            if (gtPreviewDecodeValid(pending) && !reserve(gt_comparison_detail::previewBytes(pending.image_size)))
                return false;
        }
        const auto* active = active_gt_comparison_worker_request_
                                 ? std::get_if<GTComparisonPreviewRequest>(&*active_gt_comparison_worker_request_)
                                 : nullptr;
        if (active && active_gt_comparison_image_is_prefetch_ && gtPreviewDecodeValid(*active) &&
            !(completing && gtRequestMatches(*active, request)) &&
            !reserve(gt_comparison_detail::previewBytes(active->image_size)))
            return false;
        return entries < GT_COMPARISON_IMAGE_CACHE_MAX_ENTRIES &&
               gt_comparison_detail::prefetchFits(gt_comparison_image_cache_bytes_, reserved,
                                                  bytes, GT_COMPARISON_IMAGE_CACHE_MAX_BYTES);
    }

    ViewId RenderingManager::completeGTComparisonImage(
        const GTComparisonWorkerRequest& request, std::shared_ptr<lfs::core::Tensor> image,
        std::string error, const bool stopped) {
        std::lock_guard lock(gt_comparison_image_mutex_);
        ViewId consumer = kNoView;
        if (const auto* full = std::get_if<GTComparisonFullSourceRequest>(&request)) {
            const auto* active = active_gt_comparison_worker_request_
                                     ? std::get_if<GTComparisonFullSourceRequest>(&*active_gt_comparison_worker_request_)
                                     : nullptr;
            const bool matches = active && active->owner == full->owner &&
                                 active->source_key == full->source_key && active->generation == full->generation;
            if (!stopped && matches && full->generation == gt_comparison_full_source_generation_ &&
                gt_comparison_full_source_slot_ && gt_comparison_full_source_slot_->owner == full->owner &&
                gt_comparison_full_source_slot_->source_key == full->source_key &&
                gt_comparison_full_source_slot_->generation == full->generation) {
                auto& slot = *gt_comparison_full_source_slot_;
                slot.status = image && image->is_valid() ? GTComparisonImageStatus::Ready : GTComparisonImageStatus::Failed;
                slot.cpu_source = std::move(image);
                slot.error = std::move(error);
                slot.failure_time = slot.status == GTComparisonImageStatus::Failed
                                        ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
                consumer = full->owner;
            }
            if (matches) {
                active_gt_comparison_worker_request_.reset();
                active_gt_comparison_image_is_prefetch_ = false;
            }
        } else {
            const auto& preview = std::get<GTComparisonPreviewRequest>(request);
            const auto* active = active_gt_comparison_worker_request_
                                     ? std::get_if<GTComparisonPreviewRequest>(&*active_gt_comparison_worker_request_)
                                     : nullptr;
            const bool matches = active && gtRequestMatches(*active, preview);
            const bool prefetch = active_gt_comparison_image_is_prefetch_;
            const bool current = matches && (prefetch || active->generation == gt_comparison_preview_request_generation_);
            const auto bytes = image && image->is_valid() ? image->bytes() : 0;
            if (!stopped && current && gtPreviewDecodeValid(preview) &&
                (!prefetch || gtPrefetchFits(preview, bytes, true))) {
                insertGTComparisonImageCacheEntry(preview, std::move(image), std::move(error), std::chrono::steady_clock::now());
                if (displayed_gt_comparison_request_ && gtRequestMatches(*displayed_gt_comparison_request_, preview)) {
                    consumer = displayed_gt_comparison_request_->owner;
                    if (pending_gt_comparison_image_request_ && gtRequestMatches(*pending_gt_comparison_image_request_, preview))
                        pending_gt_comparison_image_request_.reset();
                }
            }
            // This releases the active reservation on success, rejection, cancellation and failure.
            if (matches) {
                active_gt_comparison_worker_request_.reset();
                active_gt_comparison_image_is_prefetch_ = false;
            }
        }
        return consumer;
    }

    void RenderingManager::gtComparisonImageWorkerLoop(
        const std::stop_token stop_token) {
        while (true) {
            GTComparisonWorkerRequest request;
            bool is_prefetch = false;
            {
                std::unique_lock lock(gt_comparison_image_mutex_);
                gt_comparison_image_cv_.wait(lock, stop_token, [this] {
                    return pending_gt_comparison_full_source_request_.has_value() ||
                           pending_gt_comparison_image_request_.has_value() ||
                           !prefetch_gt_comparison_image_requests_.empty();
                });
                if (stop_token.stop_requested()) {
                    pending_gt_comparison_full_source_request_.reset();
                    pending_gt_comparison_image_request_.reset();
                    prefetch_gt_comparison_image_requests_.clear();
                    active_gt_comparison_worker_request_.reset();
                    active_gt_comparison_image_is_prefetch_ = false;
                    return;
                }

                if (pending_gt_comparison_full_source_request_) {
                    request = std::move(*pending_gt_comparison_full_source_request_);
                    pending_gt_comparison_full_source_request_.reset();
                } else if (pending_gt_comparison_image_request_) {
                    request = std::move(*pending_gt_comparison_image_request_);
                    pending_gt_comparison_image_request_.reset();
                } else {
                    request = std::move(prefetch_gt_comparison_image_requests_.back());
                    prefetch_gt_comparison_image_requests_.pop_back();
                    is_prefetch = true;
                }
                active_gt_comparison_worker_request_ = request;
                active_gt_comparison_image_is_prefetch_ = is_prefetch;
            }

            std::shared_ptr<lfs::core::Tensor> image;
            std::string error;
            if (const auto* full =
                    std::get_if<GTComparisonFullSourceRequest>(&request)) {
                try {
                    LOG_PERF("GT 1:1 source begin camera={} generation={} path={} queue_ms={:.2f}",
                             full->source_key.camera_uid, full->generation,
                             lfs::core::path_to_utf8(full->source_key.image_path),
                             std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - full->queued_at)
                                 .count());
                    LOG_TIMER("GT 1:1 native source load");
                    auto source = lfs::core::load_image_rgb8_chw_native_resolution(
                        full->source_key.image_path);
                    LOG_PERF("GT 1:1 source decoded camera={} generation={} extent={}x{} bytes={}",
                             full->source_key.camera_uid, full->generation,
                             source.shape()[2], source.shape()[1], source.bytes());
                    image = std::make_shared<lfs::core::Tensor>(std::move(source));
                } catch (const std::exception& exception) {
                    error = std::format(
                        "RGB GT comparison full source load failed: {}",
                        exception.what());
                    LOG_WARN("{}", error);
                } catch (...) {
                    error =
                        "RGB GT comparison full source load failed with an unknown error";
                    LOG_WARN("{}", error);
                }
            } else {
                const auto& preview =
                    std::get<GTComparisonPreviewRequest>(request);
                try {
                    lfs::core::Tensor gt_tensor;
                    if (preview.mode == GTComparisonMode::RGB) {
                        std::shared_ptr<lfs::core::Tensor> retained;
                        {
                            std::lock_guard lock(gt_comparison_image_mutex_);
                            if (gt_comparison_full_source_slot_ &&
                                gt_comparison_full_source_slot_->owner == preview.owner &&
                                gt_comparison_full_source_slot_->status == GTComparisonImageStatus::Ready &&
                                gt_comparison_full_source_slot_->source_key == detail::GTComparisonSourceKey{
                                                                                   .camera_uid = preview.camera_uid,
                                                                                   .image_path = preview.image_path,
                                                                                   .calibration_revision = preview.calibration_revision})
                                retained = gt_comparison_full_source_slot_->cpu_source;
                        }
                        if (retained) {
                            const double scale = std::min(1.0, static_cast<double>(preview.preview_max_dimension) /
                                                                   std::max(retained->size(1), retained->size(2)));
                            auto resized = resizeUInt8Preview(retained, {std::max(1, static_cast<int>(std::lround(retained->size(2) * scale))),
                                                                        std::max(1, static_cast<int>(std::lround(retained->size(1) * scale)))});
                            if (resized)
                                gt_tensor = *resized;
                        } else {
                            auto [pixels, width, height, channels] = lfs::core::load_image(
                                preview.image_path, -1, preview.preview_max_dimension);
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
                    }
                    if (preview.mode == GTComparisonMode::RGB) {
                        if (gt_tensor.is_valid() && gt_tensor.ndim() == 3) {
                            const auto gt_layout = lfs::rendering::detectImageLayout(gt_tensor);
                            if (gt_layout != lfs::rendering::ImageLayout::Unknown) {
                                const bool undistort_gt =
                                    gt_layout == lfs::rendering::ImageLayout::CHW &&
                                    preview.undistort_requested;
                                if (undistort_gt) {
                                    if (gt_tensor.device() != lfs::core::Device::GPU) {
                                        gt_tensor = gt_tensor.to(lfs::core::Device::GPU);
                                    }
                                    if (gt_tensor.dtype() == lfs::core::DataType::UInt8) {
                                        gt_tensor = gt_tensor.to(lfs::core::DataType::Float32) / 255.0f;
                                    }
                                    const auto scaled = lfs::core::scale_undistort_params(
                                        preview.undistort_params,
                                        lfs::rendering::imageWidth(gt_tensor, gt_layout),
                                        lfs::rendering::imageHeight(gt_tensor, gt_layout),
                                        preview.preview_max_dimension);
                                    gt_tensor = lfs::core::undistort_image(
                                        gt_tensor.clamp(0.0f, 1.0f).contiguous(), scaled,
                                        nullptr);
                                }
                                gt_tensor = lfs::rendering::flipImageVertical(gt_tensor, gt_layout);
                                // Static GT display images must be decoupled from the CUDA pool
                                // while training can recycle device buffers mid-frame.
                                gt_tensor = gt_tensor.cpu();
                                image = std::make_shared<lfs::core::Tensor>(std::move(gt_tensor));
                                image = resizePreview(image, preview.image_size);
                            }
                        }
                    } else if (preview.camera) {
                        if (preview.mode == GTComparisonMode::Depth) {
                            auto depth = preview.camera->load_and_get_depth(
                                -1, preview.preview_max_dimension);
                            if (depth.is_valid() && depth.ndim() == 2) {
                                if (preview.undistort_requested) {
                                    const auto scaled = lfs::core::scale_undistort_params(
                                        preview.undistort_params,
                                        static_cast<int>(depth.shape()[1]),
                                        static_cast<int>(depth.shape()[0]),
                                        preview.preview_max_dimension);
                                    depth = lfs::core::undistort_depth_area(
                                        depth, scaled, nullptr);
                                }
                                image = lfs::vis::makeDepthDisplayTensor(
                                    depth, preview.depth_visualization_mode, preview.background_color);
                                image = resizePreview(image, preview.image_size);
                                if (image) {
                                    auto flipped = lfs::rendering::flipImageVertical(
                                        *image, lfs::rendering::ImageLayout::CHW);
                                    image = std::make_shared<lfs::core::Tensor>(std::move(flipped));
                                }
                            }
                        } else {
                            auto normal = preview.camera->load_and_get_normal(
                                -1, preview.preview_max_dimension, lfs::core::Camera::NormalPriorDecode{});
                            if (normal.is_valid() && normal.ndim() == 3) {
                                const auto normal_layout = lfs::rendering::detectImageLayout(normal);
                                if (preview.undistort_requested &&
                                    normal_layout != lfs::rendering::ImageLayout::Unknown) {
                                    const auto scaled = lfs::core::scale_undistort_params(
                                        preview.undistort_params,
                                        lfs::rendering::imageWidth(normal, normal_layout),
                                        lfs::rendering::imageHeight(normal, normal_layout),
                                        preview.preview_max_dimension);
                                    normal = lfs::core::undistort_normal_area(
                                        normal, scaled, nullptr);
                                }
                                image = lfs::vis::makeNormalDisplayTensor(normal);
                                image = resizePreview(image, preview.image_size);
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
                        error = preview.mode == GTComparisonMode::RGB
                                    ? "RGB GT comparison could not load the source image"
                                : preview.mode == GTComparisonMode::Depth
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
            }

            const auto consumer = completeGTComparisonImage(request, std::move(image), std::move(error), stop_token.stop_requested());
            if (consumer != kNoView) {
                std::lock_guard lock(views_mutex_);
                if (view_states_.contains(consumer))
                    markViewDirty(consumer, DirtyFlag::SPLIT_VIEW, FrameReason::AsyncCompletion);
            }
        }
    }

} // namespace lfs::vis
