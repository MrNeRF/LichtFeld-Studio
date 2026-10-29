/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "collision_preview_cache.hpp"
#include <future>
#include <stop_token>
#include <utility>

namespace lfs::vis::detail {
    // One upload at a time bounds outstanding allocations across all volumes.
    // poll/cancel never wait. finish is exclusively for device-owner teardown.
    template <typename Resource>
    class CollisionPreviewUploadJob {
    public:
        ~CollisionPreviewUploadJob() { cancel(); }
        struct Completion {
            std::shared_ptr<Resource> resource;
            CollisionUploadStatus status;
        };

        template <typename Work>
        bool start(std::shared_ptr<Resource> resource, Work&& work) {
            if (active())
                return false;
            stop_ = std::stop_source{};
            auto future = std::async(std::launch::async,
                                     [resource, token = stop_.get_token(), work = std::forward<Work>(work)]() mutable {
                                         return work(*resource, token);
                                     });
            resource_ = std::move(resource);
            future_ = std::move(future);
            return true;
        }

        bool active() const { return future_.valid(); }
        const std::shared_ptr<Resource>& resource() const { return resource_; }
        void cancel() { stop_.request_stop(); }

        std::optional<Completion> poll() {
            if (!active() || future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                return {};
            return take();
        }

        std::optional<Completion> finish() {
            if (!active())
                return {};
            cancel();
            return take();
        }

    private:
        Completion take() {
            auto resource = std::move(resource_);
            try {
                const auto status = future_.get();
                return {std::move(resource), status};
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): report PendingFailure to the owner,
                // which records the failure and retains any unretired submission.
                // Preserve ownership conservatively: an unexpected worker
                // exception cannot prove that its GPU submission has retired.
                return {std::move(resource), CollisionUploadStatus::PendingFailure};
            }
        }
        std::shared_ptr<Resource> resource_;
        std::stop_source stop_;
        std::future<CollisionUploadStatus> future_;
    };
} // namespace lfs::vis::detail
