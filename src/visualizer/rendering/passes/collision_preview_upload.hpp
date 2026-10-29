/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>

namespace lfs::vis::detail {
    enum class CollisionUploadStatus { Ready,
                                       Canceled,
                                       Failed,
                                       PendingFailure };
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
        bool start(std::shared_ptr<Resource> resource, Work&& work, std::function<void()> wake = {}) {
            if (active())
                return false;
            stop_ = std::stop_source{};
            std::promise<CollisionUploadStatus> result;
            auto future = result.get_future();
            worker_ = std::jthread(
                [resource, token = stop_.get_token(), work = std::forward<Work>(work),
                 result = std::move(result), wake = std::move(wake)]() mutable {
                    try {
                        result.set_value(work(*resource, token));
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): forward the exception through
                        // the result future; take() preserves pending GPU ownership.
                        result.set_exception(std::current_exception());
                    }
                    // Publish before waking: an idle viewer must never consume
                    // the only wake while poll() still sees an unfinished job.
                    if (wake)
                        wake();
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
            cancel();
            auto result = active() ? take() : std::optional<Completion>{};
            if (worker_.joinable())
                worker_.join();
            return result;
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
        std::jthread worker_;
    };
} // namespace lfs::vis::detail
