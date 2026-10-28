/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lfs::vis::detail {
    enum class CollisionUploadStatus { Ready,
                                       RetryableFailure,
                                       PermanentFailure,
                                       PendingFailure };

    // CPU policy only. Submitted draws retain their own shared ownership; pending
    // failed uploads must be retained until the caller proves fence retirement.
    template <typename Source, typename Resource>
    class CollisionPreviewCache {
    public:
        using Clock = std::chrono::steady_clock;
        using Time = Clock::time_point;
        struct Stats {
            uint64_t attempts = 0, hits = 0, failures = 0, evictions = 0;
            // Logical storage bytes, excluding allocator overhead, frame-only
            // ownership and additional staging bytes of pending failed uploads.
            size_t resident_bytes = 0, pending_bytes = 0;
        };

        explicit CollisionPreviewCache(size_t budget = 128 * 1024 * 1024,
                                       Clock::duration idle = std::chrono::seconds(5))
            : budget_(budget), idle_(idle) {}

        static bool needsPreview(bool has_volume, float opacity, bool depth_view, uint32_t panel_mask) {
            return has_volume && opacity > 0 && !depth_view && panel_mask != 0;
        }

        template <typename Upload, typename Retired>
        std::shared_ptr<Resource> request(const std::shared_ptr<const Source>& source, size_t bytes,
                                          uint64_t frame, Time now, Upload&& upload, Retired&& retired) {
            auto& entry = entries_[source.get()];
            // The weak source also distinguishes a recycled address after removal.
            if (entry.source.expired())
                entry = Entry{};
            entry.source = source;
            const bool reactivated = entry.last_frame != frame && entry.last_frame + 1 < frame;
            entry.last_frame = frame;
            entry.last_used = now;
            entry.inactive_since.reset();
            if (entry.status == CollisionUploadStatus::Ready) {
                if (auto resource = entry.weak.lock()) {
                    entry.resident = resource;
                    ++stats_.hits;
                    return resource;
                }
            }
            if (entry.status == CollisionUploadStatus::PendingFailure) {
                if (!retired(*entry.resident))
                    return {};
                entry.resident.reset();
                entry.status = CollisionUploadStatus::RetryableFailure;
            }
            if (entry.status == CollisionUploadStatus::PermanentFailure)
                return {};
            if (reactivated) {
                entry.failures = 0;
                entry.next_retry = now;
            }
            if (entry.failures >= max_failures || now < entry.next_retry)
                return {};

            ++stats_.attempts;
            std::shared_ptr<Resource> resource;
            const auto status = upload(resource);
            entry.bytes = bytes;
            entry.status = status;
            if (status == CollisionUploadStatus::Ready) {
                entry.failures = 0;
                entry.next_retry = {};
                entry.weak = resource;
                entry.resident = resource;
                return resource;
            }
            ++stats_.failures;
            ++entry.failures;
            entry.next_retry = now + std::chrono::seconds(1u << (entry.failures - 1));
            if (status == CollisionUploadStatus::PendingFailure)
                entry.resident = std::move(resource);
            return {};
        }

        std::shared_ptr<Resource> ready(const Source* source) const {
            const auto it = entries_.find(source);
            return it != entries_.end() && it->second.status == CollisionUploadStatus::Ready
                       ? it->second.weak.lock()
                       : nullptr;
        }

        bool requestedInFrame(const Source* source, uint64_t frame) const {
            const auto it = entries_.find(source);
            return it != entries_.end() && it->second.last_frame == frame;
        }

        // A retired frame slot may keep its existing ownership during the
        // hidden grace period. This never allocates or promotes cache storage:
        // oversized previews otherwise disappear after two hidden UI frames,
        // even when immediately re-enabled.
        bool retainFrameOwner(const Source* source, uint64_t frame, Time now) const {
            const auto it = entries_.find(source);
            if (it == entries_.end() || it->second.status != CollisionUploadStatus::Ready)
                return false;
            const auto& entry = it->second;
            return entry.last_frame == frame || !entry.inactive_since || now - *entry.inactive_since < idle_;
        }

        // Called by the viewer's idle loop too: retries must not depend on input
        // producing another frame. Hidden entries retain their failure history
        // but never schedule rendering. Pending uploads only schedule fence polls.
        std::optional<Time> nextDeadline(uint64_t frame, Time now) const {
            std::optional<Time> result;
            const auto consider = [&](Time due) {
                if (!result || due < *result)
                    result = due;
            };
            for (const auto& [_, entry] : entries_) {
                if (entry.status == CollisionUploadStatus::Ready && entry.resident && entry.last_frame != frame)
                    consider(entry.inactive_since.value_or(entry.last_used) + idle_);
                else if (entry.status == CollisionUploadStatus::PendingFailure)
                    consider(now + std::chrono::milliseconds(250));
                else if (entry.status == CollisionUploadStatus::RetryableFailure &&
                         entry.last_frame == frame && entry.failures < max_failures)
                    consider(entry.next_retry);
            }
            return result;
        }

        bool retryDue(uint64_t frame, Time now) const {
            for (const auto& [_, entry] : entries_)
                if (entry.status == CollisionUploadStatus::RetryableFailure &&
                    entry.last_frame == frame && entry.failures < max_failures && now >= entry.next_retry)
                    return true;
            return false;
        }

        template <typename Retired>
        void maintain(Time now, Retired&& retired, std::optional<uint64_t> visible_frame = {}) {
            stats_.resident_bytes = 0;
            stats_.pending_bytes = 0;
            for (auto it = entries_.begin(); it != entries_.end();) {
                auto& entry = it->second;
                const bool visible = visible_frame && entry.last_frame == *visible_frame;
                if (visible)
                    entry.inactive_since.reset();
                else if (!entry.inactive_since)
                    entry.inactive_since = visible_frame ? now : entry.last_used;
                if (entry.status == CollisionUploadStatus::PendingFailure && retired(*entry.resident)) {
                    entry.resident.reset();
                    entry.status = CollisionUploadStatus::RetryableFailure;
                }
                if (entry.status == CollisionUploadStatus::Ready && entry.resident) {
                    if (!visible && now - *entry.inactive_since >= idle_)
                        evict(entry);
                    else
                        stats_.resident_bytes += entry.bytes;
                }
                if (entry.status == CollisionUploadStatus::PendingFailure)
                    stats_.pending_bytes += entry.bytes;
                if (entry.source.expired())
                    it = entries_.erase(it);
                else
                    ++it;
            }
            // Drop cache ownership only; frame-slot references protect GPU reads
            // and weak references let oversized/in-flight volumes remain reusable.
            if (stats_.resident_bytes > budget_) {
                std::vector<Entry*> candidates;
                for (auto& [_, entry] : entries_)
                    if (entry.status == CollisionUploadStatus::Ready && entry.resident)
                        candidates.push_back(&entry);
                std::sort(candidates.begin(), candidates.end(), [](const auto* a, const auto* b) {
                    return a->last_used < b->last_used;
                });
                for (auto* entry : candidates) {
                    if (stats_.resident_bytes <= budget_)
                        break;
                    stats_.resident_bytes -= entry->bytes;
                    evict(*entry);
                }
            }
        }

        void clear() {
            entries_.clear();
            stats_.resident_bytes = 0;
            stats_.pending_bytes = 0;
        }
        const Stats& stats() const { return stats_; }

    private:
        static constexpr unsigned max_failures = 3;
        struct Entry {
            std::weak_ptr<const Source> source;
            std::weak_ptr<Resource> weak;
            std::shared_ptr<Resource> resident;
            CollisionUploadStatus status = CollisionUploadStatus::RetryableFailure;
            Time last_used{}, next_retry{};
            std::optional<Time> inactive_since;
            uint64_t last_frame = 0;
            size_t bytes = 0;
            unsigned failures = 0;
        };
        void evict(Entry& entry) {
            entry.resident.reset();
            ++stats_.evictions;
        }
        size_t budget_;
        Clock::duration idle_;
        Stats stats_;
        std::unordered_map<const Source*, Entry> entries_;
    };
} // namespace lfs::vis::detail
