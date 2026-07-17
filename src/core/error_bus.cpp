/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_bus.hpp"

#include "core/error_reporter.hpp"

#include <utility>

namespace lfs {

    namespace {
        // Repeated frame faults collapse to a count within this window (matches
        // the five-second frame-log cadence). Product policy per the spec §9.
        constexpr std::chrono::seconds kDedupWindow{5};
    } // namespace

    Subscription::Subscription(ErrorBus* bus, const std::uint64_t id) noexcept
        : bus_(bus),
          id_(id) {}

    Subscription::Subscription(Subscription&& other) noexcept
        : bus_(std::exchange(other.bus_, nullptr)),
          id_(std::exchange(other.id_, 0)) {}

    Subscription& Subscription::operator=(Subscription&& other) noexcept {
        if (this != &other) {
            reset();
            bus_ = std::exchange(other.bus_, nullptr);
            id_ = std::exchange(other.id_, 0);
        }
        return *this;
    }

    Subscription::~Subscription() {
        reset();
    }

    void Subscription::reset() noexcept {
        if (bus_ != nullptr) {
            bus_->unsubscribe(id_);
            bus_ = nullptr;
            id_ = 0;
        }
    }

    ErrorBus& ErrorBus::instance() {
        static ErrorBus bus;
        return bus;
    }

    std::unique_ptr<ErrorBus> ErrorBus::create_isolated_for_testing() {
        return std::unique_ptr<ErrorBus>(new ErrorBus());
    }

    Subscription ErrorBus::subscribe(NativeErrorConsumer& consumer) {
        std::lock_guard lock(subscribers_mutex_);
        const std::uint64_t id = next_id_++;
        subscribers_.push_back(Subscriber{.id = id, .consumer = &consumer});
        return Subscription{this, id};
    }

    void ErrorBus::unsubscribe(const std::uint64_t id) noexcept {
        std::lock_guard lock(subscribers_mutex_);
        std::erase_if(subscribers_, [id](const Subscriber& s) { return s.id == id; });
    }

    bool ErrorBus::should_suppress(const std::uint64_t key) {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard lock(dedup_mutex_);
        std::erase_if(dedup_, [&](const auto& kv) { return now - kv.second.last >= kDedupWindow; });
        auto it = dedup_.find(key);
        if (it != dedup_.end() && now - it->second.last < kDedupWindow) {
            ++it->second.count;
            it->second.last = now;
            return true;
        }
        dedup_[key] = DedupEntry{.count = 1, .last = now};
        return false;
    }

    void ErrorBus::publish(ErrorNotification notification) noexcept {
        try {
            const std::uint64_t key =
                core::error_fingerprint(notification.error) ^ notification.operation_id.value();
            if (should_suppress(key)) {
                return;
            }

            std::vector<NativeErrorConsumer*> snapshot;
            {
                std::lock_guard lock(subscribers_mutex_);
                snapshot.reserve(subscribers_.size());
                for (const Subscriber& s : subscribers_) {
                    snapshot.push_back(s.consumer);
                }
            }

            bool delivered = false;
            for (NativeErrorConsumer* consumer : snapshot) {
                if (consumer != nullptr) {
                    consumer->on_error(notification);
                    delivered = true;
                }
            }

            if (!delivered) {
                const core::ReportChannel channel = notification.surface == ErrorSurface::Modal
                                                        ? core::ReportChannel::ProcessBoundary
                                                        : core::ReportChannel::OwnerLog;
                core::ErrorReporter::get().report(notification.error, channel);
            }
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): publish is the noexcept surfacing
            // boundary; a throwing subscriber list/dedup map (extreme OOM) must
            // never propagate to the publishing worker thread.
        }
    }

} // namespace lfs
