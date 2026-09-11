/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_bus.hpp"

#include "core/error_reporter.hpp"

#include <utility>

namespace lfs {

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

    ErrorDedup::Decision ErrorDedup::check(const std::uint64_t key,
                                           const std::chrono::steady_clock::time_point now) {
        std::lock_guard lock(mutex_);
        std::erase_if(entries_, [&](const auto& kv) {
            return now - kv.second.delivered_at >= kIdleExpiry;
        });
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            entries_[key] = Entry{.count = 0, .delivered_at = now};
            return {.suppress = false, .repeats = 0};
        }
        if (now - it->second.delivered_at < kWindow) {
            ++it->second.count;
            return {.suppress = true, .repeats = 0};
        }
        const std::uint32_t repeats = it->second.count;
        it->second.count = 0;
        it->second.delivered_at = now;
        return {.suppress = false, .repeats = repeats};
    }

    void ErrorBus::publish(ErrorNotification notification) noexcept {
        try {
            const std::uint64_t key =
                core::error_fingerprint(notification.error) ^ notification.operation_id.value();
            const ErrorDedup::Decision decision =
                dedup_.check(key, std::chrono::steady_clock::now());
            if (decision.suppress) {
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

            const ErrorDeliveryInfo delivery{.suppressed_repeats = decision.repeats};
            bool delivered = false;
            for (NativeErrorConsumer* consumer : snapshot) {
                if (consumer != nullptr) {
                    consumer->on_error(notification, delivery);
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
