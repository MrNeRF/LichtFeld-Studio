/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lfs::core::reactive {

    class Store;

    class LFS_CORE_API SubscriptionToken {
    public:
        SubscriptionToken() = default;
        SubscriptionToken(Store* store, std::uint32_t field_id, std::uint64_t subscription_id) noexcept;
        ~SubscriptionToken();

        SubscriptionToken(const SubscriptionToken&) = delete;
        SubscriptionToken& operator=(const SubscriptionToken&) = delete;

        SubscriptionToken(SubscriptionToken&& other) noexcept;
        SubscriptionToken& operator=(SubscriptionToken&& other) noexcept;

        void reset();
        [[nodiscard]] bool active() const noexcept { return store_ != nullptr; }

    private:
        Store* store_ = nullptr;
        std::uint32_t field_id_ = 0;
        std::uint64_t subscription_id_ = 0;
    };

    class LFS_CORE_API Store {
    public:
        using FieldId = std::uint32_t;
        using Subscriber = std::function<void()>;
        using WakeCallback = void (*)();

        Store() = default;
        ~Store() = default;

        Store(const Store&) = delete;
        Store& operator=(const Store&) = delete;

        [[nodiscard]] SubscriptionToken subscribe(FieldId field_id, Subscriber callback);
        void enqueue_dirty(FieldId field_id);
        [[nodiscard]] bool drain_dirty_into_frame();
        [[nodiscard]] bool has_dirty() const;

        void begin_batch();
        void end_batch();

        static void set_wake_callback(WakeCallback callback);

    private:
        friend class SubscriptionToken;

        struct SubscriberEntry {
            std::uint64_t id = 0;
            Subscriber callback;
        };

        void unsubscribe(FieldId field_id, std::uint64_t subscription_id);
        void enqueue_dirty_now(FieldId field_id);
        static void wake_main_thread();

        mutable std::mutex dirty_mutex_;
        std::vector<FieldId> dirty_fields_;

        std::mutex subscribers_mutex_;
        std::unordered_map<FieldId, std::vector<SubscriberEntry>> subscribers_;
        std::atomic_uint64_t next_subscription_id_{1};
        std::atomic_bool draining_{false};
    };

    class LFS_CORE_API BatchUpdate {
    public:
        explicit BatchUpdate(Store& store);
        ~BatchUpdate();

        BatchUpdate(const BatchUpdate&) = delete;
        BatchUpdate& operator=(const BatchUpdate&) = delete;

    private:
        Store* store_ = nullptr;
    };

} // namespace lfs::core::reactive
