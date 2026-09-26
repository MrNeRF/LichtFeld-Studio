/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <concepts>
#include <functional>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifdef LFS_EVENT_BRIDGE_EXPORTS
#define LFS_BRIDGE_API __declspec(dllexport)
#else
#define LFS_BRIDGE_API __declspec(dllimport)
#endif
#else
#define LFS_BRIDGE_API __attribute__((visibility("default")))
#endif

namespace lfs::event {

    using HandlerId = size_t;

    template <typename T>
    concept Event = requires {
        typename T::event_id;
    } && std::is_aggregate_v<T>;

    class LFS_BRIDGE_API EventBridge {
    public:
        static EventBridge& instance();

        using Handler = std::function<void(const void*)>;

        HandlerId subscribe(std::type_index type, Handler handler);
        void unsubscribe(std::type_index type, HandlerId id);
        void emit(std::type_index type, const void* data);
        size_t handler_count(std::type_index type) const;
        void clear_all();

    private:
        EventBridge() = default;
        EventBridge(const EventBridge&) = delete;
        EventBridge& operator=(const EventBridge&) = delete;

        mutable std::mutex mutex_;
        std::unordered_map<std::string_view, std::vector<std::pair<HandlerId, Handler>>> handlers_;
        std::atomic<HandlerId> next_id_{1};
    };

    template <typename E>
    HandlerId when(std::function<void(const E&)> handler) {
        return EventBridge::instance().subscribe(
            typeid(E), [h = std::move(handler)](const void* data) { h(*static_cast<const E*>(data)); });
    }

    template <typename E>
    void emit(const E& event) {
        EventBridge::instance().emit(typeid(E), &event);
    }

    template <typename E>
    size_t subscriber_count() {
        return EventBridge::instance().handler_count(typeid(E));
    }

} // namespace lfs::event
