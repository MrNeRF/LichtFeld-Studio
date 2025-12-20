/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <atomic>
#include <chrono>
#include <concepts>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <print>
#include <source_location>
#include <sstream>
#include <typeindex>
#include <unordered_map>
#include <vector>

#ifdef __GNUG__
#include <cxxabi.h>
#endif

namespace lfs::core::event {
    using HandlerId = size_t;

    // Event concept
    template <typename T>
    concept Event = requires {
        typename T::event_id;
    } && std::is_aggregate_v<T>;

    class Bus {
        template <typename T>
        using Handler = std::function<void(const T&)>;

        struct BaseChannel {
            virtual ~BaseChannel() = default;
            virtual std::string_view type_name() const = 0;
            virtual size_t handler_count() const = 0;
        };

        template <Event E>
        struct Channel : BaseChannel {
            std::vector<std::pair<HandlerId, Handler<E>>> handlers;
            mutable std::mutex mutex;

            std::string_view type_name() const override { return typeid(E).name(); }
            size_t handler_count() const override {
                std::lock_guard lock(mutex);
                return handlers.size();
            }
        };

    public:
        // Debug settings
        struct DebugConfig {
            bool enabled = false;
            bool log_emit = true;
            bool log_subscribe = true;
            bool log_unhandled = true;
            bool show_timestamp = true;
            bool show_location = true;
        };

        // Emit an event
        template <Event E>
        void emit(const E& event, std::source_location loc = std::source_location::current()) {
            if (debug_.enabled && debug_.log_emit) {
                log_emit_event<E>(loc);
            }

            if (auto it = channels_.find(typeid(E)); it != channels_.end()) {
                auto& channel = static_cast<Channel<E>&>(*it->second);

                // Copy only the handlers, not the IDs
                std::vector<Handler<E>> handlers_copy;
                {
                    std::lock_guard lock(channel.mutex);
                    handlers_copy.reserve(channel.handlers.size());
                    for (auto& [id, handler] : channel.handlers) {
                        handlers_copy.push_back(handler);
                    }
                }

                if (debug_.enabled && debug_.log_unhandled && handlers_copy.empty()) {
                    std::println("[Event::Bus] WARNING: No handlers for event: {}",
                                 demangle(typeid(E).name()));
                }

                for (auto& handler : handlers_copy) {
                    handler(event);
                }

                emit_count_++;
            } else if (debug_.enabled && debug_.log_unhandled) {
                std::println("[Event::Bus] WARNING: No channel for event: {}",
                             demangle(typeid(E).name()));
            }
        }

        // Subscribe to events
        template <Event E>
        HandlerId when(Handler<E> handler, std::source_location loc = std::source_location::current()) {
            auto& channel = get_channel<E>();
            std::lock_guard lock(channel.mutex);

            HandlerId id = next_id_++;
            channel.handlers.emplace_back(id, std::move(handler));

            if (debug_.enabled && debug_.log_subscribe) {
                log_subscribe_event<E>(id, loc);
            }
            return id;
        }

        // Unsubscribe
        template <Event E>
        void remove(HandlerId id) {
            if (auto it = channels_.find(typeid(E)); it != channels_.end()) {
                auto& channel = static_cast<Channel<E>&>(*it->second);
                std::lock_guard lock(channel.mutex);
                auto before = channel.handlers.size();
                channel.handlers.erase(
                    std::remove_if(channel.handlers.begin(), channel.handlers.end(),
                                   [id](const auto& pair) { return pair.first == id; }),
                    channel.handlers.end());

                if (debug_.enabled && before != channel.handlers.size()) {
                    std::println("[Event::Bus] Unsubscribed handler {} from {}",
                                 id, demangle(typeid(E).name()));
                }
            }
        }

        // Clear all handlers for an event type
        template <Event E>
        void clear() {
            if (auto it = channels_.find(typeid(E)); it != channels_.end()) {
                auto& channel = static_cast<Channel<E>&>(*it->second);
                std::lock_guard lock(channel.mutex);
                auto count = channel.handlers.size();
                channel.handlers.clear();

                if (debug_.enabled && count > 0) {
                    std::println("[Event::Bus] Cleared {} handlers for {}",
                                 count, demangle(typeid(E).name()));
                }
            }
        }

        // Clear all handlers
        void clear_all() {
            std::lock_guard lock(mutex_);
            if (debug_.enabled) {
                size_t total = 0;
                for (const auto& [type, channel] : channels_) {
                    total += channel->handler_count();
                }
                std::println("[Event::Bus] Clearing {} handlers across {} event types",
                             total, channels_.size());
            }
            channels_.clear();
        }

        // Get subscriber count
        template <Event E>
        size_t subscriber_count() const {
            if (auto it = channels_.find(typeid(E)); it != channels_.end()) {
                return it->second->handler_count();
            }
            return 0;
        }

        // Debug controls
        void set_debug(bool enabled) { debug_.enabled = enabled; }
        DebugConfig& debug_config() { return debug_; }

        // Debug statistics
        size_t total_emits() const { return emit_count_; }
        size_t total_channels() const {
            std::lock_guard lock(mutex_);
            return channels_.size();
        }

        void print_stats() const {
            std::lock_guard lock(mutex_);
            std::println("[Event::Bus] Statistics:");
            std::println("  Total emits: {}", emit_count_.load());
            std::println("  Active channels: {}", channels_.size());
            std::println("  Registered events:");

            for (const auto& [type, channel] : channels_) {
                std::println("    {} - {} handlers",
                             demangle(type.name()),
                             channel->handler_count());
            }
        }

    private:
        template <Event E>
        Channel<E>& get_channel() {
            std::lock_guard lock(mutex_);
            auto [it, inserted] = channels_.try_emplace(
                typeid(E),
                std::make_unique<Channel<E>>());
            return static_cast<Channel<E>&>(*it->second);
        }

        template <Event E>
        void log_emit_event(const std::source_location& loc) {
            std::string msg = std::format("[Event::Bus] EMIT: {}", demangle(typeid(E).name()));

            if (debug_.show_location) {
                msg += std::format(" @ {}:{}", loc.file_name(), loc.line());
            }

            if (debug_.show_timestamp) {
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch()) %
                          1000;

                // Use put_time for formatting
                std::stringstream time_str;
                time_str << std::put_time(std::localtime(&time_t), "%H:%M:%S");
                msg = std::format("[{}.{:03d}] {}", time_str.str(), ms.count(), msg);
            }

            std::println("{}", msg);
        }

        template <Event E>
        void log_subscribe_event(HandlerId id, const std::source_location& loc) {
            std::string msg = std::format("[Event::Bus] SUBSCRIBE: {} (id={})",
                                          demangle(typeid(E).name()), id);

            if (debug_.show_location) {
                msg += std::format(" @ {}:{}", loc.file_name(), loc.line());
            }

            std::println("{}", msg);
        }

        // Simple demangler (platform-specific implementation needed)
        static std::string demangle(const char* name) {
#ifdef __GNUG__
            int status = 0;
            std::unique_ptr<char, void (*)(void*)> res{
                abi::__cxa_demangle(name, nullptr, nullptr, &status),
                std::free};
            return (status == 0) ? res.get() : name;
#else
            return name;
#endif
        }

        mutable std::mutex mutex_;
        std::unordered_map<std::type_index, std::unique_ptr<BaseChannel>> channels_;
        std::atomic<HandlerId> next_id_{1};
        std::atomic<size_t> emit_count_{0};
        DebugConfig debug_;
    };

    // Global event bus singleton
    inline Bus& bus() {
        static Bus instance;
        return instance;
    }

    // Convenience functions
    template <Event E>
    void emit(const E& event) {
        bus().emit(event);
    }

    template <Event E>
    auto when(auto&& handler) {
        return bus().when<E>(std::forward<decltype(handler)>(handler));
    }

    // Debug helper
    inline void enable_debug(bool emit = true, bool subscribe = true, bool unhandled = true) {
        auto& b = bus();
        b.debug_config().enabled = true;
        b.debug_config().log_emit = emit;
        b.debug_config().log_subscribe = subscribe;
        b.debug_config().log_unhandled = unhandled;
    }

} // namespace lfs::core::event