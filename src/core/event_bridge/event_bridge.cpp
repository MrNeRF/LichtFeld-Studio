/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "event_bridge.hpp"
#include <algorithm>

namespace lfs::event {

    EventBridge& EventBridge::instance() {
        static EventBridge bridge;
        return bridge;
    }

    HandlerId EventBridge::subscribe(std::type_index type, Handler handler) {
        std::lock_guard lock(mutex_);
        const HandlerId id = next_id_++;
        handlers_[type.name()].emplace_back(id, std::move(handler));
        return id;
    }

    void EventBridge::unsubscribe(std::type_index type, const HandlerId id) {
        std::lock_guard lock(mutex_);
        auto it = handlers_.find(type.name());
        if (it == handlers_.end()) {
            return;
        }
        auto& vec = it->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [id](const auto& p) { return p.first == id; }),
                  vec.end());
    }

    void EventBridge::emit(std::type_index type, const void* data) {
        std::vector<Handler> handlers_copy;
        {
            std::lock_guard lock(mutex_);
            auto it = handlers_.find(type.name());
            if (it != handlers_.end()) {
                handlers_copy.reserve(it->second.size());
                for (const auto& [_, h] : it->second) {
                    handlers_copy.push_back(h);
                }
            }
        }
        for (const auto& h : handlers_copy) {
            h(data);
        }
    }

    size_t EventBridge::handler_count(std::type_index type) const {
        std::lock_guard lock(mutex_);
        auto it = handlers_.find(type.name());
        return it != handlers_.end() ? it->second.size() : 0;
    }

    void EventBridge::clear_all() {
        std::lock_guard lock(mutex_);
        handlers_.clear();
    }

} // namespace lfs::event
