/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "control_boundary.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace lfs::training {

    ControlBoundary& ControlBoundary::instance() {
        static ControlBoundary boundary;
        return boundary;
    }

    std::size_t ControlBoundary::register_callback(ControlHook hook, Callback cb) {
        if (!cb) {
            return 0;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        const auto id = next_id_.fetch_add(1, std::memory_order_relaxed);
        callbacks_[hook].push_back(Registration{.id = id, .cb = std::move(cb)});
        return id;
    }

    void ControlBoundary::unregister_callback(ControlHook hook, std::size_t handle) {
        if (handle == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = callbacks_.find(hook);
        if (it == callbacks_.end()) {
            return;
        }

        auto& vec = it->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [&](const Registration& r) { return r.id == handle; }),
                  vec.end());
        pending_callbacks_.erase(
            std::remove_if(pending_callbacks_.begin(), pending_callbacks_.end(),
                           [&](const PendingCallback& pending) {
                               return pending.registration_id == handle;
                           }),
            pending_callbacks_.end());

        if (vec.empty()) {
            callbacks_.erase(it);
        }
    }

    void ControlBoundary::notify(ControlHook hook, const HookContext& ctx) {
        std::vector<Registration> local;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = callbacks_.find(hook);
            if (it != callbacks_.end()) {
                local = it->second;
            }
        }

        if (local.empty()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_callbacks_.reserve(pending_callbacks_.size() + local.size());
            for (const auto& reg : local) {
                if (reg.cb) {
                    pending_callbacks_.push_back(PendingCallback{reg.id, reg.cb, ctx});
                }
            }
        }
    }

    void ControlBoundary::drain_callbacks() {
        std::vector<PendingCallback> local;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_callbacks_.empty()) {
                return;
            }
            local.swap(pending_callbacks_);
        }

        for (const auto& pending : local) {
            try {
                pending.cb(pending.ctx);
            } catch (...) {
                spdlog::error("Hook callback threw");
            }
        }
    }

    void ControlBoundary::clear_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_.clear();
        pending_callbacks_.clear();
    }

} // namespace lfs::training
