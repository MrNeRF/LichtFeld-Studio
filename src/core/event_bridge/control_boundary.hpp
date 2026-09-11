/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "event_bridge.hpp"

#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    class Trainer;

    enum class ControlHook {
        TrainingStart,
        IterationStart,
        PreOptimizerStep,
        PostStep,
        TrainingEnd
    };

    struct ControlHookHash {
        std::size_t operator()(ControlHook hook) const noexcept {
            return static_cast<std::size_t>(hook);
        }
    };

    struct HookContext {
        int iteration = 0;
        float loss = 0.0f;
        std::size_t num_gaussians = 0;
        bool is_refining = false;
        Trainer* trainer = nullptr;
    };

    class LFS_BRIDGE_API ControlBoundary {
    public:
        using Callback = std::function<void(const HookContext&)>;

        static ControlBoundary& instance();

        std::size_t register_callback(ControlHook hook, Callback cb);
        void unregister_callback(ControlHook hook, std::size_t handle);
        void notify(ControlHook hook, const HookContext& ctx);
        void drain_callbacks();
        void clear_all();

    private:
        ControlBoundary() = default;
        ControlBoundary(const ControlBoundary&) = delete;
        ControlBoundary& operator=(const ControlBoundary&) = delete;

        struct Registration {
            std::size_t id;
            Callback cb;
        };

        struct PendingCallback {
            std::size_t registration_id;
            Callback cb;
            HookContext ctx;
        };

        std::mutex mutex_;
        std::unordered_map<ControlHook, std::vector<Registration>, ControlHookHash> callbacks_;
        std::vector<PendingCallback> pending_callbacks_;
        std::atomic<std::size_t> next_id_{1};
    };

} // namespace lfs::training
