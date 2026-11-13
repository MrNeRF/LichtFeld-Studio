/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "event_bridge.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    class Trainer;

    enum class ControlHook {
        TrainingStart,
        IterationStart,
        PreOptimizerStep,
        PostStep,
        TrainingEnd,
        SaveStart,
        SaveEnd
    };

    // What kind of save the trainer writes.
    enum class SaveType { Save,
                          Autosave,
                          Standalone,
                          Final,
                          Stopped,
                          Emergency };

    // One project file the trainer writes, or refuses to. SaveStart fills snapshot_id, path,
    // type and the trainer's state; SaveEnd also how the write ended. Both carry the same
    // snapshot_id, the one the save's checkpoint is stored under in the file.
    struct SaveRecord {
        std::string snapshot_id;
        std::string path;
        SaveType type = SaveType::Save;
        // The trainer's state as the save began, for an end heard once the trainer is gone.
        bool is_paused = false;
        int iterations = 0;
        bool ok = false;
        std::string error;
        std::uint64_t generation = 0;
        std::uint64_t bytes = 0;
        double seconds = 0.0;
    };

    struct ControlHookHash {
        std::size_t operator()(ControlHook hook) const noexcept {
            return static_cast<std::size_t>(hook);
        }
    };

    enum class TrainingOutcome { Unspecified,
                                 Completed,
                                 Stopped,
                                 Failed };

    struct HookContext {
        int iteration = 0;
        float loss = 0.0f;
        std::size_t num_gaussians = 0;
        bool is_refining = false;
        Trainer* trainer = nullptr;
        TrainingOutcome terminal_outcome = TrainingOutcome::Unspecified;
        // SaveStart and SaveEnd only: the save, whose iteration is `iteration`.
        std::shared_ptr<const SaveRecord> save;
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
