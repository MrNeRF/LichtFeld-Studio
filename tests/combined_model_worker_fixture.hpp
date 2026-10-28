/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>

namespace lfs::test::combined_model {

    // Holds the combined-model worker inside its first allocation until released.
    // Declare it after the Scene: its destructor releases a held worker before
    // ~Scene joins it.
    class HeldWorker {
    public:
        ~HeldWorker() { release(); }

        [[nodiscard]] core::SplatTensorAllocator allocator() const {
            return [state = state_](core::TensorShape shape, size_t, core::DataType dtype, std::string_view) {
                std::unique_lock lock(state->mutex);
                state->entered = true;
                state->changed.notify_all();
                state->changed.wait(lock, [&state] { return !state->held; });
                return core::Tensor::empty(std::move(shape), core::Device::CUDA, dtype);
            };
        }

        void hold() {
            std::lock_guard lock(state_->mutex);
            state_->held = true;
            state_->entered = false;
        }

        [[nodiscard]] bool waitEntered() const {
            std::unique_lock lock(state_->mutex);
            return state_->changed.wait_for(lock, std::chrono::seconds(30), [this] { return state_->entered; });
        }

        void release() {
            {
                std::lock_guard lock(state_->mutex);
                state_->held = false;
            }
            state_->changed.notify_all();
        }

    private:
        struct State {
            std::mutex mutex;
            std::condition_variable changed;
            bool held = true;
            bool entered = false;
        };
        std::shared_ptr<State> state_ = std::make_shared<State>();
    };

    struct TimedModel {
        const core::SplatData* model = nullptr;
        bool prompt = false;
        std::chrono::milliseconds elapsed{0};
    };

    // Runs getCombinedModel off the test thread: a call that joins the held worker
    // fails the check after the limit instead of hanging the suite.
    [[nodiscard]] inline TimedModel combinedModelWithin(const core::Scene& scene, HeldWorker& worker) {
        const auto started = std::chrono::steady_clock::now();
        auto call = std::async(std::launch::async, [&scene] { return scene.getCombinedModel(); });
        const bool prompt = call.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        if (!prompt)
            worker.release();
        const auto* model = call.get();
        return {model, prompt,
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started)};
    }

    // Polls like the viewport until no build is pending, including a rebuild that
    // starts after a stale result was dropped.
    [[nodiscard]] inline const core::SplatData* settledCombinedModel(const core::Scene& scene) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        const auto* model = scene.getCombinedModel();
        while (scene.combinedModelBuildPending() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            model = scene.getCombinedModel();
        }
        return model;
    }

} // namespace lfs::test::combined_model
