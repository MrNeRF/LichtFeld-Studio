/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "scene_renderer.hpp"
#include <chrono>
#include <functional>
namespace lfs::vis {
    // Backend capability for a trainer sharing its allocation arena with a
    // renderer. Native tensor readers do not participate in this protocol.
    class SceneTrainingInterop {
    public:
        virtual ~SceneTrainingInterop() = default;
        virtual bool hasLiveTrainerReleaseFence() const { return false; }
        virtual void* renderCompleteTimeline() const { return nullptr; }
        virtual uint64_t renderCompleteValue() const { return 0; }
        virtual std::expected<void, std::string> ensureHandshakeReady() { return {}; }
        virtual std::expected<void, std::string> ensureTrainingSharedScratchReady(size_t, glm::ivec2) { return {}; }
        virtual void releaseScratchOnIdle(bool, bool = false) {}
        virtual void requestArenaHandoff() {}
        virtual void cancelArenaHandoff() {}
        virtual bool pollArenaHandoff() { return true; }
        virtual bool waitForArenaHandoff(std::chrono::milliseconds) { return true; }
        virtual void setLiveSubmitCallback(std::function<void(uint64_t)>) {}
    };
    // Preserve the idle/no-trainer behavior without teaching the scene contract
    // the shared-arena protocol or requiring native renderers to implement it.
    inline SceneTrainingInterop& rendererTrainingInterop(SceneRenderer& renderer) {
        if (auto* capability = renderer.trainingInterop())
            return *capability;
        static SceneTrainingInterop idle;
        return idle;
    }
} // namespace lfs::vis
