/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "input/view_targets.hpp"
#include <cstdint>

namespace lfs::vis {
    class TestViewTargets final : public ViewTargets {
    public:
        explicit TestViewTargets(Viewport& camera) : camera_(camera) {}
        ViewTarget activeView() override { return {1, &camera_, {0, 0}, glm::vec2(camera_.windowSize)}; }
        ViewTarget viewAt(float x, float y) override {
            const auto view = activeView();
            return view.contains(x, y) ? view : ViewTarget{};
        }
        ViewTarget findView(ViewId id) override { return id == 1 ? activeView() : ViewTarget{}; }
        ViewId viewId(const Viewport&) const override { return 1; }
        std::uint64_t viewEpoch() const override { return 1; }
        void activateView(ViewId) override {}
        bool runViewCommand(ViewId, std::string_view) override { return false; }

    private:
        Viewport& camera_;
    };
} // namespace lfs::vis
