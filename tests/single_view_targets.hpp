/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/view_targets.hpp"
#include "internal/viewport.hpp"

namespace lfs::vis {

    class SingleViewTargets : public ViewTargets {
    public:
        explicit SingleViewTargets(Viewport& viewport) : viewport_(&viewport) {}

        ViewTarget activeView() override { return target(); }
        ViewTarget viewAt(float x, float y) override {
            auto t = target();
            return t.contains(x, y) ? t : ViewTarget{};
        }
        ViewTarget findView(const ViewId id) override {
            return id == 1 ? target() : ViewTarget{};
        }
        void activateView(ViewId) override {}
        bool runViewCommand(ViewId, std::string_view) override { return false; }

    private:
        ViewTarget target() const {
            return ViewTarget{
                .id = 1,
                .viewport = viewport_,
                .pos = {0.0f, 0.0f},
                .size = {static_cast<float>(viewport_->windowSize.x),
                         static_cast<float>(viewport_->windowSize.y)},
            };
        }

        Viewport* viewport_;
    };

} // namespace lfs::vis
