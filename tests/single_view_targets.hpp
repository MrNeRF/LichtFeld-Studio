/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/view_targets.hpp"
#include "internal/viewport.hpp"

#include <optional>

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

        // Camera drag tests can use synthetic pointer travel well beyond the
        // drawable size while keeping the viewport's actual render dimensions.
        void setInputBounds(const glm::vec2 pos, const glm::vec2 size) {
            input_pos_ = pos;
            input_size_ = size;
        }

    private:
        ViewTarget target() const {
            return ViewTarget{
                .id = 1,
                .viewport = viewport_,
                .pos = input_pos_,
                .size = input_size_.value_or(glm::vec2{static_cast<float>(viewport_->windowSize.x),
                                                       static_cast<float>(viewport_->windowSize.y)}),
            };
        }

        Viewport* viewport_;
        glm::vec2 input_pos_{0.0f};
        std::optional<glm::vec2> input_size_;
    };

} // namespace lfs::vis
