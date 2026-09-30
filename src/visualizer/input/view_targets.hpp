/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "internal/viewport.hpp"
#include "rendering/view_source.hpp"

#include <glm/glm.hpp>
#include <string_view>

namespace lfs::vis {

    // A 3D view as input and tools see it: its camera and where it is on
    // screen (window pixels).
    struct ViewTarget {
        ViewId id = kNoView;
        Viewport* viewport = nullptr;
        glm::vec2 pos{0.0f};
        glm::vec2 size{0.0f};

        [[nodiscard]] bool valid() const { return viewport != nullptr; }
        [[nodiscard]] bool contains(const float x, const float y) const {
            return x >= pos.x && x < pos.x + size.x && y >= pos.y && y < pos.y + size.y;
        }
    };

    // Resolves which 3D view an interaction targets.
    class ViewTargets {
    public:
        virtual ~ViewTargets() = default;

        // The view the user last worked in. Always valid.
        [[nodiscard]] virtual ViewTarget activeView() = 0;
        // The view whose content is under a window point, if any.
        [[nodiscard]] virtual ViewTarget viewAt(float x, float y) = 0;
        [[nodiscard]] virtual ViewTarget findView(ViewId id) = 0;
        [[nodiscard]] virtual ViewId viewId(const Viewport& viewport) const = 0;
        // Makes a view the active one (a press in it does this).
        virtual void activateView(ViewId id) = 0;
        // Runs a view command (see screen::applyViewCommand, plus frame_all,
        // frame_selected, area:quad and area:side) on one view.
        virtual bool runViewCommand(ViewId id, std::string_view command) = 0;
    };

} // namespace lfs::vis
