/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "rendering/gl_resources.hpp"

struct ImFont;

namespace lfs::vis::gui {

    class StartupOverlay {
    public:
        void loadTextures();
        void destroyTextures();
        void render(const ViewportLayout& viewport, ImFont* font_small, bool drag_hovering);
        void dismiss() { visible_ = false; }
        [[nodiscard]] bool isVisible() const { return visible_; }

    private:
        bool visible_ = true;

        rendering::Texture logo_light_texture_;
        rendering::Texture logo_dark_texture_;
        rendering::Texture core11_light_texture_;
        rendering::Texture core11_dark_texture_;
        int logo_width_ = 0, logo_height_ = 0;
        int core11_width_ = 0, core11_height_ = 0;
    };

} // namespace lfs::vis::gui
