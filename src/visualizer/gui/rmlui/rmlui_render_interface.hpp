/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rmlui_gl3_backend.hpp"

namespace lfs::vis::gui {

    class RmlRenderInterface final : public RenderInterface_GL3 {
    public:
        RmlRenderInterface();
        ~RmlRenderInterface() override;

        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String& source) override;
    };

} // namespace lfs::vis::gui
