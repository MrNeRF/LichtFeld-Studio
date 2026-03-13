/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rmlui_gl3_backend.hpp"

#include <memory>

namespace lfs::core {
    class Scene;
}

namespace lfs::vis::gui {

    struct PreviewTextureCache;

    class RmlRenderInterface final : public RenderInterface_GL3 {
    public:
        RmlRenderInterface();
        ~RmlRenderInterface() override;

        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String& source) override;
        void ReleaseTexture(Rml::TextureHandle texture_handle) override;

        void set_scene(lfs::core::Scene* scene);

    private:
        Rml::TextureHandle load_preview_texture(Rml::Vector2i& dimensions, const Rml::String& source);

        lfs::core::Scene* scene_ = nullptr;
        std::unique_ptr<PreviewTextureCache> preview_cache_;
    };

} // namespace lfs::vis::gui
