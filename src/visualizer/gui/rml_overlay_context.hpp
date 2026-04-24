/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_fbo.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include <core/export.hpp>
#include <cstddef>
#include <string>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;

    class LFS_VIS_API RmlOverlayContext {
    public:
        RmlOverlayContext(RmlUIManager* mgr, const std::string& name, const std::string& rml_path);
        ~RmlOverlayContext();

        RmlOverlayContext(const RmlOverlayContext&) = delete;
        RmlOverlayContext& operator=(const RmlOverlayContext&) = delete;

        void resize(int w, int h);
        void update();
        void render(float x, float y, float w, float h, int screen_w, int screen_h);
        void forwardMouseInput(const PanelInputState& input, float overlay_x, float overlay_y);

        [[nodiscard]] Rml::ElementDocument* document() { return doc_; }
        [[nodiscard]] Rml::Element* getElementById(const std::string& id);

        void showElement(const std::string& id);
        void hideElement(const std::string& id);

        void showContextMenu(const std::string& element_id, float x, float y, const std::string& inner_rml);
        void hideContextMenu(const std::string& element_id);

        void destroyGLResources();

    private:
        void initContext();
        void syncTheme();

        RmlUIManager* mgr_;
        std::string context_name_;
        std::string rml_path_;

        Rml::Context* ctx_ = nullptr;
        Rml::ElementDocument* doc_ = nullptr;
        RmlFBO fbo_;

        std::string base_rcss_;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;

        int width_ = 0;
        int height_ = 0;
    };

} // namespace lfs::vis::gui
