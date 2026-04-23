/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_fbo.hpp"
#include <cstddef>
#include <string>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
    class EventListener;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;

    class StartupOverlay {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void setInput(const PanelInputState* input) { input_ = input; }
        void reloadResources();
        void render(const ViewportLayout& viewport, bool drag_hovering);
        void dismiss() { visible_ = false; }
        [[nodiscard]] bool isVisible() const { return visible_; }
        [[nodiscard]] bool needsAnimationFrame() const { return visible_ && shown_frames_ < 3; }

        static void openURL(const char* url);

    private:
        void populateLanguages();
        void updateTheme();
        void updateLocalizedText();
        bool forwardInput(const PanelInputState& input, float overlay_x, float overlay_y,
                          float overlay_w, float overlay_h);

        bool visible_ = true;
        int shown_frames_ = 0;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        RmlFBO fbo_;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        const PanelInputState* input_ = nullptr;

        Rml::EventListener* link_listener_ = nullptr;
        Rml::EventListener* lang_listener_ = nullptr;
    };

} // namespace lfs::vis::gui
