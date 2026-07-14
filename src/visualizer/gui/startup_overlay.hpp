/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "gui/rmlui/rmlui_manager.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
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
        void dismiss();
        void setPluginLoadState(bool started, bool active, float progress, std::string stage);
        [[nodiscard]] bool isVisible() const { return visible_; }
        [[nodiscard]] bool blocksUnderlayInput() const;
        [[nodiscard]] bool isPluginLoadComplete() const;
        [[nodiscard]] bool needsAnimationFrame() const;

        static void openURL(const char* url);

    private:
        struct InputForwardResult {
            bool escape_consumed = false;
            bool event_forwarded = false;
        };

        void populateLanguages();
        void updateTheme();
        void updateLocalizedText();
        bool updatePluginLoadUI();
        void updateClickHintUI();
        void ensureLanguageDropdownFontsLoaded();
        [[nodiscard]] bool isLanguageSelectOpen() const;
        [[nodiscard]] bool isLanguageSelectHit(float local_x, float local_y) const;
        [[nodiscard]] bool hasInputActivity(const PanelInputState& input) const;
        InputForwardResult forwardInput(const PanelInputState& input, float overlay_x, float overlay_y,
                                        float overlay_w, float overlay_h);

        struct PluginLoadState {
            bool active = false;
            float progress = 0.0f;
            std::string stage;
        };

        bool visible_ = true;
        int shown_frames_ = 0;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::uint64_t last_language_generation_ = 0;
        bool has_language_generation_ = false;
        const PanelInputState* input_ = nullptr;
        CachedVulkanContextRender direct_cache_;
        int width_ = 0;
        int height_ = 0;
        bool content_dirty_ = true;
        bool last_mouse_valid_ = false;
        float last_mouse_x_ = 0.0f;
        float last_mouse_y_ = 0.0f;
        bool language_dropdown_fonts_requested_ = false;
        mutable std::mutex plugin_load_mutex_;
        PluginLoadState plugin_load_state_;
        PluginLoadState applied_plugin_load_state_;
        bool has_applied_plugin_load_state_ = false;
        bool plugin_load_state_started_ = false;
        bool plugin_load_complete_ = true;

        Rml::EventListener* link_listener_ = nullptr;
        Rml::EventListener* lang_listener_ = nullptr;
    };

} // namespace lfs::vis::gui
