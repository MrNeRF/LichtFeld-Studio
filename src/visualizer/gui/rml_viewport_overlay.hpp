/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <cstddef>
#include <glm/glm.hpp>
#include <string>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;
    struct PanelInputState;

    class RmlViewportOverlay {
    public:
        struct GTMetricsOverlayState {
            bool visible = false;
            float x = 16.0f;
            float y = 16.0f;
            std::string psnr_text;
            bool show_ssim = false;
            std::string ssim_text;
        };

        void init(RmlUIManager* mgr);
        void shutdown();
        void setViewportBounds(glm::vec2 pos, glm::vec2 size, glm::vec2 screen_origin);
        void setToolbarPanels(float primary_x, float primary_width,
                              bool show_secondary = false,
                              float secondary_x = 0.0f,
                              float secondary_width = 0.0f);
        void setGTMetricsOverlay(GTMetricsOverlayState state);
        void reloadResources();
        void render();
        void processInput(const PanelInputState& input);
        bool wantsInput() const { return wants_input_; }
        [[nodiscard]] bool blocksPointer(double screen_x, double screen_y) const;

    private:
        bool updateTheme();
        void cacheBodyTemplate();
        void ensureBodyDataModelBound(Rml::Element* body);
        bool shouldRunDocumentHooks(bool force) const;
        void updateToolbarRoots();
        void applyGTMetricsOverlay();

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        glm::vec2 vp_pos_{0, 0};
        glm::vec2 vp_size_{0, 0};
        glm::vec2 screen_origin_{0, 0};
        float primary_toolbar_x_ = 0.0f;
        float primary_toolbar_width_ = 0.0f;
        bool show_secondary_toolbar_ = false;
        float secondary_toolbar_x_ = 0.0f;
        float secondary_toolbar_width_ = 0.0f;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;
        std::string body_template_rml_;
        bool wants_input_ = false;
        bool doc_registered_ = false;
        bool render_needed_ = true;
        bool animation_active_ = false;
        bool mouse_pos_valid_ = false;
        int last_mouse_x_ = 0;
        int last_mouse_y_ = 0;
        int last_render_w_ = 0;
        int last_render_h_ = 0;
        GTMetricsOverlayState gt_metrics_overlay_;
        std::chrono::steady_clock::time_point last_document_hook_run_{};
        static constexpr auto kDocumentHookPollInterval = std::chrono::milliseconds(100);
    };

} // namespace lfs::vis::gui
