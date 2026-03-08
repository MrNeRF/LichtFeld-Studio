/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/gpu_memory_query.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_fbo.hpp"
#include <chrono>
#include <cstddef>
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

    class RmlStatusBar {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void draw(const PanelDrawContext& ctx);

    private:
        void cacheElements();
        void updateContent(const PanelDrawContext& ctx);
        void updateTheme();
        std::string generateThemeRCSS(const lfs::vis::Theme& t) const;

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;

        RmlFBO fbo_;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;

        struct SpeedOverlayState {
            float wasd_speed = 0.0f;
            float zoom_speed = 0.0f;
            std::chrono::steady_clock::time_point wasd_start;
            std::chrono::steady_clock::time_point zoom_start;
            bool wasd_visible = false;
            bool zoom_visible = false;
            static constexpr auto DURATION = std::chrono::milliseconds(3000);
            static constexpr float FADE_MS = 500.0f;

            void showWasd(float speed);
            void showZoom(float speed);
            std::pair<float, float> getWasd() const;
            std::pair<float, float> getZoom() const;
        };

        SpeedOverlayState speed_state_;
        bool speed_events_initialized_ = false;

        // Cached element pointers
        Rml::Element* mode_text_ = nullptr;
        Rml::Element* training_section_ = nullptr;
        Rml::Element* progress_fill_ = nullptr;
        Rml::Element* progress_text_ = nullptr;
        Rml::Element* step_label_ = nullptr;
        Rml::Element* step_value_ = nullptr;
        Rml::Element* loss_label_ = nullptr;
        Rml::Element* loss_value_ = nullptr;
        Rml::Element* gaussians_label_ = nullptr;
        Rml::Element* gaussians_value_ = nullptr;
        Rml::Element* time_value_ = nullptr;
        Rml::Element* eta_label_ = nullptr;
        Rml::Element* eta_value_ = nullptr;
        Rml::Element* splat_section_ = nullptr;
        Rml::Element* splat_text_ = nullptr;
        Rml::Element* split_section_ = nullptr;
        Rml::Element* split_mode_ = nullptr;
        Rml::Element* split_detail_ = nullptr;
        Rml::Element* wasd_section_ = nullptr;
        Rml::Element* wasd_text_ = nullptr;
        Rml::Element* zoom_section_ = nullptr;
        Rml::Element* zoom_text_ = nullptr;
        Rml::Element* gpu_icon_ = nullptr;
        Rml::Element* lfs_mem_ = nullptr;
        Rml::Element* gpu_mem_ = nullptr;
        Rml::Element* fps_value_ = nullptr;
        Rml::Element* fps_label_ = nullptr;
        Rml::Element* git_commit_ = nullptr;

        // Cached last-frame values for dirty checking
        struct CachedState {
            std::string mode_rml;
            std::string mode_color;
            bool show_training = false;
            int current_iter = -1;
            int total_iter = -1;
            float loss = -1.0f;
            int num_splats = -1;
            int max_gaussians = -1;
            float elapsed = -1.0f;
            float eta = -1.0f;
            bool show_splats = false;
            std::string splat_rml;
            bool split_enabled = false;
            std::string split_mode_rml;
            std::string split_detail_rml;
            std::string wasd_rml;
            float wasd_alpha = -1.0f;
            std::string zoom_rml;
            float zoom_alpha = -1.0f;
            std::string lfs_mem_rml;
            std::string gpu_mem_rml;
            std::string gpu_mem_color;
            std::string fps_rml;
            std::string fps_color;
            bool git_set = false;
        };

        CachedState cache_;
    };

} // namespace lfs::vis::gui
