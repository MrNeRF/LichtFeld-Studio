/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"

#include <chrono>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <string>

namespace nb = nanobind;

namespace Rml {
    class ElementDocument;
}

namespace lfs::vis::gui {

    class RmlPythonPanelAdapter : public IPanel {
    public:
        RmlPythonPanelAdapter(void* manager, nb::object panel_instance,
                              const std::string& context_name, const std::string& rml_path,
                              bool has_poll = false, int height_mode = 0);
        ~RmlPythonPanelAdapter() override;

        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;
        void preload(const PanelDrawContext& ctx) override;
        void preloadDirect(float w, float h, const PanelDrawContext& ctx,
                           float clip_y_min, float clip_y_max,
                           const PanelInputState* input) override;
        bool supportsDirectDraw() const override { return true; }
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx) override;
        float getDirectDrawHeight() const override;
        void setInputClipY(float y_min, float y_max) override;
        void setInput(const PanelInputState* input) override;
        void setForcedHeight(float h) override;
        bool wantsKeyboard() const override;
        bool needsAnimationFrame() const override;
        bool wantsExternalFloatingShadow() const override { return !foreground_; }
        void setForeground(bool fg);

    private:
        bool ensureHost();
        void cachePythonCapabilities();
        void bindModelIfNeeded();
        Rml::ElementDocument* ensureDocumentInitialized();
        void syncDirectLayout(float w, float h);
        Rml::ElementDocument* prepareForRender(const PanelDrawContext* ctx);
        std::chrono::milliseconds updateInterval() const;

        void* host_ = nullptr;
        void* manager_;
        std::string context_name_;
        std::string rml_path_;
        nb::object panel_instance_;
        bool loaded_ = false;
        bool model_bound_ = false;
        bool has_bind_model_ = false;
        bool bind_model_checked_ = false;
        bool has_draw_imgui_ = false;
        bool draw_imgui_checked_ = false;
        bool has_poll_ = false;
        int height_mode_ = 0;
        bool foreground_ = false;
        uint64_t last_scene_gen_ = 0;
        uint64_t last_prepare_frame_ = 0;
        bool content_dirty_ = false;
        bool has_update_interval_ = false;
        int update_interval_ms_ = 100;
        std::chrono::steady_clock::time_point next_update_at_{};
    };

} // namespace lfs::vis::gui
