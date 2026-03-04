/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"

#include <cstdint>
#include <nanobind/nanobind.h>
#include <string>

namespace nb = nanobind;

namespace lfs::vis::gui {

    class RmlPythonPanelAdapter : public IPanel {
    public:
        RmlPythonPanelAdapter(void* manager, nb::object panel_instance,
                              const std::string& context_name, const std::string& rml_path,
                              int height_mode = 0);
        ~RmlPythonPanelAdapter() override;

        void draw(const PanelDrawContext& ctx) override;
        bool supportsDirectDraw() const override { return true; }
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx) override;
        float getDirectDrawHeight() const override;
        bool hasImguiOverlay() const override;
        void drawImguiOverlay(const PanelDrawContext& ctx) override;
        void setInputClipY(float y_min, float y_max) override;
        void setInput(const PanelInputState* input) override;
        bool wantsKeyboard() const override;
        void setForeground(bool fg);

    private:
        void* host_ = nullptr;
        void* manager_;
        std::string context_name_;
        std::string rml_path_;
        nb::object panel_instance_;
        bool loaded_ = false;
        bool model_bound_ = false;
        bool has_bind_model_ = false;
        bool has_draw_imgui_ = false;
        bool draw_imgui_checked_ = false;
        int height_mode_ = 0;
        bool foreground_ = false;
        uint64_t last_scene_gen_ = 0;
        bool content_dirty_ = false;
    };

} // namespace lfs::vis::gui
