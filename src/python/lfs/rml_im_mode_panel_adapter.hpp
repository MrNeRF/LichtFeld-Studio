/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "rml_im_mode_layout.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <optional>
#include <string>

namespace nb = nanobind;

namespace lfs::vis::gui {

    class RmlImModePanelAdapter : public IPanel {
    public:
        RmlImModePanelAdapter(void* manager, nb::object panel_instance, bool has_poll,
                              const std::string& rml_path = "rmlui/im_mode_panel.rml");
        ~RmlImModePanelAdapter() override;

        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;
        bool supportsDirectDraw() const override { return true; }
        void preloadDirect(float w, float h, const PanelDrawContext& ctx,
                           float clip_y_min, float clip_y_max,
                           const PanelInputState* input) override;
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx) override;
        float getDirectDrawHeight() const override;
        void setInputClipY(float y_min, float y_max) override;
        void setInput(const PanelInputState* input) override;
        void setForcedHeight(float h) override;
        bool needsAnimationFrame() const override;

    private:
        void ensureHost();
        void drawLayout(const PanelDrawContext* ctx);

        void* host_ = nullptr;
        void* manager_;
        std::string rml_path_;
        nb::object panel_instance_;
        bool has_poll_;
        lfs::python::RmlImModeLayout layout_;
        uint64_t last_layout_frame_ = 0;
        std::optional<PanelInputState> current_input_;
        float prev_mouse_x_ = 0.0f;
        float prev_mouse_y_ = 0.0f;
        bool have_prev_mouse_ = false;
        bool have_left_click_time_ = false;
        std::chrono::steady_clock::time_point last_left_click_at_{};
    };

} // namespace lfs::vis::gui
