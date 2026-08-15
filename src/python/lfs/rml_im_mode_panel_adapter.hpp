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
#include <string_view>

namespace nb = nanobind;

namespace lfs::vis::gui {

    class RmlImModePanelAdapter : public IPanel {
    public:
        RmlImModePanelAdapter(void* manager, nb::object panel_instance, bool has_poll,
                              const std::string& rml_path = "rmlui/im_mode_panel.rml");
        ~RmlImModePanelAdapter() override;

        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;
        PanelRenderCapabilities renderCapabilities() const override {
            return {.direct = true};
        }
        PanelDirectRenderResult renderDirect(const PanelDirectRenderRequest& request,
                                             const PanelDrawContext& ctx) override;
        bool needsAnimationFrame() const override;
        std::optional<double> nextScheduledAnimationDelay() const override;
        void reloadRmlResources() override;
        [[nodiscard]] std::string captureChromeJson() const override;
        void applyChromeJson(std::string_view json) override;

    private:
        void ensureHost();
        void drawLayout(const PanelDrawContext* ctx);
        void preloadDirect(float w, float h, const PanelDrawContext& ctx,
                           float clip_y_min, float clip_y_max,
                           const PanelInputState* input);
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx);
        bool drawDirectCached(float x, float y, float w, float h,
                              const PanelDrawContext& ctx);
        float getDirectDrawHeight() const;
        void setInput(const PanelInputState* input);
        void setInputClipY(float y_min, float y_max);
        void setForcedHeight(float h);
        void setPanelSpace(PanelSpace space);

        void* host_ = nullptr;
        void* manager_;
        std::string rml_path_;
        nb::object panel_instance_;
        bool has_poll_;
        bool floating_ = false;
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
