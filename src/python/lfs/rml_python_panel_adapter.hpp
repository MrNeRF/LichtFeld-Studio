/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "rml_im_mode_layout.hpp"

#include <chrono>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <optional>
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
                              const std::string& style = {}, bool has_poll = false,
                              int height_mode = 0, bool has_draw = false);
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
        enum class LifecycleState : uint8_t {
            AwaitingModelBind,
            ModelBound,
            Mounted,
        };

        bool ensureHost();
        void cachePythonCapabilities();
        void bindModelIfNeeded();
        Rml::ElementDocument* ensureDocumentInitialized();
        bool reloadDocumentForLanguage(const std::string& language);
        void callOnUnload(Rml::ElementDocument* doc);
        void callOnLoad(Rml::ElementDocument* doc);
        bool isModelBound() const;
        bool isMounted() const;
        void setLifecycleState(LifecycleState next_state);
        void resetLifecycle();
        void syncDirectLayout(float w, float h);
        void drawImmediateLayout(Rml::ElementDocument* doc, const PanelDrawContext* ctx);
        Rml::ElementDocument* prepareForRender(const PanelDrawContext* ctx);
        std::chrono::milliseconds updateInterval() const;

        void* host_ = nullptr;
        void* manager_;
        std::string context_name_;
        std::string rml_path_;
        std::string style_;
        nb::object panel_instance_;
        LifecycleState lifecycle_state_ = LifecycleState::AwaitingModelBind;
        bool has_bind_model_ = false;
        bool bind_model_checked_ = false;
        bool has_poll_ = false;
        bool has_draw_ = false;
        int height_mode_ = 0;
        bool foreground_ = false;
        uint64_t last_scene_gen_ = 0;
        uint64_t last_prepare_frame_ = 0;
        bool content_dirty_ = false;
        bool has_update_interval_ = false;
        int update_interval_ms_ = 100;
        std::chrono::steady_clock::time_point next_update_at_{};
        std::string last_language_;
        lfs::python::RmlImModeLayout layout_;
        std::optional<PanelInputState> current_input_;
        float prev_mouse_x_ = 0.0f;
        float prev_mouse_y_ = 0.0f;
        bool have_prev_mouse_ = false;
        bool have_left_click_time_ = false;
        std::chrono::steady_clock::time_point last_left_click_at_{};
    };

} // namespace lfs::vis::gui
