/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "view_context.hpp"
#include "python/python_runtime.hpp"

namespace lfs::vis {

    namespace py = lfs::python;

    struct ViewContextState {
        GetViewCallback view_callback;
        GetViewForPanelCallback view_for_panel_callback;
        GetViewportRenderCallback viewport_render_callback;
        CaptureViewportRenderCallback capture_viewport_render_callback;
        GetRenderSettingsCallback get_render_settings_callback;
        SetRenderSettingsCallback set_render_settings_callback;
        SetViewCallback set_view_callback;
        SetViewForPanelCallback set_view_for_panel_callback;
        SetFovCallback set_fov_callback;
    };

    static ViewContextState& state() {
        auto* p = static_cast<ViewContextState*>(py::get_view_context_state());
        if (!p) {
            p = new ViewContextState();
            py::set_view_context_state(p);
        }
        return *p;
    }

    void set_view_callback(GetViewCallback callback) {
        state().view_callback = std::move(callback);
    }

    void set_view_for_panel_callback(GetViewForPanelCallback callback) {
        state().view_for_panel_callback = std::move(callback);
    }

    void set_viewport_render_callback(GetViewportRenderCallback callback) {
        state().viewport_render_callback = std::move(callback);
    }

    void set_capture_viewport_render_callback(CaptureViewportRenderCallback callback) {
        state().capture_viewport_render_callback = std::move(callback);
    }

    std::optional<ViewInfo> get_current_view_info() {
        const auto& s = state();
        if (!s.view_callback)
            return std::nullopt;
        return s.view_callback();
    }

    std::optional<ViewInfo> get_view_info_for_panel(const SplitViewPanelId panel) {
        const auto& s = state();
        if (s.view_for_panel_callback)
            return s.view_for_panel_callback(panel);
        if (s.view_callback)
            return s.view_callback();
        return std::nullopt;
    }

    std::optional<ViewportRender> get_viewport_render() {
        const auto& s = state();
        if (!s.viewport_render_callback)
            return std::nullopt;
        return s.viewport_render_callback();
    }

    std::optional<ViewportRender> capture_viewport_render() {
        const auto& s = state();
        if (!s.capture_viewport_render_callback)
            return std::nullopt;
        return s.capture_viewport_render_callback();
    }

    void set_set_view_callback(SetViewCallback callback) {
        state().set_view_callback = std::move(callback);
    }

    void set_set_view_for_panel_callback(SetViewForPanelCallback callback) {
        state().set_view_for_panel_callback = std::move(callback);
    }

    void set_set_fov_callback(SetFovCallback callback) {
        state().set_fov_callback = std::move(callback);
    }

    void apply_set_view(const SetViewParams& params) {
        const auto& s = state();
        if (s.set_view_callback) {
            s.set_view_callback(params);
        }
    }

    void apply_set_view_for_panel(const SplitViewPanelId panel, const SetViewParams& params) {
        const auto& s = state();
        if (s.set_view_for_panel_callback) {
            s.set_view_for_panel_callback(panel, params);
            return;
        }
        if (s.set_view_callback) {
            s.set_view_callback(params);
        }
    }

    void apply_set_fov(float fov_degrees) {
        const auto& s = state();
        if (s.set_fov_callback) {
            s.set_fov_callback(fov_degrees);
        }
    }

    void set_render_settings_callbacks(GetRenderSettingsCallback get_cb, SetRenderSettingsCallback set_cb) {
        auto& s = state();
        s.get_render_settings_callback = std::move(get_cb);
        s.set_render_settings_callback = std::move(set_cb);
    }

    std::optional<RenderSettingsProxy> get_render_settings() {
        const auto& s = state();
        if (!s.get_render_settings_callback)
            return std::nullopt;
        return s.get_render_settings_callback();
    }

    void update_render_settings(const RenderSettingsProxy& settings) {
        const auto& s = state();
        if (s.set_render_settings_callback) {
            s.set_render_settings_callback(settings);
        }
    }

} // namespace lfs::vis
