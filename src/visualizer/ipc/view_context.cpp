/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "view_context.hpp"
#include "python/python_runtime.hpp"

namespace lfs::vis {

    namespace py = lfs::python;

    struct ViewContextState {
        GetViewCallback view_callback;
        GetViewportRenderCallback viewport_render_callback;
        CaptureViewportRenderCallback capture_viewport_render_callback;
        std::mutex render_settings_mutex;
        CaptureRenderSettingsTargetCallback capture_render_settings_target;
        GetViewRenderSettingsCallback get_view_render_settings;
        SetViewRenderSettingsCallback set_view_render_settings;
        GetRenderSettingsCallback get_render_settings_callback;
        SetRenderSettingsCallback set_render_settings_callback;
        SetViewCallback set_view_callback;
        SetFovCallback set_fov_callback;
        SetOrthoScaleCallback set_ortho_scale_callback;
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

    void set_viewport_render_callback(GetViewportRenderCallback callback) {
        state().viewport_render_callback = std::move(callback);
    }

    void set_capture_viewport_render_callback(CaptureViewportRenderCallback callback) {
        state().capture_viewport_render_callback = std::move(callback);
    }

    std::optional<ViewInfo> get_current_view_info() {
        if (const auto* camera = py::get_overlay_draw_context().camera)
            return *camera;
        const auto& s = state();
        if (!s.view_callback)
            return std::nullopt;
        return s.view_callback();
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

    void set_set_fov_callback(SetFovCallback callback) {
        state().set_fov_callback = std::move(callback);
    }

    void set_set_ortho_scale_callback(SetOrthoScaleCallback callback) {
        state().set_ortho_scale_callback = std::move(callback);
    }

    void apply_set_view(const SetViewParams& params) {
        const auto& s = state();
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

    void apply_set_ortho_scale(std::optional<float> scale) {
        const auto& s = state();
        if (s.set_ortho_scale_callback)
            s.set_ortho_scale_callback(scale);
    }

    void set_view_render_settings_callbacks(CaptureRenderSettingsTargetCallback capture,
                                            GetViewRenderSettingsCallback get,
                                            SetViewRenderSettingsCallback set) {
        auto& s = state();
        std::lock_guard lock(s.render_settings_mutex);
        s.capture_render_settings_target = std::move(capture);
        s.get_view_render_settings = std::move(get);
        s.set_view_render_settings = std::move(set);
    }

    std::optional<RenderSettingsTarget> capture_render_settings_target() {
        auto& s = state();
        CaptureRenderSettingsTargetCallback capture;
        {
            std::lock_guard lock(s.render_settings_mutex);
            capture = s.capture_render_settings_target;
        }
        return capture ? capture() : std::nullopt;
    }

    std::optional<RenderSettingsProxy> get_render_settings_for_view(const RenderSettingsTarget target) {
        auto& s = state();
        GetViewRenderSettingsCallback get;
        {
            std::lock_guard lock(s.render_settings_mutex);
            get = s.get_view_render_settings;
        }
        return get ? get(target) : std::nullopt;
    }

    std::optional<RenderSettingsProxy> update_render_settings_for_view(
        const RenderSettingsTarget target, const RenderSettingsProxy& settings, const RenderSettingsUpdateIntent intent) {
        auto& s = state();
        SetViewRenderSettingsCallback set;
        {
            std::lock_guard lock(s.render_settings_mutex);
            set = s.set_view_render_settings;
        }
        return set ? set(target, settings, intent) : std::nullopt;
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

    void update_render_settings(const RenderSettingsProxy& settings,
                                const RenderSettingsUpdateIntent intent) {
        const auto& s = state();
        if (s.set_render_settings_callback) {
            s.set_render_settings_callback(settings, intent);
        }
    }

} // namespace lfs::vis
