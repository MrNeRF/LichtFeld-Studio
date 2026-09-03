/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_selection.hpp"
#include "core/cuda/selection_ops.hpp"
#include "core/tensor.hpp"
#include "geometry/euclidean_transform.hpp"
#include "py_tensor.hpp"
#include "python/python_runtime.hpp"
#include "rendering/selection_ops.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/internal/viewport.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/operation/undo_entry.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/selection/selection_service.hpp"
#include "visualizer/tools/selection_tool.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <stdexcept>

namespace nb = nanobind;

namespace lfs::python {

    namespace {
        constexpr float DEPTH_FILTER_HALF_HEIGHT = 10000.0f;
        constexpr float DEFAULT_WINDOW_SCALE = 0.35f;
        constexpr float MIN_WINDOW_SCALE = 0.05f;
        constexpr float MAX_WINDOW_SCALE = 1.0f;
        constexpr float DEFAULT_LEGACY_HALF_WIDTH = 1.35f;
        // The range the depth getters report when no near/far band is in effect.
        // Matches what they already return with no rendering manager.
        constexpr float DEFAULT_LEGACY_DEPTH_NEAR = 0.0f;
        constexpr float DEFAULT_LEGACY_DEPTH_FAR = 100.0f;

        float g_last_legacy_half_width = DEFAULT_LEGACY_HALF_WIDTH;

        vis::RenderingManager* get_rm() { return get_rendering_manager(); }

        vis::SceneManager* get_sm() { return get_scene_manager(); }

        vis::SelectionService* get_ss() { return get_selection_service(); }

        auto* get_selection_tool() {
            auto* const gm = get_gui_manager();
            auto* const viewer = gm ? gm->getViewer() : nullptr;
            return viewer ? viewer->getSelectionTool() : nullptr;
        }

        template <typename Mutator>
        void apply_selection_state_with_undo(vis::SceneManager& scene_manager,
                                             const std::string& undo_label,
                                             Mutator&& mutator) {
            auto snapshot = std::make_unique<vis::op::SceneSnapshot>(scene_manager, undo_label);
            snapshot->captureSelection();
            mutator(scene_manager.getScene());
            snapshot->captureAfter();
            vis::op::pushSceneSnapshotIfChanged(std::move(snapshot));
        }

        [[nodiscard]] bool viewport_available(const std::optional<vis::ViewInfo>& view) {
            return view.has_value() && view->width > 0 && view->height > 0;
        }

        // KEEP IN SYNC: this converter pair inverts the far-plane width mapping
        // that depthWindowFarPlaneHalfExtents (selection_tool.cpp) computes from
        // pixel focal lengths / ortho scale. Known divergence: the tool honors
        // viewport.ortho_scale_override; these use the main view's ortho_scale.
        [[nodiscard]] float convert_legacy_half_width_to_scale(const float half_width,
                                                               const float depth_far,
                                                               const vis::RenderSettings& settings) {
            const auto view = vis::get_current_view_info();
            if (!viewport_available(view) || settings.equirectangular) {
                return DEFAULT_WINDOW_SCALE;
            }
            if (settings.orthographic) {
                const float ortho_width =
                    static_cast<float>(view->width) / std::max(view->ortho_scale, 1.0e-5f);
                const float denom = 0.5f * ortho_width;
                if (!(denom > 1.0e-8f)) {
                    return DEFAULT_WINDOW_SCALE;
                }
                return std::clamp(half_width / denom, MIN_WINDOW_SCALE, MAX_WINDOW_SCALE);
            }
            if (!(depth_far > 1.0e-3f)) {
                return DEFAULT_WINDOW_SCALE;
            }
            const float vfov_rad = glm::radians(view->fov);
            const float aspect =
                static_cast<float>(view->width) / static_cast<float>(view->height);
            const float tan_hfov_half = std::tan(vfov_rad * 0.5f) * aspect;
            const float denom = tan_hfov_half * depth_far;
            if (!(denom > 1.0e-8f)) {
                return DEFAULT_WINDOW_SCALE;
            }
            return std::clamp(half_width / denom, MIN_WINDOW_SCALE, MAX_WINDOW_SCALE);
        }

        [[nodiscard]] float convert_scale_to_legacy_half_width(const float scale,
                                                               const float depth_far,
                                                               const vis::RenderSettings& settings) {
            const auto view = vis::get_current_view_info();
            if (!viewport_available(view) || settings.equirectangular) {
                return g_last_legacy_half_width;
            }
            if (settings.orthographic) {
                const float ortho_width =
                    static_cast<float>(view->width) / std::max(view->ortho_scale, 1.0e-5f);
                return scale * 0.5f * ortho_width;
            }
            if (!(depth_far > 1.0e-3f)) {
                return g_last_legacy_half_width;
            }
            const float vfov_rad = glm::radians(view->fov);
            const float aspect =
                static_cast<float>(view->width) / static_cast<float>(view->height);
            const float tan_hfov_half = std::tan(vfov_rad * 0.5f) * aspect;
            return scale * tan_hfov_half * depth_far;
        }

        [[nodiscard]] float informational_half_width_from_settings(const vis::RenderSettings& settings,
                                                                   const float depth_far) {
            return convert_scale_to_legacy_half_width(settings.depth_filter_scale_x, depth_far,
                                                      settings); // legacy converters are X-only by contract
        }

        void configure_depth_filter(vis::RenderSettings& settings, const bool enabled,
                                    const float depth_near, const float depth_far,
                                    const float frustum_half_width) {
            const float clamped_near = std::max(depth_near, 0.0f);
            const float clamped_far = std::max(depth_far, clamped_near);
            const float clamped_width = std::max(frustum_half_width, 0.05f);

            settings.depth_filter_enabled = enabled;
            settings.depth_filter_min = glm::vec3(-clamped_width, -DEPTH_FILTER_HALF_HEIGHT, -clamped_far);
            settings.depth_filter_max = glm::vec3(clamped_width, DEPTH_FILTER_HALF_HEIGHT, -clamped_near);

            if (!enabled) {
                return;
            }

            if (auto view_info = vis::get_current_view_info()) {
                glm::mat3 rotation(1.0f);
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        rotation[col][row] = view_info->rotation[row * 3 + col];
                    }
                }

                settings.depth_filter_transform = lfs::geometry::EuclideanTransform(
                    glm::quat_cast(rotation),
                    glm::vec3(view_info->translation[0],
                              view_info->translation[1],
                              view_info->translation[2]));
            }
        }

        void write_legacy_window_scale(vis::RenderSettings& settings,
                                       const float frustum_half_width,
                                       const float depth_far) {
            const float clamped_scale =
                convert_legacy_half_width_to_scale(frustum_half_width, depth_far, settings);
            settings.depth_filter_scale_x = clamped_scale;
            settings.depth_filter_scale_y = clamped_scale;
        }

        void apply_legacy_depth_filter(const bool enabled,
                                       const float depth_near,
                                       const float depth_far,
                                       const float frustum_half_width) {
            g_last_legacy_half_width = std::max(frustum_half_width, 0.05f);
            auto* const tool = get_selection_tool();
            auto* const rm = get_rm();
            // Skip the scale pre-write when the tool exists but is disabled: the
            // tool ignores setDepthFilterRange in that state, and a lone scale
            // write would half-apply the request. Legacy calls defer all values
            // atomically, matching the pre-window behavior.
            if (rm && (!tool || tool->isEnabled())) {
                auto settings = rm->getSettings();
                write_legacy_window_scale(settings, frustum_half_width, depth_far);
                rm->updateSettings(settings);
            }
            if (tool) {
                tool->setDepthFilterRange(enabled, depth_near, depth_far, g_last_legacy_half_width);
                return;
            }
            if (!rm) {
                return;
            }
            auto settings = rm->getSettings();
            configure_depth_filter(settings, enabled, depth_near, depth_far, frustum_half_width);
            write_legacy_window_scale(settings, frustum_half_width, depth_far);
            rm->updateSettings(settings);
        }

        void apply_depth_filter_window(const bool enabled,
                                       const float depth_near,
                                       const float depth_far,
                                       const float scale,
                                       const float offset_x,
                                       const float offset_y,
                                       const std::optional<float> scale_y) {
            auto* const tool = get_selection_tool();
            if (tool && !tool->isEnabled()) {
                // A disable request cannot half-apply; skip everything so the call
                // stays atomic (the toolbar data-model init passes enabled=false
                // before any tool is active).
                if (!enabled) {
                    return;
                }
                // The tool ignores setDepthFilterRange while disabled; writing the window
                // fields into settings anyway would half-apply the request.
                // Known residual: this gate cannot see the tool's one-shot
                // preserve_restored_render_state_ window during project restore. An
                // enable/modify call there self-heals within one frame (the per-frame
                // update() stamp is not preserve-gated); only disabling the filter in
                // that window can leave the restored render-side flag stale until the
                // tool is next enabled (or another apply path stamps settings).
                throw std::runtime_error("Selection tool is not active/enabled; activate it before setting the depth filter window");
            }
            const float clamped_scale_x = std::clamp(scale, MIN_WINDOW_SCALE, MAX_WINDOW_SCALE);
            const float clamped_scale_y =
                std::clamp(scale_y.value_or(scale), MIN_WINDOW_SCALE, MAX_WINDOW_SCALE);
            const float clamped_offset_x = std::clamp(offset_x, -1.0f, 1.0f);
            const float clamped_offset_y = std::clamp(offset_y, -1.0f, 1.0f);
            auto* const rm = get_rm();
            if (rm) {
                auto settings = rm->getSettings();
                settings.depth_filter_scale_x = clamped_scale_x;
                settings.depth_filter_scale_y = clamped_scale_y;
                settings.depth_filter_offset_x = clamped_offset_x;
                settings.depth_filter_offset_y = clamped_offset_y;
                rm->updateSettings(settings);
            }
            const float informational_half_width = rm
                                                       ? informational_half_width_from_settings(
                                                             rm->getSettings(), std::max(depth_far, 0.0f))
                                                       : g_last_legacy_half_width;
            if (tool) {
                tool->setDepthFilterRange(enabled, depth_near, depth_far,
                                          std::max(informational_half_width, 0.05f));
                return;
            }
            if (!rm) {
                return;
            }
            auto settings = rm->getSettings();
            configure_depth_filter(settings, enabled, depth_near, depth_far,
                                   std::max(informational_half_width, 0.05f));
            settings.depth_filter_scale_x = clamped_scale_x;
            settings.depth_filter_scale_y = clamped_scale_y;
            settings.depth_filter_offset_x = clamped_offset_x;
            settings.depth_filter_offset_y = clamped_offset_y;
            rm->updateSettings(settings);
        }

        [[nodiscard]] std::tuple<float, float> fallback_depth_near_far(const vis::RenderSettings& settings) {
            // The constructor default is not in the near/far encoding - session
            // restore makes the same distinction before decoding a box. Decoding
            // it anyway reads it as near 0 / far 0, so a script that read the
            // range and wrote it back would build a zero-width band. Report the
            // legacy range instead, which is what the no-manager branches of
            // these getters already return.
            //
            // Test the sentinel, not merely the disabled flag:
            // configure_depth_filter writes a real near/far box before it returns
            // for a disabled request, so a caller can legitimately hold a band
            // with the filter switched off, and that band must still read back.
            if (!settings.depth_filter_enabled &&
                settings.depth_filter_min.z == 0.0f &&
                settings.depth_filter_max.z == 100.0f) {
                return {DEFAULT_LEGACY_DEPTH_NEAR, DEFAULT_LEGACY_DEPTH_FAR};
            }
            const float depth_near = std::max(-settings.depth_filter_max.z, 0.0f);
            const float depth_far = std::max(-settings.depth_filter_min.z, depth_near);
            return {depth_near, depth_far};
        }
    } // namespace

    void register_selection(nb::module_& m) {
        auto sel = m.def_submodule("selection", "Selection primitives for operators");

        // Selection mode enum
        nb::enum_<vis::SelectionMode>(sel, "SelectionMode")
            .value("Replace", vis::SelectionMode::Replace)
            .value("Add", vis::SelectionMode::Add)
            .value("Remove", vis::SelectionMode::Remove)
            .value("Intersect", vis::SelectionMode::Intersect);

        // ─────────────────────────────────────────────────────────────────────
        // STROKE MANAGEMENT
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "begin_stroke", []() {
                if (auto* ss = get_ss()) {
                    ss->beginStroke();
                }
            },
            "Begin a new selection stroke (saves undo state)");

        sel.def(
            "get_stroke_selection", []() -> std::optional<PyTensor> {
                auto* ss = get_ss();
                if (!ss)
                    return std::nullopt;
                auto* tensor = ss->getStrokeSelection();
                if (!tensor || !tensor->is_valid())
                    return std::nullopt;
                return PyTensor(*tensor, false);
            },
            "Get the current stroke selection tensor [N] uint8");

        sel.def(
            "commit_stroke", [](vis::SelectionMode mode) -> bool {
                auto* ss = get_ss();
                if (!ss)
                    return false;
                auto result = ss->finalizeStroke(mode);
                return result.success;
            },
            nb::arg("mode"), "Commit stroke to selection with given mode (Replace/Add/Remove)");

        sel.def(
            "cancel_stroke", []() {
                if (auto* ss = get_ss()) {
                    ss->cancelStroke();
                }
            },
            "Cancel current stroke (discard changes)");

        sel.def(
            "is_stroke_active", []() -> bool {
                auto* ss = get_ss();
                return ss && ss->isStrokeActive();
            },
            "Check if a stroke is currently active");

        // ─────────────────────────────────────────────────────────────────────
        // GPU SELECTION KERNELS
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "ring_select", [](int index, bool add) {
                auto* ss = get_ss();
                if (!ss || index < 0)
                    return;
                auto* stroke = ss->getStrokeSelection();
                if (!stroke || !stroke->is_valid())
                    return;
                if (static_cast<size_t>(index) >= stroke->numel())
                    return;
                rendering::set_selection_element(stroke->ptr<bool>(), index, add);
            },
            nb::arg("index"), nb::arg("add") = true, "Select/deselect a single gaussian by index (for ring selection mode).");

        // ─────────────────────────────────────────────────────────────────────
        // PREVIEW & VISUAL STATE
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "set_preview", [](bool add_mode) {
                auto* rm = get_rm();
                auto* ss = get_ss();
                if (!rm || !ss)
                    return;
                auto* stroke = ss->getStrokeSelection();
                rm->setPreviewSelection(stroke, add_mode);
            },
            nb::arg("add_mode") = true, "Set current stroke as preview selection (green = add, red = remove)");

        sel.def(
            "clear_preview", []() {
                if (auto* rm = get_rm()) {
                    rm->clearPreviewSelection();
                }
            },
            "Clear preview selection overlay");

        sel.def(
            "draw_brush_circle", [](float x, float y, float radius, bool add_mode) {
                auto* rm = get_rm();
                auto* ss = get_ss();
                if (!rm)
                    return;
                core::Tensor* stroke = ss ? ss->getStrokeSelection() : nullptr;
                rm->setCursorPreviewState(true, x, y, radius, add_mode, stroke);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("add_mode") = true, "Draw brush circle overlay at (x, y)");

        sel.def(
            "clear_brush_state", []() {
                if (auto* rm = get_rm()) {
                    rm->clearCursorPreviewState();
                }
            },
            "Clear brush circle overlay");

        // Rectangle preview
        sel.def(
            "draw_rect_preview", [](float x0, float y0, float x1, float y1, bool add_mode) {
                if (auto* rm = get_rm()) {
                    rm->setRectPreview(x0, y0, x1, y1, add_mode);
                }
            },
            nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("add_mode") = true, "Draw rectangle selection preview");

        sel.def(
            "clear_rect_preview", []() {
                if (auto* rm = get_rm()) {
                    rm->clearRectPreview();
                }
            },
            "Clear rectangle selection preview");

        // Polygon preview
        sel.def(
            "draw_polygon_preview", [](const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode) {
                if (auto* rm = get_rm()) {
                    rm->setPolygonPreview(points, closed, add_mode);
                }
            },
            nb::arg("points"), nb::arg("closed") = false, nb::arg("add_mode") = true, "Draw polygon selection preview (render-space 2D points)");

        sel.def(
            "clear_polygon_preview", []() {
                if (auto* rm = get_rm()) {
                    rm->clearPolygonPreview();
                }
            },
            "Clear polygon selection preview");

        // Lasso preview
        sel.def(
            "draw_lasso_preview", [](const std::vector<std::pair<float, float>>& points, bool add_mode) {
                if (auto* rm = get_rm()) {
                    rm->setLassoPreview(points, add_mode);
                }
            },
            nb::arg("points"), nb::arg("add_mode") = true, "Draw lasso selection preview");

        sel.def(
            "clear_lasso_preview", []() {
                if (auto* rm = get_rm()) {
                    rm->clearLassoPreview();
                }
            },
            "Clear lasso selection preview");

        // ─────────────────────────────────────────────────────────────────────
        // SCREEN POSITIONS
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "has_screen_positions", []() -> bool {
                auto* ss = get_ss();
                return ss && ss->hasScreenPositions();
            },
            "Check if screen positions are available");

        sel.def(
            "get_screen_positions", []() -> std::optional<PyTensor> {
                auto* ss = get_ss();
                if (!ss)
                    return std::nullopt;
                auto positions = ss->getScreenPositions();
                if (!positions || !positions->is_valid())
                    return std::nullopt;
                return PyTensor(*positions, false);
            },
            "Get screen positions tensor [N, 2]");

        // ─────────────────────────────────────────────────────────────────────
        // DEPTH FILTER
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "set_depth_filter", [](bool enabled, float depth_far, float frustum_half_width, float depth_near) {
                apply_legacy_depth_filter(enabled, depth_near, depth_far, frustum_half_width);
            },
            nb::arg("enabled"), nb::arg("depth_far") = 100.0f, nb::arg("frustum_half_width") = 50.0f, nb::arg("depth_near") = 0.0f, "Deprecated. Set selection depth filter in camera space.\n"
                                                                                                                                    "frustum_half_width is converted to a screen-space window scale:\n"
                                                                                                                                    "- pinhole: scale = clamp(half_width / (tan(hfov/2) * far), 0.05, 1.0) when far > 1e-3 "
                                                                                                                                    "and a viewport is available; otherwise scale = 0.35\n"
                                                                                                                                    "- ortho: scale = clamp(half_width / (0.5 * ortho_width), 0.05, 1.0)\n"
                                                                                                                                    "- equirect or no viewport: scale = 0.35 (no single equivalent exists)\n"
                                                                                                                                    "Prefer set_depth_filter_window.");

        sel.def(
            "set_depth_filter_range", [](bool enabled, float depth_near, float depth_far, float frustum_half_width) {
                apply_legacy_depth_filter(enabled, depth_near, depth_far, frustum_half_width);
            },
            nb::arg("enabled"), nb::arg("depth_near") = 0.0f, nb::arg("depth_far") = 100.0f, nb::arg("frustum_half_width") = 50.0f, "Deprecated. Set selection depth filter range in camera space as (near, far, width).\n"
                                                                                                                                    "frustum_half_width is converted to a screen-space window scale:\n"
                                                                                                                                    "- pinhole: scale = clamp(half_width / (tan(hfov/2) * far), 0.05, 1.0) when far > 1e-3 "
                                                                                                                                    "and a viewport is available; otherwise scale = 0.35\n"
                                                                                                                                    "- ortho: scale = clamp(half_width / (0.5 * ortho_width), 0.05, 1.0)\n"
                                                                                                                                    "- equirect or no viewport: scale = 0.35 (no single equivalent exists)\n"
                                                                                                                                    "Prefer set_depth_filter_window.");

        sel.def(
            "set_depth_filter_window", [](bool enabled, float depth_near, float depth_far, float scale, float offset_x, float offset_y, std::optional<float> scale_y) {
                apply_depth_filter_window(enabled, depth_near, depth_far, scale, offset_x, offset_y, scale_y);
            },
            nb::arg("enabled"), nb::arg("depth_near") = 0.0f, nb::arg("depth_far") = 100.0f, nb::arg("scale") = 0.35f, nb::arg("offset_x") = 0.0f, nb::arg("offset_y") = 0.0f, nb::arg("scale_y") = nb::none(), "Set the screen-space selection depth window.\n"
                                                                                                                                                                                                                "scale is the X-axis on-screen fraction of the viewport (0.05-1.0, default 0.35).\n"
                                                                                                                                                                                                                "scale_y is the Y-axis fraction; None uses scale for isotropic compatibility.\n"
                                                                                                                                                                                                                "offset_x/offset_y are fractions of available travel (-1 to 1, default 0).\n"
                                                                                                                                                                                                                "When the Selection tool exists but is not active/enabled, enable/modify\n"
                                                                                                                                                                                                                "requests (enabled=True) raise RuntimeError because they cannot be applied\n"
                                                                                                                                                                                                                "atomically; disable requests (enabled=False) are silent atomic no-ops,\n"
                                                                                                                                                                                                                "matching the legacy calls' contract.");

        sel.def(
            "get_depth_filter", []() -> std::tuple<bool, float, float> {
                if (const auto* const tool = get_selection_tool()) {
                    const auto* const rm = get_rm();
                    const float informational_half_width = rm
                                                               ? informational_half_width_from_settings(
                                                                     rm->getSettings(), tool->getDepthFar())
                                                               : g_last_legacy_half_width;
                    return {tool->isDepthFilterEnabled(),
                            tool->getDepthFar(),
                            informational_half_width};
                }
                auto* rm = get_rm();
                if (!rm)
                    return {false, 100.0f, g_last_legacy_half_width};
                const auto& settings = rm->getSettings();
                const auto [depth_near, depth_far] = fallback_depth_near_far(settings);
                (void)depth_near;
                return {settings.depth_filter_enabled, depth_far,
                        informational_half_width_from_settings(settings, depth_far)};
            },
            "Get depth filter state: (enabled, depth_far, frustum_half_width).\n"
            "frustum_half_width is a derived informational read-back of the far-plane-equivalent "
            "window half-width (inverse of the set_depth_filter_range conversion).");

        sel.def(
            "get_depth_filter_range", []() -> std::tuple<bool, float, float, float> {
                if (const auto* const tool = get_selection_tool()) {
                    const auto* const rm = get_rm();
                    const float informational_half_width = rm
                                                               ? informational_half_width_from_settings(
                                                                     rm->getSettings(), tool->getDepthFar())
                                                               : g_last_legacy_half_width;
                    return {tool->isDepthFilterEnabled(),
                            tool->getDepthNear(),
                            tool->getDepthFar(),
                            informational_half_width};
                }
                auto* rm = get_rm();
                if (!rm)
                    return {false, 0.0f, 100.0f, g_last_legacy_half_width};
                const auto& settings = rm->getSettings();
                const auto [depth_near, depth_far] = fallback_depth_near_far(settings);
                return {settings.depth_filter_enabled,
                        depth_near,
                        depth_far,
                        informational_half_width_from_settings(settings, depth_far)};
            },
            "Get selection depth filter state: (enabled, depth_near, depth_far, frustum_half_width).\n"
            "frustum_half_width is a derived informational read-back of the far-plane-equivalent "
            "window half-width (inverse of the set_depth_filter_range conversion).");

        sel.def(
            "get_depth_filter_window", []() -> std::tuple<bool, float, float, float, float, float, float> {
                if (const auto* const tool = get_selection_tool()) {
                    const auto* const rm = get_rm();
                    float scale_x = DEFAULT_WINDOW_SCALE;
                    float scale_y = DEFAULT_WINDOW_SCALE;
                    float offset_x = 0.0f;
                    float offset_y = 0.0f;
                    if (rm) {
                        const auto& settings = rm->getSettings();
                        scale_x = settings.depth_filter_scale_x;
                        scale_y = settings.depth_filter_scale_y;
                        offset_x = settings.depth_filter_offset_x;
                        offset_y = settings.depth_filter_offset_y;
                    }
                    return {tool->isDepthFilterEnabled(),
                            tool->getDepthNear(),
                            tool->getDepthFar(),
                            scale_x,
                            scale_y,
                            offset_x,
                            offset_y};
                }
                auto* rm = get_rm();
                if (!rm)
                    return {false, 0.0f, 100.0f, DEFAULT_WINDOW_SCALE, DEFAULT_WINDOW_SCALE, 0.0f, 0.0f};
                const auto& settings = rm->getSettings();
                const auto [depth_near, depth_far] = fallback_depth_near_far(settings);
                return {settings.depth_filter_enabled,
                        depth_near,
                        depth_far,
                        settings.depth_filter_scale_x,
                        settings.depth_filter_scale_y,
                        settings.depth_filter_offset_x,
                        settings.depth_filter_offset_y};
            },
            "Get the screen-space selection depth window:\n"
            "(enabled, near, far, scale_x, scale_y, offset_x, offset_y).");

        // ─────────────────────────────────────────────────────────────────────
        // CROP FILTER
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "set_crop_filter", [](bool enabled) {
                auto* rm = get_rm();
                if (!rm)
                    return;
                auto settings = rm->getSettings();
                settings.crop_filter_for_selection = enabled;
                rm->updateSettings(settings);
            },
            nb::arg("enabled"), "Enable/disable crop box filtering for selection");

        sel.def(
            "apply_crop_filter", []() {
                auto* ss = get_ss();
                if (!ss)
                    return;
                ss->applyCropFilterToStroke();
            },
            "Apply crop box filter to current stroke selection");

        // ─────────────────────────────────────────────────────────────────────
        // VIEWPORT & COORDINATES
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "get_viewport_bounds", []() -> std::tuple<float, float, float, float> {
                float x, y, w, h;
                get_viewport_bounds(x, y, w, h);
                return {x, y, w, h};
            },
            "Get viewport bounds (x, y, width, height)");

        sel.def(
            "get_render_scale", []() -> float {
                auto* rm = get_rm();
                return rm ? rm->getSettings().render_scale : 1.0f;
            },
            "Get current render scale factor");

        sel.def(
            "screen_to_render", [](float screen_x, float screen_y) -> std::pair<float, float> {
                auto* rm = get_rm();
                const float scale = rm ? rm->getSettings().render_scale : 1.0f;
                float vx, vy, vw, vh;
                get_viewport_bounds(vx, vy, vw, vh);
                const float local_x = screen_x - vx;
                const float local_y = screen_y - vy;
                return {local_x * scale, local_y * scale};
            },
            nb::arg("screen_x"), nb::arg("screen_y"), "Convert screen coordinates to render coordinates");

        sel.def(
            "get_hovered_gaussian_id", []() -> int {
                auto* rm = get_rm();
                return rm ? rm->getHoveredGaussianId() : -1;
            },
            "Get ID of gaussian under cursor (-1 if none)");

        // PickResult struct
        struct PyPickResult {
            int index;
            float depth;
            std::tuple<float, float, float> world_position;
        };

        nb::class_<PyPickResult>(sel, "PickResult")
            .def_ro("index", &PyPickResult::index, "Gaussian index at current cursor position (-1 if unavailable)")
            .def_ro("depth", &PyPickResult::depth, "Camera-space depth")
            .def_ro("world_position", &PyPickResult::world_position, "Hit point in world coordinates");

        sel.def(
            "pick_at_screen", [](float screen_x, float screen_y) -> std::optional<PyPickResult> {
                auto* rm = get_rm();
                if (!rm)
                    return std::nullopt;

                float vx, vy, vw, vh;
                get_viewport_bounds(vx, vy, vw, vh);
                const float local_x = screen_x - vx;
                const float local_y = screen_y - vy;

                const float depth = rm->getDepthAtPixel(
                    static_cast<int>(local_x), static_cast<int>(local_y));
                if (depth <= 0.0f)
                    return std::nullopt;

                auto view_info = vis::get_current_view_info();
                if (!view_info)
                    return std::nullopt;

                Viewport vp(static_cast<size_t>(vw), static_cast<size_t>(vh));
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        vp.camera.R[i][j] = view_info->rotation[j * 3 + i];
                vp.camera.t = glm::vec3(
                    view_info->translation[0],
                    view_info->translation[1],
                    view_info->translation[2]);

                const glm::vec3 world_pos = vp.unprojectPixel(
                    local_x,
                    local_y,
                    depth,
                    rm->getFocalLengthMm(),
                    view_info->orthographic,
                    view_info->ortho_scale);

                constexpr float INVALID = -1e10f;
                if (world_pos.x <= INVALID)
                    return std::nullopt;

                const int gaussian_id = rm->getHoveredGaussianId();

                return PyPickResult{
                    gaussian_id,
                    depth,
                    {world_pos.x, world_pos.y, world_pos.z}};
            },
            nb::arg("screen_x"), nb::arg("screen_y"), "Pick at screen coordinates. Returns PickResult with depth and world_position at the given coords. "
                                                      "The index field reflects the gaussian under the current cursor, not the queried coordinates.");

        // ─────────────────────────────────────────────────────────────────────
        // SELECTION GROUPS
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "get_active_group", []() -> int {
                auto* sm = get_sm();
                if (!sm)
                    return 0;
                return static_cast<int>(sm->getScene().getActiveSelectionGroup());
            },
            "Get the active selection group ID");

        sel.def(
            "set_active_group", [](int group_id) {
                auto* sm = get_sm();
                if (!sm)
                    return;
                apply_selection_state_with_undo(
                    *sm, "selection_group.set_active",
                    [group_id](core::Scene& scene) { scene.setActiveSelectionGroup(static_cast<uint8_t>(group_id)); });
            },
            nb::arg("group_id"), "Set the active selection group ID");

        sel.def(
            "is_group_locked", [](int group_id) -> bool {
                auto* sm = get_sm();
                if (!sm)
                    return false;
                return sm->getScene().isSelectionGroupLocked(static_cast<uint8_t>(group_id));
            },
            nb::arg("group_id"), "Check if a selection group is locked");

        // ─────────────────────────────────────────────────────────────────────
        // SPATIAL SELECTION OPERATIONS (GPU)
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "grow", [](float radius, int iterations) {
                auto* sm = get_sm();
                if (!sm)
                    return;
                auto& scene = sm->getScene();
                auto mask = scene.getSelectionMask();
                if (!mask || !scene.hasSelection())
                    return;
                auto* model = scene.getCombinedModel();
                if (!model)
                    return;
                const auto group_id = scene.getActiveSelectionGroup();
                auto current = *mask;
                for (int i = 0; i < iterations; ++i)
                    current = core::cuda::selection_grow(current, model->means(), radius, group_id);
                apply_selection_state_with_undo(
                    *sm, "selection.grow",
                    [updated = std::move(current)](core::Scene& target_scene) mutable {
                        target_scene.setSelectionMask(std::make_shared<core::Tensor>(std::move(updated)));
                    });
                if (auto* rm = get_rm())
                    rm->markDirty(vis::DirtyFlag::SELECTION);
            },
            nb::arg("radius"), nb::arg("iterations") = 1, "Grow selection by radius (scene units). Uses spatial hashing, O(N).");

        sel.def(
            "shrink", [](float radius, int iterations) {
                auto* sm = get_sm();
                if (!sm)
                    return;
                auto& scene = sm->getScene();
                auto mask = scene.getSelectionMask();
                if (!mask || !scene.hasSelection())
                    return;
                auto* model = scene.getCombinedModel();
                if (!model)
                    return;
                auto current = *mask;
                for (int i = 0; i < iterations; ++i)
                    current = core::cuda::selection_shrink(current, model->means(), radius);
                apply_selection_state_with_undo(
                    *sm, "selection.shrink",
                    [updated = std::move(current)](core::Scene& target_scene) mutable {
                        target_scene.setSelectionMask(std::make_shared<core::Tensor>(std::move(updated)));
                    });
                if (auto* rm = get_rm())
                    rm->markDirty(vis::DirtyFlag::SELECTION);
            },
            nb::arg("radius"), nb::arg("iterations") = 1, "Shrink selection by radius (scene units). Uses spatial hashing, O(N).");

        sel.def(
            "by_opacity", [](float min_opacity, float max_opacity) {
                auto* sm = get_sm();
                if (!sm)
                    return;
                auto& scene = sm->getScene();
                auto* model = scene.getCombinedModel();
                if (!model)
                    return;
                const auto group_id = scene.getActiveSelectionGroup();
                auto mask = core::cuda::select_by_opacity(model->opacity_raw(), min_opacity, max_opacity, group_id);
                apply_selection_state_with_undo(
                    *sm, "selection.by_opacity",
                    [updated = std::move(mask)](core::Scene& target_scene) mutable {
                        target_scene.setSelectionMask(std::make_shared<core::Tensor>(std::move(updated)));
                    });
                if (auto* rm = get_rm())
                    rm->markDirty(vis::DirtyFlag::SELECTION);
            },
            nb::arg("min_opacity") = 0.0f, nb::arg("max_opacity") = 1.0f, "Select gaussians by activated opacity range [min, max].");

        sel.def(
            "by_scale", [](float max_scale) {
                auto* sm = get_sm();
                if (!sm)
                    return;
                auto& scene = sm->getScene();
                auto* model = scene.getCombinedModel();
                if (!model)
                    return;
                const auto group_id = scene.getActiveSelectionGroup();
                auto mask = core::cuda::select_by_scale(model->scaling_raw(), max_scale, group_id);
                apply_selection_state_with_undo(
                    *sm, "selection.by_scale",
                    [updated = std::move(mask)](core::Scene& target_scene) mutable {
                        target_scene.setSelectionMask(std::make_shared<core::Tensor>(std::move(updated)));
                    });
                if (auto* rm = get_rm())
                    rm->markDirty(vis::DirtyFlag::SELECTION);
            },
            nb::arg("max_scale"), "Select gaussians with max activated scale <= threshold.");

        sel.def(
            "by_color", [](int gaussian_index, float threshold) {
                auto* sm = get_sm();
                if (!sm)
                    return;
                auto& scene = sm->getScene();
                auto* model = scene.getCombinedModel();
                if (!model)
                    return;
                const auto& sh0 = model->sh0();
                if (!sh0.is_valid() || gaussian_index < 0 ||
                    static_cast<size_t>(gaussian_index) >= sh0.size(0))
                    return;

                // Read the reference gaussian's SH0 coefficients from GPU
                const auto sh0_row = sh0.slice(0, static_cast<size_t>(gaussian_index),
                                               static_cast<size_t>(gaussian_index) + 1)
                                         .cpu()
                                         .contiguous();
                const float* sh0_data = sh0_row.ptr<float>();
                if (!sh0_data)
                    return;

                // Decode SH DC to RGB: color = clamp(0.5 + sh_val * SH_C0, 0, 1)
                constexpr float SH_C0 = 0.28209479177387814f;
                const float ref_r = std::clamp(0.5f + sh0_data[0] * SH_C0, 0.0f, 1.0f);
                const float ref_g = std::clamp(0.5f + sh0_data[1] * SH_C0, 0.0f, 1.0f);
                const float ref_b = std::clamp(0.5f + sh0_data[2] * SH_C0, 0.0f, 1.0f);

                const auto group_id = scene.getActiveSelectionGroup();
                auto mask = core::cuda::select_by_color(sh0, ref_r, ref_g, ref_b,
                                                        std::clamp(threshold, 0.0f, 1.0f), group_id);
                apply_selection_state_with_undo(
                    *sm, "selection.by_color",
                    [updated = std::move(mask)](core::Scene& target_scene) mutable {
                        target_scene.setSelectionMask(std::make_shared<core::Tensor>(std::move(updated)));
                    });
                if (auto* rm = get_rm())
                    rm->markDirty(vis::DirtyFlag::SELECTION);
            },
            nb::arg("gaussian_index"), nb::arg("threshold") = 0.2f, "Select gaussians by color similarity to a reference gaussian.\n"
                                                                    "Picks the SH DC color of the gaussian at the given index and selects all\n"
                                                                    "gaussians whose per-channel color difference is within the threshold (0-1).");

        // ─────────────────────────────────────────────────────────────────────
        // FLASH & FEEDBACK
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "trigger_flash", []() {
                if (auto* rm = get_rm()) {
                    rm->triggerSelectionFlash();
                }
            },
            "Trigger selection flash animation feedback");
    }

} // namespace lfs::python
