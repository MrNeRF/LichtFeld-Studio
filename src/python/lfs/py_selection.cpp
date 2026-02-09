/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_selection.hpp"
#include "core/tensor.hpp"
#include "py_tensor.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rasterizer/rasterization/include/forward.h"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "visualizer/internal/viewport.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/selection/selection_service.hpp"

#include <glm/glm.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace lfs::python {

    namespace {
        vis::RenderingManager* get_rm() { return get_rendering_manager(); }

        vis::SceneManager* get_sm() { return get_scene_manager(); }

        vis::SelectionService* get_ss() { return get_selection_service(); }
    } // namespace

    void register_selection(nb::module_& m) {
        auto sel = m.def_submodule("selection", "Selection primitives for operators");

        // Selection mode enum
        nb::enum_<vis::SelectionMode>(sel, "SelectionMode")
            .value("Replace", vis::SelectionMode::Replace)
            .value("Add", vis::SelectionMode::Add)
            .value("Remove", vis::SelectionMode::Remove);

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
            "brush_select", [](float x, float y, float radius) {
                auto* rm = get_rm();
                auto* ss = get_ss();
                if (!rm || !ss)
                    return;
                auto* stroke = ss->getStrokeSelection();
                if (!stroke || !stroke->is_valid())
                    return;
                rm->brushSelect(x, y, radius, *stroke);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("radius"), "Brush select at (x, y) with given radius. Accumulates into stroke selection.");

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

        sel.def(
            "rect_select", [](float x0, float y0, float x1, float y1) {
                auto* ss = get_ss();
                if (!ss)
                    return;
                auto screen_pos = ss->getScreenPositions();
                auto* stroke = ss->getStrokeSelection();
                if (!screen_pos || !stroke || !stroke->is_valid())
                    return;
                rendering::rect_select_tensor(*screen_pos, x0, y0, x1, y1, *stroke);
            },
            nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), "Rectangle select from (x0, y0) to (x1, y1). Sets stroke selection.");

        sel.def(
            "polygon_select", [](const std::vector<std::pair<float, float>>& vertices) {
                auto* ss = get_ss();
                if (!ss || vertices.size() < 3)
                    return;
                auto screen_pos = ss->getScreenPositions();
                auto* stroke = ss->getStrokeSelection();
                if (!screen_pos || !stroke || !stroke->is_valid())
                    return;

                // Convert vertices to GPU tensor [N, 2]
                auto poly_cpu = core::Tensor::empty({vertices.size(), size_t{2}},
                                                    core::Device::CPU, core::DataType::Float32);
                auto* data = poly_cpu.ptr<float>();
                for (size_t i = 0; i < vertices.size(); ++i) {
                    data[i * 2 + 0] = vertices[i].first;
                    data[i * 2 + 1] = vertices[i].second;
                }
                auto poly_gpu = poly_cpu.cuda();

                rendering::polygon_select_tensor(*screen_pos, poly_gpu, *stroke);
            },
            nb::arg("vertices"), "Polygon select with given vertices [(x, y), ...]. Sets stroke selection.");

        sel.def(
            "lasso_select", [](const std::vector<std::pair<float, float>>& points) {
                auto* ss = get_ss();
                if (!ss || points.size() < 3)
                    return;
                auto screen_pos = ss->getScreenPositions();
                auto* stroke = ss->getStrokeSelection();
                if (!screen_pos || !stroke || !stroke->is_valid())
                    return;

                auto poly_cpu = core::Tensor::empty({points.size(), size_t{2}},
                                                    core::Device::CPU, core::DataType::Float32);
                auto* data = poly_cpu.ptr<float>();
                for (size_t i = 0; i < points.size(); ++i) {
                    data[i * 2 + 0] = points[i].first;
                    data[i * 2 + 1] = points[i].second;
                }
                auto poly_gpu = poly_cpu.cuda();

                rendering::polygon_select_tensor(*screen_pos, poly_gpu, *stroke);
            },
            nb::arg("points"), "Lasso (freehand polygon) select. Sets stroke selection.");

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
                rm->setBrushState(true, x, y, radius, add_mode, stroke);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("add_mode") = true, "Draw brush circle overlay at (x, y)");

        sel.def(
            "clear_brush_state", []() {
                if (auto* rm = get_rm()) {
                    rm->clearBrushState();
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
            nb::arg("points"), nb::arg("closed") = false, nb::arg("add_mode") = true, "Draw polygon selection preview");

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
        // SCREEN POSITIONS OUTPUT
        // ─────────────────────────────────────────────────────────────────────

        sel.def(
            "set_output_screen_positions", [](bool enable) {
                if (auto* rm = get_rm()) {
                    rm->setOutputScreenPositions(enable);
                    rm->markDirty();
                }
            },
            nb::arg("enable"), "Enable/disable screen positions output during rendering");

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
            "set_depth_filter", [](bool enabled, float depth_far, float frustum_half_width) {
                auto* rm = get_rm();
                if (!rm)
                    return;
                auto settings = rm->getSettings();
                settings.depth_filter_enabled = enabled;
                settings.depth_filter_min = glm::vec3(-frustum_half_width, -10000.0f, 0.0f);
                settings.depth_filter_max = glm::vec3(frustum_half_width, 10000.0f, depth_far);
                rm->updateSettings(settings);
            },
            nb::arg("enabled"), nb::arg("depth_far") = 100.0f, nb::arg("frustum_half_width") = 50.0f, "Set depth filter for selection (frustum-shaped filter in camera space)");

        sel.def(
            "get_depth_filter", []() -> std::tuple<bool, float, float> {
                auto* rm = get_rm();
                if (!rm)
                    return {false, 100.0f, 50.0f};
                const auto& settings = rm->getSettings();
                return {settings.depth_filter_enabled, settings.depth_filter_max.z,
                        settings.depth_filter_max.x};
            },
            "Get depth filter state: (enabled, depth_far, frustum_half_width)");

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
                auto* rm = get_rm();
                auto* ss = get_ss();
                if (!rm || !ss)
                    return;
                auto* stroke = ss->getStrokeSelection();
                if (stroke && stroke->is_valid()) {
                    rm->applyCropFilter(*stroke);
                }
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
                    local_x, local_y, depth, rm->getFocalLengthMm());

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
                sm->getScene().setActiveSelectionGroup(static_cast<uint8_t>(group_id));
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
