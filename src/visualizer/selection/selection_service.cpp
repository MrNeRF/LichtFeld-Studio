/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_service.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rasterizer/rasterization/include/forward.h"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection_group_mask.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <limits>

namespace lfs::vis {

    namespace {
        constexpr float POLYGON_CLOSE_DISTANCE_PX = 12.0f;
        constexpr float POLYGON_CURSOR_APPEND_EPSILON_PX = 0.5f;
        constexpr float POLYGON_VERTEX_HIT_RADIUS_PX = 8.0f;
        constexpr float POLYGON_EDGE_HIT_RADIUS_PX = 10.0f;

        [[nodiscard]] glm::vec2 screenToRender(const glm::vec2& screen, const SelectionService::ViewportInfo& info) {
            const float scale_x = static_cast<float>(info.render_width) / info.width;
            const float scale_y = static_cast<float>(info.render_height) / info.height;
            return {
                (screen.x - info.x) * scale_x,
                (screen.y - info.y) * scale_y,
            };
        }

        [[nodiscard]] std::vector<std::pair<float, float>> screenPointsToRender(
            const std::vector<glm::vec2>& points, const SelectionService::ViewportInfo& info) {
            std::vector<std::pair<float, float>> render_points;
            render_points.reserve(points.size());
            for (const auto& point : points) {
                const auto render = screenToRender(point, info);
                render_points.emplace_back(render.x, render.y);
            }
            return render_points;
        }

        [[nodiscard]] core::Tensor& uploadRenderPointsToBuffer(
            const std::vector<glm::vec2>& points,
            const SelectionService::ViewportInfo& info,
            std::vector<float>& host_buffer,
            core::Tensor& device_buffer) {
            host_buffer.resize(points.size() * 2);
            for (size_t i = 0; i < points.size(); ++i) {
                const auto render = screenToRender(points[i], info);
                host_buffer[i * 2] = render.x;
                host_buffer[i * 2 + 1] = render.y;
            }

            const bool needs_realloc = !device_buffer.is_valid() ||
                                       device_buffer.device() != core::Device::CUDA ||
                                       device_buffer.dtype() != core::DataType::Float32 ||
                                       device_buffer.shape().rank() != 2 ||
                                       device_buffer.size(0) != points.size() ||
                                       device_buffer.size(1) != 2;
            if (needs_realloc) {
                device_buffer = core::Tensor::empty({points.size(), size_t{2}},
                                                    core::Device::CUDA,
                                                    core::DataType::Float32);
            }

            auto host_view = core::Tensor::from_blob(host_buffer.data(),
                                                     {points.size(), size_t{2}},
                                                     core::Device::CPU,
                                                     core::DataType::Float32);
            device_buffer.copy_from(host_view);
            return device_buffer;
        }

        [[nodiscard]] std::optional<core::Tensor> projectWorldPolygonToRenderSpace(
            const std::vector<glm::vec3>& world_points,
            const Viewport& viewport,
            const float focal_mm,
            const int render_width,
            const int render_height) {
            auto polygon = core::Tensor::empty({world_points.size(), size_t{2}},
                                               core::Device::CPU,
                                               core::DataType::Float32);
            auto* data = polygon.ptr<float>();
            const glm::mat4 vp = viewport.getProjectionMatrix(focal_mm) * viewport.getViewMatrix();
            for (size_t i = 0; i < world_points.size(); ++i) {
                const glm::vec4 clip = vp * glm::vec4(world_points[i], 1.0f);
                if (clip.w <= 0.0f) {
                    return std::nullopt;
                }
                const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                data[i * 2] = (ndc.x * 0.5f + 0.5f) * static_cast<float>(render_width);
                data[i * 2 + 1] = (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(render_height);
            }
            return polygon;
        }

        [[nodiscard]] size_t countSelected(const core::Tensor& mask) {
            if (!mask.is_valid()) {
                return 0;
            }
            const auto bool_mask = (mask.dtype() == core::DataType::Bool) ? mask : mask.to(core::DataType::Bool);
            return static_cast<size_t>(bool_mask.sum_scalar());
        }

        [[nodiscard]] float distanceSquaredToSegment(const glm::vec2 point,
                                                     const glm::vec2 segment_start,
                                                     const glm::vec2 segment_end,
                                                     float* out_t = nullptr) {
            const glm::vec2 delta = segment_end - segment_start;
            const float length_sq = glm::dot(delta, delta);
            const float t =
                (length_sq > 0.0f)
                    ? glm::clamp(glm::dot(point - segment_start, delta) / length_sq, 0.0f, 1.0f)
                    : 0.0f;
            if (out_t) {
                *out_t = t;
            }

            const glm::vec2 closest = segment_start + delta * t;
            const glm::vec2 offset = point - closest;
            return glm::dot(offset, offset);
        }

        [[nodiscard]] core::Tensor ensureCudaBoolMask(const core::Tensor& mask) {
            auto result = (mask.dtype() == core::DataType::Bool) ? mask : mask.to(core::DataType::Bool);
            if (result.device() != core::Device::CUDA) {
                result = result.cuda();
            }
            return result;
        }

        [[nodiscard]] core::Tensor& resetCudaByteScratchBuffer(core::Tensor& buffer, const size_t size) {
            const bool needs_realloc = !buffer.is_valid() ||
                                       buffer.device() != core::Device::CUDA ||
                                       buffer.dtype() != core::DataType::UInt8 ||
                                       buffer.numel() != size;
            if (needs_realloc) {
                buffer = core::Tensor::zeros({size}, core::Device::CUDA, core::DataType::UInt8);
                return buffer;
            }

            buffer.zero_();
            return buffer;
        }

        [[nodiscard]] core::Tensor& acquireSelectionOutputBuffer(std::array<core::Tensor, 2>& buffers,
                                                                 size_t& next_index,
                                                                 const size_t size) {
            auto& buffer = resetCudaByteScratchBuffer(buffers[next_index], size);
            next_index = (next_index + 1) % buffers.size();
            return buffer;
        }

        [[nodiscard]] rendering::ViewportData viewportDataFromCamera(const core::Camera& camera) {
            const auto rotation_cpu = camera.R().cpu().to(core::DataType::Float32);
            const auto position_cpu = camera.cam_position().cpu().to(core::DataType::Float32);
            const float* const rotation = rotation_cpu.ptr<float>();
            const float* const position = position_cpu.ptr<float>();

            glm::mat3 view_rotation(1.0f);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    view_rotation[col][row] = rotation[row * 3 + col];
                }
            }

            const int width = std::max(camera.image_width(), camera.camera_width());
            const int height = std::max(camera.image_height(), camera.camera_height());

            return rendering::ViewportData{
                .rotation = view_rotation,
                .translation = glm::vec3(position[0], position[1], position[2]),
                .size = glm::ivec2(width, height),
                .focal_length_mm = rendering::vFovToFocalLength(glm::degrees(camera.FoVy())),
                .orthographic = false,
                .ortho_scale = 1.0f,
            };
        }

        [[nodiscard]] rendering::ViewportData viewportDataFromViewer(
            const Viewport& viewport,
            const SelectionService::ViewportInfo& info,
            const RenderSettings& settings) {
            return rendering::ViewportData{
                .rotation = viewport.camera.R,
                .translation = viewport.camera.t,
                .size = glm::ivec2(info.render_width, info.render_height),
                .focal_length_mm = settings.focal_length_mm,
                .orthographic = settings.orthographic,
                .ortho_scale = settings.ortho_scale,
            };
        }

        template <typename RenderableT>
        [[nodiscard]] const RenderableT* findRenderableByNodeId(const std::vector<RenderableT>& items,
                                                                const core::NodeId node_id) {
            if (node_id == core::NULL_NODE) {
                return nullptr;
            }
            const auto it = std::find_if(items.begin(), items.end(),
                                         [node_id](const auto& item) { return item.node_id == node_id; });
            return (it == items.end()) ? nullptr : &(*it);
        }

    } // namespace

    SelectionService::SelectionService(SceneManager* scene_manager, RenderingManager* rendering_manager)
        : scene_manager_(scene_manager),
          rendering_manager_(rendering_manager) {
        assert(scene_manager_);
        assert(rendering_manager_);
    }

    SelectionService::~SelectionService() {
        if (hovered_depth_id_device_) {
            cudaFree(hovered_depth_id_device_);
            hovered_depth_id_device_ = nullptr;
        }
        if (hovered_depth_id_host_) {
            cudaFreeHost(hovered_depth_id_host_);
            hovered_depth_id_host_ = nullptr;
        }
    }

    SelectionResult SelectionService::selectBrush(float x, float y, float radius, SelectionMode mode,
                                                  int camera_index) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto screen_positions = resolveCommandScreenPositions(camera_index);
        if (!screen_positions || !screen_positions->is_valid()) {
            return {false, 0, "No screen positions"};
        }

        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, screen_positions->size(0));
        rendering::brush_select_tensor(*screen_positions, x, y, radius, selection);
        return commitSelection(selection, mode, effectiveNodeMask(true), defaultFilterState(), "selection.brush");
    }

    SelectionResult SelectionService::selectRect(float x0, float y0, float x1, float y1, SelectionMode mode,
                                                 int camera_index) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto screen_positions = resolveCommandScreenPositions(camera_index);
        if (!screen_positions || !screen_positions->is_valid()) {
            return {false, 0, "No screen positions"};
        }

        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, screen_positions->size(0));
        rendering::rect_select_tensor(*screen_positions, x0, y0, x1, y1, selection);
        return commitSelection(selection, mode, effectiveNodeMask(true), defaultFilterState(), "selection.rect");
    }

    SelectionResult SelectionService::selectPolygon(const core::Tensor& vertices, SelectionMode mode,
                                                    int camera_index) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto screen_positions = resolveCommandScreenPositions(camera_index);
        if (!screen_positions || !screen_positions->is_valid()) {
            return {false, 0, "No screen positions"};
        }
        if (!vertices.is_valid() || vertices.size(0) < 3) {
            return {false, 0, "Polygon requires at least 3 vertices"};
        }

        const auto polygon = (vertices.device() == core::Device::CUDA) ? vertices : vertices.cuda();
        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, screen_positions->size(0));
        rendering::polygon_select_tensor(*screen_positions, polygon, selection);
        return commitSelection(selection, mode, effectiveNodeMask(true), defaultFilterState(), "selection.polygon");
    }

    SelectionResult SelectionService::selectLasso(const core::Tensor& vertices, const SelectionMode mode,
                                                  const int camera_index) {
        auto result = selectPolygon(vertices, mode, camera_index);
        if (result.success) {
            return result;
        }
        if (result.error == "Polygon requires at least 3 vertices") {
            result.error = "Lasso requires at least 3 points";
        }
        return result;
    }

    SelectionResult SelectionService::selectRing(const float x, const float y, const SelectionMode mode,
                                                 const int camera_index) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto filters = defaultFilterState();
        const auto hovered_id = resolveCommandHoveredGaussianId(x, y, camera_index, filters);
        if (!hovered_id) {
            return {false, 0, "No hovered gaussian"};
        }

        const size_t total = scene_manager_->getScene().getTotalGaussianCount();
        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, total);
        rendering::set_selection_element(selection.ptr<bool>(), *hovered_id, true);
        return commitSelection(selection, mode, effectiveNodeMask(true), filters, "selection.ring");
    }

    SelectionResult SelectionService::applyMask(const std::vector<uint8_t>& mask, SelectionMode mode) {
        if (!scene_manager_) {
            return {false, 0, "Missing scene manager"};
        }

        const size_t total = scene_manager_->getScene().getTotalGaussianCount();
        if (total == 0 || mask.size() != total) {
            return {false, 0, "Mask size mismatch"};
        }

        auto tensor_mask = core::Tensor::empty({total}, core::Device::CPU, core::DataType::UInt8);
        std::memcpy(tensor_mask.ptr<uint8_t>(), mask.data(), mask.size() * sizeof(uint8_t));
        return applyMask(tensor_mask, mode);
    }

    SelectionResult SelectionService::applyMask(const core::Tensor& mask, SelectionMode mode) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const size_t total = scene_manager_->getScene().getTotalGaussianCount();
        if (total == 0 || mask.numel() != total) {
            return {false, 0, "Mask size mismatch"};
        }

        return commitSelection(mask, mode, {}, SelectionFilterState{}, "selection.mask");
    }

    void SelectionService::beginStroke() {
        if (!scene_manager_) {
            return;
        }

        const size_t n = scene_manager_->getScene().getTotalGaussianCount();
        if (n == 0) {
            return;
        }

        const auto existing = scene_manager_->getScene().getSelectionMask();
        selection_before_stroke_ =
            (existing && existing->is_valid()) ? std::make_shared<core::Tensor>(existing->clone()) : nullptr;

        (void)resetBoolScratchBuffer(stroke_selection_, n);
        stroke_active_ = true;
    }

    core::Tensor* SelectionService::getStrokeSelection() {
        return stroke_active_ ? &stroke_selection_ : nullptr;
    }

    SelectionResult SelectionService::finalizeStroke(SelectionMode mode, const std::vector<bool>& node_mask) {
        if (!stroke_active_ || !stroke_selection_.is_valid()) {
            return {false, 0, "No active stroke"};
        }

        const auto result = commitSelection(stroke_selection_, mode,
                                            node_mask.empty() ? effectiveNodeMask(true) : node_mask,
                                            defaultFilterState(), "selection.stroke");

        selection_before_stroke_.reset();
        stroke_selection_ = core::Tensor();
        stroke_active_ = false;

        if (rendering_manager_) {
            rendering_manager_->clearSelectionPreviews();
            rendering_manager_->markDirty(DirtyFlag::SELECTION);
        }

        return result;
    }

    void SelectionService::cancelStroke() {
        if (selection_before_stroke_ && scene_manager_) {
            scene_manager_->getScene().setSelectionMask(std::make_shared<core::Tensor>(selection_before_stroke_->clone()));
        }
        selection_before_stroke_.reset();
        stroke_selection_ = core::Tensor();
        stroke_active_ = false;

        if (rendering_manager_) {
            rendering_manager_->clearSelectionPreviews();
            rendering_manager_->markDirty(DirtyFlag::SELECTION);
        }
    }

    size_t SelectionService::getTotalGaussianCount() const {
        return scene_manager_ ? scene_manager_->getScene().getTotalGaussianCount() : 0;
    }

    bool SelectionService::hasScreenPositions() const {
        const auto screen_positions = getScreenPositions();
        return screen_positions && screen_positions->is_valid();
    }

    std::shared_ptr<core::Tensor> SelectionService::getScreenPositions() const {
        if (testing_screen_positions_ && testing_screen_positions_->is_valid()) {
            return testing_screen_positions_;
        }
        return rendering_manager_ ? rendering_manager_->getScreenPositions() : nullptr;
    }

    void SelectionService::setTestingScreenPositions(std::shared_ptr<core::Tensor> screen_positions) {
        testing_screen_positions_ = std::move(screen_positions);
    }

    void SelectionService::setTestingScreenPositionsForCamera(const int camera_index,
                                                              std::shared_ptr<core::Tensor> screen_positions) {
        if (camera_index < 0) {
            return;
        }
        if (screen_positions && screen_positions->is_valid()) {
            testing_camera_screen_positions_[camera_index] = std::move(screen_positions);
            return;
        }
        testing_camera_screen_positions_.erase(camera_index);
    }

    void SelectionService::setTestingViewport(ViewportInfo viewport) {
        testing_viewport_ = std::move(viewport);
    }

    void SelectionService::setTestingHoveredGaussianId(std::optional<int> hovered_gaussian_id) {
        testing_hovered_gaussian_id_ = hovered_gaussian_id;
    }

    void SelectionService::clearTestingOverrides() {
        testing_screen_positions_.reset();
        testing_camera_screen_positions_.clear();
        testing_viewport_.reset();
        testing_hovered_gaussian_id_.reset();
    }

    bool SelectionService::beginInteractiveSelection(const SelectionShape shape, const SelectionMode mode,
                                                     const glm::vec2 start_pos, const float brush_radius,
                                                     const SelectionFilterState filters) {
        if (!scene_manager_ || !rendering_manager_) {
            return false;
        }

        const size_t total = scene_manager_->getScene().getTotalGaussianCount();
        if (total == 0) {
            return false;
        }

        cancelInteractiveSelection();

        interactive_selection_ = {};
        interactive_selection_.active = true;
        interactive_selection_.shape = shape;
        interactive_selection_.mode = mode;
        interactive_selection_.filters = filters;
        interactive_selection_.brush_radius = brush_radius;
        interactive_selection_.start_pos = start_pos;
        interactive_selection_.cursor_pos = start_pos;
        (void)resetBoolScratchBuffer(interactive_selection_.working_selection, total);

        switch (shape) {
        case SelectionShape::Brush:
        case SelectionShape::Lasso:
        case SelectionShape::Polygon:
        case SelectionShape::Rings:
            interactive_selection_.points.push_back(start_pos);
            break;
        case SelectionShape::Rectangle:
            break;
        }

        if (shape == SelectionShape::Polygon) {
            if (const auto world_point = resolveInteractivePolygonWorldPoint(start_pos)) {
                interactive_selection_.polygon_world_points.push_back(*world_point);
            }
        }

        refreshInteractivePreview();
        return true;
    }

    void SelectionService::updateInteractiveSelection(const glm::vec2 cursor_pos) {
        if (!interactive_selection_.active) {
            return;
        }

        auto& session = interactive_selection_;
        session.cursor_pos = cursor_pos;

        switch (session.shape) {
        case SelectionShape::Brush:
            if (session.points.empty() || glm::distance(session.points.back(), cursor_pos) > 1.0f) {
                session.points.push_back(cursor_pos);
            }
            break;
        case SelectionShape::Lasso:
            if (session.points.empty() || glm::distance(session.points.back(), cursor_pos) > 3.0f) {
                session.points.push_back(cursor_pos);
            }
            break;
        case SelectionShape::Rectangle:
        case SelectionShape::Rings:
            break;
        case SelectionShape::Polygon:
            if (session.dragged_polygon_vertex >= 0 &&
                static_cast<size_t>(session.dragged_polygon_vertex) < session.points.size()) {
                session.points[session.dragged_polygon_vertex] = cursor_pos;
                if (static_cast<size_t>(session.dragged_polygon_vertex) < session.polygon_world_points.size()) {
                    if (const auto world_point = resolveInteractivePolygonWorldPoint(cursor_pos)) {
                        session.polygon_world_points[session.dragged_polygon_vertex] = *world_point;
                    }
                }
            }
            break;
        }

        session.preview_dirty = true;
    }

    bool SelectionService::appendInteractivePolygonVertex(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || session.polygon_closed) {
            return false;
        }

        session.cursor_pos = point;
        glm::vec2 close_anchor = session.points.front();
        if (!session.polygon_world_points.empty()) {
            if (const auto projected = projectInteractivePolygonWorldPoint(session.polygon_world_points.front())) {
                close_anchor = *projected;
            }
        }

        if (session.points.size() >= 3 &&
            glm::distance(close_anchor, point) < POLYGON_CLOSE_DISTANCE_PX) {
            session.polygon_closed = true;
        } else {
            session.points.push_back(point);
            if (const auto world_point = resolveInteractivePolygonWorldPoint(point)) {
                session.polygon_world_points.push_back(*world_point);
            }
        }

        session.preview_dirty = true;
        return true;
    }

    bool SelectionService::beginInteractivePolygonVertexDrag(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed) {
            return false;
        }

        const int vertex = findInteractivePolygonVertexAt(point);
        if (vertex < 0) {
            return false;
        }

        session.dragged_polygon_vertex = vertex;
        session.cursor_pos = point;
        session.preview_dirty = true;
        return true;
    }

    bool SelectionService::insertInteractivePolygonVertex(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed) {
            return false;
        }

        const int edge = findInteractivePolygonEdgeAt(point);
        if (edge < 0) {
            return false;
        }

        const size_t insert_at = static_cast<size_t>(edge + 1);
        session.points.insert(session.points.begin() + static_cast<std::ptrdiff_t>(insert_at), point);

        if (!session.polygon_world_points.empty()) {
            const auto world_point = resolveInteractivePolygonWorldPoint(point);
            if (!world_point || session.polygon_world_points.size() + 1 != session.points.size()) {
                session.points.erase(session.points.begin() + static_cast<std::ptrdiff_t>(insert_at));
                return false;
            }

            session.polygon_world_points.insert(
                session.polygon_world_points.begin() + static_cast<std::ptrdiff_t>(insert_at), *world_point);
        }

        session.dragged_polygon_vertex = static_cast<int>(insert_at);
        session.cursor_pos = point;
        session.preview_dirty = true;
        return true;
    }

    void SelectionService::endInteractivePolygonVertexDrag() {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon) {
            return;
        }
        session.dragged_polygon_vertex = -1;
        session.preview_dirty = true;
    }

    bool SelectionService::removeInteractivePolygonVertex(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed ||
            session.points.size() <= 3) {
            return false;
        }

        const int vertex = findInteractivePolygonVertexAt(point);
        if (vertex < 0) {
            return false;
        }

        if (!session.polygon_world_points.empty()) {
            if (static_cast<size_t>(vertex) >= session.polygon_world_points.size()) {
                return false;
            }
            session.polygon_world_points.erase(
                session.polygon_world_points.begin() + static_cast<std::ptrdiff_t>(vertex));
        }
        session.points.erase(session.points.begin() + static_cast<std::ptrdiff_t>(vertex));

        session.dragged_polygon_vertex = -1;
        session.cursor_pos = point;
        session.preview_dirty = true;
        return true;
    }

    bool SelectionService::undoInteractivePolygonVertex() {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || session.points.empty()) {
            return false;
        }
        if (session.points.size() <= 1) {
            return false;
        }

        session.points.pop_back();
        if (session.polygon_world_points.size() >= session.points.size() + 1) {
            session.polygon_world_points.pop_back();
        }
        if (session.dragged_polygon_vertex >= static_cast<int>(session.points.size())) {
            session.dragged_polygon_vertex = -1;
        }
        session.polygon_closed = false;
        session.preview_dirty = true;
        return true;
    }

    SelectionResult SelectionService::finishInteractiveSelection() {
        auto& session = interactive_selection_;
        if (!session.active) {
            return {false, 0, "No active interactive selection"};
        }

        core::Tensor selection;
        if (!buildSelectionMaskForInteractiveSession(selection)) {
            return {false, 0, "Interactive selection is incomplete"};
        }

        const auto result = commitSelection(selection, session.mode,
                                            effectiveNodeMask(session.filters.restrict_to_selected_nodes),
                                            session.filters, "select.stroke");
        clearInteractivePreviewState();
        interactive_selection_ = {};
        return result;
    }

    void SelectionService::cancelInteractiveSelection() {
        clearInteractivePreviewState();
        interactive_selection_ = {};
    }

    void SelectionService::refreshInteractivePreview() {
        auto& session = interactive_selection_;
        if (!session.active || !rendering_manager_) {
            return;
        }

        const bool continuous_refresh = (session.shape == SelectionShape::Polygon) ||
                                        (session.shape == SelectionShape::Rings);
        if (!session.preview_dirty && !continuous_refresh) {
            return;
        }

        const auto info = resolveViewportInfo();
        if (!info) {
            return;
        }

        rendering_manager_->clearRectPreview();
        rendering_manager_->clearPolygonPreview();
        rendering_manager_->clearLassoPreview();
        rendering_manager_->clearPreviewSelection();
        if (session.shape != SelectionShape::Brush && session.shape != SelectionShape::Rings) {
            rendering_manager_->clearBrushState();
        }

        const bool add_mode = (session.mode != SelectionMode::Remove);
        switch (session.shape) {
        case SelectionShape::Brush: {
            const auto render_cursor = screenToRender(session.cursor_pos, *info);
            const float radius = session.brush_radius * (static_cast<float>(info->render_width) / info->width);
            rendering_manager_->setBrushState(true, render_cursor.x, render_cursor.y, radius, add_mode, nullptr);
            break;
        }
        case SelectionShape::Rectangle: {
            const auto render_start = screenToRender(session.start_pos, *info);
            const auto render_end = screenToRender(session.cursor_pos, *info);
            rendering_manager_->setRectPreview(render_start.x, render_start.y, render_end.x, render_end.y, add_mode);
            break;
        }
        case SelectionShape::Polygon: {
            if (!session.polygon_world_points.empty()) {
                rendering_manager_->setPolygonPreviewWorldSpace(
                    session.polygon_world_points,
                    shouldClosePolygonPreview(),
                    add_mode);
            } else {
                rendering_manager_->setPolygonPreview(
                    screenPointsToRender(getPolygonPreviewPoints(), *info),
                    shouldClosePolygonPreview(),
                    add_mode);
            }
            break;
        }
        case SelectionShape::Lasso:
            rendering_manager_->setLassoPreview(screenPointsToRender(session.points, *info), add_mode);
            break;
        case SelectionShape::Rings: {
            const auto render_cursor = screenToRender(session.cursor_pos, *info);
            rendering_manager_->setBrushState(true, render_cursor.x, render_cursor.y, 0.0f, add_mode, nullptr);
            break;
        }
        }

        core::Tensor selection;
        if (buildSelectionMaskForInteractiveSession(selection, true)) {
            rendering_manager_->setPreviewSelection(&interactive_selection_.working_selection, add_mode);
        }

        rendering_manager_->markDirty(DirtyFlag::SELECTION);
        session.preview_dirty = false;
    }

    SelectionResult SelectionService::commitSelection(const core::Tensor& selection, const SelectionMode mode,
                                                      const std::vector<bool>& node_mask,
                                                      const SelectionFilterState& filters,
                                                      const char* undo_name) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        auto selection_mask = ensureCudaBoolMask(selection);
        if (!selection_mask.is_valid()) {
            return {false, 0, "Invalid selection mask"};
        }

        applyFilters(selection_mask, filters, node_mask);

        auto& scene = scene_manager_->getScene();
        const auto existing_mask = scene.getSelectionMask();
        const size_t n = selection_mask.numel();
        const uint8_t group_id = scene.getActiveSelectionGroup();

        auto locked_groups = selection::upload_locked_group_mask(scene, locked_groups_device_mask_);
        if (!locked_groups) {
            return {false, 0, locked_groups.error()};
        }

        const core::Tensor empty_mask;
        const auto& existing_ref = (existing_mask && existing_mask->is_valid()) ? *existing_mask : empty_mask;
        const auto transform_indices = scene.getTransformIndices();
        const bool add_mode = (mode != SelectionMode::Remove);
        const bool replace_mode = (mode == SelectionMode::Replace);
        auto& output_mask = acquireSelectionOutputBuffer(selection_output_buffers_, selection_output_buffer_index_, n);

        rendering::apply_selection_group_tensor_mask(
            selection_mask, existing_ref, output_mask, group_id, *locked_groups,
            add_mode, transform_indices.get(), node_mask, replace_mode);

        auto entry = std::make_unique<op::SceneSnapshot>(*scene_manager_, undo_name);
        entry->captureSelection();

        auto new_selection = std::make_shared<core::Tensor>(output_mask);
        scene.setSelectionMask(new_selection);

        entry->captureAfter();
        op::undoHistory().push(std::move(entry));

        rendering_manager_->markDirty(DirtyFlag::SELECTION);
        return {true, countSelected(*new_selection), {}};
    }

    std::shared_ptr<core::Tensor> SelectionService::resolveCommandScreenPositions(const int camera_index) const {
        if (camera_index >= 0) {
            if (const auto it = testing_camera_screen_positions_.find(camera_index);
                it != testing_camera_screen_positions_.end() &&
                it->second &&
                it->second->is_valid()) {
                return it->second;
            }
            if (auto remote_positions = renderScreenPositionsForCamera(camera_index);
                remote_positions && remote_positions->is_valid()) {
                return remote_positions;
            }
        }
        return getScreenPositions();
    }

    std::shared_ptr<core::Tensor> SelectionService::renderScreenPositionsForCamera(const int camera_index) const {
        if (!scene_manager_ || !rendering_manager_ || camera_index < 0) {
            return nullptr;
        }

        auto cameras = scene_manager_->getScene().getAllCameras();
        if (camera_index >= static_cast<int>(cameras.size()) || !cameras[camera_index]) {
            return nullptr;
        }

        auto* const engine = rendering_manager_->getRenderingEngine();
        if (!engine || !engine->isInitialized()) {
            return nullptr;
        }

        auto scene_state = scene_manager_->buildRenderState();
        if (!scene_state.combined_model || scene_state.combined_model->size() == 0) {
            return nullptr;
        }

        const auto settings = rendering_manager_->getSettings();
        rendering::RenderRequest request{
            .viewport = viewportDataFromCamera(*cameras[camera_index]),
            .scaling_modifier = settings.scaling_modifier,
            .antialiasing = settings.antialiasing,
            .mip_filter = settings.mip_filter,
            .sh_degree = scene_state.combined_model->get_active_sh_degree(),
            .background_color = settings.background_color,
            .crop_box = std::nullopt,
            .show_rings = false,
            .ring_width = settings.ring_width,
            .show_center_markers = false,
            .model_transforms = &scene_state.model_transforms,
            .transform_indices = scene_state.transform_indices,
            .selection_mask = nullptr,
            .output_screen_positions = true,
            .ellipsoid = std::nullopt,
            .depth_filter = std::nullopt,
            .selected_node_mask = {},
            .node_visibility_mask = scene_state.node_visibility_mask,
        };

        auto render_result = engine->renderGaussians(*scene_state.combined_model, request);
        if (!render_result) {
            LOG_WARN("SelectionService: failed to render screen positions for camera {}: {}",
                     camera_index, render_result.error());
            return nullptr;
        }

        if (!render_result->screen_positions || !render_result->screen_positions->is_valid()) {
            LOG_WARN("SelectionService: no screen positions produced for camera {}", camera_index);
            return nullptr;
        }

        return render_result->screen_positions;
    }

    std::optional<int> SelectionService::resolveCommandHoveredGaussianId(const float x, const float y,
                                                                         const int camera_index,
                                                                         const SelectionFilterState& filters) {
        if (testing_hovered_gaussian_id_.has_value()) {
            return testing_hovered_gaussian_id_;
        }

        if (camera_index >= 0) {
            if (auto hovered_id = renderHoveredGaussianIdForCamera(x, y, camera_index, filters);
                hovered_id.has_value()) {
                return hovered_id;
            }
        }

        return renderHoveredGaussianIdForCurrentViewport(x, y, filters);
    }

    std::optional<int> SelectionService::renderHoveredGaussianIdForCamera(const float x, const float y,
                                                                          const int camera_index,
                                                                          const SelectionFilterState& filters) {
        if (!scene_manager_ || camera_index < 0) {
            return std::nullopt;
        }

        auto cameras = scene_manager_->getScene().getAllCameras();
        if (camera_index >= static_cast<int>(cameras.size()) || !cameras[camera_index]) {
            return std::nullopt;
        }

        return renderHoveredGaussianId(viewportDataFromCamera(*cameras[camera_index]), {x, y}, filters);
    }

    std::optional<int> SelectionService::renderHoveredGaussianIdForCurrentViewport(
        const float x, const float y, const SelectionFilterState& filters) {
        auto* const gm = services().guiOrNull();
        if (!gm || !gm->getViewer() || !rendering_manager_) {
            return std::nullopt;
        }

        const auto info = resolveViewportInfo();
        if (!info) {
            return std::nullopt;
        }

        const auto settings = rendering_manager_->getSettings();
        return renderHoveredGaussianId(
            viewportDataFromViewer(gm->getViewer()->getViewport(), *info, settings),
            {x, y},
            filters);
    }

    bool SelectionService::ensureHoveredDepthBuffersAllocated() {
        if (!hovered_depth_id_device_) {
            if (cudaMalloc(&hovered_depth_id_device_, sizeof(unsigned long long)) != cudaSuccess) {
                LOG_WARN("SelectionService: failed to allocate hovered depth device buffer");
                hovered_depth_id_device_ = nullptr;
                return false;
            }
        }

        if (!hovered_depth_id_host_) {
            if (cudaMallocHost(&hovered_depth_id_host_, sizeof(unsigned long long)) != cudaSuccess) {
                LOG_WARN("SelectionService: failed to allocate hovered depth host buffer");
                cudaFree(hovered_depth_id_device_);
                hovered_depth_id_device_ = nullptr;
                hovered_depth_id_host_ = nullptr;
                return false;
            }
        }

        return true;
    }

    std::optional<int> SelectionService::renderHoveredGaussianId(const rendering::ViewportData& viewport,
                                                                 const glm::vec2 cursor_pos,
                                                                 const SelectionFilterState& filters) {
        if (!scene_manager_ || !rendering_manager_) {
            return std::nullopt;
        }

        auto* const engine = rendering_manager_->getRenderingEngine();
        if (!engine || !engine->isInitialized()) {
            return std::nullopt;
        }
        if (!ensureHoveredDepthBuffersAllocated()) {
            return std::nullopt;
        }

        auto scene_state = scene_manager_->buildRenderState();
        if (!scene_state.combined_model || scene_state.combined_model->size() == 0) {
            return std::nullopt;
        }

        const auto settings = rendering_manager_->getSettings();
        rendering::RenderRequest request{
            .viewport = viewport,
            .scaling_modifier = settings.scaling_modifier,
            .antialiasing = settings.antialiasing,
            .mip_filter = settings.mip_filter,
            .sh_degree = scene_state.combined_model->get_active_sh_degree(),
            .background_color = settings.background_color,
            .crop_box = std::nullopt,
            .show_rings = false,
            .ring_width = settings.ring_width,
            .show_center_markers = false,
            .model_transforms = &scene_state.model_transforms,
            .transform_indices = scene_state.transform_indices,
            .selection_mask = nullptr,
            .output_screen_positions = false,
            .brush_active = true,
            .brush_x = cursor_pos.x,
            .brush_y = cursor_pos.y,
            .brush_radius = 0.0f,
            .brush_add_mode = true,
            .brush_selection_tensor = nullptr,
            .brush_saturation_mode = false,
            .brush_saturation_amount = 0.0f,
            .selection_mode_rings = true,
            .ellipsoid = std::nullopt,
            .depth_filter = std::nullopt,
            .selected_node_mask = {},
            .node_visibility_mask = scene_state.node_visibility_mask,
            .desaturate_unselected = false,
            .selection_flash_intensity = 0.0f,
            .hovered_depth_id = hovered_depth_id_device_,
            .highlight_gaussian_id = -1,
            .far_plane = settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE,
            .orthographic = viewport.orthographic,
            .ortho_scale = viewport.ortho_scale,
        };

        if (filters.crop_filter) {
            const auto& scene = scene_manager_->getScene();
            const auto& cropboxes = scene.getVisibleCropBoxes();
            if (const auto* const cb = findRenderableByNodeId(cropboxes, scene_manager_->getActiveSelectionCropBoxId());
                cb && cb->data) {
                request.crop_box = rendering::BoundingBox{
                    .min = cb->data->min,
                    .max = cb->data->max,
                    .transform = glm::inverse(cb->world_transform),
                };
                request.crop_inverse = cb->data->inverse;
                request.crop_parent_node_index = scene.getVisibleNodeIndex(cb->parent_splat_id);
            }

            const auto& ellipsoids = scene.getVisibleEllipsoids();
            if (const auto* const el = findRenderableByNodeId(ellipsoids, scene_manager_->getActiveSelectionEllipsoidId());
                el && el->data) {
                request.ellipsoid = rendering::Ellipsoid{
                    .radii = el->data->radii,
                    .transform = glm::inverse(el->world_transform),
                };
                request.ellipsoid_inverse = el->data->inverse;
                request.ellipsoid_parent_node_index = scene.getVisibleNodeIndex(el->parent_splat_id);
            }
        }

        if (filters.depth_filter && settings.depth_filter_enabled) {
            request.depth_filter = rendering::BoundingBox{
                .min = settings.depth_filter_min,
                .max = settings.depth_filter_max,
                .transform = settings.depth_filter_transform.inv().toMat4(),
            };
        }

        constexpr auto NO_HOVERED_RESULT = std::numeric_limits<unsigned long long>::max();
        if (cudaMemset(hovered_depth_id_device_, 0xFF, sizeof(unsigned long long)) != cudaSuccess) {
            LOG_WARN("SelectionService: failed to reset hovered depth buffer");
            return std::nullopt;
        }

        auto render_result = engine->renderGaussians(*scene_state.combined_model, request);
        if (!render_result) {
            LOG_WARN("SelectionService: failed to render hovered gaussian id: {}", render_result.error());
            return std::nullopt;
        }

        if (cudaMemcpy(hovered_depth_id_host_, hovered_depth_id_device_,
                       sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
            LOG_WARN("SelectionService: failed to read back hovered gaussian id");
            return std::nullopt;
        }

        const unsigned long long packed = *hovered_depth_id_host_;
        if (packed == NO_HOVERED_RESULT) {
            return std::nullopt;
        }

        const int hovered_id = static_cast<int>(packed & 0xFFFFFFFFu);
        if (hovered_id < 0 ||
            static_cast<size_t>(hovered_id) >= scene_manager_->getScene().getTotalGaussianCount()) {
            return std::nullopt;
        }
        return hovered_id;
    }

    core::Tensor& SelectionService::resetBoolScratchBuffer(core::Tensor& buffer, const size_t size) {
        const bool needs_realloc = !buffer.is_valid() ||
                                   buffer.device() != core::Device::CUDA ||
                                   buffer.dtype() != core::DataType::Bool ||
                                   buffer.numel() != size;
        if (needs_realloc) {
            buffer = core::Tensor::zeros({size}, core::Device::CUDA, core::DataType::Bool);
            return buffer;
        }

        buffer.zero_();
        return buffer;
    }

    std::optional<SelectionService::ViewportInfo> SelectionService::resolveViewportInfo() const {
        if (testing_viewport_ && testing_viewport_->valid()) {
            return testing_viewport_;
        }

        auto* const gm = services().guiOrNull();
        if (!rendering_manager_ || !gm || !gm->getViewer()) {
            return std::nullopt;
        }

        const auto pos = gm->getViewportPos();
        const auto size = gm->getViewportSize();
        const auto& viewport = gm->getViewer()->getViewport();
        const auto& cached = rendering_manager_->getCachedResult();

        ViewportInfo info{
            .x = pos.x,
            .y = pos.y,
            .width = size.x,
            .height = size.y,
            .render_width = cached.image
                                ? static_cast<int>(cached.image->size(2))
                                : static_cast<int>(viewport.windowSize.x),
            .render_height = cached.image
                                 ? static_cast<int>(cached.image->size(1))
                                 : static_cast<int>(viewport.windowSize.y),
        };
        if (!info.valid()) {
            return std::nullopt;
        }
        return info;
    }

    bool SelectionService::buildSelectionMaskForInteractiveSession(core::Tensor& selection_out,
                                                                   const bool include_polygon_cursor) {
        auto& session = interactive_selection_;
        if (!session.active || !scene_manager_ || !rendering_manager_) {
            return false;
        }

        const size_t total = scene_manager_->getScene().getTotalGaussianCount();
        if (total == 0) {
            return false;
        }

        selection_out = resetBoolScratchBuffer(session.working_selection, total);

        bool success = false;
        switch (session.shape) {
        case SelectionShape::Brush:
            success = buildBrushSelection(session.points, session.brush_radius, selection_out);
            break;
        case SelectionShape::Rectangle:
            success = buildRectangleSelection(session.start_pos, session.cursor_pos, selection_out);
            break;
        case SelectionShape::Polygon: {
            if (!session.polygon_world_points.empty()) {
                success = buildWorldPolygonSelection(session.polygon_world_points, selection_out);
            } else {
                const auto polygon_points = include_polygon_cursor ? getPolygonPreviewPoints() : session.points;
                success = buildPolygonSelection(polygon_points, selection_out);
            }
            break;
        }
        case SelectionShape::Lasso:
            success = buildPolygonSelection(session.points, selection_out);
            break;
        case SelectionShape::Rings:
            success = buildRingSelection(session.cursor_pos, selection_out);
            break;
        }

        if (!success) {
            return false;
        }

        applyFilters(selection_out, session.filters, effectiveNodeMask(session.filters.restrict_to_selected_nodes));
        return true;
    }

    bool SelectionService::buildBrushSelection(const std::vector<glm::vec2>& points, const float radius,
                                               core::Tensor& selection_out) const {
        if (points.empty()) {
            return false;
        }

        const auto screen_positions = getScreenPositions();
        const auto info = resolveViewportInfo();
        if (!screen_positions || !screen_positions->is_valid() || !info) {
            return false;
        }

        const float scale_x = static_cast<float>(info->render_width) / info->width;
        const float scaled_radius = radius * scale_x;
        constexpr float STEP_FACTOR = 0.5f;

        for (size_t i = 0; i < points.size(); ++i) {
            const glm::vec2 from = (i == 0) ? points[i] : points[i - 1];
            const glm::vec2 to = points[i];
            const glm::vec2 delta = to - from;
            constexpr int MAX_BRUSH_STEPS = 128;
            const float step_spacing = std::max(radius * STEP_FACTOR, 1.0f);
            const int num_steps = std::min(
                MAX_BRUSH_STEPS,
                std::max(1, static_cast<int>(std::ceil(glm::length(delta) / step_spacing))));
            for (int step = 0; step < num_steps; ++step) {
                const float t = (num_steps == 1) ? 1.0f : static_cast<float>(step + 1) / static_cast<float>(num_steps);
                const glm::vec2 sample = from + delta * t;
                const auto render = screenToRender(sample, *info);
                rendering::brush_select_tensor(*screen_positions, render.x, render.y, scaled_radius, selection_out);
            }
        }

        return true;
    }

    bool SelectionService::buildRectangleSelection(const glm::vec2 start, const glm::vec2 end,
                                                   core::Tensor& selection_out) const {
        const auto screen_positions = getScreenPositions();
        const auto info = resolveViewportInfo();
        if (!screen_positions || !screen_positions->is_valid() || !info) {
            return false;
        }

        const auto render_start = screenToRender(start, *info);
        const auto render_end = screenToRender(end, *info);
        rendering::rect_select_tensor(*screen_positions,
                                      std::min(render_start.x, render_end.x),
                                      std::min(render_start.y, render_end.y),
                                      std::max(render_start.x, render_end.x),
                                      std::max(render_start.y, render_end.y),
                                      selection_out);
        return true;
    }

    bool SelectionService::buildPolygonSelection(const std::vector<glm::vec2>& points,
                                                 core::Tensor& selection_out) const {
        if (points.size() < 3) {
            return false;
        }

        const auto screen_positions = getScreenPositions();
        const auto info = resolveViewportInfo();
        if (!screen_positions || !screen_positions->is_valid() || !info) {
            return false;
        }

        const auto& polygon = uploadRenderPointsToBuffer(points, *info,
                                                         polygon_vertex_host_buffer_,
                                                         polygon_vertex_device_buffer_);
        rendering::polygon_select_tensor(*screen_positions, polygon, selection_out);
        return true;
    }

    bool SelectionService::buildWorldPolygonSelection(const std::vector<glm::vec3>& world_points,
                                                      core::Tensor& selection_out) const {
        if (world_points.size() < 3) {
            return false;
        }

        auto* const gm = services().guiOrNull();
        const auto info = resolveViewportInfo();
        const auto screen_positions = getScreenPositions();
        if (!gm || !gm->getViewer() || !info || !screen_positions || !screen_positions->is_valid() ||
            !rendering_manager_) {
            return false;
        }

        const auto polygon = projectWorldPolygonToRenderSpace(world_points,
                                                              gm->getViewer()->getViewport(),
                                                              rendering_manager_->getFocalLengthMm(),
                                                              info->render_width,
                                                              info->render_height);
        if (!polygon) {
            return false;
        }

        rendering::polygon_select_tensor(*screen_positions, polygon->cuda(), selection_out);
        return true;
    }

    bool SelectionService::buildRingSelection(const glm::vec2 /*cursor_pos*/, core::Tensor& selection_out) const {
        if (!rendering_manager_) {
            return false;
        }

        const int hovered_id = testing_hovered_gaussian_id_.value_or(rendering_manager_->getHoveredGaussianId());
        if (hovered_id < 0 || static_cast<size_t>(hovered_id) >= selection_out.numel()) {
            return false;
        }

        rendering::set_selection_element(selection_out.ptr<bool>(), hovered_id, true);
        return true;
    }

    std::vector<glm::vec2> SelectionService::getPolygonPreviewPoints() const {
        const auto& session = interactive_selection_;
        std::vector<glm::vec2> preview_points = session.points;
        if (!session.active || session.shape != SelectionShape::Polygon || session.polygon_closed) {
            return preview_points;
        }

        if (preview_points.empty()) {
            preview_points.push_back(session.cursor_pos);
            return preview_points;
        }

        glm::vec2 close_anchor = preview_points.front();
        if (const auto projected = resolveInteractivePolygonDisplayPoint(0)) {
            close_anchor = *projected;
        }

        const bool can_preview_close = preview_points.size() >= 3 &&
                                       glm::distance(close_anchor, session.cursor_pos) < POLYGON_CLOSE_DISTANCE_PX;
        if (!can_preview_close &&
            glm::distance(preview_points.back(), session.cursor_pos) > POLYGON_CURSOR_APPEND_EPSILON_PX) {
            preview_points.push_back(session.cursor_pos);
        }
        return preview_points;
    }

    std::optional<glm::vec2> SelectionService::resolveInteractivePolygonDisplayPoint(const size_t index) const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || index >= session.points.size()) {
            return std::nullopt;
        }

        if (index < session.polygon_world_points.size()) {
            if (const auto projected = projectInteractivePolygonWorldPoint(session.polygon_world_points[index])) {
                return projected;
            }
        }

        return session.points[index];
    }

    int SelectionService::findInteractivePolygonVertexAt(const glm::vec2 screen_point) const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon) {
            return -1;
        }

        const float radius_sq = POLYGON_VERTEX_HIT_RADIUS_PX * POLYGON_VERTEX_HIT_RADIUS_PX;
        for (size_t i = 0; i < session.points.size(); ++i) {
            if (const auto vertex = resolveInteractivePolygonDisplayPoint(i)) {
                const glm::vec2 delta = screen_point - *vertex;
                if (glm::dot(delta, delta) <= radius_sq) {
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    }

    int SelectionService::findInteractivePolygonEdgeAt(const glm::vec2 screen_point) const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed ||
            session.points.size() < 3) {
            return -1;
        }

        const float edge_radius_sq = POLYGON_EDGE_HIT_RADIUS_PX * POLYGON_EDGE_HIT_RADIUS_PX;
        const float vertex_radius_sq = POLYGON_VERTEX_HIT_RADIUS_PX * POLYGON_VERTEX_HIT_RADIUS_PX;
        float best_distance_sq = edge_radius_sq;
        int best_edge = -1;

        for (size_t i = 0; i < session.points.size(); ++i) {
            const size_t next = (i + 1) % session.points.size();
            const auto start = resolveInteractivePolygonDisplayPoint(i);
            const auto end = resolveInteractivePolygonDisplayPoint(next);
            if (!start || !end) {
                continue;
            }

            if (glm::dot(screen_point - *start, screen_point - *start) <= vertex_radius_sq ||
                glm::dot(screen_point - *end, screen_point - *end) <= vertex_radius_sq) {
                continue;
            }

            float t = 0.0f;
            const float distance_sq = distanceSquaredToSegment(screen_point, *start, *end, &t);
            if (distance_sq <= best_distance_sq && t > 0.0f && t < 1.0f) {
                best_distance_sq = distance_sq;
                best_edge = static_cast<int>(i);
            }
        }

        return best_edge;
    }

    std::optional<glm::vec3> SelectionService::resolveInteractivePolygonWorldPoint(const glm::vec2 screen_point) const {
        auto* const gm = services().guiOrNull();
        const auto info = resolveViewportInfo();
        if (!gm || !gm->getViewer() || !info || !rendering_manager_) {
            return std::nullopt;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        const float local_x = screen_point.x - info->x;
        const float local_y = screen_point.y - info->y;
        const float focal_length_mm = rendering_manager_->getFocalLengthMm();
        const float depth = rendering_manager_->getDepthAtPixel(
            static_cast<int>(local_x), static_cast<int>(local_y));

        if (depth > 0.0f) {
            const glm::vec3 world = viewport.unprojectPixel(local_x, local_y, depth, focal_length_mm);
            if (Viewport::isValidWorldPosition(world)) {
                return world;
            }
        }

        const float pivot_distance = glm::length(viewport.camera.pivot - viewport.camera.t);
        const float fallback_distance = pivot_distance > 0.1f ? pivot_distance : 10.0f;
        const glm::vec3 fallback_world =
            viewport.unprojectPixel(local_x, local_y, fallback_distance, focal_length_mm);
        if (Viewport::isValidWorldPosition(fallback_world)) {
            return fallback_world;
        }

        const glm::vec3 forward = glm::normalize(viewport.camera.R * glm::vec3(0.0f, 0.0f, 1.0f));
        return viewport.camera.t + forward * fallback_distance;
    }

    std::optional<glm::vec2> SelectionService::projectInteractivePolygonWorldPoint(const glm::vec3 world_point) const {
        auto* const gm = services().guiOrNull();
        const auto info = resolveViewportInfo();
        if (!gm || !gm->getViewer() || !info || !rendering_manager_) {
            return std::nullopt;
        }

        const auto& viewport = gm->getViewer()->getViewport();
        const glm::vec4 clip =
            viewport.getProjectionMatrix(rendering_manager_->getFocalLengthMm()) *
            viewport.getViewMatrix() * glm::vec4(world_point, 1.0f);
        if (clip.w <= 0.0f) {
            return std::nullopt;
        }

        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        return glm::vec2(info->x + (ndc.x * 0.5f + 0.5f) * info->width,
                         info->y + (1.0f - (ndc.y * 0.5f + 0.5f)) * info->height);
    }

    bool SelectionService::shouldClosePolygonPreview() const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon) {
            return false;
        }
        if (session.polygon_closed) {
            return true;
        }

        glm::vec2 close_anchor = session.points.front();
        if (!session.polygon_world_points.empty()) {
            if (const auto projected = projectInteractivePolygonWorldPoint(session.polygon_world_points.front())) {
                close_anchor = *projected;
            }
        }

        return session.points.size() >= 3 &&
               glm::distance(close_anchor, session.cursor_pos) < POLYGON_CLOSE_DISTANCE_PX;
    }

    void SelectionService::applyFilters(core::Tensor& selection, const SelectionFilterState& filters,
                                        const std::vector<bool>& node_mask) const {
        if (!scene_manager_ || !rendering_manager_ || !selection.is_valid()) {
            return;
        }

        if (!node_mask.empty()) {
            if (const auto transform_indices = scene_manager_->getScene().getTransformIndices();
                transform_indices && transform_indices->is_valid()) {
                rendering::filter_selection_by_node_mask(selection, *transform_indices, node_mask);
            }
        }

        rendering_manager_->applySelectionFilters(selection, filters.crop_filter, filters.depth_filter);
    }

    void SelectionService::clearInteractivePreviewState() {
        if (rendering_manager_) {
            rendering_manager_->clearSelectionPreviews();
            rendering_manager_->markDirty(DirtyFlag::SELECTION);
        }
    }

    std::vector<bool> SelectionService::effectiveNodeMask(const bool restrict_to_selected_nodes) const {
        if (!scene_manager_ || !restrict_to_selected_nodes || !scene_manager_->hasSelectedNode()) {
            return {};
        }
        return scene_manager_->getSelectedNodeMask();
    }

    SelectionFilterState SelectionService::defaultFilterState() const {
        SelectionFilterState filters{};
        if (!rendering_manager_) {
            return filters;
        }

        const auto settings = rendering_manager_->getSettings();
        filters.crop_filter = settings.crop_filter_for_selection;
        filters.depth_filter = settings.depth_filter_enabled;
        filters.restrict_to_selected_nodes = true;
        return filters;
    }

} // namespace lfs::vis
