/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include "rendering/rendering_types.hpp"
#include <array>
#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::rendering {
    struct ViewportData;
}

class Viewport;

namespace lfs::vis {

    class SceneManager;
    class RenderingManager;

    enum class SelectionMode {
        Replace,
        Add,
        Remove
    };

    enum class SelectionShape : uint8_t {
        Brush,
        Rectangle,
        Polygon,
        Lasso,
        Rings
    };

    struct SelectionResult {
        bool success = false;
        size_t affected_count = 0;
        std::string error;
    };

    struct SelectionFilterState {
        bool crop_filter = false;
        bool depth_filter = false;
        bool restrict_to_selected_nodes = true;
    };

    class LFS_VIS_API SelectionService {
    public:
        struct ViewportInfo {
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
            int render_width = 0;
            int render_height = 0;

            [[nodiscard]] bool valid() const {
                return width > 0.0f && height > 0.0f && render_width > 0 && render_height > 0;
            }
        };

        SelectionService(SceneManager* scene_manager, RenderingManager* rendering_manager);
        ~SelectionService();

        SelectionService(const SelectionService&) = delete;
        SelectionService& operator=(const SelectionService&) = delete;

        [[nodiscard]] SelectionResult selectBrush(float x, float y, float radius, SelectionMode mode,
                                                  int camera_index = 0);
        [[nodiscard]] SelectionResult selectRect(float x0, float y0, float x1, float y1, SelectionMode mode,
                                                 int camera_index = 0);
        [[nodiscard]] SelectionResult selectPolygon(const core::Tensor& vertices, SelectionMode mode,
                                                    int camera_index = 0);
        [[nodiscard]] SelectionResult selectLasso(const core::Tensor& vertices, SelectionMode mode,
                                                  int camera_index = 0);
        [[nodiscard]] SelectionResult selectRing(float x, float y, SelectionMode mode, int camera_index = 0);

        [[nodiscard]] SelectionResult applyMask(const std::vector<uint8_t>& mask, SelectionMode mode);
        [[nodiscard]] SelectionResult applyMask(const core::Tensor& mask, SelectionMode mode);

        void beginStroke();
        [[nodiscard]] core::Tensor* getStrokeSelection();
        void applyCropFilterToStroke();
        [[nodiscard]] SelectionResult finalizeStroke(SelectionMode mode, const std::vector<bool>& node_mask = {});
        void cancelStroke();

        [[nodiscard]] bool isStrokeActive() const { return stroke_active_; }
        [[nodiscard]] size_t getTotalGaussianCount() const;
        [[nodiscard]] bool hasScreenPositions() const;
        [[nodiscard]] std::shared_ptr<core::Tensor> getScreenPositions() const;

        bool beginInteractiveSelection(SelectionShape shape, SelectionMode mode, glm::vec2 start_pos,
                                       float brush_radius, SelectionFilterState filters = {});
        void updateInteractiveSelection(glm::vec2 cursor_pos);
        bool appendInteractivePolygonVertex(glm::vec2 point);
        bool insertInteractivePolygonVertex(glm::vec2 point);
        bool beginInteractivePolygonVertexDrag(glm::vec2 point);
        bool removeInteractivePolygonVertex(glm::vec2 point);
        void endInteractivePolygonVertexDrag();
        bool undoInteractivePolygonVertex();
        [[nodiscard]] SelectionResult finishInteractiveSelection();
        void cancelInteractiveSelection();
        void refreshInteractivePreview();
        [[nodiscard]] bool isInteractiveSelectionActive() const { return interactive_selection_.active; }
        [[nodiscard]] SelectionShape getInteractiveSelectionShape() const { return interactive_selection_.shape; }
        [[nodiscard]] bool isInteractiveSelectionClosed() const { return interactive_selection_.polygon_closed; }
        [[nodiscard]] bool isInteractivePolygonVertexDragActive() const {
            return interactive_selection_.dragged_polygon_vertex >= 0;
        }
        void setInteractiveSelectionMode(SelectionMode mode) { interactive_selection_.mode = mode; }
        void setTestingScreenPositions(std::shared_ptr<core::Tensor> screen_positions);
        void setTestingScreenPositionsForCamera(int camera_index, std::shared_ptr<core::Tensor> screen_positions);
        void setTestingViewport(ViewportInfo viewport);
        void setTestingHoveredGaussianId(std::optional<int> hovered_gaussian_id);
        void clearTestingOverrides();

    private:
        struct ViewerViewportContext {
            SplitViewPanelId panel = SplitViewPanelId::Left;
            ViewportInfo info;
            const Viewport* viewport = nullptr;

            [[nodiscard]] bool valid() const { return viewport != nullptr && info.valid(); }
        };

        struct InteractiveSelectionState {
            bool active = false;
            SelectionShape shape = SelectionShape::Brush;
            SelectionMode mode = SelectionMode::Replace;
            SelectionFilterState filters{};
            float brush_radius = 20.0f;
            glm::vec2 start_pos{0.0f};
            glm::vec2 cursor_pos{0.0f};
            std::optional<ViewerViewportContext> viewport_context;
            std::vector<glm::vec2> points;
            std::vector<glm::vec3> polygon_world_points;
            bool polygon_closed = false;
            int dragged_polygon_vertex = -1;
            bool preview_dirty = false;
            core::Tensor working_selection;
        };

        [[nodiscard]] SelectionResult commitSelection(const core::Tensor& selection, SelectionMode mode,
                                                      const std::vector<bool>& node_mask,
                                                      const SelectionFilterState& filters,
                                                      const char* undo_name);
        [[nodiscard]] core::Tensor& resetBoolScratchBuffer(core::Tensor& buffer, size_t size);
        [[nodiscard]] std::optional<ViewerViewportContext> resolveViewerViewportContext(
            std::optional<glm::vec2> screen_point = std::nullopt,
            std::optional<SplitViewPanelId> panel_override = std::nullopt) const;
        [[nodiscard]] std::optional<ViewportInfo> resolveViewportInfo() const;
        [[nodiscard]] std::shared_ptr<core::Tensor> getScreenPositionsForContext(
            const ViewerViewportContext& context) const;
        [[nodiscard]] std::shared_ptr<core::Tensor> resolveCommandScreenPositions(int camera_index) const;
        [[nodiscard]] std::shared_ptr<core::Tensor> renderScreenPositionsForCamera(int camera_index) const;
        [[nodiscard]] std::shared_ptr<core::Tensor> renderScreenPositionsForViewerContext(
            const ViewerViewportContext& context) const;
        [[nodiscard]] std::shared_ptr<core::Tensor> renderScreenPositionsForCurrentViewport() const;
        [[nodiscard]] std::optional<int> resolveCommandHoveredGaussianId(float x, float y, int camera_index,
                                                                         const SelectionFilterState& filters);
        [[nodiscard]] std::optional<int> renderHoveredGaussianIdForViewerContext(
            const ViewerViewportContext& context,
            glm::vec2 cursor_pos,
            const SelectionFilterState& filters) const;
        [[nodiscard]] std::optional<int> renderHoveredGaussianId(const rendering::ViewportData& viewport,
                                                                 glm::vec2 cursor_pos,
                                                                 const SelectionFilterState& filters) const;
        [[nodiscard]] std::optional<int> renderHoveredGaussianIdForCamera(float x, float y, int camera_index,
                                                                          const SelectionFilterState& filters);
        [[nodiscard]] std::optional<int> renderHoveredGaussianIdForCurrentViewport(float x, float y,
                                                                                   const SelectionFilterState& filters);
        [[nodiscard]] bool buildSelectionMaskForInteractiveSession(core::Tensor& selection_out,
                                                                   bool include_polygon_cursor = false);
        [[nodiscard]] bool buildBrushSelection(const std::vector<glm::vec2>& points, float radius,
                                               core::Tensor& selection_out) const;
        [[nodiscard]] bool buildRectangleSelection(glm::vec2 start, glm::vec2 end,
                                                   core::Tensor& selection_out) const;
        [[nodiscard]] bool buildPolygonSelection(const std::vector<glm::vec2>& points,
                                                 core::Tensor& selection_out) const;
        [[nodiscard]] bool buildWorldPolygonSelection(const std::vector<glm::vec3>& world_points,
                                                      core::Tensor& selection_out) const;
        [[nodiscard]] bool buildRingSelection(glm::vec2 cursor_pos, core::Tensor& selection_out) const;
        [[nodiscard]] std::vector<glm::vec2> getPolygonPreviewPoints() const;
        [[nodiscard]] std::optional<glm::vec2> resolveInteractivePolygonDisplayPoint(size_t index) const;
        [[nodiscard]] int findInteractivePolygonVertexAt(glm::vec2 screen_point) const;
        [[nodiscard]] int findInteractivePolygonEdgeAt(glm::vec2 screen_point) const;
        [[nodiscard]] std::optional<glm::vec3> resolveInteractivePolygonWorldPoint(glm::vec2 screen_point) const;
        [[nodiscard]] std::optional<glm::vec2> projectInteractivePolygonWorldPoint(glm::vec3 world_point) const;
        [[nodiscard]] bool shouldClosePolygonPreview() const;
        void applyFilters(core::Tensor& selection, const SelectionFilterState& filters,
                          const std::vector<bool>& node_mask) const;
        void applyCropFilter(core::Tensor& selection) const;
        void applyDepthFilter(core::Tensor& selection) const;
        void clearInteractivePreviewState();
        [[nodiscard]] std::vector<bool> effectiveNodeMask(bool restrict_to_selected_nodes) const;
        [[nodiscard]] SelectionFilterState defaultFilterState() const;

        SceneManager* scene_manager_;
        RenderingManager* rendering_manager_;

        bool stroke_active_ = false;
        core::Tensor stroke_selection_;
        std::shared_ptr<core::Tensor> selection_before_stroke_;
        core::Tensor command_selection_buffer_;
        core::Tensor locked_groups_device_mask_;
        std::array<core::Tensor, 2> selection_output_buffers_;
        size_t selection_output_buffer_index_ = 0;
        std::shared_ptr<core::Tensor> testing_screen_positions_;
        std::unordered_map<int, std::shared_ptr<core::Tensor>> testing_camera_screen_positions_;
        std::optional<ViewportInfo> testing_viewport_;
        std::optional<int> testing_hovered_gaussian_id_;
        mutable std::array<std::shared_ptr<core::Tensor>, 2> viewport_screen_positions_;
        mutable std::array<uint64_t, 2> viewport_screen_positions_generation_{0, 0};
        mutable std::vector<float> polygon_vertex_host_buffer_;
        mutable core::Tensor polygon_vertex_device_buffer_;

        InteractiveSelectionState interactive_selection_;
    };

} // namespace lfs::vis
