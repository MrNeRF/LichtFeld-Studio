/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_types.hpp"
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <expected>
#include <glm/mat4x4.hpp>
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

    namespace op {
        class SceneSnapshot;
    }

    class SceneManager;
    class RenderingManager;

    enum class SelectionMode {
        Replace,
        Add,
        Remove,
        Intersect
    };

    enum class SelectionShape : uint8_t {
        Brush,
        Rectangle,
        Polygon,
        Lasso,
        Rings,
        Box,
        Sphere
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

    struct SelectionProjectionContext {
        struct ViewerLayout {
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
            int render_width = 0;
            int render_height = 0;

            [[nodiscard]] bool valid() const {
                return width > 0.0f && height > 0.0f && render_width > 0 && render_height > 0;
            }

            [[nodiscard]] friend bool operator==(const ViewerLayout&, const ViewerLayout&) = default;
        };

        rendering::ViewportData viewport{};
        bool equirectangular = false;
        float far_plane = rendering::DEFAULT_FAR_PLANE;
        // Real unjittered intrinsics of the displayed camera for depth-window containment.
        // Empty means "image-centred with FoV-derived focals", which is correct for every
        // synthetic-intrinsics viewport. Populated for the explicit-camera lane (M2) and the
        // GT compare panel (M3).
        std::optional<rendering::CameraIntrinsics> containment_intrinsics;
        std::optional<ViewerLayout> viewer_layout;
        std::optional<SplitViewPanelId> panel;

        [[nodiscard]] bool valid() const { return viewport.size.x > 0 && viewport.size.y > 0; }
    };

    // Typed failure for command projection resolution, following
    // PlySequenceResolveError in the sequencer: a code callers can branch on,
    // and the message a caller surfaces to the user. The codes are exactly the
    // distinct failures resolveCommandProjectionSnapshot can return.
    enum class SelectionProjectionErrorCode : uint8_t {
        CAMERA_PROJECTION_UNAVAILABLE,
        CAMERA_INDEX_OUT_OF_RANGE,
        VIEWER_VIEWPORT_UNAVAILABLE,
        VIEWER_PROJECTION_UNAVAILABLE,
    };

    struct SelectionProjectionError {
        SelectionProjectionErrorCode code = SelectionProjectionErrorCode::CAMERA_PROJECTION_UNAVAILABLE;
        std::string message;
    };

    struct SelectionCommitOptions {
        const core::Tensor* base_selection = nullptr;
        bool push_undo = true;
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
                                                  int camera_index = -1);
        [[nodiscard]] SelectionResult selectRect(float x0, float y0, float x1, float y1, SelectionMode mode,
                                                 int camera_index = -1);
        [[nodiscard]] SelectionResult selectPolygon(const std::vector<glm::vec2>& vertices, SelectionMode mode,
                                                    int camera_index = -1);
        [[nodiscard]] SelectionResult selectLasso(const std::vector<glm::vec2>& vertices, SelectionMode mode,
                                                  int camera_index = -1);
        [[nodiscard]] SelectionResult selectRing(float x, float y, SelectionMode mode, int camera_index = -1);
        [[nodiscard]] SelectionResult selectByColorAt(float x, float y, SelectionMode mode,
                                                      SelectionFilterState filters = {},
                                                      int camera_index = -1);
        [[nodiscard]] SelectionResult selectBoxVolume(SelectionMode mode,
                                                      SelectionCommitOptions options = {});
        [[nodiscard]] SelectionResult selectSphereVolume(SelectionMode mode,
                                                         SelectionCommitOptions options = {});
        [[nodiscard]] SelectionResult selectAllFiltered();
        [[nodiscard]] SelectionResult invertFiltered();

        [[nodiscard]] SelectionResult applyMask(const std::vector<uint8_t>& mask, SelectionMode mode);
        [[nodiscard]] SelectionResult applyMask(const core::Tensor& mask, SelectionMode mode);
        [[nodiscard]] SelectionResult previewMask(const core::Tensor& mask, SelectionMode mode);

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
        // Forces the next interactive-preview refresh to restart the incremental
        // brush filter from point zero and re-apply filters (used when the depth
        // window changes under an unchanged brush point set, e.g. a drag commit).
        void invalidateInteractiveBrushFilterCache();
        [[nodiscard]] bool isInteractiveSelectionActive() const { return interactive_selection_.active; }
        [[nodiscard]] SelectionShape getInteractiveSelectionShape() const { return interactive_selection_.shape; }
        [[nodiscard]] bool isInteractiveSelectionClosed() const { return interactive_selection_.polygon_closed; }
        [[nodiscard]] bool isInteractivePolygonVertexDragActive() const {
            return interactive_selection_.dragged_polygon_vertex >= 0;
        }
        void updatePassiveRingHoverPreview(glm::vec2 cursor_pos, SelectionMode mode,
                                           SelectionFilterState filters = {});
        void setInteractiveSelectionMode(SelectionMode mode) { interactive_selection_.mode = mode; }
        // Test-only observable for the incremental brush-preview cache: the number of
        // brush points already folded into the interactive preview. Zero after
        // invalidateInteractiveBrushFilterCache(); restored by the next refresh.
        [[nodiscard]] std::size_t interactiveBrushPreviewPointCountForTest() const {
            return interactive_selection_.preview_brush_point_count;
        }
        void setTestingScreenPositions(std::shared_ptr<core::Tensor> screen_positions);
        void setTestingScreenPositionsForCamera(int camera_index, std::shared_ptr<core::Tensor> screen_positions);
        void setTestingViewport(ViewportInfo viewport);
        void setTestingContainmentIntrinsics(std::optional<rendering::CameraIntrinsics> intrinsics);
        void setTestingHoveredGaussianId(std::optional<int> hovered_gaussian_id);
        // Applies completed GPU count readbacks without waiting. The scene
        // manager calls this once per render-state build; selection commands
        // also poll before starting a new commit.
        void pollPendingSelectionCounts() const;
        // MCP and history boundaries use this synchronous variant. The
        // histogram is only 257 integers; the interactive path remains on the
        // non-blocking poll above.
        void completePendingSelectionCounts() const;

    private:
        struct PendingSelectionCounts {
            std::shared_ptr<core::Tensor> mask;
            std::unique_ptr<op::SceneSnapshot> undo_entry;
            core::Tensor scratch;
            int* host_counts = nullptr;
            cudaEvent_t ready_event = nullptr;
            bool pending = false;
            bool apply_to_scene = true;
            uint64_t sequence = 0;
            core::Scene::SelectionStateMetadata after_metadata;
        };

        bool queueSelectionCounts(const std::shared_ptr<core::Tensor>& mask,
                                  std::unique_ptr<op::SceneSnapshot>& undo_entry,
                                  const core::Scene::SelectionStateMetadata& after_metadata) const;
        void completePendingSelectionCount(PendingSelectionCounts& pending, bool wait) const;
        bool pollPendingPassiveRingCount() const;

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
            std::optional<glm::vec3> volume_center_world;
            bool polygon_closed = false;
            int dragged_polygon_vertex = -1;
            bool preview_dirty = false;
            core::Tensor working_selection;
            core::Tensor live_delta_selection;
            std::vector<bool> live_preview_node_mask;
            size_t preview_brush_point_count = 0;
            std::size_t preview_brush_projection_signature = 0;
            bool preview_brush_projection_signature_valid = false;
            uint64_t generation = 0;
        };
        struct InteractiveVolumeGeometry {
            glm::vec3 center_world{0.0f};
            float radius = 0.0f;
            glm::mat4 visualizer_transform{1.0f};
            glm::vec3 box_min{0.0f};
            glm::vec3 box_max{0.0f};
            glm::vec3 ellipsoid_radii{0.0f};
        };
        struct ScreenPositionCacheKey {
            bool valid = false;
            std::size_t signature = 0;

            [[nodiscard]] friend bool operator==(const ScreenPositionCacheKey& a,
                                                 const ScreenPositionCacheKey& b) = default;
        };

        [[nodiscard]] SelectionResult commitSelection(const core::Tensor& selection, SelectionMode mode,
                                                      const std::vector<bool>& node_mask,
                                                      const SelectionFilterState& filters,
                                                      const SelectionProjectionContext& projection_context,
                                                      const char* undo_name,
                                                      SelectionCommitOptions options = {});
        [[nodiscard]] core::Tensor& resetBoolScratchBuffer(core::Tensor& buffer, size_t size);
        [[nodiscard]] std::optional<ViewerViewportContext> resolveViewerViewportContext(
            std::optional<glm::vec2> screen_point = std::nullopt,
            std::optional<SplitViewPanelId> panel_override = std::nullopt) const;
        [[nodiscard]] std::optional<ViewportInfo> resolveViewportInfo() const;
        [[nodiscard]] std::optional<int> resolveCommandHoveredGaussianId(float x, float y, int camera_index,
                                                                         const SelectionFilterState& filters,
                                                                         const SelectionProjectionContext& projection_context);
        [[nodiscard]] std::optional<int> renderHoveredGaussianIdForViewerContext(
            const ViewerViewportContext& context,
            glm::vec2 cursor_pos,
            const SelectionFilterState& filters,
            const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] std::optional<int> pickHoveredGaussianIdFromScreenPositions(
            const core::Tensor& screen_positions,
            glm::vec2 cursor_pos,
            const SelectionFilterState& filters,
            const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] std::optional<int> renderHoveredGaussianIdForCamera(float x, float y, int camera_index,
                                                                          const SelectionFilterState& filters,
                                                                          const SelectionProjectionContext& projection_context);
        [[nodiscard]] std::optional<int> renderHoveredGaussianIdForCurrentViewport(
            float x, float y, const SelectionFilterState& filters,
            const SelectionProjectionContext& projection_context);
        [[nodiscard]] bool buildSelectionMaskForInteractiveSession(
            core::Tensor& selection_out,
            const SelectionProjectionContext& projection_context,
            bool include_polygon_cursor = false,
            int* picked_ring_id_out = nullptr,
            bool for_commit = false);
        [[nodiscard]] bool buildInteractiveBrushPreviewIncremental(
            const SelectionProjectionContext& projection_context);
        [[nodiscard]] bool buildBrushSelection(const std::vector<glm::vec2>& points, float radius,
                                               core::Tensor& selection_out,
                                               const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] bool buildRectangleSelection(glm::vec2 start, glm::vec2 end,
                                                   core::Tensor& selection_out,
                                                   const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] bool buildPolygonSelection(const std::vector<glm::vec2>& points,
                                                 core::Tensor& selection_out,
                                                 const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] bool buildWorldPolygonSelection(const std::vector<glm::vec3>& world_points,
                                                      core::Tensor& selection_out,
                                                      const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] std::optional<bool> buildRingSelectionForContext(const SelectionProjectionContext& projection_context,
                                                                       glm::vec2 cursor_pos,
                                                                       core::Tensor& selection_out,
                                                                       int* picked_ring_id_out = nullptr) const;
        [[nodiscard]] bool buildRingSelection(glm::vec2 cursor_pos, core::Tensor& selection_out,
                                              const SelectionProjectionContext& projection_context,
                                              bool try_exact_ring_pick = true,
                                              bool require_exact_ring_hit = true,
                                              int* picked_ring_id_out = nullptr) const;
        [[nodiscard]] std::optional<InteractiveVolumeGeometry> buildInteractiveVolumeGeometry(
            const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] bool buildVolumeSelection(const InteractiveVolumeGeometry& geometry,
                                                core::Tensor& selection_out) const;
        void publishInteractiveVolumeGeometry(const InteractiveVolumeGeometry& geometry) const;
        [[nodiscard]] std::vector<glm::vec2> getPolygonPreviewPoints() const;
        [[nodiscard]] std::optional<glm::vec2> resolveInteractivePolygonDisplayPoint(size_t index) const;
        [[nodiscard]] int findInteractivePolygonVertexAt(glm::vec2 screen_point) const;
        [[nodiscard]] int findInteractivePolygonEdgeAt(glm::vec2 screen_point) const;
        [[nodiscard]] std::optional<glm::vec3> resolveInteractivePolygonWorldPoint(glm::vec2 screen_point) const;
        [[nodiscard]] std::optional<glm::vec2> projectInteractivePolygonWorldPoint(glm::vec3 world_point) const;
        [[nodiscard]] bool shouldClosePolygonPreview() const;
        [[nodiscard]] bool applyFilters(core::Tensor& selection, const SelectionFilterState& filters,
                                        const std::vector<bool>& node_mask,
                                        const SelectionProjectionContext& projection_context) const;
        void applyCropFilter(core::Tensor& selection,
                             const core::Tensor* crop_box_transform = nullptr,
                             const core::Tensor* crop_box_min = nullptr,
                             const core::Tensor* crop_box_max = nullptr,
                             const core::Tensor* ellipsoid_transform = nullptr,
                             const core::Tensor* ellipsoid_radii = nullptr,
                             bool use_scene_filters = true) const;
        void applyDepthFilter(core::Tensor& selection,
                              const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] std::optional<ViewerViewportContext> resolveInteractiveSessionViewportContext(
            std::optional<glm::vec2> screen_point = std::nullopt) const;
        [[nodiscard]] std::expected<SelectionProjectionContext, SelectionProjectionError> resolveCommandProjectionSnapshot(
            int camera_index,
            std::optional<glm::vec2> query_point = std::nullopt,
            bool skip_explicit_camera_validation = false) const;
        [[nodiscard]] std::optional<SelectionProjectionContext> projectionContextFromViewerContext(
            const ViewerViewportContext& context) const;
        [[nodiscard]] std::optional<rendering::FrameView> frameViewFromProjectionContext(
            const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] std::shared_ptr<core::Tensor> screenPositionsForCommandPass(
            int camera_index, const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] std::shared_ptr<core::Tensor> renderScreenPositionsForProjectionContext(
            const SelectionProjectionContext& projection_context) const;
        [[nodiscard]] bool hasTestingScreenPositionsForCamera(int camera_index) const;
        [[nodiscard]] bool commandCameraValidationRequired(int camera_index) const;
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
        std::array<uint32_t, 8> locked_groups_host_mask_{};
        bool locked_groups_host_mask_valid_ = false;
        mutable std::array<PendingSelectionCounts, 2> pending_selection_counts_{};
        mutable PendingSelectionCounts pending_passive_ring_count_{};
        mutable uint64_t selection_count_sequence_ = 0;
        std::array<std::shared_ptr<core::Tensor>, 4> selection_output_buffers_;
        size_t selection_output_buffer_index_ = 0;
        uint64_t interactive_selection_generation_ = 0;
        std::shared_ptr<core::Tensor> testing_screen_positions_;
        std::unordered_map<int, std::shared_ptr<core::Tensor>> testing_camera_screen_positions_;
        std::optional<ViewportInfo> testing_viewport_;
        std::optional<rendering::CameraIntrinsics> testing_containment_intrinsics_;
        std::optional<int> testing_hovered_gaussian_id_;
        mutable bool passive_ring_preview_key_valid_ = false;
        mutable std::size_t passive_ring_preview_key_ = 0;
        mutable bool passive_ring_has_hit_ = false;
        mutable std::array<std::shared_ptr<core::Tensor>, 2> viewport_screen_positions_;
        mutable std::array<ScreenPositionCacheKey, 2> viewport_screen_position_keys_{};
        mutable std::vector<float> polygon_vertex_host_buffer_;
        mutable core::Tensor polygon_vertex_device_buffer_;

        InteractiveSelectionState interactive_selection_;
    };

} // namespace lfs::vis
