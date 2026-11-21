/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_bus.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "viewport_request_builder.hpp"
#include "visualizer/app_store.hpp"
#include <algorithm>

namespace lfs::vis {

    using namespace lfs::core::events;

    namespace {
        void restoreAlignmentCamera(Viewport& viewport, const Viewport& saved) {
            viewport.camera = saved.camera;
            viewport.ortho_scale_override = saved.ortho_scale_override;
        }

        [[nodiscard]] constexpr bool hasSceneMutation(const uint32_t flags, const lfs::core::Scene::MutationType type) {
            return (flags & static_cast<uint32_t>(type)) != 0;
        }

        [[nodiscard]] constexpr DirtyMask dirtyMaskForSceneMutations(const uint32_t flags) {
            using Mutation = lfs::core::Scene::MutationType;

            if (flags == 0 || hasSceneMutation(flags, Mutation::CLEARED)) {
                return DirtyFlag::ALL;
            }

            DirtyMask dirty = 0;
            if (hasSceneMutation(flags, Mutation::NODE_ADDED) ||
                hasSceneMutation(flags, Mutation::NODE_REMOVED) ||
                hasSceneMutation(flags, Mutation::VISIBILITY_CHANGED) ||
                hasSceneMutation(flags, Mutation::MODEL_CHANGED)) {
                dirty |= DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY | DirtyFlag::SPLIT_VIEW;
            }
            if (hasSceneMutation(flags, Mutation::TRANSFORM_CHANGED) ||
                hasSceneMutation(flags, Mutation::NODE_REPARENTED)) {
                dirty |= DirtyFlag::MESH | DirtyFlag::OVERLAY;
            }
            if (hasSceneMutation(flags, Mutation::SELECTION_CHANGED)) {
                dirty |= DirtyFlag::SELECTION | DirtyFlag::OVERLAY;
            }
            if (hasSceneMutation(flags, Mutation::NODE_RENAMED)) {
                dirty |= DirtyFlag::OVERLAY | DirtyFlag::SPLIT_VIEW;
            }

            return dirty == 0 ? DirtyFlag::ALL : dirty;
        }
    } // namespace

    void RenderingManager::setupEventHandlers() {
        event_handlers_.subscribe<cmd::ToggleSplitView>([this](const auto&) { handleToggleSplitView(); });
        event_handlers_.subscribe<cmd::ToggleIndependentSplitView>(
            [this](const auto& event) { handleToggleIndependentSplitView(event); });
        event_handlers_.subscribe<cmd::ToggleGTComparison>([this](const auto&) { handleToggleGTComparison(); });
        event_handlers_.subscribe<cmd::GoToCamView>([this](const auto& event) { handleGoToCamView(event.cam_id); });
        event_handlers_.subscribe<ui::SplitPositionChanged>(
            [this](const auto& event) { handleSplitPositionChanged(event.position); });
        event_handlers_.subscribe<ui::RenderSettingsChanged>(
            [this](const auto& event) { handleRenderSettingsChanged(event); });
        event_handlers_.subscribe<ui::WindowResized>([this](const auto&) { handleWindowResized(); });
        event_handlers_.subscribe<ui::WindowResizeInteraction>(
            [this](const auto& event) { setViewportResizeActive(event.active); });
        event_handlers_.subscribe<ui::GridSettingsChanged>(
            [this](const auto& event) { handleGridSettingsChanged(event); });
        event_handlers_.subscribe<state::TrainingStarted>([this](const auto&) { handleTrainingStarted(); });
        event_handlers_.subscribe<state::TrainingCompleted>([this](const auto&) { handleTrainingCompleted(); });
        event_handlers_.subscribe<state::SceneLoaded>([this](const auto&) { handleSceneLoaded(); });
        event_handlers_.subscribe<state::SceneChanged>(
            [this](const auto& event) { handleSceneChanged(event.mutation_flags); });
        event_handlers_.subscribe<state::SceneCleared>([this](const auto&) { handleSceneCleared(); });
        event_handlers_.subscribe<cmd::SetPLYVisibility>([this](const auto&) { handlePLYVisibilityChanged(); });
        event_handlers_.subscribe<state::PLYAdded>([this](const auto&) { handlePLYAdded(); });
        event_handlers_.subscribe<state::PLYRemoved>([this](const auto&) { handlePLYRemoved(); });
        event_handlers_.subscribe<ui::CropBoxChanged>(
            [this](const auto& event) { handleCropBoxChanged(event.enabled); });
        event_handlers_.subscribe<ui::EllipsoidChanged>(
            [this](const auto& event) { handleEllipsoidChanged(event.enabled); });
        event_handlers_.subscribe<ui::PointCloudModeChanged>(
            [this](const auto& event) { handlePointCloudModeChanged(event); });
    }

    void RenderingManager::handleToggleSplitView() {
        if (modelAlignmentView())
            return;
        if (!isPLYComparisonActive()) {
            const auto* const scene_manager = services().sceneOrNull();
            if (!scene_manager || plyComparisonNodes(scene_manager->getScene()).size() < 2) {
                lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                    .error = lfs::make_error({
                        .code = lfs::ErrorCode::FailedPrecondition,
                        .domain = lfs::ErrorDomain::Rendering,
                        .severity = lfs::Severity::Info,
                        .user_message = LOC("notification.split_view_requires_two_models"),
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }),
                    .surface = lfs::ErrorSurface::StatusOnly,
                    .actions = {},
                    .operation_id = lfs::OperationId::generate(),
                });
                return;
            }
        }

        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.toggleMode(settings_, SplitViewMode::PLYComparison);
            markDirty(DirtyFlag::SPLIT_VIEW, lfs::vis::FrameReason::SettingsChange);
        }
        applySplitModeChange(result);
        LOG_INFO("Split view: {}", result.current_mode == SplitViewMode::PLYComparison ? "PLY comparison mode" : "disabled");
    }

    void RenderingManager::handleToggleIndependentSplitView(const cmd::ToggleIndependentSplitView& event) {
        if (modelAlignmentView())
            return;
        if (!event.viewport) {
            return;
        }

        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.toggleMode(settings_, SplitViewMode::IndependentDual, event.viewport);
            syncGridPlanesLocked(settings_.grid_plane);
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::CAMERA, lfs::vis::FrameReason::CameraMotion);
        }
        applySplitModeChange(result);
        LOG_INFO("Split view: {}", result.current_mode == SplitViewMode::IndependentDual
                                       ? "independent dual-camera mode"
                                       : "disabled");
    }

    void RenderingManager::handleToggleGTComparison() {
        if (modelAlignmentView())
            return;
        if (!isGTComparisonActive() && !hasGTComparisonAvailable()) {
            LOG_WARN("GT comparison requires a loaded dataset with source images");
            return;
        }

        SplitViewService::ModeChangeResult result;

        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.toggleMode(settings_, SplitViewMode::GTComparison);
            markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS, lfs::vis::FrameReason::SceneChange);
        }

        applySplitModeChange(result);
        if (!splitViewUsesGTComparison(result.current_mode)) {
            invalidateCameraMetricsRequests(true);
        }
    }

    void RenderingManager::restoreSplitViewMode(
        const SplitViewMode mode,
        Viewport& primary_viewport) {
        std::vector<SplitViewService::ModeChangeResult>
            changes;
        {
            std::lock_guard<std::mutex> lock(
                settings_mutex_);
            if (settings_.split_view_mode == mode)
                return;
            if (settings_.split_view_mode !=
                SplitViewMode::Disabled) {
                changes.push_back(
                    split_view_service_.toggleMode(
                        settings_,
                        settings_.split_view_mode,
                        &primary_viewport));
            }
            if (mode != SplitViewMode::Disabled) {
                changes.push_back(
                    split_view_service_.toggleMode(
                        settings_, mode,
                        &primary_viewport));
            }
            if (mode == SplitViewMode::IndependentDual)
                syncGridPlanesLocked(settings_.grid_plane);
            markDirty(DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
        }
        for (const auto& change : changes)
            applySplitModeChange(change);
        if (mode != SplitViewMode::GTComparison)
            invalidateCameraMetricsRequests(true);
    }

    bool RenderingManager::beginModelAlignment(
        Viewport& primary, const core::Scene& scene,
        const core::SceneNode& reference, const core::SceneNode& mobile,
        const Viewport& left, const Viewport& right) {
        if (model_alignment_view_ || reference.type != core::NodeType::SPLAT ||
            mobile.type != core::NodeType::SPLAT || !reference.model || !mobile.model ||
            !scene.isNodeEffectivelyVisible(reference.id) || !scene.isNodeEffectivelyVisible(mobile.id) ||
            scene.isNodeEffectivelyLocked(mobile.id))
            return false;
        for (auto* parent = &reference; parent; parent = scene.getNodeById(parent->parent_id)) {
            if (parent->id == mobile.id)
                return false;
        }

        const auto settings = getSettings();
        const auto focus = getFocusedSplitPanel();
        const auto secondary = projectSecondaryViewport();
        model_alignment_view_.emplace(ModelAlignmentView{
            .nodes = {reference.uuid, mobile.uuid},
            .previous_mode = settings.split_view_mode,
            .previous_position = settings.split_position,
            .previous_offset = settings.split_view_offset,
            .previous_focus = focus,
            .previous_primary = primary,
            .previous_secondary = secondary,
        });
        restoreSplitViewMode(SplitViewMode::IndependentDual, primary);
        primary.camera = left.camera;
        auto& secondary_viewport = projectSecondaryViewport();
        secondary_viewport.camera = right.camera;
        primary.ortho_scale_override = left.ortho_scale_override.value_or(settings.ortho_scale);
        secondary_viewport.ortho_scale_override = right.ortho_scale_override.value_or(settings.ortho_scale);
        setFocusedSplitPanel(SplitViewPanelId::Left);
        handleSplitPositionChanged(0.5f);
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
        return true;
    }

    void RenderingManager::endModelAlignment(Viewport& primary) {
        if (!model_alignment_view_)
            return;
        auto saved = std::move(*model_alignment_view_);
        model_alignment_view_.reset();
        restoreSplitViewMode(saved.previous_mode, primary);
        {
            std::lock_guard lock(settings_mutex_);
            settings_.split_position = saved.previous_position;
            settings_.split_view_offset = saved.previous_offset;
        }
        restoreAlignmentCamera(primary, saved.previous_primary);
        restoreAlignmentCamera(projectSecondaryViewport(), saved.previous_secondary);
        setFocusedSplitPanel(saved.previous_focus);
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
    }

    void RenderingManager::setModelAlignmentPreview(Viewport& primary, std::optional<glm::mat4> transform) {
        if (!model_alignment_view_)
            return;
        auto& view = *model_alignment_view_;
        auto& right = projectSecondaryViewport();
        if (transform && !view.preview) {
            view.free_views = std::array{primary, right};
            const float scale = glm::length(glm::vec3((*transform)[0]));
            const glm::mat3 rotation = glm::mat3(*transform) / scale;
            right.camera.t = glm::vec3(*transform * glm::vec4(right.camera.t, 1.0f));
            right.camera.pivot = glm::vec3(*transform * glm::vec4(right.camera.pivot, 1.0f));
            right.camera.R = rotation * right.camera.R;
            if (right.ortho_scale_override)
                *right.ortho_scale_override /= scale;
            view.composite = SplitViewPanelId::Right;
        } else if (!transform && view.preview) {
            if (view.free_views) {
                restoreAlignmentCamera(primary, (*view.free_views)[0]);
                restoreAlignmentCamera(right, (*view.free_views)[1]);
            }
            view.free_views.reset();
            view.unlinked_preview_views.reset();
            view.composite.reset();
            view.linked = false;
        }
        view.preview = std::move(transform);
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
    }

    bool RenderingManager::canSwapModelAlignmentRoles(const core::Scene& scene) const {
        if (!model_alignment_view_ || model_alignment_view_->preview)
            return false;
        const auto& nodes = model_alignment_view_->nodes;
        const auto* reference = scene.getNodeByUuid(nodes[0]);
        const auto* mobile = scene.getNodeByUuid(nodes[1]);
        if (!reference || !mobile || scene.isNodeEffectivelyLocked(reference->id))
            return false;
        for (auto* parent = mobile; parent; parent = scene.getNodeById(parent->parent_id)) {
            if (parent->id == reference->id)
                return false;
        }
        return true;
    }

    void RenderingManager::swapModelAlignmentRoles(Viewport& primary, const core::Scene& scene) {
        if (!canSwapModelAlignmentRoles(scene))
            return;
        std::swap(model_alignment_view_->nodes[0], model_alignment_view_->nodes[1]);
        std::swap(primary.camera, projectSecondaryViewport().camera);
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
    }

    void RenderingManager::setModelAlignmentComposite(Viewport& primary, std::optional<SplitViewPanelId> panel) {
        if (!model_alignment_view_ || !model_alignment_view_->preview || model_alignment_view_->composite == panel)
            return;
        auto& view = *model_alignment_view_;
        if (panel && view.composite && panel != view.composite && !view.linked) {
            auto& right = projectSecondaryViewport();
            std::swap(primary.camera, right.camera);
            std::swap(primary.ortho_scale_override, right.ortho_scale_override);
        }
        view.composite = panel;
        if (panel)
            setFocusedSplitPanel(*panel);
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
    }

    void RenderingManager::toggleModelAlignmentCameraLink(Viewport& primary) {
        if (!model_alignment_view_ || !model_alignment_view_->preview)
            return;
        auto& view = *model_alignment_view_;
        auto& right = projectSecondaryViewport();
        view.linked = !view.linked;
        if (view.linked) {
            view.unlinked_preview_views = std::array{primary, right};
            syncModelAlignmentCamera(primary, getFocusedSplitPanel() == SplitViewPanelId::Right ? right : primary);
        } else if (view.unlinked_preview_views) {
            restoreAlignmentCamera(primary, (*view.unlinked_preview_views)[0]);
            restoreAlignmentCamera(right, (*view.unlinked_preview_views)[1]);
            view.unlinked_preview_views.reset();
        }
        markDirty(DirtyFlag::ALL, FrameReason::SceneChange);
    }

    void RenderingManager::syncModelAlignmentCamera(Viewport& primary, const Viewport& source) {
        if (!model_alignment_view_ || !model_alignment_view_->linked)
            return;
        auto& right = projectSecondaryViewport();
        auto& target = &source == &right ? primary : right;
        target.camera.t = source.camera.t;
        target.camera.pivot = source.camera.pivot;
        target.camera.R = source.camera.R;
        target.ortho_scale_override = source.ortho_scale_override;
    }

    void RenderingManager::handleGoToCamView(const int cam_id) {
        setCurrentCameraId(cam_id);
        LOG_DEBUG("Current camera ID set to: {}", cam_id);

        if (isGTComparisonActive() && cam_id >= 0) {
            markDirty(DirtyFlag::SPLIT_VIEW, lfs::vis::FrameReason::SettingsChange);
        }
    }

    void RenderingManager::handleSplitPositionChanged(const float position) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.split_position = std::clamp(position, 0.0f, 1.0f);
        LOG_TRACE("Split position changed to: {}", position);
        markDirty(DirtyFlag::SPLIT_POSITION, lfs::vis::FrameReason::SettingsChange);
    }

    void RenderingManager::handleRenderSettingsChanged(const ui::RenderSettingsChanged& event) {
        std::unique_lock<std::mutex> lock(settings_mutex_);
        const bool projection_changed = event.equirectangular &&
                                        (*event.equirectangular != settings_.equirectangular ||
                                         (*event.equirectangular && settings_.orthographic));
        if (event.sh_degree) {
            settings_.sh_degree = *event.sh_degree;
            LOG_TRACE("SH_DEGREE changed to: {}", settings_.sh_degree);
        }
        if (event.focal_length_mm) {
            settings_.focal_length_mm = *event.focal_length_mm;
            LOG_TRACE("Focal length changed to: {} mm", settings_.focal_length_mm);
        }
        if (event.scaling_modifier) {
            settings_.scaling_modifier = *event.scaling_modifier;
            LOG_TRACE("Scaling modifier changed to: {}", settings_.scaling_modifier);
        }
        if (event.antialiasing) {
            settings_.antialiasing = *event.antialiasing;
            LOG_TRACE("Antialiasing: {}", settings_.antialiasing ? "enabled" : "disabled");
        }
        if (event.background_color) {
            settings_.background_color = *event.background_color;
            LOG_TRACE("Background color changed");
        }
        if (event.equirectangular) {
            settings_.equirectangular = *event.equirectangular;
            if (settings_.equirectangular) {
                settings_.orthographic = false;
            }
            enforceProjectionBackend(settings_, true);
            LOG_TRACE("Equirectangular rendering: {}", settings_.equirectangular ? "enabled" : "disabled");
        }
        markDirty(DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::BACKGROUND, lfs::vis::FrameReason::SettingsChange);
        lock.unlock();

        if (projection_changed) {
            auto& generation = app_store().render_settings_generation;
            generation.set(generation.get() + 1);
        }
    }

    void RenderingManager::handleWindowResized() {
        LOG_DEBUG("RenderingManager window resize: deferring viewport refresh");
        markDirty(frame_lifecycle_service_.deferViewportRefresh(), lfs::vis::FrameReason::SceneChange);
    }

    void RenderingManager::handleGridSettingsChanged(const ui::GridSettingsChanged& event) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.show_grid = event.enabled;
        settings_.grid_plane = clampGridPlane(event.plane);
        settings_.grid_opacity = event.opacity;
        if (split_view_service_.isIndependentDualActive(settings_)) {
            panel_grid_planes_[splitViewPanelIndex(split_view_service_.focusedPanel())] = settings_.grid_plane;
        } else {
            syncGridPlanesLocked(settings_.grid_plane);
        }
        LOG_TRACE("Grid settings updated - enabled: {}, plane: {}, opacity: {}",
                  event.enabled, settings_.grid_plane, event.opacity);
        markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
    }

    void RenderingManager::handleTrainingStarted() {
        // The worker completion handoff only invalidates overlay state. Any
        // renderer setup is consumed by the next render cadence tick.
        markDirty(DirtyFlag::OVERLAY, lfs::vis::FrameReason::Overlay);
    }

    void RenderingManager::handleTrainingCompleted() {
        // TrainingCompleted is delivered from the training side. Defer Vulkan
        // destruction to renderVulkanFrame, which runs on the Vulkan thread and
        // also has the final trainer/viewer completion ordering in hand.
        vksplat_terminal_release_pending_.store(true, std::memory_order_release);
        markDirty(DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
    }

    void RenderingManager::handleSceneLoaded() {
        LOG_DEBUG("Scene loaded, marking render dirty");
        markDirty(DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
        invalidateCameraMetricsRequests(true);
        camera_interaction_service_.clearCurrentCamera();
        camera_interaction_service_.clearHoveredCamera();

        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.handleSceneLoaded(settings_);
            syncGridPlanesLocked(settings_.grid_plane);
        }
        applySplitModeChange(result);
        if (splitViewUsesGTComparison(result.previous_mode) && !splitViewUsesGTComparison(result.current_mode)) {
            LOG_INFO("Scene loaded, disabling GT comparison (camera selection reset)");
        }
    }

    void RenderingManager::handleSceneChanged(const uint32_t mutation_flags) {
        auto dirty = dirtyMaskForSceneMutations(mutation_flags);
        using Mutation = lfs::core::Scene::MutationType;
        if (hasSceneMutation(mutation_flags, Mutation::VISIBILITY_CHANGED) ||
            hasSceneMutation(mutation_flags, Mutation::NODE_REPARENTED)) {
            SplitViewService::ModeChangeResult result;
            {
                std::lock_guard<std::mutex> lock(settings_mutex_);
                result = split_view_service_.handleVisiblePLYsChanged(settings_, services().sceneOrNull());
            }
            applySplitModeChange(result);
            if (result.mode_changed) {
                dirty |= DirtyFlag::SPLIT_VIEW;
            }
        }
        markDirty(dirty, lfs::vis::FrameReason::SceneChange);
    }

    void RenderingManager::handleSceneCleared() {
        releaseSceneRenderResources();
        invalidateCameraMetricsRequests(true);
        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            settings_.raster_backend_before_equirectangular.reset();
            result = split_view_service_.handleSceneCleared(settings_);
            syncGridPlanesLocked(settings_.grid_plane);
        }
        camera_interaction_service_.clearCurrentCamera();
        camera_interaction_service_.clearHoveredCamera();
        applySplitModeChange(result);
        markDirty(DirtyFlag::ALL, lfs::vis::FrameReason::SceneChange);
    }

    void RenderingManager::handlePLYVisibilityChanged() {
        markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
    }

    void RenderingManager::handlePLYAdded() {
        LOG_DEBUG("PLY added, marking render dirty");
        markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SceneChange);
    }

    void RenderingManager::handlePLYRemoved() {
        SplitViewService::ModeChangeResult result;
        {
            std::lock_guard<std::mutex> lock(settings_mutex_);
            result = split_view_service_.handleVisiblePLYsChanged(settings_, services().sceneOrNull());
            markDirty(DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY | DirtyFlag::SPLIT_VIEW, lfs::vis::FrameReason::SceneChange);
        }
        applySplitModeChange(result);
        if (result.mode_changed) {
            LOG_DEBUG("PLY removed, disabling split view (not enough PLYs)");
        }
    }

    void RenderingManager::handleCropBoxChanged(const bool enabled) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.use_crop_box = enabled;
        markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SettingsChange);
    }

    void RenderingManager::handleEllipsoidChanged(const bool enabled) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.use_ellipsoid = enabled;
        markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY, lfs::vis::FrameReason::SettingsChange);
    }

    void RenderingManager::handlePointCloudModeChanged(const ui::PointCloudModeChanged& event) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.point_cloud_mode = event.enabled;
        settings_.voxel_size = event.voxel_size;
        LOG_DEBUG("Point cloud mode: {}, voxel size: {}",
                  event.enabled ? "enabled" : "disabled", event.voxel_size);
        viewport_artifact_service_.clearViewportOutput();
        markDirty(DirtyFlag::SPLATS, lfs::vis::FrameReason::SettingsChange);
    }

} // namespace lfs::vis
