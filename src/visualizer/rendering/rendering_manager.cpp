/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering/rendering.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "theme/theme.hpp"
#include <stdexcept>

namespace lfs::vis {

    // RenderingManager Implementation
    RenderingManager::RenderingManager() {
        setupEventHandlers();
    }

    RenderingManager::~RenderingManager() {
    }

    void RenderingManager::initialize() {
        if (initialized_)
            return;

        LOG_TIMER("RenderingEngine initialization");

        engine_ = lfs::rendering::RenderingEngine::create();
        auto init_result = engine_->initialize();
        if (!init_result) {
            LOG_ERROR("Failed to initialize rendering engine: {}", init_result.error());
            throw std::runtime_error("Failed to initialize rendering engine: " + init_result.error());
        }

        initialized_ = true;
        LOG_INFO("Rendering engine initialized successfully");
    }

    void RenderingManager::markDirty() {
        markDirty(DirtyFlag::ALL);
    }

    void RenderingManager::markDirty(const DirtyMask flags) {
        dirty_mask_.fetch_or(flags, std::memory_order_relaxed);

        LOG_TRACE("Render marked dirty (flags: 0x{:x})", flags);
    }

    void RenderingManager::setViewportResizeActive(bool active) {
        if (const DirtyMask dirty = frame_lifecycle_service_.setViewportResizeActive(active); dirty) {
            markDirty(dirty);
        }
    }

    void RenderingManager::updateSettings(const RenderSettings& new_settings) {
        std::lock_guard<std::mutex> lock(settings_mutex_);

        // Update preview color if changed
        if (settings_.selection_color_preview != new_settings.selection_color_preview) {
            const auto& p = new_settings.selection_color_preview;
            lfs::rendering::config::setSelectionPreviewColor(make_float3(p.x, p.y, p.z));
        }

        // Update center marker color (group 0) if changed
        if (settings_.selection_color_center_marker != new_settings.selection_color_center_marker) {
            const auto& m = new_settings.selection_color_center_marker;
            lfs::rendering::config::setSelectionGroupColor(0, make_float3(m.x, m.y, m.z));
        }

        settings_ = new_settings;
        markDirty();
    }

    RenderSettings RenderingManager::getSettings() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_;
    }

    void RenderingManager::setOrthographic(const bool enabled, const float viewport_height, const float distance_to_pivot) {
        std::lock_guard<std::mutex> lock(settings_mutex_);

        // Calculate ortho_scale to preserve apparent size at pivot distance
        if (enabled && !settings_.orthographic) {
            constexpr float MIN_DISTANCE = 0.01f;
            constexpr float MIN_SCALE = 1.0f;
            constexpr float MAX_SCALE = 10000.0f;
            constexpr float DEFAULT_SCALE = 100.0f;

            if (viewport_height <= 0.0f || distance_to_pivot <= MIN_DISTANCE) {
                LOG_WARN("setOrthographic: invalid viewport_height={} or distance={}", viewport_height, distance_to_pivot);
                settings_.ortho_scale = DEFAULT_SCALE;
            } else {
                const float vfov = lfs::rendering::focalLengthToVFov(settings_.focal_length_mm);
                const float half_tan_fov = std::tan(glm::radians(vfov) * 0.5f);
                settings_.ortho_scale = std::clamp(
                    viewport_height / (2.0f * distance_to_pivot * half_tan_fov),
                    MIN_SCALE, MAX_SCALE);
            }
        }

        settings_.orthographic = enabled;
        markDirty(DirtyFlag::CAMERA);
    }

    float RenderingManager::getFovDegrees() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return lfs::rendering::focalLengthToVFov(settings_.focal_length_mm);
    }

    float RenderingManager::getFocalLengthMm() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.focal_length_mm;
    }

    void RenderingManager::setFocalLength(const float focal_mm) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.focal_length_mm = std::clamp(focal_mm,
                                               lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                               lfs::rendering::MAX_FOCAL_LENGTH_MM);
        markDirty(DirtyFlag::CAMERA);
    }

    float RenderingManager::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.scaling_modifier;
    }

    void RenderingManager::setScalingModifier(const float s) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        settings_.scaling_modifier = s;
        markDirty(DirtyFlag::SPLATS);
    }

    void RenderingManager::syncSelectionGroupColor(const int group_id, const glm::vec3& color) {
        lfs::rendering::config::setSelectionGroupColor(group_id, make_float3(color.x, color.y, color.z));
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::advanceSplitOffset() {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        split_view_service_.advanceSplitOffset(settings_);
        markDirty(DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS);
    }

    SplitViewInfo RenderingManager::getSplitViewInfo() const {
        return split_view_service_.getInfo();
    }

    bool RenderingManager::isSplitViewActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isActive(settings_);
    }

    bool RenderingManager::isGTComparisonActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isGTComparisonActive(settings_);
    }

    bool RenderingManager::isIndependentSplitViewActive() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return split_view_service_.isIndependentDualActive(settings_);
    }

    float RenderingManager::getSplitPosition() const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        return settings_.split_position;
    }

    std::optional<float> RenderingManager::getSplitDividerScreenX(const glm::vec2& viewport_pos,
                                                                  const glm::vec2& viewport_size) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (!split_view_service_.isActive(settings_)) {
            return std::nullopt;
        }

        const auto content_bounds = getContentBounds(glm::ivec2(
            std::max(static_cast<int>(viewport_size.x), 0),
            std::max(static_cast<int>(viewport_size.y), 0)));
        return viewport_pos.x + content_bounds.x + content_bounds.width * settings_.split_position;
    }

    Viewport& RenderingManager::resolvePanelViewport(Viewport& primary_viewport, const SplitViewPanelId panel) {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (split_view_service_.isIndependentDualActive(settings_) &&
            panel == SplitViewPanelId::Right) {
            return split_view_service_.secondaryViewport();
        }
        return primary_viewport;
    }

    const Viewport& RenderingManager::resolvePanelViewport(
        const Viewport& primary_viewport,
        const SplitViewPanelId panel) const {
        std::lock_guard<std::mutex> lock(settings_mutex_);
        if (split_view_service_.isIndependentDualActive(settings_) &&
            panel == SplitViewPanelId::Right) {
            return split_view_service_.secondaryViewport();
        }
        return primary_viewport;
    }

    void RenderingManager::applySplitModeChange(const SplitViewService::ModeChangeResult& result) {
        if (!result.mode_changed) {
            return;
        }

        if (result.clear_viewport_output) {
            viewport_artifact_service_.clearViewportOutput();
        }

        if (result.restore_equirectangular) {
            auto event = lfs::core::events::ui::RenderSettingsChanged{};
            event.equirectangular = *result.restore_equirectangular;
            event.emit();
        }
    }

    Viewport& RenderingManager::resolveFocusedViewport(Viewport& primary_viewport) {
        return resolvePanelViewport(primary_viewport, split_view_service_.focusedPanel());
    }

    const Viewport& RenderingManager::resolveFocusedViewport(const Viewport& primary_viewport) const {
        return resolvePanelViewport(primary_viewport, split_view_service_.focusedPanel());
    }

    void RenderingManager::setCursorPreviewState(const bool active, const float x, const float y, const float radius,
                                                 const bool add_mode, lfs::core::Tensor* selection_tensor,
                                                 const bool saturation_mode, const float saturation_amount,
                                                 const std::optional<SplitViewPanelId> panel,
                                                 const int focused_gaussian_id) {
        viewport_overlay_service_.setCursorPreview(active, x, y, radius, add_mode, selection_tensor,
                                                   saturation_mode, saturation_amount, panel, focused_gaussian_id);
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::clearCursorPreviewState() {
        viewport_overlay_service_.clearCursorPreview();
        markDirty(DirtyFlag::SELECTION);
    }

    void RenderingManager::setRectPreview(float x0, float y0, float x1, float y1, bool add_mode,
                                          const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setRect(x0, y0, x1, y1, add_mode, panel);
    }

    void RenderingManager::clearRectPreview() {
        viewport_overlay_service_.clearRect();
    }

    void RenderingManager::setPolygonPreview(const std::vector<std::pair<float, float>>& points, bool closed,
                                             bool add_mode, const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setPolygon(points, closed, add_mode, panel);
    }

    void RenderingManager::setPolygonPreviewWorldSpace(const std::vector<glm::vec3>& world_points,
                                                       const bool closed, const bool add_mode,
                                                       const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setPolygonWorldSpace(world_points, closed, add_mode, panel);
    }

    void RenderingManager::clearPolygonPreview() {
        viewport_overlay_service_.clearPolygon();
    }

    void RenderingManager::setLassoPreview(const std::vector<std::pair<float, float>>& points, bool add_mode,
                                           const std::optional<SplitViewPanelId> panel) {
        viewport_overlay_service_.setLasso(points, add_mode, panel);
    }

    void RenderingManager::clearLassoPreview() {
        viewport_overlay_service_.clearLasso();
    }

    void RenderingManager::clearSelectionPreviews() {
        viewport_overlay_service_.clearSelectionPreviews();
        markDirty(DirtyFlag::SELECTION);
    }

} // namespace lfs::vis
