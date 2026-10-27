/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lfs::vis {

    // Forward declarations
    class SceneManager;
    class TrainerManager;
    class RenderingManager;
    class WindowManager;
    class ParameterManager;
    class EditorContext;

    namespace gui {
        class GuiManager;
    }

    class PointPickState {
    public:
        PointPickState(void (*notify)(), const bool notify_status)
            : notify_(notify), notify_status_(notify_status) {}

        enum class UiAction : uint8_t {
            None = 0,
            Apply,
            Clear,
            TogglePreview,
            RefreshPreview,
        };

        void setPickedPoints(std::vector<glm::vec3> points) {
            if (picked_points_ == points)
                return;
            picked_points_ = std::move(points);
            if (!selected_point_ ||
                *selected_point_ < 0 ||
                static_cast<size_t>(*selected_point_) >= picked_points_.size()) {
                selected_point_.reset();
            }
            notify_();
        }
        [[nodiscard]] const std::vector<glm::vec3>& getPickedPoints() const { return picked_points_; }
        void clearPickedPoints() {
            const bool changed = !picked_points_.empty() || !status_message_.empty();
            picked_points_.clear();
            selected_point_.reset();
            status_message_.clear();
            status_until_ = {};
            if (changed)
                notify_();
        }

        void setSelectedPoint(std::optional<int> index) {
            if (!index || *index < 0 ||
                static_cast<size_t>(*index) >= picked_points_.size()) {
                selected_point_.reset();
                return;
            }
            selected_point_ = index;
        }
        [[nodiscard]] std::optional<int> getSelectedPoint() const { return selected_point_; }
        void clearSelectedPoint() { selected_point_.reset(); }

        void requestUiAction(const UiAction action, const float target_distance = 0.0f) {
            ui_action_ = action;
            if (action == UiAction::Apply) {
                target_distance_ = target_distance;
            }
        }
        [[nodiscard]] UiAction takeUiAction(float* out_target_distance = nullptr) {
            const UiAction action = ui_action_;
            if (out_target_distance) {
                *out_target_distance = target_distance_;
            }
            ui_action_ = UiAction::None;
            return action;
        }
        [[nodiscard]] bool hasUiAction() const {
            return ui_action_ != UiAction::None;
        }

        void setStatusMessage(std::string message, const double duration_seconds = 1.5) {
            status_message_ = std::move(message);
            status_until_ = std::chrono::steady_clock::now() +
                            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                std::chrono::duration<double>(duration_seconds));
            if (notify_status_)
                notify_();
        }

        [[nodiscard]] const std::string* getStatusMessage() const {
            if (status_message_.empty()) {
                return nullptr;
            }
            if (std::chrono::steady_clock::now() >= status_until_) {
                return nullptr;
            }
            return &status_message_;
        }

        void clearStatusMessage() {
            if (notify_status_ && status_message_.empty())
                return;
            status_message_.clear();
            status_until_ = {};
            if (notify_status_)
                notify_();
        }

        void reset() {
            clearPickedPoints();
            ui_action_ = UiAction::None;
            target_distance_ = 0.0f;
        }

        void notifyChanged() const { notify_(); }

    private:
        void (*notify_)();
        bool notify_status_;
        std::vector<glm::vec3> picked_points_;
        std::optional<int> selected_point_;
        UiAction ui_action_ = UiAction::None;
        float target_distance_ = 0.0f;
        std::string status_message_;
        std::chrono::steady_clock::time_point status_until_{};
    };

    // Service locator — registration on main thread only, access from any thread.
    class LFS_VIS_API Services {
    public:
        static Services& instance();

        // Registration
        void set(SceneManager* sm) { scene_manager_ = sm; }
        void set(TrainerManager* tm) { trainer_manager_ = tm; }
        void set(RenderingManager* rm) { rendering_manager_ = rm; }
        void set(WindowManager* wm) { window_manager_ = wm; }
        void set(gui::GuiManager* gm) { gui_manager_ = gm; }
        void set(ParameterManager* pm) { parameter_manager_ = pm; }
        void set(EditorContext* ec) { editor_context_ = ec; }

        // Access
        [[nodiscard]] SceneManager& scene() {
            assert(scene_manager_ && "SceneManager not registered");
            return *scene_manager_;
        }

        [[nodiscard]] TrainerManager& trainer() {
            assert(trainer_manager_ && "TrainerManager not registered");
            return *trainer_manager_;
        }

        [[nodiscard]] RenderingManager& rendering() {
            assert(rendering_manager_ && "RenderingManager not registered");
            return *rendering_manager_;
        }

        [[nodiscard]] WindowManager& window() {
            assert(window_manager_ && "WindowManager not registered");
            return *window_manager_;
        }

        [[nodiscard]] gui::GuiManager& gui() {
            assert(gui_manager_ && "GuiManager not registered");
            return *gui_manager_;
        }

        [[nodiscard]] ParameterManager& params() {
            assert(parameter_manager_ && "ParameterManager not registered");
            return *parameter_manager_;
        }

        [[nodiscard]] EditorContext& editor() {
            assert(editor_context_ && "EditorContext not registered");
            return *editor_context_;
        }

        // Nullable access
        [[nodiscard]] SceneManager* sceneOrNull() { return scene_manager_; }
        [[nodiscard]] TrainerManager* trainerOrNull() { return trainer_manager_; }
        [[nodiscard]] RenderingManager* renderingOrNull() { return rendering_manager_; }
        [[nodiscard]] WindowManager* windowOrNull() { return window_manager_; }
        [[nodiscard]] gui::GuiManager* guiOrNull() { return gui_manager_; }
        [[nodiscard]] ParameterManager* paramsOrNull() { return parameter_manager_; }
        [[nodiscard]] EditorContext* editorOrNull() { return editor_context_; }

        // Check if all core services are registered
        [[nodiscard]] bool isInitialized() const {
            return scene_manager_ && trainer_manager_ && rendering_manager_ && window_manager_;
        }

        PointPickState& alignPointPick() { return align_point_pick_; }
        const PointPickState& alignPointPick() const { return align_point_pick_; }
        PointPickState& scaleDistancePointPick() { return scale_distance_point_pick_; }
        const PointPickState& scaleDistancePointPick() const { return scale_distance_point_pick_; }

        void setAlignAxisSnapEnabled(const bool enabled) {
            if (align_axis_snap_enabled_ == enabled)
                return;
            align_axis_snap_enabled_ = enabled;
            align_point_pick_.notifyChanged();
        }
        [[nodiscard]] bool getAlignAxisSnapEnabled() const { return align_axis_snap_enabled_; }

        void setAlignEdgeToAxisEnabled(const bool enabled) {
            if (align_edge_to_axis_enabled_ == enabled)
                return;
            align_edge_to_axis_enabled_ = enabled;
            align_point_pick_.notifyChanged();
        }
        [[nodiscard]] bool getAlignEdgeToAxisEnabled() const { return align_edge_to_axis_enabled_; }

        void setAlignCameraPosition(const glm::vec3& position) { align_camera_position_ = position; }
        [[nodiscard]] const glm::vec3& getAlignCameraPosition() const { return align_camera_position_; }

        void setAlignPreviewEnabled(const bool enabled) {
            if (align_preview_enabled_ == enabled)
                return;
            align_preview_enabled_ = enabled;
            align_point_pick_.notifyChanged();
        }
        [[nodiscard]] bool getAlignPreviewEnabled() const { return align_preview_enabled_; }

        void clear() {
            scene_manager_ = nullptr;
            trainer_manager_ = nullptr;
            rendering_manager_ = nullptr;
            window_manager_ = nullptr;
            gui_manager_ = nullptr;
            parameter_manager_ = nullptr;
            editor_context_ = nullptr;
            align_point_pick_.reset();
            setAlignPreviewEnabled(false);
            scale_distance_point_pick_.reset();
        }

    private:
        Services();
        ~Services() = default;
        Services(const Services&) = delete;
        Services& operator=(const Services&) = delete;

        SceneManager* scene_manager_ = nullptr;
        TrainerManager* trainer_manager_ = nullptr;
        RenderingManager* rendering_manager_ = nullptr;
        WindowManager* window_manager_ = nullptr;
        gui::GuiManager* gui_manager_ = nullptr;
        ParameterManager* parameter_manager_ = nullptr;
        EditorContext* editor_context_ = nullptr;

        // Tool state
        PointPickState align_point_pick_;
        PointPickState scale_distance_point_pick_;
        bool align_axis_snap_enabled_ = true;
        bool align_edge_to_axis_enabled_ = false;
        bool align_preview_enabled_ = false;
        glm::vec3 align_camera_position_{};
    };

    inline Services& services() { return Services::instance(); }

} // namespace lfs::vis
