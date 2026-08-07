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

        // Align tool state
        enum class AlignUiAction : uint8_t {
            None = 0,
            Apply,
            Clear,
        };

        void setAlignPickedPoints(std::vector<glm::vec3> points) {
            align_picked_points_ = std::move(points);
            if (!align_selected_point_ ||
                *align_selected_point_ < 0 ||
                static_cast<size_t>(*align_selected_point_) >= align_picked_points_.size()) {
                align_selected_point_.reset();
            }
        }
        [[nodiscard]] const std::vector<glm::vec3>& getAlignPickedPoints() const { return align_picked_points_; }
        void clearAlignPickedPoints() {
            align_picked_points_.clear();
            align_selected_point_.reset();
            clearAlignStatusMessage();
        }

        void setAlignSelectedPoint(std::optional<int> index) {
            if (!index || *index < 0 ||
                static_cast<size_t>(*index) >= align_picked_points_.size()) {
                align_selected_point_.reset();
                return;
            }
            align_selected_point_ = index;
        }
        [[nodiscard]] std::optional<int> getAlignSelectedPoint() const { return align_selected_point_; }
        void clearAlignSelectedPoint() { align_selected_point_.reset(); }

        void setAlignAxisSnapEnabled(const bool enabled) { align_axis_snap_enabled_ = enabled; }
        [[nodiscard]] bool getAlignAxisSnapEnabled() const { return align_axis_snap_enabled_; }

        void setAlignEdgeToAxisEnabled(const bool enabled) { align_edge_to_axis_enabled_ = enabled; }
        [[nodiscard]] bool getAlignEdgeToAxisEnabled() const { return align_edge_to_axis_enabled_; }

        void requestAlignUiAction(const AlignUiAction action) { align_ui_action_ = action; }
        [[nodiscard]] AlignUiAction takeAlignUiAction() {
            const AlignUiAction action = align_ui_action_;
            align_ui_action_ = AlignUiAction::None;
            return action;
        }
        [[nodiscard]] bool hasAlignUiAction() const {
            return align_ui_action_ != AlignUiAction::None;
        }

        void setAlignStatusMessage(std::string message, const double duration_seconds = 1.5) {
            align_status_message_ = std::move(message);
            align_status_until_ = std::chrono::steady_clock::now() +
                                  std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(duration_seconds));
        }

        [[nodiscard]] const std::string* getAlignStatusMessage() const {
            if (align_status_message_.empty()) {
                return nullptr;
            }
            if (std::chrono::steady_clock::now() >= align_status_until_) {
                return nullptr;
            }
            return &align_status_message_;
        }

        void clearAlignStatusMessage() {
            align_status_message_.clear();
            align_status_until_ = {};
        }

        void clear() {
            scene_manager_ = nullptr;
            trainer_manager_ = nullptr;
            rendering_manager_ = nullptr;
            window_manager_ = nullptr;
            gui_manager_ = nullptr;
            parameter_manager_ = nullptr;
            editor_context_ = nullptr;
            align_picked_points_.clear();
            align_selected_point_.reset();
            align_ui_action_ = AlignUiAction::None;
            clearAlignStatusMessage();
        }

    private:
        Services() = default;
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
        std::vector<glm::vec3> align_picked_points_;
        std::optional<int> align_selected_point_;
        bool align_axis_snap_enabled_ = true;
        bool align_edge_to_axis_enabled_ = false;
        AlignUiAction align_ui_action_ = AlignUiAction::None;
        std::string align_status_message_;
        std::chrono::steady_clock::time_point align_status_until_{};
    };

    inline Services& services() { return Services::instance(); }

} // namespace lfs::vis
