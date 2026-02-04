/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cassert>
#include <glm/glm.hpp>
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
        void setAlignPickedPoints(std::vector<glm::vec3> points) { align_picked_points_ = std::move(points); }
        [[nodiscard]] const std::vector<glm::vec3>& getAlignPickedPoints() const { return align_picked_points_; }
        void clearAlignPickedPoints() { align_picked_points_.clear(); }

        void clear() {
            scene_manager_ = nullptr;
            trainer_manager_ = nullptr;
            rendering_manager_ = nullptr;
            window_manager_ = nullptr;
            gui_manager_ = nullptr;
            parameter_manager_ = nullptr;
            editor_context_ = nullptr;
            align_picked_points_.clear();
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
    };

    inline Services& services() { return Services::instance(); }

} // namespace lfs::vis
