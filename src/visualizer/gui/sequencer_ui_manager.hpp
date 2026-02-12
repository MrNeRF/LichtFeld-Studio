/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/keyframe_scene_sync.hpp"
#include "gui/panel_layout.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "gui/ui_context.hpp"
#include "rendering/gl_resources.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "sequencer/sequencer_panel.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <ImGuizmo.h>

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {

        class SequencerUIManager {
        public:
            SequencerUIManager(VisualizerImpl* viewer, panels::SequencerUIState& ui_state);
            ~SequencerUIManager();

            void setupEvents();
            void render(const UIContext& ctx, const ViewportLayout& viewport);

            void destroyGLResources();

            [[nodiscard]] SequencerController& controller() { return controller_; }
            [[nodiscard]] const SequencerController& controller() const { return controller_; }

        private:
            void renderSequencerPanel(const UIContext& ctx, const ViewportLayout& viewport);
            void renderCameraPath(const ViewportLayout& viewport);
            void renderKeyframeGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void renderContextMenu();
            void initPipPreview();
            void renderKeyframePreview(const UIContext& ctx);
            void drawPipPreviewWindow(const ViewportLayout& viewport);

            VisualizerImpl* viewer_;
            panels::SequencerUIState& ui_state_;
            SequencerController controller_;
            std::unique_ptr<SequencerPanel> panel_;
            std::unique_ptr<KeyframeSceneSync> scene_sync_;

            bool context_menu_open_ = false;
            std::optional<size_t> context_menu_keyframe_;

            ImGuizmo::OPERATION keyframe_gizmo_op_ = ImGuizmo::OPERATION(0);
            bool keyframe_gizmo_active_ = false;
            glm::vec3 keyframe_pos_before_drag_{0.0f};
            glm::quat keyframe_rot_before_drag_{1.0f, 0.0f, 0.0f, 0.0f};

            static constexpr int PREVIEW_WIDTH = 320;
            static constexpr int PREVIEW_HEIGHT = 180;
            static constexpr float PREVIEW_TARGET_FPS = 30.0f;
            rendering::FBO pip_fbo_;
            rendering::Texture pip_texture_;
            rendering::RBO pip_depth_rbo_;
            bool pip_initialized_ = false;
            bool pip_init_failed_ = false;
            std::optional<size_t> pip_last_keyframe_;
            bool pip_needs_update_ = true;
            std::chrono::steady_clock::time_point pip_last_render_time_ = std::chrono::steady_clock::now();
        };

    } // namespace gui
} // namespace lfs::vis
