/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/film_strip_renderer.hpp"
#include "gui/gl_line_renderer.hpp"
#include "gui/keyframe_scene_sync.hpp"
#include "gui/panel_layout.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "gui/ui_context.hpp"
#include "rendering/gl_resources.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/sequencer_controller.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui {
    class RmlSequencerOverlay;
}

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {

        class SequencerUIManager {
        public:
            SequencerUIManager(VisualizerImpl* viewer, panels::SequencerUIState& ui_state,
                               gui::RmlUIManager* rml_manager);
            ~SequencerUIManager();

            void setupEvents();
            void render(const UIContext& ctx, const ViewportLayout& viewport);
            void compositeOverlays(int screen_w, int screen_h) const;
            void setSequencerEnabled(bool enabled);

            void destroyGLResources();

            [[nodiscard]] SequencerController& controller() { return controller_; }
            [[nodiscard]] const SequencerController& controller() const { return controller_; }
            [[nodiscard]] float panelTopY() const { return panel_ ? panel_->cachedPanelY() : -1.0f; }
            [[nodiscard]] bool blocksPointer(double x, double y) const;
            [[nodiscard]] bool blocksKeyboard() const;

        private:
            void renderSequencerPanel(const UIContext& ctx, const ViewportLayout& viewport);
            void renderCameraPath(const ViewportLayout& viewport);
            void renderKeyframeGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void handleOverlayActions();
            void renderKeyframeEditOverlay(const ViewportLayout& viewport);
            void renderFilmStrip(const UIContext& ctx);
            void drawTimelineGuides();
            void drawTimelineTooltip();
            void drawEasingCurves();
            void initPipPreview();
            void renderKeyframePreview(const UIContext& ctx);
            void drawPipPreviewWindow(const ViewportLayout& viewport);
            void beginViewportKeyframeEdit(size_t keyframe_index);
            void endViewportKeyframeEdit();
            [[nodiscard]] sequencer::CameraState currentViewportCameraState() const;
            void restoreViewportCameraState(const sequencer::CameraState& state) const;

            VisualizerImpl* viewer_;
            panels::SequencerUIState& ui_state_;
            SequencerController controller_;
            std::unique_ptr<RmlSequencerPanel> panel_;
            std::unique_ptr<gui::RmlSequencerOverlay> overlay_;
            std::unique_ptr<KeyframeSceneSync> scene_sync_;
            GLLineRenderer line_renderer_;
            FilmStripRenderer film_strip_;

            ImGuizmo::OPERATION keyframe_gizmo_op_ = ImGuizmo::OPERATION(0);
            bool keyframe_gizmo_active_ = false;
            bool edit_entered_mouse_down_ = false;

            bool film_strip_scrubbing_ = false;
            bool timeline_tooltip_active_ = false;
            ImVec2 timeline_tooltip_pos_{0.0f, 0.0f};
            std::string timeline_tooltip_text_;

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
            bool last_equirectangular_ = false;
            std::chrono::steady_clock::time_point pip_last_render_time_ = std::chrono::steady_clock::now();
            std::optional<sequencer::Keyframe> viewport_keyframe_edit_snapshot_;

            struct TimelineGeometry {
                float timeline_x = 0.0f;
                float timeline_width = 0.0f;
                float panel_x = 0.0f;
                float panel_width = 0.0f;
                float panel_y = 0.0f;
                float dp = 1.0f;
            };
            TimelineGeometry tl_geo_;

            enum class TransportMenuType { NONE,
                                           SNAP,
                                           PREVIEW,
                                           FORMAT,
                                           CLEAR_CONFIRM };
            TransportMenuType active_transport_menu_ = TransportMenuType::NONE;
        };

    } // namespace gui
} // namespace lfs::vis
