/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/film_strip_renderer.hpp"
#include "gui/keyframe_scene_sync.hpp"
#include "gui/line_renderer.hpp"
#include "gui/panel_layout.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "gui/sequencer_viewport_edit_mode.hpp"
#include "gui/ui_context.hpp"
#include "gui/vulkan_ui_texture.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/sequencer_controller.hpp"
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>

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
            void render(const UIContext& ctx, const ViewportLayout& viewport,
                        float panel_x, float panel_y, float panel_width, float panel_height,
                        const PanelInputState& panel_input);
            void compositeOverlays(int screen_w, int screen_h);
            void setSequencerEnabled(bool enabled);
            void reloadRmlResources();

            void destroyGraphicsResources();

            [[nodiscard]] SequencerController& controller() { return controller_; }
            [[nodiscard]] const SequencerController& controller() const { return controller_; }
            void setFloating(bool floating);
            [[nodiscard]] float panelTopY() const { return panel_ && !panel_->isFloating() ? panel_->cachedPanelY() : -1.0f; }
            [[nodiscard]] bool blocksPointer(double x, double y) const;
            [[nodiscard]] bool blocksKeyboard() const;
            [[nodiscard]] float preferredFloatingHeight() const;

        private:
            void renderSequencerPanel(const UIContext& ctx, const ViewportLayout& viewport,
                                      float panel_x, float panel_y, float panel_width,
                                      float panel_height, const PanelInputState& panel_input);
            void renderCameraPath(const ViewportLayout& viewport);
            void renderKeyframeGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void handleOverlayActions();
            void renderKeyframeEditOverlay(const ViewportLayout& viewport);
            void initPipPreview();
            void renderKeyframePreview(const UIContext& ctx);
            void syncPipPreviewWindow(const ViewportLayout& viewport);
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
            LineRenderer line_renderer_;
            FilmStripRenderer film_strip_;

            SequencerViewportEditMode viewport_edit_mode_ = SequencerViewportEditMode::None;
            bool keyframe_gizmo_active_ = false;
            bool edit_entered_mouse_down_ = false;

            lfs::vis::PanelInputState panel_input_{};
            std::chrono::steady_clock::time_point last_panel_frame_time_ = std::chrono::steady_clock::now();
            float panel_elapsed_time_ = 0.0f;

            static constexpr int PREVIEW_WIDTH = 320;
            static constexpr int PREVIEW_HEIGHT = 180;
            static constexpr float PREVIEW_TARGET_FPS = 30.0f;
            VulkanUiTexture pip_texture_;
            bool pip_initialized_ = false;
            std::optional<size_t> pip_last_keyframe_;
            bool pip_needs_update_ = true;
            bool last_equirectangular_ = false;
            std::chrono::steady_clock::time_point pip_last_render_time_ = std::chrono::steady_clock::now();
            std::optional<sequencer::Keyframe> viewport_keyframe_edit_snapshot_;
        };

    } // namespace gui
} // namespace lfs::vis
