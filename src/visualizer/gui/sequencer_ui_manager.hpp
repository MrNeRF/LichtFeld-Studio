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
#include "io/loader.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/sequencer_controller.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <core/export.hpp>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

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
            void tickPlaybackBeforeSceneRender();

            [[nodiscard]] SequencerController& controller() { return controller_; }
            [[nodiscard]] const SequencerController& controller() const { return controller_; }
            void setFloating(bool floating);
            [[nodiscard]] float panelTopY() const { return panel_ && !panel_->isFloating() ? panel_->cachedPanelY() : -1.0f; }
            [[nodiscard]] bool blocksPointer(double x, double y) const;
            [[nodiscard]] bool blocksKeyboard() const;
            [[nodiscard]] bool needsAnimationFrame() const;
            [[nodiscard]] float preferredFloatingHeight() const;
            // Serialized status of the active PLY sequence (empty when inactive).
            // Used by MCP tooling to verify playback/scrub behaviour.
            [[nodiscard]] LFS_VIS_API std::string plyPlayerStatusJson() const;

        private:
            void renderSequencerPanel(const UIContext& ctx, const ViewportLayout& viewport,
                                      float panel_x, float panel_y, float panel_width,
                                      float panel_height, const PanelInputState& panel_input);
            void renderCameraPath(const ViewportLayout& viewport);
            void renderKeyframeGizmo(const UIContext& ctx, const ViewportLayout& viewport);
            void handleOverlayActions();
            void loadPlySequenceFromDirectory(const std::filesystem::path& directory);
            void applyPlySequenceFrame();
            void startPlySequenceStreaming(std::vector<std::filesystem::path> paths,
                                           lfs::io::SplatTensorAllocator allocator);
            void stopPlySequenceStreaming();
            void plySequenceStreamWorker(uint64_t generation);
            void drainPlySequenceStream();
            void requestPlySequenceFrame(size_t frame_index, bool priority);
            void requestPlySequenceWindow(size_t frame_index);
            void prunePlySequenceRequests(size_t frame_index);
            void evictPlySequenceFrames(size_t keep_frame_index);
            [[nodiscard]] bool isPlySequenceFrameResident(size_t frame_index) const;
            [[nodiscard]] std::optional<size_t> selectPlySequenceDisplayFrame(size_t requested_frame) const;
            [[nodiscard]] bool isPlySequenceFrameInWindow(size_t frame_index, size_t center_frame, size_t frame_count) const;
            [[nodiscard]] bool isPlySequenceFrameInWindow(size_t frame_index,
                                                          size_t center_frame,
                                                          size_t frame_count,
                                                          bool loop) const;
            [[nodiscard]] size_t plySequenceFrameDistance(size_t lhs, size_t rhs, size_t frame_count) const;
            [[nodiscard]] bool plySequenceStreamHasWork() const;
            float advancePanelClock();
            float advancePlaybackClock();
            [[nodiscard]] float playbackDelta(float delta_time) const;
            void advancePlayback(float delta_time);
            void applyPlaybackCameraFollow();
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
            std::optional<std::chrono::steady_clock::time_point> last_playback_tick_time_;
            float last_panel_delta_time_ = 0.0f;
            float panel_elapsed_time_ = 0.0f;
            bool playback_ticked_before_scene_ = false;

            static constexpr int PREVIEW_WIDTH = 320;
            static constexpr int PREVIEW_HEIGHT = 180;
            static constexpr float PREVIEW_TARGET_FPS = 30.0f;
            VulkanUiTexture pip_texture_;
            bool pip_initialized_ = false;
            std::optional<size_t> pip_last_keyframe_;
            bool pip_needs_update_ = true;
            bool last_equirectangular_ = false;
            std::optional<size_t> last_ply_sequence_frame_;
            std::vector<size_t> loaded_ply_sequence_frames_;

            enum class PlyStreamFrameState : uint8_t {
                Empty,
                Queued,
                Loading,
                Resident,
                Failed
            };

            struct PlyStreamResult {
                uint64_t generation = 0;
                size_t frame_index = 0;
                std::unique_ptr<lfs::core::SplatData> model;
                std::string error;
                double load_ms = 0.0;
                bool cache_hit = false;
                bool cache_miss = false;
                bool cache_written = false;
                bool cache_write_failed = false;
                bool cancelled = false;
            };

            static constexpr size_t MAX_STREAM_RESIDENT_FRAMES = 64;
            static constexpr size_t STREAM_PREFETCH_AHEAD = 48;
            static constexpr size_t STREAM_PREFETCH_BEHIND = 12;

            mutable std::mutex ply_stream_mutex_;
            std::condition_variable ply_stream_cv_;
            std::thread ply_stream_thread_;
            std::atomic<bool> ply_stream_stop_{false};
            std::atomic<uint64_t> ply_stream_generation_{0};
            std::atomic<size_t> ply_stream_target_frame_{0};
            std::atomic<bool> ply_stream_target_loop_{false};
            std::vector<std::filesystem::path> ply_stream_paths_;
            lfs::io::SplatTensorAllocator ply_stream_allocator_;
            std::vector<PlyStreamFrameState> ply_stream_states_;
            std::deque<size_t> ply_stream_requests_;
            std::deque<PlyStreamResult> ply_stream_completed_;
            bool ply_stream_inflight_ = false;
            size_t ply_stream_inflight_frame_ = 0;
            double ply_stream_last_load_ms_ = 0.0;
            size_t ply_stream_failed_count_ = 0;
            size_t ply_stream_miss_count_ = 0;
            size_t ply_stream_fallback_count_ = 0;
            size_t ply_stream_eviction_count_ = 0;
            size_t ply_stream_stale_request_drop_count_ = 0;
            size_t ply_stream_cache_hit_count_ = 0;
            size_t ply_stream_cache_miss_count_ = 0;
            size_t ply_stream_cache_write_count_ = 0;
            size_t ply_stream_cache_write_fail_count_ = 0;
            std::chrono::steady_clock::time_point pip_last_render_time_ = std::chrono::steady_clock::now();
            std::optional<sequencer::Keyframe> viewport_keyframe_edit_snapshot_;
        };

    } // namespace gui
} // namespace lfs::vis
