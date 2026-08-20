/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error_bus.hpp"
#include "core/events.hpp"
#include "core/export.hpp"
#include "gui/async_task_manager.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_error_consumer.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/menu_bar.hpp"
#include "gui/perf_sampler.hpp"
#include "gui/rml_menu_bar.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "gui/rml_right_panel.hpp"
#include "gui/rml_shell_frame.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/rml_toast_overlay.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/scene_tree_session.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "gui/startup_overlay.hpp"
#include "gui/ui_context.hpp"
#include "gui/utils/drag_drop_native.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/gui/video_widget_interface.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

struct SDL_Cursor;

namespace lfs::vis {
    class VisualizerImpl;
    class WindowManager;
    class VisualizerImplResetTest_RecoveryDeclineKeepsSidecarSuppressesRepeatAndExplicitSaveDeletesIt_Test;
    class VisualizerImplResetTest_NewProjectClearsRecoveryPromptPendingSoNextOpenProceeds_Test;
    class VisualizerImplResetTest_RecoveredPublishUsesRecoveredCommitKind_Test;
    class VisualizerImplResetTest_RecoveredProjectSwitchDeletesTempOnlyAfterReplacement_Test;
    class VisualizerImplResetTest_FailedNewProjectKeepsRecoveredSessionTemp_Test;
    class VisualizerImplResetTest_RecoveredCloseDeletesTempAfterDocumentTeardown_Test;
    class VisualizerImplResetTest_StartupOffersRecoveryAfterUncleanShutdown_Test;
    class VisualizerImplResetTest_StartupWithCleanLastSessionLeavesBlankSession_Test;
    class VisualizerImplResetTest_UiVisibilityWaitsForMatchingFrame_Test;
    class VisualizerImplResetTest_UiVisibilityTimeoutCommitsRequestedLayout_Test;
    class VisualizerImplResetTest_RecoveryDismissalPersistsAndNewerCandidateIsOffered_Test;
    class VisualizerImplResetTest_StartupOffersScratchRecoveryAsUntitled_Test;
    class VisualizerImplResetTest_StartupSweepsEmptyScratchAndDoesNotOffer_Test;

    namespace gui {
        class NativeScenePanel;

        LFS_VIS_API void openPreferencesPanel(std::string section = {});
        [[nodiscard]] LFS_VIS_API std::string consumePreferencesSectionRequest();

        struct GuiHitTestResult {
            bool blocks_pointer = false;
            bool blocks_mouse_button = false;
            bool takes_keyboard_focus = false;
        };

        struct GuiInputState {
            bool has_keyboard_focus = false;
            bool text_input_active = false;
            bool modal_open = false;
        };

        class LFS_VIS_API GuiManager {
        public:
            GuiManager(VisualizerImpl* viewer);
            ~GuiManager();

            // Lifecycle
            void init();
            void shutdown();
            void render();
            void updateInteractiveTransitions();
            [[nodiscard]] bool isInteractiveTransitionSettling() const;
            void syncVisiblePanelsBeforeSceneRender();
            void setRmlResizeDeferring(bool defer) { rmlui_manager_.setResizeDeferring(defer); }
            void ensureCjkFontsLoaded() { rmlui_manager_.ensureCjkFontsLoaded(); }

            // Sub-manager access
            [[nodiscard]] AsyncTaskManager& asyncTasks() { return async_tasks_; }
            [[nodiscard]] const AsyncTaskManager& asyncTasks() const { return async_tasks_; }
            void enqueueModal(lfs::core::ModalRequest request);
            void enqueueToast(ToastRequest request);
            [[nodiscard]] GizmoManager& gizmo() { return gizmo_manager_; }
            [[nodiscard]] const GizmoManager& gizmo() const { return gizmo_manager_; }
            [[nodiscard]] PanelLayoutManager& panelLayout() { return panel_layout_; }
            [[nodiscard]] const PanelLayoutManager& panelLayout() const { return panel_layout_; }
            [[nodiscard]] GlobalContextMenu& globalContextMenu() { return *global_context_menu_; }

            // State queries
            bool needsAnimationFrame() const;
            // Min finite scheduled GUI animation/update delay (seconds). Used by the
            // idle wait path so CSS transitions / timers wake on time without spinning.
            [[nodiscard]] std::optional<double> secondsUntilNextAnimationFrame() const;
            [[nodiscard]] bool isViewportExportLocked() const;

            // Window visibility
            void showWindow(const std::string& name, bool show = true);

            // Viewport region access
            glm::vec2 getViewportPos() const;
            glm::vec2 getViewportSize() const;
            glm::vec2 getSceneRenderViewportPos() const;
            glm::vec2 getSceneRenderViewportSize() const;
            void commitUiVisibilityTransitionIfFrameReady(bool frame_ready);
            bool isViewportFocused() const;
            bool isPositionInViewport(double x, double y) const;
            bool isPositionOverFloatingPanel(double x, double y) const;
            [[nodiscard]] GuiHitTestResult hitTestPointer(double x, double y) const;
            [[nodiscard]] GuiInputState inputState() const;

            bool isForceExit() const { return force_exit_; }
            void setForceExit(bool value) { force_exit_ = value; }

            [[nodiscard]] SequencerController& sequencer() { return sequencer_ui_.controller(); }
            [[nodiscard]] const SequencerController& sequencer() const { return sequencer_ui_.controller(); }
            [[nodiscard]] SequencerUIManager& sequencerUI() { return sequencer_ui_; }
            [[nodiscard]] const SequencerUIManager& sequencerUI() const { return sequencer_ui_; }

            [[nodiscard]] panels::SequencerUIState& getSequencerUIState() { return sequencer_ui_state_; }
            [[nodiscard]] const panels::SequencerUIState& getSequencerUIState() const { return sequencer_ui_state_; }

            [[nodiscard]] VisualizerImpl* getViewer() const { return viewer_; }
            [[nodiscard]] std::unordered_map<std::string, bool>* getWindowStates() { return &window_states_; }
            [[nodiscard]] const std::unordered_map<std::string, bool>&
            getWindowStates() const {
                return window_states_;
            }
            [[nodiscard]] std::string scenePanelActiveTab() const;
            void setScenePanelActiveTab(std::string_view tab);
            [[nodiscard]] SceneTreeSessionChrome captureSceneTreeChrome(
                const lfs::core::Scene& scene) const;
            void applySceneTreeChrome(const SceneTreeSessionChrome& chrome);
            void resetSceneTreeChrome();
            [[nodiscard]] float tabStripScroll() const;
            void setTabStripScroll(float value);

            void requestExitConfirmation(
                bool training_in_progress = false);
            void openPreferences();
            [[nodiscard]] std::expected<void, std::string> resetLayout();
            [[nodiscard]] std::expected<void, std::string> resetWindowState();
            void dismissExitConfirmation();
            void noteExitPopupMirror(bool open);
            bool isExitConfirmationPending() const;

            bool isCapturingInput() const;
            bool isModalWindowOpen() const;
            [[nodiscard]] bool passiveMouseMoveNeedsRender(float mouse_x, float mouse_y) const;
            [[nodiscard]] std::optional<double> secondsUntilTooltipReveal() const;
            [[nodiscard]] bool isStartupVisible() const { return startup_overlay_.isVisible(); }
            void dismissStartupOverlay() { startup_overlay_.dismiss(); }
            [[nodiscard]] bool isStartupBlockingInput() const {
                return startup_overlay_.blocksUnderlayInput();
            }
            void setStartupPluginLoadState(bool started, bool active, float progress,
                                           const std::string& stage);
            // Rebuild static @tr: RML content after a runtime language switch.
            // The reload is deferred until no RML interaction is active.
            void requestLocalizationUiRefresh();
            void captureKey(int physical_key, int logical_key, int mods);
            void captureMouseButton(int button, int mods, double x, double y, std::optional<int> chord_key = std::nullopt);
            void captureMouseButtonRelease(int button);
            void captureMouseMove(double x, double y);

            // Thumbnail system (delegates to MenuBar)
            void requestThumbnail(const std::string& video_id);
            void processThumbnails();
            bool isThumbnailReady(const std::string& video_id) const;
            uint64_t getThumbnailTexture(const std::string& video_id) const;

            int getHighlightedCameraUid() const;

            // Drag-drop state for overlays
            [[nodiscard]] bool isDragHovering() const { return drag_drop_hovering_; }

            // Used by native panel wrappers
            void renderSelectionOverlays(const UIContext& ctx);
            void renderViewportDecorations();

        private:
            friend class lfs::vis::VisualizerImplResetTest_RecoveryDeclineKeepsSidecarSuppressesRepeatAndExplicitSaveDeletesIt_Test;
            friend class lfs::vis::VisualizerImplResetTest_NewProjectClearsRecoveryPromptPendingSoNextOpenProceeds_Test;
            friend class lfs::vis::VisualizerImplResetTest_RecoveredPublishUsesRecoveredCommitKind_Test;
            friend class lfs::vis::VisualizerImplResetTest_RecoveredProjectSwitchDeletesTempOnlyAfterReplacement_Test;
            friend class lfs::vis::VisualizerImplResetTest_FailedNewProjectKeepsRecoveredSessionTemp_Test;
            friend class lfs::vis::VisualizerImplResetTest_RecoveredCloseDeletesTempAfterDocumentTeardown_Test;
            friend class lfs::vis::VisualizerImplResetTest_StartupOffersRecoveryAfterUncleanShutdown_Test;
            friend class lfs::vis::VisualizerImplResetTest_StartupWithCleanLastSessionLeavesBlankSession_Test;
            friend class lfs::vis::VisualizerImplResetTest_UiVisibilityWaitsForMatchingFrame_Test;
            friend class lfs::vis::VisualizerImplResetTest_UiVisibilityTimeoutCommitsRequestedLayout_Test;
            friend class lfs::vis::VisualizerImplResetTest_RecoveryDismissalPersistsAndNewerCandidateIsOffered_Test;
            friend class lfs::vis::VisualizerImplResetTest_StartupOffersScratchRecoveryAsUntitled_Test;
            friend class lfs::vis::VisualizerImplResetTest_StartupSweepsEmptyScratchAndDoesNotOffer_Test;
            [[nodiscard]] bool isPositionOverRightPanelResizeEdge(double x, double y) const;
            [[nodiscard]] VulkanViewportPassParams buildVulkanViewportParams(VkExtent2D extent,
                                                                             std::size_t frame_slot) const;
            void recordVulkanViewport(VkCommandBuffer command_buffer,
                                      VkExtent2D extent,
                                      const VulkanViewportPassParams& params);
            void setupEventHandlers();
            void applyDefaultStyle();
            void initMenuBar();
            void registerNativePanels();
            void updateInputOverrides(const PanelInputState& input, bool mouse_in_viewport);
            void applyUiScale(float scale);
            void rebuildFonts(float scale);
            void initCustomCursors();
            void destroyCustomCursors();
            void applyRmlCursorRequest(RmlCursorRequest req);
            struct DevResourceScanResult {
                std::unordered_map<std::string, std::filesystem::file_time_type> file_times;
                bool rml_changed = false;
                bool locale_changed = false;
                bool scan_failed = false;
            };
            void initDevResourceHotReload();
            void pollDevResourceHotReload();
            DevResourceScanResult scanDevResourceFiles(bool detect_changes);
            static DevResourceScanResult scanDevResourceFilesSnapshot(
                std::filesystem::path rml_dir,
                std::filesystem::path locale_dir,
                std::unordered_map<std::string, std::filesystem::file_time_type> previous_times,
                bool detect_changes);
            void launchDevResourceScan();
            bool consumeDevResourceScanResult();
            bool shouldDeferDevResourceHotReload() const;
            bool reloadLocalizationResources();
            void reloadRmlResources();

            [[nodiscard]] bool isVramHudOverlayVisible() const;
            [[nodiscard]] bool isVramHudPublishDue(std::chrono::steady_clock::time_point now) const;
            [[nodiscard]] PanelAnimationVisibility panelAnimationVisibility() const;
            [[nodiscard]] bool drainVulkanFramesForInteractiveTransition(
                lfs::vis::WindowManager& window_manager,
                const char* transition_name);
            void applyInteractiveTransitionCooldown(
                std::chrono::steady_clock::time_point& next_allowed_at,
                std::chrono::steady_clock::time_point now,
                bool training_active);
            void queueUiVisibilityToggle();
            void requestUiVisibilityToggle();
            void updateUiVisibilityTransition();
            void commitUiVisibilityTransition(bool matched_frame);
            void queueFullscreenToggle();
            void requestFullscreenToggle();
            void updateFullscreenTransition();
            enum class InteractiveTransitionTrainingPolicy {
                KeepRunning,
                PauseAndResume,
            };
            void beginInteractiveTransitionGuard(InteractiveTransitionTrainingPolicy training_policy);
            void updateInteractiveTransitionGuard();
            void endInteractiveTransitionGuard();

            struct EditorContextUpdateStamp {
                bool valid = false;
                bool has_scene_manager = false;
                bool has_trainer_manager = false;
                bool has_dataset = false;
                bool has_training_model = false;
                bool trainer_running = false;
                bool trainer_paused = false;
                bool trainer_finished = false;
                std::uint64_t scene_generation = 0;
                std::uint64_t selection_generation = 0;
                std::uint64_t scene_node_count = 0;

                bool operator==(const EditorContextUpdateStamp&) const = default;
            };

            // Core dependencies
            VisualizerImpl* viewer_;

            // Owned components
            std::unique_ptr<RmlModalOverlay> rml_modal_overlay_;
            std::unique_ptr<RmlToastOverlay> rml_toast_overlay_;
            std::unique_ptr<lfs::gui::IVideoExtractorWidget> video_widget_;

            // UI state only
            std::unordered_map<std::string, bool> window_states_;
            bool show_main_panel_ = true;
            bool show_vram_hud_ = false;
            bool perf_hud_expanded_ = true;
            bool vram_hud_visible_published_ = false;
            bool perf_hud_visible_published_ = false;
            std::chrono::steady_clock::time_point next_vram_hud_publish_{};
            PerfSampler perf_sampler_;
            std::chrono::steady_clock::time_point ui_toggle_next_allowed_at_{};
            bool ui_toggle_pending_ = false;
            bool ui_visibility_resize_active_ = false;
            bool ui_visibility_layout_committed_ = false;
            bool ui_visibility_target_ready_ = false;
            bool ui_visibility_target_hidden_ = false;
            ViewportLayout ui_visibility_target_layout_{};
            std::chrono::steady_clock::time_point fullscreen_toggle_next_allowed_at_{};
            std::chrono::steady_clock::time_point interactive_transition_guard_until_{};
            bool fullscreen_toggle_pending_ = false;
            bool fullscreen_target_state_ = false;
            bool interactive_transition_resume_training_ = false;
            std::optional<AppStore::GTMetricsOverlayConfig> published_gt_metrics_overlay_config_;
            bool menu_labels_synced_ = false;
            std::uint64_t synced_menu_entries_version_ = 0;
            std::uint64_t synced_menu_language_generation_ = 0;

            // Panel layout and viewport
            PanelLayoutManager panel_layout_;
            ViewportLayout viewport_layout_;
            float menu_toolbar_right_edge_ = 0.0f;
            bool force_exit_ = false;
            bool exit_confirmation_requested_ = false;
            bool exit_confirmation_dismissed_ = false;

            std::unique_ptr<MenuBar> menu_bar_;

            panels::SequencerUIState sequencer_ui_state_;
            SequencerUIManager sequencer_ui_;
            GizmoManager gizmo_manager_;

            std::string focus_panel_name_;
            bool ui_hidden_ = false;

            // Font storage
            FontSet::FontHandle font_regular_ = nullptr;
            FontSet::FontHandle font_bold_ = nullptr;
            FontSet::FontHandle font_heading_ = nullptr;
            FontSet::FontHandle font_small_ = nullptr;
            FontSet::FontHandle font_section_ = nullptr;
            FontSet::FontHandle font_monospace_ = nullptr;
            FontSet::FontHandle mono_fonts_[FontSet::MONO_SIZE_COUNT] = {};
            float mono_font_scales_[FontSet::MONO_SIZE_COUNT] = {};
            FontSet buildFontSet() const;

            // Async task management
            AsyncTaskManager async_tasks_;

            StartupOverlay startup_overlay_;
            RmlShellFrame rml_shell_frame_;
            RmlRightPanel rml_right_panel_;
            RmlViewportOverlay rml_viewport_overlay_;
            RmlMenuBar rml_menu_bar_;
            RmlStatusBar rml_status_bar_;
            std::unique_ptr<GlobalContextMenu> global_context_menu_;

            // Native drag-drop handler
            NativeDragDrop drag_drop_;
            bool drag_drop_hovering_ = false;

            // DPI scaling
            float current_ui_scale_ = 1.0f;
            float pending_ui_scale_ = 0.0f;

            bool cuda_unavailable_notified_ = false;

            // File association prompt (Windows only, one-shot)
            bool file_association_checked_ = false;
            void promptFileAssociation();

            // RmlUI integration
            RmlUIManager rmlui_manager_;
            std::unique_ptr<lfs::vis::VulkanViewportPass> vulkan_viewport_pass_;
            bool vulkan_gui_ = false;
            SDL_Cursor* pipette_cursor_ = nullptr;

            // Native panel wrapper storage (registered with PanelRegistry)
            std::vector<std::shared_ptr<IPanel>> native_panel_storage_;
            std::shared_ptr<NativeScenePanel> native_scene_panel_;
            SceneTreeSessionChrome pending_scene_tree_chrome_;
            uint64_t panel_frame_serial_ = 0;
            uint8_t ui_layout_settle_frames_ = 0;
            EditorContextUpdateStamp last_editor_context_update_stamp_;
            glm::vec2 last_ui_layout_work_pos_{-1.0f, -1.0f};
            glm::vec2 last_ui_layout_work_size_{-1.0f, -1.0f};
            float last_ui_layout_right_panel_w_ = -1.0f;
            float last_ui_layout_scene_ratio_ = -1.0f;
            float last_ui_layout_python_console_w_ = -1.0f;
            float last_ui_layout_bottom_dock_h_ = -1.0f;
            float last_ui_layout_left_dock_w_ = -1.0f;
            bool last_ui_layout_show_main_panel_ = false;
            bool last_ui_layout_ui_hidden_ = false;
            bool last_ui_layout_python_console_visible_ = false;
            bool last_ui_layout_bottom_dock_visible_ = false;
            bool last_ui_layout_left_dock_visible_ = false;
            enum class RightPanelPointerRegion : uint8_t {
                None,
                Resize,
                SceneHeader,
                ActiveTab,
                Chrome,
            };
            bool right_panel_pointer_live_capture_ = false;
            RightPanelPointerRegion right_panel_pointer_capture_region_ =
                RightPanelPointerRegion::None;
            bool right_panel_resize_edge_was_hovered_ = false;
            bool bottom_dock_pointer_live_capture_ = false;
            bool left_dock_pointer_live_capture_ = false;
            bool dock_resize_interaction_active_ = false;
            std::string last_ui_layout_active_tab_;
            std::uint64_t last_pre_scene_panel_sync_generation_ = 0;

            struct DevResourceWatchState {
                bool enabled = false;
                std::filesystem::path rml_dir;
                std::filesystem::path locale_dir;
                std::unordered_map<std::string, std::filesystem::file_time_type> file_times;
                std::chrono::steady_clock::time_point next_scan{};
                std::future<DevResourceScanResult> scan_future;
                bool pending_rml_reload = false;
                bool pending_locale_reload = false;
            };

            DevResourceWatchState dev_resource_watch_;
            bool pending_localization_ui_refresh_ = false;
            std::uint64_t localized_rml_language_generation_ = std::numeric_limits<std::uint64_t>::max();

            // Native ErrorBus surfacing (Phase 8). Declared last so
            // error_subscription_ unsubscribes before any other member (the
            // modal overlay included) is torn down; error_consumer_ outlives
            // its subscription per the frozen lifetime rule.
            std::unique_ptr<GuiErrorConsumer> error_consumer_;
            lfs::Subscription error_subscription_;
        };
    } // namespace gui
} // namespace lfs::vis
