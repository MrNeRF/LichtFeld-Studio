/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_version.hpp"
#include "core/events.hpp"
#include "core/export.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "gui/async_task_manager.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/menu_bar.hpp"
#include "gui/rml_menu_bar.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "gui/rml_right_panel.hpp"
#include "gui/rml_shell_frame.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
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
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>
#include <imgui.h>

struct SDL_Cursor;

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {
    class VisualizerImpl;
    class VulkanContext;
    class WindowManager;
    struct VulkanSceneInteropTarget;

    namespace gui {
        struct GuiHitTestResult {
            bool blocks_pointer = false;
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

            // Sub-manager access
            [[nodiscard]] AsyncTaskManager& asyncTasks() { return async_tasks_; }
            [[nodiscard]] const AsyncTaskManager& asyncTasks() const { return async_tasks_; }
            void enqueueModal(lfs::core::ModalRequest request);
            [[nodiscard]] GizmoManager& gizmo() { return gizmo_manager_; }
            [[nodiscard]] const GizmoManager& gizmo() const { return gizmo_manager_; }
            [[nodiscard]] PanelLayoutManager& panelLayout() { return panel_layout_; }
            [[nodiscard]] const PanelLayoutManager& panelLayout() const { return panel_layout_; }
            [[nodiscard]] GlobalContextMenu& globalContextMenu() { return *global_context_menu_; }

            // State queries
            bool needsAnimationFrame() const;
            [[nodiscard]] bool isViewportExportLocked() const;

            // Window visibility
            void showWindow(const std::string& name, bool show = true);

            // Viewport region access
            glm::vec2 getViewportPos() const;
            glm::vec2 getViewportSize() const;
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

            void requestExitConfirmation();
            bool isExitConfirmationPending() const;

            bool isCapturingInput() const;
            bool isModalWindowOpen() const;
            [[nodiscard]] bool passiveMouseMoveNeedsRender(float mouse_x, float mouse_y) const;
            [[nodiscard]] bool isStartupVisible() const { return startup_overlay_.isVisible(); }
            void dismissStartupOverlay();
            void setStartupPluginLoadState(bool active, float progress, const std::string& stage);
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
            void setVulkanSceneImage(std::shared_ptr<const lfs::core::Tensor> image,
                                     glm::ivec2 size,
                                     bool flip_y,
                                     std::uint64_t generation,
                                     VkSemaphore completion_semaphore = VK_NULL_HANDLE,
                                     std::uint64_t completion_value = 0);
            void setVulkanExternalSceneImage(VkImage image,
                                             VkImageView image_view,
                                             VkImageLayout layout,
                                             glm::ivec2 size,
                                             bool flip_y,
                                             std::uint64_t generation,
                                             VkSemaphore completion_semaphore = VK_NULL_HANDLE,
                                             std::uint64_t completion_value = 0);

            // Split-view's right panel routes through a parallel CUDA/Vulkan interop
            // slot so we don't pay PCIe staging cost for it; the left panel reuses the
            // existing scene image interop above.
            void setVulkanSplitRightImage(std::shared_ptr<const lfs::core::Tensor> image,
                                          glm::ivec2 size,
                                          bool flip_y,
                                          std::uint64_t generation);
            void clearVulkanSplitRightImage();

            // Splat depth -> R32_SFLOAT external image for the depth-blit pass to sample.
            void setVulkanDepthBlitImage(std::shared_ptr<const lfs::core::Tensor> depth,
                                         glm::ivec2 size,
                                         std::uint64_t generation);
            void clearVulkanDepthBlitImage();

            // Used by native panel wrappers
            void renderSelectionOverlays(const UIContext& ctx);
            void renderViewportDecorations();

        private:
            [[nodiscard]] VulkanViewportPassParams buildVulkanViewportParams(VkExtent2D extent,
                                                                             std::size_t frame_slot) const;
            void recordVulkanViewport(VkCommandBuffer command_buffer,
                                      VkExtent2D extent,
                                      const VulkanViewportPassParams& params);
            void prepareVulkanSceneInterop(VulkanContext& context);
            void resetVulkanSceneInterop();
            void prepareVulkanSplitRightInterop(VulkanContext& context);
            void resetVulkanSplitRightInterop();
            void prepareVulkanDepthBlitInterop(VulkanContext& context);
            void resetVulkanDepthBlitInterop();
            [[nodiscard]] bool shouldDeferVulkanInteropResize() const;
            void setupEventHandlers();
            void checkCudaVersionAndNotify();
            void applyDefaultStyle();
            void initMenuBar();
            void registerNativePanels();
            void updateInputOverrides(const PanelInputState& input, bool mouse_in_viewport);
            void applyUiScale(float scale);
            void rebuildFonts(float scale);
            void loadImGuiSettings();
            void saveImGuiSettings() const;
            void persistImGuiSettingsIfNeeded();
            void beginImGuiPlatformFrame(WindowManager* window_manager,
                                         VulkanContext* vulkan_context);
            [[nodiscard]] bool shouldUseCachedImGuiResizeFrame(
                const WindowManager* window_manager,
                const VulkanContext* vulkan_context) const;
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
            void queueFullscreenToggle();
            void requestFullscreenToggle();
            void updateFullscreenTransition();
            void beginInteractiveTransitionGuard();
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
            std::unique_ptr<lfs::gui::IVideoExtractorWidget> video_widget_;

            // UI state only
            std::unordered_map<std::string, bool> window_states_;
            bool show_main_panel_ = true;
            bool show_vram_hud_ = true;
            bool vram_hud_visible_published_ = false;
            std::chrono::steady_clock::time_point next_vram_hud_publish_{};
            std::chrono::steady_clock::time_point ui_toggle_next_allowed_at_{};
            bool ui_toggle_pending_ = false;
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

            std::unique_ptr<MenuBar> menu_bar_;

            panels::SequencerUIState sequencer_ui_state_;
            SequencerUIManager sequencer_ui_;
            GizmoManager gizmo_manager_;

            std::string focus_panel_name_;
            bool ui_hidden_ = false;

            // Font storage
            ImFont* font_regular_ = nullptr;
            ImFont* font_bold_ = nullptr;
            ImFont* font_heading_ = nullptr;
            ImFont* font_small_ = nullptr;
            ImFont* font_section_ = nullptr;
            ImFont* font_monospace_ = nullptr;
            ImFont* mono_fonts_[FontSet::MONO_SIZE_COUNT] = {};
            float mono_font_scales_[FontSet::MONO_SIZE_COUNT] = {};
            std::filesystem::path imgui_ini_path_;
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

            // Deferred CUDA version warning (emitted on first drawFrame)
            std::optional<lfs::core::CudaVersionInfo> pending_cuda_warning_;

            // File association prompt (Windows only, one-shot)
            bool file_association_checked_ = false;
            void promptFileAssociation();

            // RmlUI integration
            RmlUIManager rmlui_manager_;
            std::chrono::steady_clock::time_point last_imgui_platform_frame_time_{};
            std::uint64_t cached_imgui_resize_frame_count_ = 0;
            bool used_cached_imgui_resize_frame_ = false;
            std::unique_ptr<lfs::vis::VulkanViewportPass> vulkan_viewport_pass_;
            std::vector<std::unique_ptr<VulkanSceneInteropTarget>> vulkan_scene_interop_;
            std::shared_ptr<const lfs::core::Tensor> vulkan_scene_image_;
            std::uint64_t vulkan_scene_image_generation_ = 0;
            glm::ivec2 vulkan_scene_image_size_{0, 0};
            bool vulkan_scene_image_flip_y_ = false;
            VkImage vulkan_external_scene_image_ = VK_NULL_HANDLE;
            VkImageView vulkan_external_scene_image_view_ = VK_NULL_HANDLE;
            VkImageLayout vulkan_external_scene_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            glm::ivec2 vulkan_external_scene_image_size_{0, 0};
            bool vulkan_external_scene_image_flip_y_ = false;
            std::uint64_t vulkan_external_scene_image_generation_ = 0;
            VkSemaphore vulkan_frame_completion_semaphore_ = VK_NULL_HANDLE;
            std::uint64_t vulkan_frame_completion_value_ = 0;
            bool vulkan_scene_interop_disabled_ = false;

            // Parallel slot for split-view's right panel.
            std::vector<std::unique_ptr<VulkanSceneInteropTarget>> vulkan_split_right_interop_;
            std::shared_ptr<const lfs::core::Tensor> vulkan_split_right_image_;
            std::uint64_t vulkan_split_right_image_generation_ = 0;
            glm::ivec2 vulkan_split_right_image_size_{0, 0};
            bool vulkan_split_right_image_flip_y_ = false;
            VkImage vulkan_split_right_external_image_ = VK_NULL_HANDLE;
            VkImageView vulkan_split_right_external_image_view_ = VK_NULL_HANDLE;
            VkImageLayout vulkan_split_right_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t vulkan_split_right_external_image_generation_ = 0;
            bool vulkan_split_right_interop_disabled_ = false;

            // R32_SFLOAT slot for splat depth (consumed by VulkanDepthBlitPass).
            std::vector<std::unique_ptr<VulkanSceneInteropTarget>> vulkan_depth_blit_interop_;
            std::shared_ptr<const lfs::core::Tensor> vulkan_depth_blit_image_;
            std::uint64_t vulkan_depth_blit_image_generation_ = 0;
            glm::ivec2 vulkan_depth_blit_image_size_{0, 0};
            VkImage vulkan_depth_blit_external_image_ = VK_NULL_HANDLE;
            VkImageView vulkan_depth_blit_external_image_view_ = VK_NULL_HANDLE;
            VkImageLayout vulkan_depth_blit_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t vulkan_depth_blit_external_image_generation_ = 0;
            bool vulkan_depth_blit_interop_disabled_ = false;
            bool vulkan_gui_ = false;
            SDL_Cursor* pipette_cursor_ = nullptr;

            // Native panel wrapper storage (registered with PanelRegistry)
            std::vector<std::shared_ptr<IPanel>> native_panel_storage_;
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
        };
    } // namespace gui
} // namespace lfs::vis
