/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/gamepad_service.hpp"
#include "input/input_router.hpp"
#include "visualizer/visualizer.hpp"
#include <chrono>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct SDL_Window;
union SDL_Event;

namespace lfs::vis {

    class InputController;
    class GraphicsContext;
    struct VulkanDeviceHooks;
    namespace gui {
        class NativeDragContext;
        class NativeDragScheduler;
    } // namespace gui

    class LFS_VIS_API WindowManager {
    public:
        struct PersistentWindowState {
            int x = 0;
            int y = 0;
            int width = 1280;
            int height = 720;
            bool maximized = false;
        };

        enum class ResizeIntent {
            Interactive,
            Exact,
        };

        struct HitTestRect {
            int x = 0;
            int y = 0;
            int w = 0;
            int h = 0;
        };

        WindowManager(const std::string& title, int width, int height,
                      int monitor_x = 0, int monitor_y = 0,
                      int monitor_width = 0, int monitor_height = 0);
        ~WindowManager();

        WindowManager(const WindowManager&) = delete;
        WindowManager& operator=(const WindowManager&) = delete;

        // When Vulkan cannot start through device_hooks, init retries with plain
        // Vulkan and deviceHooksError() keeps the reason. Only the Vulkan graphics
        // backend accepts hooks; other backends report them unusable there.
        bool init(const VulkanDeviceHooks* device_hooks = nullptr);
        [[nodiscard]] const std::string& deviceHooksError() const { return device_hooks_error_; }

        void setInitialWindowState(PersistentWindowState state);
        [[nodiscard]] PersistentWindowState persistentWindowState() const;
        [[nodiscard]] bool resetPersistentWindowState();

        void showWindow();
        void updateWindowSize(const char* reason = "manual",
                              ResizeIntent intent = ResizeIntent::Exact);
        void pollEvents();
        void waitEvents(std::optional<double> timeout_seconds = std::nullopt);
        bool shouldClose() const;
        void requestClose() { should_close_ = true; }
        void cancelClose();
        void wakeEventLoop();
        void refreshResizeCursor();
        [[nodiscard]] unsigned manualResizeEdgeMask() const;

        gui::NativeDragContext& nativeFileDragContext();
        gui::NativeDragScheduler& nativeDragScheduler();
        void dispatchNativeDrags(std::function<bool()> frame);
        bool advanceNativeTransferFrame();
        SDL_Window* getWindow() const { return window_; }
        GraphicsContext* getGraphicsContext() const { return graphics_context_.get(); }
        glm::ivec2 getWindowSize() const { return window_size_; }
        glm::ivec2 getFramebufferSize() const { return framebuffer_size_; }
        [[nodiscard]] bool hasRecentWindowSizeChange(std::chrono::steady_clock::duration max_age) const;
        bool isFullscreen() const { return is_fullscreen_; }
        bool isMaximized() const;
        void minimize();
        void toggleMaximized();
        void setTitlebarDragRegion(int height_px, std::vector<HitTestRect> excluded_rects);
        void clearTitlebarDragRegion();
        [[nodiscard]] bool isTitlebarDragPoint(int x, int y) const;
        [[nodiscard]] bool usesWayland() const { return is_wayland_; }
        [[nodiscard]] bool usesEventDrivenTitlebarDrag() const { return native_titlebar_move_available_; }
        void setFullscreen(bool fullscreen);
        void setInputController(InputController* ic);
        [[nodiscard]] InputController* inputController() const { return input_controller_; }
        [[nodiscard]] const FrameInputBuffer& frameInput() const { return frame_input_; }
        [[nodiscard]] const input::InputRouter& inputRouter() const { return input_router_; }
        [[nodiscard]] input::GamepadService& gamepads() { return gamepads_; }
        void updateGamepadService(bool navigation_enabled);
        void setGamepadDiagnosticsActive(bool active);

    private:
        friend class WindowInputDispatchTest;
        void processEvent(const ::SDL_Event& event);
        void dispatchQueuedEvent(const ::SDL_Event& event);
        void dispatchPolledEvent(const ::SDL_Event& event);
        void flushGamepadEvents();
        std::vector<SDL_Event> gamepad_events_;
        bool drainQueuedEvents();
        static bool watchEvent(void* userdata, ::SDL_Event* event);
        bool pumping_events_ = false;
        bool watching_event_ = false;
        std::vector<std::pair<Uint32, Uint64>> dispatched_events_;
        [[nodiscard]] bool shouldSuppressGuiRoutingForResize(const ::SDL_Event& event,
                                                             unsigned int main_window_id) const;

        enum class ResizeEdge : unsigned {
            NoEdge = 0,
            Left = 1u << 0,
            Right = 1u << 1,
            Top = 1u << 2,
            Bottom = 1u << 3,
        };

        SDL_Window* window_ = nullptr;
        input::GamepadService gamepads_;
        input::GamepadSample gamepad_wake_sample_;
        bool gamepad_diagnostics_active_ = false;
        std::unique_ptr<GraphicsContext> graphics_context_;
        std::string device_hooks_error_;
        std::unique_ptr<gui::NativeDragContext> native_file_drag_;
        std::unique_ptr<gui::NativeDragScheduler> native_drag_scheduler_;
        std::function<bool()> native_transfer_frame_;
        std::string title_;
        glm::ivec2 window_size_;
        glm::ivec2 framebuffer_size_;
        std::chrono::steady_clock::time_point last_window_size_change_time_{};

        glm::ivec2 monitor_pos_{0, 0};
        glm::ivec2 monitor_size_{0, 0};

        bool is_fullscreen_ = false;
        glm::ivec2 windowed_pos_{0, 0};
        glm::ivec2 windowed_size_{1280, 720};
        int titlebar_drag_height_px_ = 0;
        std::vector<HitTestRect> titlebar_drag_excluded_rects_;
        bool native_titlebar_move_available_ = false;
        bool is_wayland_ = false;
        bool pending_titlebar_double_click_ = false;
        bool titlebar_drag_active_ = false;
        bool titlebar_drag_started_maximized_ = false;
        glm::ivec2 titlebar_drag_start_global_{0, 0};
        glm::ivec2 titlebar_drag_start_local_{0, 0};
        glm::ivec2 titlebar_drag_window_offset_{0, 0};
        ResizeEdge manual_resize_edge_ = ResizeEdge::NoEdge;
        glm::ivec2 manual_resize_start_global_{0, 0};
        glm::ivec2 manual_resize_start_pos_{0, 0};
        glm::ivec2 manual_resize_start_size_{0, 0};
        bool is_borderless_maximized_ = false;
        glm::ivec2 borderless_restore_pos_{0, 0};
        glm::ivec2 borderless_restore_size_{1280, 720};
        std::optional<PersistentWindowState> initial_window_state_;
        bool should_close_ = false;

        InputController* input_controller_ = nullptr;
        input::InputRouter input_router_;
        FrameInputBuffer frame_input_;
        std::vector<std::string> pending_drop_files_;

        void beginTitlebarNativeMove();
        void installX11ErrorHandlers();
        [[nodiscard]] ResizeEdge resizeEdgeAt(int x, int y) const;
        void setResizeCursorForEdge(ResizeEdge edge);
        void updateResizeCursor(int x, int y);
        void beginManualResize(ResizeEdge edge);
        void updateManualResize();
        void finishManualResize();
        void suppressFrameInputForManualResize();
        void pinFrameInputForValueDrag();
        [[nodiscard]] bool isManualResizeActive() const { return manual_resize_edge_ != ResizeEdge::NoEdge; }
        void beginTitlebarDrag(int local_x, int local_y);
        void updateTitlebarDrag();
        void finishTitlebarDrag();
        void finishTitlebarDragIfReleased();
        [[nodiscard]] bool isSdlMaximized() const;
        void saveBorderlessRestoreGeometry();
        void normalizeNativeMaximize(const char* reason);
        void maximizeBorderless(const char* reason, bool save_restore_geometry);
        void restoreMaximized(const char* reason);
        void restoreMaximizedForTitlebarDrag();
        void updateManualTitlebarMove();
        [[nodiscard]] bool titlebarDragMovedEnough() const;
        [[nodiscard]] bool isTitlebarDragAtDisplayTop() const;
        void flushPendingTitlebarDoubleClick();
    };

} // namespace lfs::vis
