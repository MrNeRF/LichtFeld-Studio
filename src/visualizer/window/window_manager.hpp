/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/frame_input_buffer.hpp"
#include "input/input_router.hpp"
#include "visualizer/visualizer.hpp"
#include <chrono>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

struct SDL_Window;
union SDL_Event;

namespace lfs::vis {

    class InputController;
    class VulkanContext;

    class WindowManager {
    public:
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
                      int monitor_width = 0, int monitor_height = 0,
                      GraphicsBackend graphics_backend = GraphicsBackend::Vulkan);
        ~WindowManager();

        WindowManager(const WindowManager&) = delete;
        WindowManager& operator=(const WindowManager&) = delete;

        bool init();

        void showWindow();
        void updateWindowSize(const char* reason = "manual",
                              ResizeIntent intent = ResizeIntent::Exact);
        void swapBuffers();
        void pollEvents();
        void waitEvents(double timeout_seconds);
        bool shouldClose() const;
        void requestClose() { should_close_ = true; }
        void cancelClose();
        void wakeEventLoop();
        void refreshResizeCursor();
        [[nodiscard]] unsigned manualResizeEdgeMask() const;

        SDL_Window* getWindow() const { return window_; }
        VulkanContext* getVulkanContext() const { return vulkan_context_.get(); }
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
        [[nodiscard]] bool usesEventDrivenTitlebarDrag() const { return native_titlebar_move_available_; }
        void setFullscreen(bool fullscreen);
        GraphicsBackend graphicsBackend() const { return graphics_backend_; }
        bool isVulkan() const { return true; }

        void setCallbackHandler(void* handler) { callback_handler_ = handler; }
        void setInputController(InputController* ic);
        [[nodiscard]] const FrameInputBuffer& frameInput() const { return frame_input_; }
        [[nodiscard]] const input::InputRouter& inputRouter() const { return input_router_; }

    private:
        void processEvent(const ::SDL_Event& event);
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
        std::unique_ptr<VulkanContext> vulkan_context_;
        GraphicsBackend graphics_backend_ = GraphicsBackend::Vulkan;
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
        bool should_close_ = false;

        static void* callback_handler_;
        InputController* input_controller_ = nullptr;
        input::InputRouter input_router_;
        FrameInputBuffer frame_input_;
        std::vector<std::string> pending_drop_files_;

        void beginTitlebarNativeMove();
        [[nodiscard]] ResizeEdge resizeEdgeAt(int x, int y) const;
        void setResizeCursorForEdge(ResizeEdge edge);
        void updateResizeCursor(int x, int y);
        void beginManualResize(ResizeEdge edge);
        void updateManualResize();
        void finishManualResize();
        void suppressFrameInputForManualResize();
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
