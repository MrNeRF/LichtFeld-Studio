/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/frame_input_buffer.hpp"
#include "input/input_router.hpp"
#include "visualizer/visualizer.hpp"
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
        WindowManager(const std::string& title, int width, int height,
                      int monitor_x = 0, int monitor_y = 0,
                      int monitor_width = 0, int monitor_height = 0,
                      GraphicsBackend graphics_backend = GraphicsBackend::Vulkan);
        ~WindowManager();

        WindowManager(const WindowManager&) = delete;
        WindowManager& operator=(const WindowManager&) = delete;

        bool init();

        void showWindow();
        void updateWindowSize();
        void swapBuffers();
        void pollEvents();
        void waitEvents(double timeout_seconds);
        bool shouldClose() const;
        void requestClose() { should_close_ = true; }
        void cancelClose();
        void wakeEventLoop();

        SDL_Window* getWindow() const { return window_; }
        VulkanContext* getVulkanContext() const { return vulkan_context_.get(); }
        glm::ivec2 getWindowSize() const { return window_size_; }
        glm::ivec2 getFramebufferSize() const { return framebuffer_size_; }
        bool isFullscreen() const { return is_fullscreen_; }
        void toggleFullscreen();
        GraphicsBackend graphicsBackend() const { return graphics_backend_; }
        bool isVulkan() const { return true; }

        void setCallbackHandler(void* handler) { callback_handler_ = handler; }
        void setInputController(InputController* ic);
        [[nodiscard]] const FrameInputBuffer& frameInput() const { return frame_input_; }
        [[nodiscard]] const input::InputRouter& inputRouter() const { return input_router_; }

    private:
        void processEvent(const ::SDL_Event& event);

        SDL_Window* window_ = nullptr;
        std::unique_ptr<VulkanContext> vulkan_context_;
        GraphicsBackend graphics_backend_ = GraphicsBackend::Vulkan;
        std::string title_;
        glm::ivec2 window_size_;
        glm::ivec2 framebuffer_size_;

        glm::ivec2 monitor_pos_{0, 0};
        glm::ivec2 monitor_size_{0, 0};

        bool is_fullscreen_ = false;
        glm::ivec2 windowed_pos_{0, 0};
        glm::ivec2 windowed_size_{1280, 720};
        bool should_close_ = false;

        static void* callback_handler_;
        InputController* input_controller_ = nullptr;
        input::InputRouter input_router_;
        FrameInputBuffer frame_input_;
        std::vector<std::string> pending_drop_files_;
    };

} // namespace lfs::vis
