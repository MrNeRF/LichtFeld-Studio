/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <filesystem>
#include <glm/glm.hpp>
#include <string>

// Forward declarations
struct GLFWwindow;

namespace lfs::vis {

    class WindowManager {
    public:
        WindowManager(const std::string& title, int width, int height,
                      int monitor_x = 0, int monitor_y = 0,
                      int monitor_width = 0, int monitor_height = 0);
        ~WindowManager();

        // Delete copy operations
        WindowManager(const WindowManager&) = delete;
        WindowManager& operator=(const WindowManager&) = delete;

        // Initialize GLFW and create window
        bool init();

        // Window operations
        void showWindow(); // Show window (call after initialization complete)
        void updateWindowSize();
        void swapBuffers();
        void pollEvents();
        void waitEvents(double timeout_seconds); // Sleep until event or timeout
        bool shouldClose() const;
        void cancelClose();
        void requestRedraw();
        bool needsRedraw() const;

        // Getters
        GLFWwindow* getWindow() const { return window_; }
        glm::ivec2 getWindowSize() const { return window_size_; }
        glm::ivec2 getFramebufferSize() const { return framebuffer_size_; }
        bool isFullscreen() const { return is_fullscreen_; }
        void toggleFullscreen();

        // Set the callback handler (typically the viewer instance)
        void setCallbackHandler(void* handler) { callback_handler_ = handler; }

    private:
        GLFWwindow* window_ = nullptr;
        std::string title_;
        glm::ivec2 window_size_;
        glm::ivec2 framebuffer_size_;

        glm::ivec2 monitor_pos_{0, 0};
        glm::ivec2 monitor_size_{0, 0};

        // Fullscreen state
        bool is_fullscreen_ = false;
        glm::ivec2 windowed_pos_{0, 0};
        glm::ivec2 windowed_size_{1280, 720};

        // Static callback handler pointer
        static void* callback_handler_;
        mutable std::atomic<bool> needs_redraw_{false}; // Redraw flag

        // Static GLFW callbacks
        static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        static void cursorPosCallback(GLFWwindow* window, double x, double y);
        static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void dropCallback(GLFWwindow* window, int count, const char** paths);
    };

} // namespace lfs::vis