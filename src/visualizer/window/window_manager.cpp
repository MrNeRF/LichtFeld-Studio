/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "window_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
// clang-format off
// GLAD must be included before GLFW
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <iostream>

namespace lfs::vis {

    void* WindowManager::callback_handler_ = nullptr;

    static void window_focus_callback(GLFWwindow*, int focused) {
        if (!focused) {
            lfs::core::events::internal::WindowFocusLost{}.emit();
            LOG_DEBUG("Window lost focus");
        } else {
            LOG_DEBUG("Window gained focus");
        }
    }

    WindowManager::WindowManager(const std::string& title, const int width, const int height,
                                 const int monitor_x, const int monitor_y,
                                 const int monitor_width, const int monitor_height)
        : title_(title),
          window_size_(width, height),
          framebuffer_size_(width, height),
          monitor_pos_(monitor_x, monitor_y),
          monitor_size_(monitor_width, monitor_height) {
    }

    WindowManager::~WindowManager() {
        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    bool WindowManager::init() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW!" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_SAMPLES, 8);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_FALSE);
        glfwWindowHint(GLFW_DEPTH_BITS, 24);

        window_ = glfwCreateWindow(
            window_size_.x,
            window_size_.y,
            title_.c_str(),
            nullptr,
            nullptr);

        if (!window_) {
            std::cerr << "Failed to create GLFW window!" << std::endl;
            glfwTerminate();
            return false;
        }

        // Position window on specified monitor (if provided)
        if (monitor_size_.x > 0 && monitor_size_.y > 0) {
            const int xpos = monitor_pos_.x + (monitor_size_.x - window_size_.x) / 2;
            const int ypos = monitor_pos_.y + (monitor_size_.y - window_size_.y) / 2;
            glfwSetWindowPos(window_, xpos, ypos);
        }

        glfwMakeContextCurrent(window_);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "GLAD init failed" << std::endl;
            glfwTerminate();
            return false;
        }

        // Set window focus callback
        glfwSetWindowFocusCallback(window_, window_focus_callback);

        // Enable VSync for smooth frame delivery and reduced GPU load
        glfwSwapInterval(1);

        // Set up OpenGL state
        glEnable(GL_LINE_SMOOTH);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glEnable(GL_PROGRAM_POINT_SIZE);

        // Clear to dark background immediately
        glClearColor(0.11f, 0.11f, 0.14f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glfwSwapBuffers(window_);
        glfwPollEvents();

        return true;
    }

    void WindowManager::showWindow() {
        if (window_) {
            glfwShowWindow(window_);
            glfwFocusWindow(window_);
        }
    }

    void WindowManager::updateWindowSize() {
        int winW, winH, fbW, fbH;
        glfwGetWindowSize(window_, &winW, &winH);
        glfwGetFramebufferSize(window_, &fbW, &fbH);
        window_size_ = glm::ivec2(winW, winH);
        framebuffer_size_ = glm::ivec2(fbW, fbH);
        glViewport(0, 0, fbW, fbH);
    }

    void WindowManager::swapBuffers() {
        glfwSwapBuffers(window_);
    }

    void WindowManager::pollEvents() {
        glfwPollEvents();
    }

    void WindowManager::waitEvents(double timeout_seconds) {
        glfwWaitEventsTimeout(timeout_seconds);
    }

    bool WindowManager::shouldClose() const {
        return glfwWindowShouldClose(window_);
    }

    void WindowManager::cancelClose() {
        glfwSetWindowShouldClose(window_, false);
    }

    void WindowManager::requestRedraw() {
        // Set a flag that we need a redraw
        needs_redraw_ = true;
        // Post an empty event to wake up the event loop
        glfwPostEmptyEvent();
    }

    bool WindowManager::needsRedraw() const {
        bool result = needs_redraw_;
        if (result) {
            needs_redraw_ = false; // Reset the flag
        }
        return result;
    }

    void WindowManager::toggleFullscreen() {
        if (!window_)
            return;

        if (is_fullscreen_) {
            glfwSetWindowMonitor(window_, nullptr,
                                 windowed_pos_.x, windowed_pos_.y,
                                 windowed_size_.x, windowed_size_.y,
                                 GLFW_DONT_CARE);
            is_fullscreen_ = false;
        } else {
            glfwGetWindowPos(window_, &windowed_pos_.x, &windowed_pos_.y);
            glfwGetWindowSize(window_, &windowed_size_.x, &windowed_size_.y);

            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            int monitor_count = 0;
            GLFWmonitor** const monitors = glfwGetMonitors(&monitor_count);
            const int cx = windowed_pos_.x + windowed_size_.x / 2;
            const int cy = windowed_pos_.y + windowed_size_.y / 2;

            for (int i = 0; i < monitor_count; ++i) {
                int mx = 0, my = 0;
                glfwGetMonitorPos(monitors[i], &mx, &my);
                const auto* const mode = glfwGetVideoMode(monitors[i]);
                if (cx >= mx && cx < mx + mode->width && cy >= my && cy < my + mode->height) {
                    monitor = monitors[i];
                    break;
                }
            }

            const auto* const mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(window_, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
            is_fullscreen_ = true;
        }

        updateWindowSize();
        requestRedraw();
    }

} // namespace lfs::vis
