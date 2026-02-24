/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "window_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "input/input_controller.hpp"
#include "input/sdl_key_mapping.hpp"
#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <imgui_impl_sdl3.h>
#include <iostream>
#include <imgui.h>

namespace lfs::vis {

    void* WindowManager::callback_handler_ = nullptr;

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
        if (gl_context_) {
            SDL_GL_DestroyContext(gl_context_);
        }
        if (window_) {
            SDL_DestroyWindow(window_);
        }
        SDL_Quit();
    }

    bool WindowManager::init() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
            return false;
        }

        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 8);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

        window_ = SDL_CreateWindow(
            title_.c_str(),
            window_size_.x,
            window_size_.y,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN);

        if (!window_) {
            std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return false;
        }

        // Position window on specified monitor (if provided)
        if (monitor_size_.x > 0 && monitor_size_.y > 0) {
            const int xpos = monitor_pos_.x + (monitor_size_.x - window_size_.x) / 2;
            const int ypos = monitor_pos_.y + (monitor_size_.y - window_size_.y) / 2;
            SDL_SetWindowPosition(window_, xpos, ypos);
        }

        gl_context_ = SDL_GL_CreateContext(window_);
        if (!gl_context_) {
            std::cerr << "Failed to create GL context: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window_);
            window_ = nullptr;
            SDL_Quit();
            return false;
        }
        SDL_GL_MakeCurrent(window_, gl_context_);

        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
            std::cerr << "GLAD init failed" << std::endl;
            SDL_Quit();
            return false;
        }

        SDL_GL_SetSwapInterval(1);

        glEnable(GL_LINE_SMOOTH);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glEnable(GL_PROGRAM_POINT_SIZE);

        glClearColor(0.11f, 0.11f, 0.14f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        SDL_GL_SwapWindow(window_);

        return true;
    }

    void WindowManager::showWindow() {
        if (window_) {
            SDL_ShowWindow(window_);
            SDL_RaiseWindow(window_);
        }
    }

    void WindowManager::updateWindowSize() {
        int winW, winH, fbW, fbH;
        SDL_GetWindowSize(window_, &winW, &winH);
        SDL_GetWindowSizeInPixels(window_, &fbW, &fbH);
        window_size_ = glm::ivec2(winW, winH);
        framebuffer_size_ = glm::ivec2(fbW, fbH);
        glViewport(0, 0, fbW, fbH);
    }

    void WindowManager::swapBuffers() {
        SDL_GL_SwapWindow(window_);
    }

    void WindowManager::pollEvents() {
        const bool imgui_ready = ImGui::GetCurrentContext() != nullptr;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (imgui_ready)
                ImGui_ImplSDL3_ProcessEvent(&event);
            processEvent(event);
        }
    }

    void WindowManager::waitEvents(double timeout_seconds) {
        const bool imgui_ready = ImGui::GetCurrentContext() != nullptr;
        SDL_Event event;
        const int timeout_ms = static_cast<int>(timeout_seconds * 1000.0);
        if (SDL_WaitEventTimeout(&event, timeout_ms)) {
            if (imgui_ready)
                ImGui_ImplSDL3_ProcessEvent(&event);
            processEvent(event);
            while (SDL_PollEvent(&event)) {
                if (imgui_ready)
                    ImGui_ImplSDL3_ProcessEvent(&event);
                processEvent(event);
            }
        }
    }

    bool WindowManager::shouldClose() const {
        return should_close_;
    }

    void WindowManager::cancelClose() {
        should_close_ = false;
    }

    void WindowManager::requestRedraw() {
        needs_redraw_ = true;
        SDL_Event event{};
        event.type = SDL_EVENT_USER;
        SDL_PushEvent(&event);
    }

    bool WindowManager::needsRedraw() const {
        bool result = needs_redraw_;
        if (result) {
            needs_redraw_ = false;
        }
        return result;
    }

    void WindowManager::processEvent(const SDL_Event& event) {
        switch (event.type) {
        case SDL_EVENT_QUIT:
            should_close_ = true;
            break;

        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            should_close_ = true;
            break;

        case SDL_EVENT_WINDOW_FOCUS_LOST:
            lfs::core::events::internal::WindowFocusLost{}.emit();
            if (input_controller_) {
                input_controller_->onWindowFocusLost();
            }
            break;

        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            if (window_) {
                const float scale = SDL_GetWindowDisplayScale(window_);
                lfs::core::events::internal::DisplayScaleChanged{.scale = scale}.emit();
            }
            break;

        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP: {
            if (!input_controller_)
                break;
            const int button = input::sdlMouseButtonToApp(event.button.button);
            const int action = (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) ? input::ACTION_PRESS : input::ACTION_RELEASE;
            input_controller_->handleMouseButton(button, action, event.button.x, event.button.y);
            break;
        }

        case SDL_EVENT_MOUSE_MOTION:
            if (input_controller_) {
                input_controller_->handleMouseMove(event.motion.x, event.motion.y);
            }
            break;

        case SDL_EVENT_MOUSE_WHEEL:
            if (input_controller_) {
                input_controller_->handleScroll(event.wheel.x, event.wheel.y);
            }
            break;

        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            if (!input_controller_)
                break;
            const int key = input::sdlKeycodeToAppKey(event.key.key);
            const int action = event.key.down
                                   ? (event.key.repeat ? input::ACTION_REPEAT : input::ACTION_PRESS)
                                   : input::ACTION_RELEASE;
            const int mods = input::sdlModsToAppMods(event.key.mod);
            input_controller_->handleKey(key, action, mods);
            break;
        }

        case SDL_EVENT_DROP_FILE:
            if (event.drop.data) {
                pending_drop_files_.emplace_back(event.drop.data);
            }
            break;

        case SDL_EVENT_DROP_COMPLETE:
            if (input_controller_ && !pending_drop_files_.empty()) {
                input_controller_->handleFileDrop(pending_drop_files_);
                pending_drop_files_.clear();
            }
            break;

        default:
            break;
        }
    }

    void WindowManager::toggleFullscreen() {
        if (!window_)
            return;

        if (is_fullscreen_) {
            SDL_SetWindowFullscreen(window_, false);
            SDL_SetWindowPosition(window_, windowed_pos_.x, windowed_pos_.y);
            SDL_SetWindowSize(window_, windowed_size_.x, windowed_size_.y);
            is_fullscreen_ = false;
        } else {
            SDL_GetWindowPosition(window_, &windowed_pos_.x, &windowed_pos_.y);
            SDL_GetWindowSize(window_, &windowed_size_.x, &windowed_size_.y);
            SDL_SetWindowFullscreen(window_, true);
            is_fullscreen_ = true;
        }

        updateWindowSize();
        requestRedraw();
    }

} // namespace lfs::vis
