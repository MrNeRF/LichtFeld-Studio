/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "window_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "input/input_controller.hpp"
#include "input/sdl_key_mapping.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "vulkan_context.hpp"
#include "vulkan_loader_probe.hpp"
#include <SDL3/SDL.h>
#include <cstdlib>
#include <cstring>
#include <imgui_impl_sdl3.h>
#include <iostream>
#include <string>
#include <imgui.h>

namespace lfs::vis {

    namespace {
        bool eventTargetsWindow(const SDL_Event& event, const SDL_WindowID target_window_id) {
            if (target_window_id == 0)
                return true;

            switch (event.type) {
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            case SDL_EVENT_WINDOW_FOCUS_LOST:
            case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
                return event.window.windowID == target_window_id;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
            case SDL_EVENT_MOUSE_BUTTON_UP:
                return event.button.windowID == target_window_id;
            case SDL_EVENT_MOUSE_MOTION:
                return event.motion.windowID == target_window_id;
            case SDL_EVENT_MOUSE_WHEEL:
                return event.wheel.windowID == target_window_id;
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP:
                return event.key.windowID == target_window_id;
            case SDL_EVENT_TEXT_INPUT:
                return event.text.windowID == target_window_id;
            case SDL_EVENT_DROP_FILE:
            case SDL_EVENT_DROP_COMPLETE:
                return event.drop.windowID == target_window_id;
            default:
                return true;
            }
        }

        std::string compiledVideoDrivers() {
            const int num_drivers = SDL_GetNumVideoDrivers();
            if (num_drivers <= 0) {
                return "<none>";
            }

            std::string result;
            for (int i = 0; i < num_drivers; ++i) {
                const char* const driver = SDL_GetVideoDriver(i);
                if (i > 0) {
                    result += ", ";
                }
                result += driver ? driver : "<null>";
            }
            return result;
        }

        bool hasCompiledVideoDriver(const char* const expected_driver) {
            const int num_drivers = SDL_GetNumVideoDrivers();
            for (int i = 0; i < num_drivers; ++i) {
                const char* const driver = SDL_GetVideoDriver(i);
                if (driver && std::strcmp(driver, expected_driver) == 0) {
                    return true;
                }
            }
            return false;
        }

        bool containsToken(const char* const haystack, const char* const needle) {
            return haystack && needle && std::strstr(haystack, needle) != nullptr;
        }

        bool shouldPreferX11OnGnome() {
#if defined(__linux__)
            // GNOME on Wayland can present undecorated SDL toplevels when the
            // compositor expects client-side decorations but libdecor is not
            // available at runtime. Prefer X11/Xwayland in that case so the
            // native min/max/close buttons remain available.
            const char* const current_desktop = std::getenv("XDG_CURRENT_DESKTOP");
            const char* const session_desktop = std::getenv("XDG_SESSION_DESKTOP");
            const bool is_gnome = containsToken(current_desktop, "GNOME") ||
                                  containsToken(session_desktop, "gnome") ||
                                  containsToken(session_desktop, "GNOME");
            const bool has_wayland = std::getenv("WAYLAND_DISPLAY") != nullptr;
            const bool has_x11 = std::getenv("DISPLAY") != nullptr;
            const bool explicit_driver = std::getenv("SDL_VIDEO_DRIVER") != nullptr;
            return is_gnome && has_wayland && has_x11 && !explicit_driver;
#else
            return false;
#endif
        }

        void reportSdlVideoInitFailure() {
            std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;

#if defined(__linux__)
            std::cerr << "Compiled SDL video drivers: " << compiledVideoDrivers() << std::endl;
            if (!hasCompiledVideoDriver("x11") && !hasCompiledVideoDriver("wayland")) {
                std::cerr
                    << "This SDL build lacks both X11 and Wayland support. Install the Linux GUI build "
                       "dependencies and rebuild SDL3."
                    << std::endl;
            }
#endif
        }
    } // namespace

    void* WindowManager::callback_handler_ = nullptr;

    WindowManager::WindowManager(const std::string& title, const int width, const int height,
                                 const int monitor_x, const int monitor_y,
                                 const int monitor_width, const int monitor_height,
                                 const GraphicsBackend graphics_backend)
        : graphics_backend_(graphics_backend),
          title_(title),
          window_size_(width, height),
          framebuffer_size_(width, height),
          monitor_pos_(monitor_x, monitor_y),
          monitor_size_(monitor_width, monitor_height) {
    }

    WindowManager::~WindowManager() {
        vulkan_context_.reset();
        if (window_) {
            SDL_DestroyWindow(window_);
        }
        SDL_Quit();
    }

    void WindowManager::setInputController(InputController* ic) {
        input_controller_ = ic;
        input_router_.setInputController(ic);
        if (input_controller_) {
            input_controller_->setInputRouter(&input_router_);
        }
    }

    bool WindowManager::init() {
        if (shouldPreferX11OnGnome()) {
            SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11,wayland");
            LOG_INFO("GNOME Wayland session detected; preferring X11/Xwayland for native window decorations");
        }

        if (!SDL_Init(SDL_INIT_VIDEO)) {
            reportSdlVideoInitFailure();
            return false;
        }

        if (const char* const video_driver = SDL_GetCurrentVideoDriver(); video_driver) {
            LOG_INFO("SDL video driver: {}", video_driver);
        }

        const auto vulkan_info = probeVulkanLoader();
        if (vulkan_info.enabled) {
            if (vulkan_info.loader_available) {
                LOG_INFO("Vulkan loader available: API {}", formatVulkanApiVersion(vulkan_info.api_version));
            } else {
                LOG_WARN("Vulkan viewer dependency is enabled, but the loader probe failed: {}", vulkan_info.error);
            }
        }

        window_ = SDL_CreateWindow(
            title_.c_str(),
            window_size_.x,
            window_size_.y,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN);

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

        int fb_w = 0;
        int fb_h = 0;
        SDL_GetWindowSizeInPixels(window_, &fb_w, &fb_h);
        framebuffer_size_ = glm::ivec2(fb_w, fb_h);

        vulkan_context_ = std::make_unique<VulkanContext>();
        if (!vulkan_context_->init(window_, framebuffer_size_.x, framebuffer_size_.y)) {
            std::cerr << "Failed to initialize Vulkan context: " << vulkan_context_->lastError() << std::endl;
            vulkan_context_.reset();
            SDL_DestroyWindow(window_);
            window_ = nullptr;
            SDL_Quit();
            return false;
        }
        lfs::rendering::setExpectedVulkanDeviceUuid(vulkan_context_->deviceUUID());
        if (!vulkan_context_->presentBootstrapFrame(0.11f, 0.11f, 0.14f, 1.0f)) {
            std::cerr << "Failed to present Vulkan bootstrap frame: " << vulkan_context_->lastError() << std::endl;
            vulkan_context_.reset();
            SDL_DestroyWindow(window_);
            window_ = nullptr;
            SDL_Quit();
            return false;
        }
        LOG_INFO("Vulkan window context initialized");
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
        if (vulkan_context_) {
            vulkan_context_->notifyFramebufferResized(fbW, fbH);
        }
    }

    void WindowManager::swapBuffers() {
        if (vulkan_context_) {
            if (!vulkan_context_->presentBootstrapFrame(0.11f, 0.11f, 0.14f, 1.0f)) {
                LOG_WARN("Vulkan bootstrap present failed: {}", vulkan_context_->lastError());
            }
        }
    }

    void WindowManager::pollEvents() {
        frame_input_.beginFrame();
        const bool imgui_ready = ImGui::GetCurrentContext() != nullptr;
        const SDL_WindowID main_window_id = window_ ? SDL_GetWindowID(window_) : 0;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (imgui_ready)
                ImGui_ImplSDL3_ProcessEvent(&event);
            frame_input_.processEvent(event, main_window_id);
            processEvent(event);
        }
        frame_input_.finalize(window_);
    }

    void WindowManager::waitEvents(double timeout_seconds) {
        frame_input_.beginFrame();
        const bool imgui_ready = ImGui::GetCurrentContext() != nullptr;
        const SDL_WindowID main_window_id = window_ ? SDL_GetWindowID(window_) : 0;
        SDL_Event event;
        const int timeout_ms = static_cast<int>(timeout_seconds * 1000.0);
        if (SDL_WaitEventTimeout(&event, timeout_ms)) {
            if (imgui_ready)
                ImGui_ImplSDL3_ProcessEvent(&event);
            frame_input_.processEvent(event, main_window_id);
            processEvent(event);
            while (SDL_PollEvent(&event)) {
                if (imgui_ready)
                    ImGui_ImplSDL3_ProcessEvent(&event);
                frame_input_.processEvent(event, main_window_id);
                processEvent(event);
            }
        }
        frame_input_.finalize(window_);
    }

    bool WindowManager::shouldClose() const {
        return should_close_;
    }

    void WindowManager::cancelClose() {
        should_close_ = false;
    }

    void WindowManager::wakeEventLoop() {
        if (!SDL_WasInit(SDL_INIT_EVENTS)) {
            return;
        }

        // Wake SDL_WaitEventTimeout so queued viewer-thread work is serviced promptly.
        SDL_Event event{};
        event.type = SDL_EVENT_USER;
        SDL_PushEvent(&event);
    }

    void WindowManager::processEvent(const SDL_Event& event) {
        const SDL_WindowID main_window_id = window_ ? SDL_GetWindowID(window_) : 0;

        switch (event.type) {
        case SDL_EVENT_QUIT:
            should_close_ = true;
            break;

        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            should_close_ = true;
            break;

        case SDL_EVENT_WINDOW_FOCUS_LOST:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            lfs::core::events::internal::WindowFocusLost{}.emit();
            input_router_.onWindowFocusLost();
            if (input_controller_) {
                input_controller_->onWindowFocusLost();
            }
            break;

        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (window_) {
                const float scale = SDL_GetWindowDisplayScale(window_);
                lfs::core::events::internal::DisplayScaleChanged{.scale = scale}.emit();
            }
            break;

        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP: {
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (!input_controller_)
                break;
            const int button = input::sdlMouseButtonToApp(event.button.button);
            const int action = (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) ? input::ACTION_PRESS : input::ACTION_RELEASE;
            input_router_.beginMouseButton(action, event.button.x, event.button.y);
            input_controller_->handleMouseButton(button, action, event.button.x, event.button.y);
            input_router_.endMouseButton(action);
            break;
        }

        case SDL_EVENT_MOUSE_MOTION:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (input_controller_) {
                input_controller_->handleMouseMove(event.motion.x, event.motion.y);
            }
            break;

        case SDL_EVENT_MOUSE_WHEEL:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (input_controller_) {
                input_controller_->handleScroll(event.wheel.x, event.wheel.y);
            }
            break;

        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (!input_controller_)
                break;
            const int physical_key = input::sdlScancodeToAppKey(event.key.scancode);
            // Resolve the unmodified layout key so bindings keep modifiers separate
            // (for example, '=' + Shift stays KEY_EQUAL plus a Shift modifier).
            int logical_key = input::sdlKeycodeToAppKey(
                SDL_GetKeyFromScancode(event.key.scancode, SDL_KMOD_NONE, false));
            if (logical_key == input::KEY_UNKNOWN) {
                logical_key = physical_key;
            }
            const int action = event.key.down
                                   ? (event.key.repeat ? input::ACTION_REPEAT : input::ACTION_PRESS)
                                   : input::ACTION_RELEASE;
            const int mods = input::sdlModsToAppMods(event.key.mod);
            input_controller_->handleKey(
                physical_key, logical_key, static_cast<int>(event.key.scancode), action, mods);
            break;
        }

        case SDL_EVENT_DROP_FILE:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (event.drop.data) {
                pending_drop_files_.emplace_back(event.drop.data);
            }
            break;

        case SDL_EVENT_DROP_COMPLETE:
            if (!eventTargetsWindow(event, main_window_id))
                break;
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
        wakeEventLoop();
    }

} // namespace lfs::vis
