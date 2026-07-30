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
#if defined(__linux__)
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

namespace lfs::vis {

    namespace {
        constexpr double kPendingResizeMinWaitSeconds = 0.001;
        constexpr int kResizeBorder = 6;
        constexpr int kMinWindowWidth = 640;
        constexpr int kMinWindowHeight = 360;
#if defined(_WIN32)
        constexpr bool kUseManualBorderlessResize = true;
#else
        constexpr bool kUseManualBorderlessResize = false;
#endif
        constexpr unsigned kResizeLeft = 1u << 0;
        constexpr unsigned kResizeRight = 1u << 1;
        constexpr unsigned kResizeTop = 1u << 2;
        constexpr unsigned kResizeBottom = 1u << 3;

        void configureValidationLayerSearchPath() {
#ifdef LFS_VULKAN_VALIDATION_LAYER_DIR
#ifdef _WIN32
            constexpr char path_separator = ';';
#else
            constexpr char path_separator = ':';
#endif
            std::string layer_path = LFS_VULKAN_VALIDATION_LAYER_DIR;
            if (const char* const existing_path = std::getenv("VK_ADD_LAYER_PATH");
                existing_path && *existing_path) {
                layer_path += path_separator;
                layer_path += existing_path;
            }

#ifdef _WIN32
            const bool configured = _putenv_s("VK_ADD_LAYER_PATH", layer_path.c_str()) == 0;
#else
            const bool configured = ::setenv("VK_ADD_LAYER_PATH", layer_path.c_str(), 1) == 0;
#endif
            if (!configured) {
                LOG_WARN("Failed to configure the pinned Vulkan validation layer path");
            } else if (const char* const override_path = std::getenv("VK_LAYER_PATH");
                       override_path && *override_path) {
                LOG_WARN("VK_LAYER_PATH overrides the pinned Vulkan validation layer path: {}", override_path);
            } else {
                LOG_INFO("Vulkan validation layer path: {}", LFS_VULKAN_VALIDATION_LAYER_DIR);
            }
#endif
        }

        const char* windowEventName(const Uint32 event_type) {
            switch (event_type) {
            case SDL_EVENT_WINDOW_RESIZED:
                return "WINDOW_RESIZED";
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                return "WINDOW_PIXEL_SIZE_CHANGED";
            case SDL_EVENT_WINDOW_MINIMIZED:
                return "WINDOW_MINIMIZED";
            case SDL_EVENT_WINDOW_MAXIMIZED:
                return "WINDOW_MAXIMIZED";
            case SDL_EVENT_WINDOW_RESTORED:
                return "WINDOW_RESTORED";
            case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
                return "WINDOW_ENTER_FULLSCREEN";
            case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
                return "WINDOW_LEAVE_FULLSCREEN";
            case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
                return "WINDOW_DISPLAY_SCALE_CHANGED";
            default:
                return "WINDOW_EVENT";
            }
        }

        bool eventTargetsWindow(const SDL_Event& event, const SDL_WindowID target_window_id) {
            if (target_window_id == 0)
                return true;

            switch (event.type) {
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            case SDL_EVENT_WINDOW_FOCUS_LOST:
            case SDL_EVENT_WINDOW_RESIZED:
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_MINIMIZED:
            case SDL_EVENT_WINDOW_MAXIMIZED:
            case SDL_EVENT_WINDOW_RESTORED:
            case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
            case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
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

        glm::ivec2 globalMousePosition() {
            float global_x = 0.0f;
            float global_y = 0.0f;
            SDL_GetGlobalMouseState(&global_x, &global_y);
            return {
                static_cast<int>(std::lround(global_x)),
                static_cast<int>(std::lround(global_y)),
            };
        }

#if defined(__linux__)
        bool getX11WindowHandle(SDL_Window* const window, Display*& display, ::Window& xwindow) {
            if (!window)
                return false;

            const SDL_PropertiesID props = SDL_GetWindowProperties(window);
            if (!props)
                return false;

            display = static_cast<Display*>(
                SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr));
            xwindow = static_cast<::Window>(
                SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
            return display != nullptr && xwindow != 0;
        }

        bool hasX11NativeMoveSupport(SDL_Window* const window) {
            Display* display = nullptr;
            ::Window xwindow = 0;
            return getX11WindowHandle(window, display, xwindow);
        }

        bool requestX11WindowMove(SDL_Window* const window) {
            Display* display = nullptr;
            ::Window xwindow = 0;
            if (!getX11WindowHandle(window, display, xwindow))
                return false;

            const glm::ivec2 global_mouse = globalMousePosition();

            const Atom moveresize = XInternAtom(display, "_NET_WM_MOVERESIZE", False);
            if (moveresize == None)
                return false;

            XEvent event{};
            event.xclient.type = ClientMessage;
            event.xclient.display = display;
            event.xclient.window = xwindow;
            event.xclient.message_type = moveresize;
            event.xclient.format = 32;
            event.xclient.data.l[0] = static_cast<long>(global_mouse.x);
            event.xclient.data.l[1] = static_cast<long>(global_mouse.y);
            event.xclient.data.l[2] = 8; // _NET_WM_MOVERESIZE_MOVE
            event.xclient.data.l[3] = 1; // left mouse button
            event.xclient.data.l[4] = 1; // normal application source

            XUngrabPointer(display, CurrentTime);
            const int sent = XSendEvent(display,
                                        DefaultRootWindow(display),
                                        False,
                                        SubstructureRedirectMask | SubstructureNotifyMask,
                                        &event);
            XFlush(display);
            return sent != 0;
        }
#else
        bool hasX11NativeMoveSupport(SDL_Window*) {
            return false;
        }

        bool requestX11WindowMove(SDL_Window*) {
            return false;
        }
#endif

        SDL_HitTestResult resizeHitTestResult(const unsigned edge_mask) {
            const bool left = (edge_mask & kResizeLeft) != 0;
            const bool right = (edge_mask & kResizeRight) != 0;
            const bool top = (edge_mask & kResizeTop) != 0;
            const bool bottom = (edge_mask & kResizeBottom) != 0;

            if (top && left)
                return SDL_HITTEST_RESIZE_TOPLEFT;
            if (top && right)
                return SDL_HITTEST_RESIZE_TOPRIGHT;
            if (bottom && left)
                return SDL_HITTEST_RESIZE_BOTTOMLEFT;
            if (bottom && right)
                return SDL_HITTEST_RESIZE_BOTTOMRIGHT;
            if (left)
                return SDL_HITTEST_RESIZE_LEFT;
            if (right)
                return SDL_HITTEST_RESIZE_RIGHT;
            if (top)
                return SDL_HITTEST_RESIZE_TOP;
            if (bottom)
                return SDL_HITTEST_RESIZE_BOTTOM;
            return SDL_HITTEST_NORMAL;
        }

        SDL_HitTestResult SDLCALL borderlessWindowHitTest(SDL_Window* window, const SDL_Point* const area, void* data) {
            auto* const self = static_cast<WindowManager*>(data);
            if (!self || !area)
                return SDL_HITTEST_NORMAL;
            if (self->isFullscreen())
                return SDL_HITTEST_NORMAL;

            if constexpr (kUseManualBorderlessResize) {
                const unsigned active_resize_edge = self->manualResizeEdgeMask();
                if (active_resize_edge != 0)
                    return resizeHitTestResult(active_resize_edge);
            }

            glm::ivec2 size = self->getWindowSize();
            if (window)
                SDL_GetWindowSize(window, &size.x, &size.y);

            const bool left = area->x >= 0 && area->x < kResizeBorder;
            const bool right = area->x >= size.x - kResizeBorder && area->x < size.x;
            const bool top = area->y >= 0 && area->y < kResizeBorder;
            const bool bottom = area->y >= size.y - kResizeBorder && area->y < size.y;
            const bool titlebar_point = self->isTitlebarDragPoint(area->x, area->y);

            if (!self->isMaximized()) {
                unsigned edge_mask = 0;
                if (left)
                    edge_mask |= kResizeLeft;
                if (right)
                    edge_mask |= kResizeRight;
                if (top && !titlebar_point)
                    edge_mask |= kResizeTop;
                if (bottom)
                    edge_mask |= kResizeBottom;

                if constexpr (kUseManualBorderlessResize) {
                    if (edge_mask != 0)
                        return SDL_HITTEST_NORMAL;
                } else {
                    const SDL_HitTestResult resize_result = resizeHitTestResult(edge_mask);
                    if (resize_result != SDL_HITTEST_NORMAL)
                        return resize_result;
                }
            }

            if (titlebar_point) {
                if (self->isMaximized())
                    return SDL_HITTEST_NORMAL;
                if (self->usesEventDrivenTitlebarDrag())
                    return SDL_HITTEST_NORMAL;
                return SDL_HITTEST_DRAGGABLE;
            }

            return SDL_HITTEST_NORMAL;
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
        configureValidationLayerSearchPath();

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
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN |
                SDL_WINDOW_BORDERLESS);

        if (!window_) {
            std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return false;
        }
        if (!SDL_SetWindowMinimumSize(window_, kMinWindowWidth, kMinWindowHeight)) {
            LOG_DEBUG("Failed to set window minimum size: {}", SDL_GetError());
        }

        native_titlebar_move_available_ = hasX11NativeMoveSupport(window_);
        if (native_titlebar_move_available_) {
            LOG_DEBUG("Using X11 native titlebar move for borderless window drag");
        }

        if (!SDL_SetWindowHitTest(window_, borderlessWindowHitTest, this)) {
            LOG_DEBUG("SDL window hit testing unavailable: {}", SDL_GetError());
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

    void WindowManager::updateWindowSize(const char* const reason, const ResizeIntent intent) {
        if (!window_) {
            return;
        }

        int winW, winH, fbW, fbH;
        SDL_GetWindowSize(window_, &winW, &winH);
        SDL_GetWindowSizeInPixels(window_, &fbW, &fbH);
        const glm::ivec2 next_window_size(winW, winH);
        const glm::ivec2 next_framebuffer_size(fbW, fbH);
        const bool window_size_changed = next_window_size != window_size_;
        const bool framebuffer_size_changed = next_framebuffer_size != framebuffer_size_;
        const bool size_changed = window_size_changed || framebuffer_size_changed;

        const auto flags = SDL_GetWindowFlags(window_);
        if (size_changed) {
            last_window_size_change_time_ = std::chrono::steady_clock::now();
            LOG_DEBUG("Window size update [{}]: logical {}x{} -> {}x{}, framebuffer {}x{} -> {}x{}, fullscreen={}, flags=0x{:x}",
                      reason,
                      window_size_.x,
                      window_size_.y,
                      next_window_size.x,
                      next_window_size.y,
                      framebuffer_size_.x,
                      framebuffer_size_.y,
                      next_framebuffer_size.x,
                      next_framebuffer_size.y,
                      is_fullscreen_,
                      static_cast<unsigned>(flags));
        } else {
            LOG_DEBUG("Window size update [{}]: unchanged logical {}x{}, framebuffer {}x{}, fullscreen={}, flags=0x{:x}",
                      reason,
                      next_window_size.x,
                      next_window_size.y,
                      next_framebuffer_size.x,
                      next_framebuffer_size.y,
                      is_fullscreen_,
                      static_cast<unsigned>(flags));
        }

        window_size_ = next_window_size;
        framebuffer_size_ = next_framebuffer_size;
        if (vulkan_context_ && framebuffer_size_changed) {
            const ResizeIntent effective_intent =
                intent == ResizeIntent::Interactive && (is_fullscreen_ || isMaximized())
                    ? ResizeIntent::Exact
                    : intent;
            const auto vulkan_resize_intent =
                effective_intent == ResizeIntent::Interactive
                    ? VulkanContext::ResizeIntent::Interactive
                    : VulkanContext::ResizeIntent::Exact;
            vulkan_context_->notifyFramebufferResized(fbW, fbH, vulkan_resize_intent);
        }
        if (size_changed) {
            lfs::core::events::ui::WindowResized{.width = fbW, .height = fbH}.emit();
        }
    }

    bool WindowManager::hasRecentWindowSizeChange(
        const std::chrono::steady_clock::duration max_age) const {
        if (last_window_size_change_time_ == std::chrono::steady_clock::time_point{}) {
            return false;
        }
        return std::chrono::steady_clock::now() - last_window_size_change_time_ <= max_age;
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
        const SDL_WindowID main_window_id = window_ ? SDL_GetWindowID(window_) : 0;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            const bool suppress_gui_route = shouldSuppressGuiRoutingForResize(event, main_window_id);
            if (!suppress_gui_route)
                frame_input_.processEvent(event, main_window_id);
            processEvent(event);
        }
        if (isManualResizeActive())
            updateManualResize();
        finishTitlebarDragIfReleased();
        frame_input_.finalize(window_);
        suppressFrameInputForManualResize();
        flushPendingTitlebarDoubleClick();
    }

    void WindowManager::waitEvents(double timeout_seconds) {
        frame_input_.beginFrame();
        const SDL_WindowID main_window_id = window_ ? SDL_GetWindowID(window_) : 0;
        SDL_Event event;
        if (vulkan_context_ &&
            vulkan_context_->hasPendingSwapchainResize()) {
            const double resize_wait = vulkan_context_->secondsUntilPendingSwapchainResizeReady();
            timeout_seconds = std::min(timeout_seconds,
                                       resize_wait > 0.0
                                           ? std::max(kPendingResizeMinWaitSeconds, resize_wait)
                                           : 0.0);
        }
        if (isManualResizeActive())
            timeout_seconds = std::min(timeout_seconds, 1.0 / 60.0);
        const int timeout_ms = static_cast<int>(timeout_seconds * 1000.0);
        if (SDL_WaitEventTimeout(&event, timeout_ms)) {
            bool suppress_gui_route = shouldSuppressGuiRoutingForResize(event, main_window_id);
            if (!suppress_gui_route)
                frame_input_.processEvent(event, main_window_id);
            processEvent(event);
            while (SDL_PollEvent(&event)) {
                suppress_gui_route = shouldSuppressGuiRoutingForResize(event, main_window_id);
                if (!suppress_gui_route)
                    frame_input_.processEvent(event, main_window_id);
                processEvent(event);
            }
        }
        if (isManualResizeActive())
            updateManualResize();
        finishTitlebarDragIfReleased();
        frame_input_.finalize(window_);
        suppressFrameInputForManualResize();
        flushPendingTitlebarDoubleClick();
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

    bool WindowManager::shouldSuppressGuiRoutingForResize(const SDL_Event& event,
                                                          const unsigned int main_window_id) const {
        if constexpr (!kUseManualBorderlessResize)
            return false;

        switch (event.type) {
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
        case SDL_EVENT_MOUSE_MOTION:
        case SDL_EVENT_MOUSE_WHEEL:
            if (!eventTargetsWindow(event, main_window_id))
                return false;
            break;
        default:
            return false;
        }

        if (isManualResizeActive())
            return true;

        if (event.type != SDL_EVENT_MOUSE_BUTTON_DOWN ||
            event.button.button != SDL_BUTTON_LEFT) {
            return false;
        }

        const int mouse_x = static_cast<int>(std::round(event.button.x));
        const int mouse_y = static_cast<int>(std::round(event.button.y));
        return resizeEdgeAt(mouse_x, mouse_y) != ResizeEdge::NoEdge;
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
            titlebar_drag_active_ = false;
            finishManualResize();
            break;

        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (window_) {
                const float scale = SDL_GetWindowDisplayScale(window_);
                lfs::core::events::internal::DisplayScaleChanged{.scale = scale}.emit();
            }
            updateWindowSize(windowEventName(event.type), ResizeIntent::Exact);
            break;

        case SDL_EVENT_WINDOW_MAXIMIZED:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            LOG_DEBUG("SDL window event: {} data1={} data2={} fullscreen={}",
                      windowEventName(event.type),
                      event.window.data1,
                      event.window.data2,
                      is_fullscreen_);
            normalizeNativeMaximize("native-window-maximized");
            break;

        case SDL_EVENT_WINDOW_RESIZED:
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_MINIMIZED:
        case SDL_EVENT_WINDOW_RESTORED:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            LOG_DEBUG("SDL window event: {} data1={} data2={} fullscreen={}",
                      windowEventName(event.type),
                      event.window.data1,
                      event.window.data2,
                      is_fullscreen_);
            updateWindowSize(windowEventName(event.type),
                             event.type == SDL_EVENT_WINDOW_RESTORED ||
                                     event.type == SDL_EVENT_WINDOW_MINIMIZED
                                 ? ResizeIntent::Exact
                                 : ResizeIntent::Interactive);
            break;

        case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            is_fullscreen_ = true;
            LOG_DEBUG("SDL window event: {} data1={} data2={}",
                      windowEventName(event.type),
                      event.window.data1,
                      event.window.data2);
            updateWindowSize(windowEventName(event.type), ResizeIntent::Exact);
            break;

        case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
            if (!eventTargetsWindow(event, main_window_id))
                break;
            is_fullscreen_ = false;
            LOG_DEBUG("SDL window event: {} data1={} data2={}",
                      windowEventName(event.type),
                      event.window.data1,
                      event.window.data2);
            updateWindowSize(windowEventName(event.type), ResizeIntent::Exact);
            break;

        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP: {
            if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
                event.button.button == SDL_BUTTON_LEFT &&
                isManualResizeActive()) {
                finishManualResize();
                break;
            }

            if (!eventTargetsWindow(event, main_window_id))
                break;
            const int mouse_x = static_cast<int>(std::round(event.button.x));
            const int mouse_y = static_cast<int>(std::round(event.button.y));
            const bool titlebar_point = isTitlebarDragPoint(mouse_x, mouse_y);
            if (event.button.button == SDL_BUTTON_LEFT) {
                if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                    const ResizeEdge resize_edge = resizeEdgeAt(mouse_x, mouse_y);
                    if (resize_edge != ResizeEdge::NoEdge) {
                        if constexpr (kUseManualBorderlessResize) {
                            beginManualResize(resize_edge);
                            break;
                        }
                        break;
                    }
                }

                if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && titlebar_drag_active_) {
                    finishTitlebarDrag();
                    break;
                }

                if (titlebar_point && event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                    if (event.button.clicks >= 2 || pending_titlebar_double_click_) {
                        titlebar_drag_active_ = false;
                        pending_titlebar_double_click_ = true;
                    } else {
                        beginTitlebarDrag(mouse_x, mouse_y);
                        if (native_titlebar_move_available_ && !titlebar_drag_started_maximized_) {
                            beginTitlebarNativeMove();
                        }
                    }
                    break;
                }
            }
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
            if (isManualResizeActive()) {
                updateManualResize();
                break;
            }
            if (!eventTargetsWindow(event, main_window_id))
                break;
            if (titlebar_drag_active_) {
                updateTitlebarDrag();
                break;
            }
            if (input_controller_) {
                input_controller_->handleMouseMove(event.motion.x, event.motion.y);
            }
            updateResizeCursor(static_cast<int>(std::round(event.motion.x)),
                               static_cast<int>(std::round(event.motion.y)));
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

    void WindowManager::setFullscreen(const bool fullscreen) {
        if (!window_)
            return;

        LOG_DEBUG("setFullscreen request: target={}, current={}, logical={}x{}, framebuffer={}x{}",
                  fullscreen,
                  is_fullscreen_,
                  window_size_.x,
                  window_size_.y,
                  framebuffer_size_.x,
                  framebuffer_size_.y);

        if (fullscreen == is_fullscreen_) {
            LOG_DEBUG("setFullscreen request already satisfied: fullscreen={}", is_fullscreen_);
            return;
        }

        if (!fullscreen) {
            if (!SDL_SetWindowFullscreen(window_, false)) {
                LOG_WARN("Failed to leave fullscreen: {}", SDL_GetError());
                return;
            }
            SDL_SetWindowPosition(window_, windowed_pos_.x, windowed_pos_.y);
            SDL_SetWindowSize(window_, windowed_size_.x, windowed_size_.y);
            is_fullscreen_ = false;
            LOG_DEBUG("setFullscreen leave requested: restoring windowed pos={}x{}, size={}x{}",
                      windowed_pos_.x,
                      windowed_pos_.y,
                      windowed_size_.x,
                      windowed_size_.y);
        } else {
            SDL_GetWindowPosition(window_, &windowed_pos_.x, &windowed_pos_.y);
            SDL_GetWindowSize(window_, &windowed_size_.x, &windowed_size_.y);
            if (!SDL_SetWindowFullscreen(window_, true)) {
                LOG_WARN("Failed to enter fullscreen: {}", SDL_GetError());
                return;
            }
            is_fullscreen_ = true;
            LOG_DEBUG("setFullscreen enter requested: saved windowed pos={}x{}, size={}x{}",
                      windowed_pos_.x,
                      windowed_pos_.y,
                      windowed_size_.x,
                      windowed_size_.y);
        }

        updateWindowSize(fullscreen ? "setFullscreen-enter" : "setFullscreen-leave",
                         ResizeIntent::Exact);
        wakeEventLoop();
    }

    bool WindowManager::isMaximized() const {
        return is_borderless_maximized_ || isSdlMaximized();
    }

    void WindowManager::minimize() {
        if (!window_)
            return;
        if (!SDL_MinimizeWindow(window_))
            LOG_WARN("Failed to minimize window: {}", SDL_GetError());
    }

    void WindowManager::toggleMaximized() {
        if (!window_)
            return;

        if (isMaximized()) {
            restoreMaximized("toggleMaximized-restore");
            return;
        }

        maximizeBorderless("toggleMaximized-maximize", true);
    }

    void WindowManager::setTitlebarDragRegion(const int height_px, std::vector<HitTestRect> excluded_rects) {
        titlebar_drag_height_px_ = std::max(0, height_px);
        titlebar_drag_excluded_rects_ = std::move(excluded_rects);
    }

    void WindowManager::clearTitlebarDragRegion() {
        titlebar_drag_height_px_ = 0;
        titlebar_drag_excluded_rects_.clear();
        pending_titlebar_double_click_ = false;
        titlebar_drag_active_ = false;
    }

    bool WindowManager::isTitlebarDragPoint(const int x, const int y) const {
        if (titlebar_drag_height_px_ <= 0 || y < 0 || y >= titlebar_drag_height_px_)
            return false;

        for (const auto& rect : titlebar_drag_excluded_rects_) {
            if (rect.w <= 0 || rect.h <= 0)
                continue;
            if (x >= rect.x && x < rect.x + rect.w &&
                y >= rect.y && y < rect.y + rect.h)
                return false;
        }

        return true;
    }

    void WindowManager::beginTitlebarNativeMove() {
        if (!window_ || is_fullscreen_ || !native_titlebar_move_available_)
            return;

        if (!requestX11WindowMove(window_)) {
            native_titlebar_move_available_ = false;
            LOG_WARN("Failed to start native titlebar window move; falling back to SDL hit-testing");
        }
    }

    unsigned WindowManager::manualResizeEdgeMask() const {
        return static_cast<unsigned>(manual_resize_edge_);
    }

    WindowManager::ResizeEdge WindowManager::resizeEdgeAt(const int x, const int y) const {
        if (!window_ || is_fullscreen_ || isMaximized())
            return ResizeEdge::NoEdge;

        glm::ivec2 size = getWindowSize();
        SDL_GetWindowSize(window_, &size.x, &size.y);
        if (size.x <= 0 || size.y <= 0)
            return ResizeEdge::NoEdge;

        const bool left = x >= 0 && x < kResizeBorder;
        const bool right = x >= size.x - kResizeBorder && x < size.x;
        const bool top = y >= 0 && y < kResizeBorder;
        const bool bottom = y >= size.y - kResizeBorder && y < size.y;

        unsigned edge = 0;
        if (left)
            edge |= static_cast<unsigned>(ResizeEdge::Left);
        if (right)
            edge |= static_cast<unsigned>(ResizeEdge::Right);
        if (top && !isTitlebarDragPoint(x, y))
            edge |= static_cast<unsigned>(ResizeEdge::Top);
        if (bottom)
            edge |= static_cast<unsigned>(ResizeEdge::Bottom);

        return static_cast<ResizeEdge>(edge);
    }

    void WindowManager::setResizeCursorForEdge(const ResizeEdge edge) {
        if constexpr (!kUseManualBorderlessResize)
            return;

        if (edge == ResizeEdge::NoEdge)
            return;

        const auto has_edge = [edge](const ResizeEdge flag) {
            return (static_cast<unsigned>(edge) & static_cast<unsigned>(flag)) != 0;
        };
        SDL_Cursor* cursor = nullptr;
        static SDL_Cursor* ew_cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_EW_RESIZE);
        static SDL_Cursor* ns_cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NS_RESIZE);
        static SDL_Cursor* nwse_cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NWSE_RESIZE);
        static SDL_Cursor* nesw_cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NESW_RESIZE);

        if ((has_edge(ResizeEdge::Left) && has_edge(ResizeEdge::Top)) ||
            (has_edge(ResizeEdge::Right) && has_edge(ResizeEdge::Bottom))) {
            cursor = nwse_cursor;
        } else if ((has_edge(ResizeEdge::Right) && has_edge(ResizeEdge::Top)) ||
                   (has_edge(ResizeEdge::Left) && has_edge(ResizeEdge::Bottom))) {
            cursor = nesw_cursor;
        } else if (has_edge(ResizeEdge::Left) || has_edge(ResizeEdge::Right)) {
            cursor = ew_cursor;
        } else if (has_edge(ResizeEdge::Top) || has_edge(ResizeEdge::Bottom)) {
            cursor = ns_cursor;
        }
        if (cursor)
            SDL_SetCursor(cursor);
    }

    void WindowManager::refreshResizeCursor() {
        if constexpr (!kUseManualBorderlessResize)
            return;

        if (!window_ || is_fullscreen_ || isMaximized())
            return;

        if (isManualResizeActive()) {
            setResizeCursorForEdge(manual_resize_edge_);
            return;
        }

        if ((SDL_GetWindowFlags(window_) & SDL_WINDOW_MOUSE_FOCUS) == 0)
            return;

        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        setResizeCursorForEdge(resizeEdgeAt(static_cast<int>(std::round(mouse_x)),
                                            static_cast<int>(std::round(mouse_y))));
    }

    void WindowManager::updateResizeCursor(const int x, const int y) {
        if constexpr (!kUseManualBorderlessResize)
            return;

        if (!window_ || isManualResizeActive() || is_fullscreen_ || isMaximized())
            return;

        setResizeCursorForEdge(resizeEdgeAt(x, y));
    }

    void WindowManager::beginManualResize(const ResizeEdge edge) {
        if constexpr (!kUseManualBorderlessResize)
            return;

        if (!window_ || edge == ResizeEdge::NoEdge || is_fullscreen_ || isMaximized())
            return;

        manual_resize_edge_ = edge;
        manual_resize_start_global_ = globalMousePosition();
        SDL_GetWindowPosition(window_, &manual_resize_start_pos_.x, &manual_resize_start_pos_.y);
        SDL_GetWindowSize(window_, &manual_resize_start_size_.x, &manual_resize_start_size_.y);
        saveBorderlessRestoreGeometry();
        SDL_CaptureMouse(true);
        lfs::core::events::ui::WindowResizeInteraction{.active = true}.emit();
    }

    void WindowManager::updateManualResize() {
        if (!window_ || !isManualResizeActive())
            return;

        const auto has_edge = [this](const ResizeEdge flag) {
            return (static_cast<unsigned>(manual_resize_edge_) & static_cast<unsigned>(flag)) != 0;
        };
        const glm::ivec2 current_global = globalMousePosition();
        const glm::ivec2 delta = current_global - manual_resize_start_global_;

        int next_x = manual_resize_start_pos_.x;
        int next_y = manual_resize_start_pos_.y;
        int next_w = manual_resize_start_size_.x;
        int next_h = manual_resize_start_size_.y;

        if (has_edge(ResizeEdge::Left)) {
            next_w = manual_resize_start_size_.x - delta.x;
            next_x = manual_resize_start_pos_.x + delta.x;
            if (next_w < kMinWindowWidth) {
                next_w = kMinWindowWidth;
                next_x = manual_resize_start_pos_.x + manual_resize_start_size_.x - next_w;
            }
        } else if (has_edge(ResizeEdge::Right)) {
            next_w = manual_resize_start_size_.x + delta.x;
            next_w = std::max(next_w, kMinWindowWidth);
        }

        if (has_edge(ResizeEdge::Top)) {
            next_h = manual_resize_start_size_.y - delta.y;
            next_y = manual_resize_start_pos_.y + delta.y;
            if (next_h < kMinWindowHeight) {
                next_h = kMinWindowHeight;
                next_y = manual_resize_start_pos_.y + manual_resize_start_size_.y - next_h;
            }
        } else if (has_edge(ResizeEdge::Bottom)) {
            next_h = manual_resize_start_size_.y + delta.y;
            next_h = std::max(next_h, kMinWindowHeight);
        }

        int current_x = 0;
        int current_y = 0;
        int current_w = 0;
        int current_h = 0;
        SDL_GetWindowPosition(window_, &current_x, &current_y);
        SDL_GetWindowSize(window_, &current_w, &current_h);

        const bool position_changed = next_x != current_x || next_y != current_y;
        const bool size_changed = next_w != current_w || next_h != current_h;
        if (!position_changed && !size_changed) {
            setResizeCursorForEdge(manual_resize_edge_);
            return;
        }

        if (position_changed) {
            SDL_SetWindowPosition(window_, next_x, next_y);
        }
        if (size_changed) {
            SDL_SetWindowSize(window_, next_w, next_h);
        }
        updateWindowSize("manual-resize-drag", ResizeIntent::Interactive);
        setResizeCursorForEdge(manual_resize_edge_);
    }

    void WindowManager::finishManualResize() {
        if (!isManualResizeActive())
            return;

        manual_resize_edge_ = ResizeEdge::NoEdge;
        SDL_CaptureMouse(false);
        SDL_SetCursor(SDL_GetDefaultCursor());
        updateWindowSize("manual-resize-end", ResizeIntent::Exact);
        lfs::core::events::ui::WindowResizeInteraction{.active = false}.emit();
        wakeEventLoop();
    }

    void WindowManager::suppressFrameInputForManualResize() {
        if (!isManualResizeActive())
            return;

        frame_input_.mouse_x = -100000.0f;
        frame_input_.mouse_y = -100000.0f;
        frame_input_.mouse_down[0] = false;
        frame_input_.mouse_down[1] = false;
        frame_input_.mouse_down[2] = false;
        frame_input_.mouse_clicked[0] = false;
        frame_input_.mouse_clicked[1] = false;
        frame_input_.mouse_clicked[2] = false;
        frame_input_.mouse_released[0] = false;
        frame_input_.mouse_released[1] = false;
        frame_input_.mouse_released[2] = false;
        frame_input_.mouse_wheel = 0.0f;
        frame_input_.mouse_moved = false;
    }

    void WindowManager::beginTitlebarDrag(const int local_x, const int local_y) {
        if (!window_ || is_fullscreen_)
            return;

        titlebar_drag_start_global_ = globalMousePosition();
        titlebar_drag_start_local_ = glm::ivec2(local_x, local_y);
        titlebar_drag_started_maximized_ = isMaximized();

        if (!titlebar_drag_started_maximized_) {
            saveBorderlessRestoreGeometry();
        }
        titlebar_drag_active_ = true;
    }

    void WindowManager::updateTitlebarDrag() {
        if (!window_ || !titlebar_drag_active_)
            return;

        if (titlebar_drag_started_maximized_) {
            if (!titlebarDragMovedEnough())
                return;

            if (isMaximized()) {
                restoreMaximizedForTitlebarDrag();
                if (isMaximized())
                    return;
            }

            if (!native_titlebar_move_available_) {
                updateManualTitlebarMove();
            }
        }
    }

    void WindowManager::finishTitlebarDrag() {
        if (!titlebar_drag_active_)
            return;

        const bool should_maximize = window_ &&
                                     !is_fullscreen_ &&
                                     !isMaximized() &&
                                     titlebarDragMovedEnough() &&
                                     isTitlebarDragAtDisplayTop();
        titlebar_drag_active_ = false;

        if (!should_maximize)
            return;

        maximizeBorderless("titlebar-drag-top-maximize", false);
    }

    void WindowManager::finishTitlebarDragIfReleased() {
        if (!titlebar_drag_active_ && !isManualResizeActive())
            return;

        const SDL_MouseButtonFlags buttons = SDL_GetGlobalMouseState(nullptr, nullptr);
        if ((buttons & SDL_BUTTON_LMASK) != 0)
            return;

        if (isManualResizeActive())
            finishManualResize();
        if (titlebar_drag_active_)
            finishTitlebarDrag();
    }

    bool WindowManager::isSdlMaximized() const {
        return window_ && (SDL_GetWindowFlags(window_) & SDL_WINDOW_MAXIMIZED) != 0;
    }

    void WindowManager::saveBorderlessRestoreGeometry() {
        if (!window_)
            return;

        SDL_GetWindowPosition(window_, &borderless_restore_pos_.x, &borderless_restore_pos_.y);
        SDL_GetWindowSize(window_, &borderless_restore_size_.x, &borderless_restore_size_.y);
    }

    void WindowManager::normalizeNativeMaximize(const char* const reason) {
        if (!window_ || is_fullscreen_) {
            updateWindowSize(reason, ResizeIntent::Exact);
            return;
        }

        bool restored_sdl_maximize = false;
        if (isSdlMaximized()) {
            if (!SDL_RestoreWindow(window_)) {
                LOG_WARN("Failed to restore SDL-maximized window before normalizing maximize: {}", SDL_GetError());
                updateWindowSize(reason, ResizeIntent::Exact);
                return;
            }
            restored_sdl_maximize = true;
        }

        if (is_borderless_maximized_) {
            updateWindowSize(reason, ResizeIntent::Exact);
            return;
        }

        if (restored_sdl_maximize) {
            saveBorderlessRestoreGeometry();
        }

        maximizeBorderless(reason, !restored_sdl_maximize);
    }

    void WindowManager::maximizeBorderless(const char* const reason, const bool save_restore_geometry) {
        if (!window_ || is_fullscreen_)
            return;

        if (save_restore_geometry) {
            saveBorderlessRestoreGeometry();
        }

        if (isSdlMaximized() && !SDL_RestoreWindow(window_)) {
            LOG_WARN("Failed to restore SDL-maximized window before borderless maximize: {}", SDL_GetError());
            return;
        }

        const SDL_DisplayID display_id = SDL_GetDisplayForWindow(window_);
        SDL_Rect display_bounds{};
        SDL_Rect usable_bounds{};
        if (display_id == 0 ||
            !SDL_GetDisplayBounds(display_id, &display_bounds) ||
            !SDL_GetDisplayUsableBounds(display_id, &usable_bounds) ||
            display_bounds.w <= 0 || display_bounds.h <= 0 ||
            usable_bounds.w <= 0 || usable_bounds.h <= 0) {
            LOG_WARN("Failed to query display work area for borderless maximize: {}", SDL_GetError());
            if (!SDL_MaximizeWindow(window_)) {
                LOG_WARN("Failed to maximize window: {}", SDL_GetError());
                return;
            }
            is_borderless_maximized_ = false;
            updateWindowSize(reason, ResizeIntent::Exact);
            wakeEventLoop();
            return;
        }

        SDL_Rect target_bounds = usable_bounds;
        // Keep work-area dimensions so taskbars stay visible. Ask for the
        // display top; some WMs may still clamp managed windows to work-area y.
        target_bounds.y = display_bounds.y;
        target_bounds.h = std::min(usable_bounds.h, display_bounds.h);

        const bool size_set = SDL_SetWindowSize(window_, target_bounds.w, target_bounds.h);
        const bool position_set = SDL_SetWindowPosition(window_, target_bounds.x, target_bounds.y);
        if (!position_set || !size_set) {
            LOG_WARN("Failed to apply borderless maximize bounds {}x{} at {},{}: {}",
                     target_bounds.w,
                     target_bounds.h,
                     target_bounds.x,
                     target_bounds.y,
                     SDL_GetError());
            return;
        }

        is_borderless_maximized_ = true;
        updateWindowSize(reason, ResizeIntent::Exact);
        wakeEventLoop();
    }

    void WindowManager::restoreMaximized(const char* const reason) {
        if (!window_)
            return;

        if (is_borderless_maximized_) {
            if (isSdlMaximized() && !SDL_RestoreWindow(window_)) {
                LOG_WARN("Failed to restore SDL-maximized window before borderless restore: {}", SDL_GetError());
                return;
            }

            const bool position_set = SDL_SetWindowPosition(
                window_, borderless_restore_pos_.x, borderless_restore_pos_.y);
            const bool size_set = SDL_SetWindowSize(
                window_, borderless_restore_size_.x, borderless_restore_size_.y);
            if (!position_set || !size_set) {
                LOG_WARN("Failed to restore borderless maximized window to {}x{} at {},{}: {}",
                         borderless_restore_size_.x,
                         borderless_restore_size_.y,
                         borderless_restore_pos_.x,
                         borderless_restore_pos_.y,
                         SDL_GetError());
                return;
            }

            is_borderless_maximized_ = false;
            updateWindowSize(reason, ResizeIntent::Exact);
            wakeEventLoop();
            return;
        }

        if (!isSdlMaximized())
            return;

        if (!SDL_RestoreWindow(window_)) {
            LOG_WARN("Failed to restore window: {}", SDL_GetError());
            return;
        }

        updateWindowSize(reason, ResizeIntent::Exact);
        wakeEventLoop();
    }

    void WindowManager::restoreMaximizedForTitlebarDrag() {
        if (!window_ || !isMaximized())
            return;

        const glm::ivec2 current_global = globalMousePosition();

        const float anchor_ratio_x = window_size_.x > 0
                                         ? std::clamp(static_cast<float>(titlebar_drag_start_local_.x) /
                                                          static_cast<float>(window_size_.x),
                                                      0.0f,
                                                      1.0f)
                                         : 0.5f;

        restoreMaximized("titlebar-drag-restore");
        if (isMaximized())
            return;

        int restored_w = 0;
        int restored_h = 0;
        SDL_GetWindowSize(window_, &restored_w, &restored_h);
        if (restored_w <= 0 || restored_h <= 0)
            return;

        const int pointer_offset_x = std::clamp(
            static_cast<int>(std::lround(static_cast<float>(restored_w) * anchor_ratio_x)),
            0,
            std::max(0, restored_w - 1));
        const int max_titlebar_offset_y = titlebar_drag_height_px_ > 0
                                              ? std::min(titlebar_drag_height_px_ - 1, restored_h - 1)
                                              : restored_h - 1;
        const int pointer_offset_y = std::clamp(
            titlebar_drag_start_local_.y,
            0,
            std::max(0, max_titlebar_offset_y));

        titlebar_drag_window_offset_ = glm::ivec2(pointer_offset_x, pointer_offset_y);
        SDL_SetWindowPosition(window_,
                              current_global.x - titlebar_drag_window_offset_.x,
                              current_global.y - titlebar_drag_window_offset_.y);
        wakeEventLoop();

        if (native_titlebar_move_available_) {
            beginTitlebarNativeMove();
        }
    }

    void WindowManager::updateManualTitlebarMove() {
        if (!window_)
            return;

        const glm::ivec2 current_global = globalMousePosition();

        SDL_SetWindowPosition(window_,
                              current_global.x - titlebar_drag_window_offset_.x,
                              current_global.y - titlebar_drag_window_offset_.y);
        wakeEventLoop();
    }

    bool WindowManager::titlebarDragMovedEnough() const {
        if (!window_ || !titlebar_drag_active_)
            return false;

        const glm::ivec2 current_global = globalMousePosition();

        constexpr int kPointerDragThresholdPx = 4;
        const glm::ivec2 pointer_delta = glm::abs(current_global - titlebar_drag_start_global_);
        return pointer_delta.x >= kPointerDragThresholdPx ||
               pointer_delta.y >= kPointerDragThresholdPx;
    }

    bool WindowManager::isTitlebarDragAtDisplayTop() const {
        if (!window_)
            return false;

        const glm::ivec2 global_mouse = globalMousePosition();
        const SDL_Point global_point{global_mouse.x, global_mouse.y};

        SDL_DisplayID display_id = SDL_GetDisplayForPoint(&global_point);
        if (display_id == 0) {
            display_id = SDL_GetDisplayForWindow(window_);
        }
        if (display_id == 0)
            return false;

        SDL_Rect display_bounds{};
        if (!SDL_GetDisplayBounds(display_id, &display_bounds))
            return false;

        constexpr int kTopEdgeSnapSlopPx = 8;
        const int display_right = display_bounds.x + display_bounds.w;
        const bool within_display_x =
            global_point.x >= display_bounds.x - kTopEdgeSnapSlopPx &&
            global_point.x < display_right + kTopEdgeSnapSlopPx;
        const bool at_display_top =
            global_point.y >= display_bounds.y - kTopEdgeSnapSlopPx &&
            global_point.y <= display_bounds.y + kTopEdgeSnapSlopPx;
        return within_display_x && at_display_top;
    }

    void WindowManager::flushPendingTitlebarDoubleClick() {
        if (!pending_titlebar_double_click_)
            return;

        pending_titlebar_double_click_ = false;
        toggleMaximized();
    }

} // namespace lfs::vis
