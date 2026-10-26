/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "native_file_drag.hpp"
#include <SDL3/SDL.h>
#ifdef LFS_HAS_WAYLAND_DRAG
#include "native_drag_icon.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <wayland-client.h>
#endif
namespace lfs::vis::gui {
    struct NativeDragContext::Impl {
#ifdef LFS_HAS_WAYLAND_DRAG
        struct Seat {
            Impl* owner;
            uint32_t name;
            wl_seat* seat = nullptr;
            wl_pointer* pointer = nullptr;
            wl_keyboard* keyboard = nullptr;
            wl_data_device* device = nullptr;
            wl_surface* focus = nullptr;
            wl_surface* origin = nullptr;
            uint32_t serial = 0;
            float x = 0, y = 0;
            bool held = false;
            ~Seat() {
                if (device)
                    wl_data_device_release(device);
                if (keyboard)
                    wl_keyboard_release(keyboard);
                if (pointer)
                    wl_pointer_release(pointer);
                if (seat)
                    wl_seat_release(seat);
            }
        };
        // Borrow SDL's connection, never disconnect it or replace SDL listeners.
        wl_display* display = nullptr;
        wl_event_queue* queue = nullptr;
        wl_registry* registry = nullptr;
        wl_data_device_manager* manager = nullptr;
        uint32_t manager_name = 0;
        std::vector<std::unique_ptr<Seat>> seats;
        wl_data_source* source = nullptr;
        wl_compositor* compositor = nullptr;
        wl_shm* shm = nullptr;
        wl_buffer* icon_buffers[2] = {};
        int icon_state = -1;
        wl_surface* drag_icon = nullptr;
        static void shmFormat(void*, wl_shm*, uint32_t) {}
        static constexpr wl_shm_listener shm_listener = {shmFormat};
        static constexpr int icon_size = 72;
        void updateIcon(bool accepted) {
            const int state = accepted ? 1 : 0;
            if (!drag_icon || !icon_buffers[state] || icon_state == state)
                return;
            icon_state = state;
            wl_surface_attach(drag_icon, icon_buffers[state], 0, 0);
            wl_surface_damage(drag_icon, 0, 0, icon_size, icon_size);
            wl_surface_commit(drag_icon);
        }
        void destroyIcon() {
            if (drag_icon)
                wl_surface_destroy(drag_icon);
            drag_icon = nullptr;
            for (auto& buffer : icon_buffers) {
                if (buffer)
                    wl_buffer_destroy(buffer);
                buffer = nullptr;
            }
            icon_state = -1;
        }
        void createIcon() {
            if (!compositor || !shm)
                return;
            auto* image = loadNativeDragIcon(64);
            if (!image)
                return;
            const int fd = memfd_create("lichtfeld-drag-icon", MFD_CLOEXEC);
            constexpr int bytes = icon_size * icon_size * 4;
            void* memory = MAP_FAILED;
            if (fd >= 0 && ftruncate(fd, bytes * 2) == 0)
                memory = mmap(nullptr, bytes * 2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (memory != MAP_FAILED) {
                std::memset(memory, 0, bytes * 2);
                const auto* pixels = gdk_pixbuf_get_pixels(image);
                const int channels = gdk_pixbuf_get_n_channels(image);
                const int stride = gdk_pixbuf_get_rowstride(image);
                for (int state = 0; state < 2; ++state) {
                    auto* output = static_cast<uint32_t*>(memory) + state * icon_size * icon_size;
                    for (int y = 0; y < std::min(64, gdk_pixbuf_get_height(image)); ++y)
                        for (int x = 0; x < std::min(64, gdk_pixbuf_get_width(image)); ++x) {
                            const auto* pixel = pixels + y * stride + x * channels;
                            const uint32_t alpha = channels == 4 ? pixel[3] : 255;
                            output[y * icon_size + x] = alpha << 24 | (pixel[0] * alpha / 255) << 16 | (pixel[1] * alpha / 255) << 8 | pixel[2] * alpha / 255;
                        }
                    for (int y = 48; y < 72; ++y)
                        for (int x = 48; x < 72; ++x) {
                            const int dx = x - 59, dy = y - 59;
                            if (dx * dx + dy * dy > 121)
                                continue;
                            const bool mark = state ? (std::abs(dx) <= 1 && std::abs(dy) <= 6) || (std::abs(dy) <= 1 && std::abs(dx) <= 6)
                                                    : std::abs(dx + dy) <= 1 && std::abs(dx) <= 7;
                            output[y * icon_size + x] = mark ? 0xffffffff : state ? 0xff28843b
                                                                                  : 0xffb82b32;
                        }
                }
                if (auto* pool = wl_shm_create_pool(shm, fd, bytes * 2)) {
                    for (int state = 0; state < 2; ++state)
                        icon_buffers[state] = wl_shm_pool_create_buffer(pool, state * bytes, icon_size, icon_size, icon_size * 4, WL_SHM_FORMAT_ARGB8888);
                    wl_shm_pool_destroy(pool);
                }
                munmap(memory, bytes * 2);
            }
            if (fd >= 0)
                close(fd);
            g_object_unref(image);
            if (icon_buffers[0] && icon_buffers[1])
                drag_icon = wl_compositor_create_surface(compositor);
        }
        bool attempted = false, done = false, failed = false, active = false, broken = false, provider_failed = false;
        std::string payload;
        FileProvider provider;
        bool dispatching = false;
        bool dropped = false;

        static void enter(void* data, wl_pointer*, uint32_t, wl_surface* surface, wl_fixed_t x, wl_fixed_t y) {
            auto& s = *static_cast<Seat*>(data);
            s.focus = surface;
            s.x = wl_fixed_to_double(x);
            s.y = wl_fixed_to_double(y);
        }
        static void leave(void* data, wl_pointer*, uint32_t, wl_surface*) { static_cast<Seat*>(data)->focus = nullptr; }
        static void motion(void* data, wl_pointer*, uint32_t, wl_fixed_t x, wl_fixed_t y) {
            auto& s = *static_cast<Seat*>(data);
            s.x = wl_fixed_to_double(x);
            s.y = wl_fixed_to_double(y);
        }
        static void button(void* data, wl_pointer*, uint32_t serial, uint32_t, uint32_t button, uint32_t state) {
            if (button != 0x110)
                return; // Linux BTN_LEFT, protocol-defined evdev code.
            auto& s = *static_cast<Seat*>(data);
            s.held = state == WL_POINTER_BUTTON_STATE_PRESSED;
            if (s.held) {
                s.serial = serial;
                s.origin = s.focus;
            } else
                s.serial = 0;
        }
        static void axis(void*, wl_pointer*, uint32_t, uint32_t, wl_fixed_t) {}
        static void frame(void*, wl_pointer*) {}
        static void axisSource(void*, wl_pointer*, uint32_t) {}
        static void axisStop(void*, wl_pointer*, uint32_t, uint32_t) {}
        static void axisDiscrete(void*, wl_pointer*, uint32_t, int32_t) {}
        static constexpr auto pointer_listener = [] {
            wl_pointer_listener listener{};
            listener.enter = enter;
            listener.leave = leave;
            listener.motion = motion;
            listener.button = button;
            listener.axis = axis;
            listener.frame = frame;
            listener.axis_source = axisSource;
            listener.axis_stop = axisStop;
            listener.axis_discrete = axisDiscrete;
            return listener;
        }();
        static void keymap(void*, wl_keyboard*, uint32_t, int fd, uint32_t) { close(fd); }
        static void keyEnter(void*, wl_keyboard*, uint32_t, wl_surface*, wl_array*) {}
        static void keyLeave(void*, wl_keyboard*, uint32_t, wl_surface*) {}
        static void key(void* data, wl_keyboard*, uint32_t, uint32_t, uint32_t key, uint32_t state) {
            auto& self = *static_cast<Seat*>(data)->owner;
            if (self.active && key == 1 && state == WL_KEYBOARD_KEY_STATE_PRESSED) {
                self.failed = true;
                self.done = true;
            }
        }
        static void modifiers(void*, wl_keyboard*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}
        static void repeat(void*, wl_keyboard*, int32_t, int32_t) {}
        static constexpr wl_keyboard_listener keyboard_listener = {keymap, keyEnter, keyLeave, key, modifiers, repeat};
        static void offer(void*, wl_data_device*, wl_data_offer* value) { wl_data_offer_destroy(value); }
        static void deviceEnter(void*, wl_data_device*, uint32_t, wl_surface*, wl_fixed_t, wl_fixed_t, wl_data_offer*) {}
        static void deviceLeave(void*, wl_data_device*) {}
        static void deviceMotion(void*, wl_data_device*, uint32_t, wl_fixed_t, wl_fixed_t) {}
        static void deviceDrop(void*, wl_data_device*) {}
        static void selection(void*, wl_data_device*, wl_data_offer*) {}
        static constexpr wl_data_device_listener device_listener = {offer, deviceEnter, deviceLeave, deviceMotion, deviceDrop, selection};
        void attachDevice(Seat& s) {
            if (manager && !s.device) {
                s.device = wl_data_device_manager_get_data_device(manager, s.seat);
                if (!s.device) {
                    broken = true;
                    done = true;
                    return;
                }
                wl_data_device_add_listener(s.device, &device_listener, &s);
            }
        }
        static void capabilities(void* data, wl_seat*, uint32_t flags) {
            auto& s = *static_cast<Seat*>(data);
            if ((flags & WL_SEAT_CAPABILITY_POINTER) && !s.pointer) {
                s.pointer = wl_seat_get_pointer(s.seat);
                if (!s.pointer) {
                    s.owner->broken = true;
                    s.owner->done = true;
                    return;
                }
                wl_pointer_add_listener(s.pointer, &pointer_listener, &s);
            } else if (!(flags & WL_SEAT_CAPABILITY_POINTER) && s.pointer) {
                if (s.owner->active) {
                    s.owner->failed = true;
                    s.owner->done = true;
                }
                wl_pointer_release(s.pointer);
                s.pointer = nullptr;
                s.held = false;
                s.serial = 0;
                s.focus = s.origin = nullptr;
            }
            if ((flags & WL_SEAT_CAPABILITY_KEYBOARD) && !s.keyboard) {
                s.keyboard = wl_seat_get_keyboard(s.seat);
                if (!s.keyboard) {
                    s.owner->broken = true;
                    s.owner->done = true;
                    return;
                }
                wl_keyboard_add_listener(s.keyboard, &keyboard_listener, &s);
            } else if (!(flags & WL_SEAT_CAPABILITY_KEYBOARD) && s.keyboard) {
                wl_keyboard_release(s.keyboard);
                s.keyboard = nullptr;
            }
        }
        static void seatName(void*, wl_seat*, const char*) {}
        static constexpr wl_seat_listener seat_listener = {capabilities, seatName};
        static void global(void* data, wl_registry* registry, uint32_t name, const char* interface, uint32_t version) noexcept {
            auto& self = *static_cast<Impl*>(data);
            try {
                if (!std::strcmp(interface, wl_compositor_interface.name) && !self.compositor) {
                    self.compositor = static_cast<wl_compositor*>(wl_registry_bind(registry, name, &wl_compositor_interface, 1));
                } else if (!std::strcmp(interface, wl_shm_interface.name) && !self.shm) {
                    self.shm = static_cast<wl_shm*>(wl_registry_bind(registry, name, &wl_shm_interface, 1));
                    if (self.shm)
                        wl_shm_add_listener(self.shm, &shm_listener, &self);
                } else if (!std::strcmp(interface, wl_data_device_manager_interface.name) && version >= 3 && !self.manager) {
                    self.manager = static_cast<wl_data_device_manager*>(wl_registry_bind(registry, name, &wl_data_device_manager_interface, 3));
                    self.manager_name = name;
                    if (!self.manager)
                        throw std::bad_alloc();
                    for (auto& s : self.seats)
                        self.attachDevice(*s);
                } else if (!std::strcmp(interface, wl_seat_interface.name) && version >= 5) {
                    auto s = std::make_unique<Seat>();
                    s->owner = &self;
                    s->name = name;
                    s->seat = static_cast<wl_seat*>(wl_registry_bind(registry, name, &wl_seat_interface, 5));
                    if (!s->seat)
                        throw std::bad_alloc();
                    wl_seat_add_listener(s->seat, &seat_listener, s.get());
                    self.attachDevice(*s);
                    self.seats.push_back(std::move(s));
                }
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): never unwind across a libwayland C callback; fail the operation.
                self.broken = true;
                self.done = true;
            }
        }
        static void remove(void* data, wl_registry*, uint32_t name) {
            auto& self = *static_cast<Impl*>(data);
            if (name == self.manager_name) {
                self.broken = true;
                self.done = true;
            }
            for (auto it = self.seats.begin(); it != self.seats.end(); ++it)
                if ((*it)->name == name) {
                    if (self.active) {
                        self.failed = true;
                        self.done = true;
                    }
                    self.seats.erase(it);
                    break;
                }
        }
        static constexpr wl_registry_listener registry_listener = {global, remove};
        void init() {
            if (attempted)
                return;
            const char* driver = SDL_GetCurrentVideoDriver();
            if (!driver || std::string_view(driver) != "wayland")
                return;
            display = static_cast<wl_display*>(SDL_GetPointerProperty(SDL_GetGlobalProperties(), SDL_PROP_GLOBAL_VIDEO_WAYLAND_WL_DISPLAY_POINTER, nullptr));
            if (!display)
                return;
            attempted = true;
            queue = wl_display_create_queue(display);
            if (!queue) {
                broken = true;
                return;
            }
            auto* wrapper = static_cast<wl_display*>(wl_proxy_create_wrapper(display));
            if (!wrapper) {
                broken = true;
                return;
            }
            wl_proxy_set_queue(reinterpret_cast<wl_proxy*>(wrapper), queue);
            registry = wl_display_get_registry(wrapper);
            wl_proxy_wrapper_destroy(wrapper);
            if (!registry) {
                broken = true;
                return;
            }
            wl_registry_add_listener(registry, &registry_listener, this);
            // Initialize at panel creation, before a press. These round trips
            // dispatch only our queue; SDL's proxies/listeners remain untouched.
            if (wl_display_roundtrip_queue(display, queue) < 0 || wl_display_roundtrip_queue(display, queue) < 0)
                broken = true;
        }
        void pump() {
            if (dispatching)
                return;
            struct DispatchScope {
                bool& value;
                explicit DispatchScope(bool& flag) : value(flag) { value = true; }
                ~DispatchScope() { value = false; }
            } scope(dispatching);
            init();
            if (queue && (wl_display_dispatch_queue_pending(display, queue) < 0 || (wl_display_flush(display) < 0 && errno != EAGAIN))) {
                broken = true;
                done = true;
            }
        }
        // No allocation and no exception may leave a C callback. The descriptor
        // owns the receiver pipe, including when it closes early (SIGPIPE).
        static void send(void* data, wl_data_source*, const char* mime, int fd) noexcept {
            auto& self = *static_cast<Impl*>(data);
            if (!mime || std::strcmp(mime, "text/uri-list")) {
                close(fd);
                self.failed = true;
                self.done = true;
                return;
            }
            if (self.done || self.provider_failed) {
                close(fd);
                return;
            }
            try {
                if (self.payload.empty()) {
                    auto provider = std::move(self.provider);
                    if (!provider) {
                        close(fd);
                        self.failed = true;
                        self.done = true;
                        return;
                    }

                    constexpr char hex[] = "0123456789ABCDEF";
                    for (const auto& path : provider(FileDragTarget::Unknown)) {
                        self.payload += "file://";
                        for (const unsigned char c : path.native()) {
                            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' || c == '.' || c == '~')
                                self.payload += char(c);
                            else {
                                self.payload += '%';
                                self.payload += hex[c >> 4];
                                self.payload += hex[c & 15];
                            }
                        }
                        self.payload += "\r\n";
                    }
                }
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): close the receiver pipe and report cancellation across the C boundary.
                close(fd);
                self.failed = true;
                self.provider_failed = true;
                self.done = true;
                return;
            }
            sigset_t blocked, previous, pending;
            sigemptyset(&blocked);
            sigaddset(&blocked, SIGPIPE);
            const bool masked = pthread_sigmask(SIG_BLOCK, &blocked, &previous) == 0;
            sigpending(&pending);
            const bool already_pending = sigismember(&pending, SIGPIPE);
            bool ok = masked && mime && !std::strcmp(mime, "text/uri-list");
            const int flags = fcntl(fd, F_GETFL);
            ok = ok && flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
            size_t offset = 0;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (ok && offset < self.payload.size() && std::chrono::steady_clock::now() < deadline) {
                const auto n = write(fd, self.payload.data() + offset, self.payload.size() - offset);
                if (n > 0)
                    offset += size_t(n);
                else if (n < 0 && errno == EINTR)
                    continue;
                else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                    pollfd pipe{fd, POLLOUT, 0};
                    ok = poll(&pipe, 1, 1000) > 0 && (pipe.revents & POLLOUT);
                } else
                    ok = false;
            }
            close(fd);
            if (masked) {
                if (!already_pending) {
                    timespec zero{};
                    sigtimedwait(&blocked, nullptr, &zero);
                }
                pthread_sigmask(SIG_SETMASK, &previous, nullptr);
            }
            if (offset != self.payload.size())
                ok = false;
            if (!ok) {
                self.failed = true;
                self.done = true;
            }
        }
        static void target(void* data, wl_data_source*, const char* mime) {
            static_cast<Impl*>(data)->updateIcon(mime && !std::strcmp(mime, "text/uri-list"));
        }
        static void cancelled(void* data, wl_data_source*) {
            auto& s = *static_cast<Impl*>(data);
            s.failed = true;
            s.done = true;
        }
        static void dropPerformed(void* data, wl_data_source*) { static_cast<Impl*>(data)->dropped = true; }
        static void finished(void* data, wl_data_source*) { static_cast<Impl*>(data)->done = true; }
        static void action(void* data, wl_data_source*, uint32_t action) {
            static_cast<Impl*>(data)->updateIcon(action == WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);
            if (action != WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY && action != WL_DATA_DEVICE_MANAGER_DND_ACTION_NONE)
                static_cast<Impl*>(data)->failed = true;
        }
        static constexpr wl_data_source_listener source_listener = {target, send, cancelled, dropPerformed, finished, action};
        ~Impl() {
            destroyIcon();
            if (shm)
                wl_shm_destroy(shm);
            if (compositor)
                wl_compositor_destroy(compositor);
            if (source)
                wl_data_source_destroy(source);
            seats.clear();
            if (manager)
                wl_data_device_manager_destroy(manager);
            if (registry)
                wl_registry_destroy(registry);
            if (queue)
                wl_event_queue_destroy(queue);
        }
#endif
    };
    NativeDragContext::NativeDragContext() : impl_(std::make_unique<Impl>()) { pump(); }
    NativeDragContext::~NativeDragContext() = default;
    void NativeDragContext::pump() {
#ifdef LFS_HAS_WAYLAND_DRAG
        impl_->pump();
#endif
    }
    std::optional<NativeDragPointer> NativeDragContext::pointer(SDL_Window* window) {
#ifdef LFS_HAS_WAYLAND_DRAG
        pump();
        if (impl_->queue && window) {
            auto* surface = static_cast<wl_surface*>(SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr));
            for (const auto& seat : impl_->seats)
                if (surface && (seat->origin == surface || seat->focus == surface)) {
                    int width = 0, height = 0;
                    SDL_GetWindowSize(window, &width, &height);
                    return NativeDragPointer{seat->x, seat->y, seat->held, seat->x < 0 || seat->y < 0 || seat->x >= width || seat->y >= height};
                }
            return NativeDragPointer{};
        }
#else
        (void)window;
#endif
        return std::nullopt;
    }
    FileDragResult NativeDragContext::drag(FileProvider files, SDL_Window* window) {
#ifdef LFS_HAS_WAYLAND_DRAG
        pump();
        auto& self = *impl_;
        if (!window || !self.manager || self.broken || self.active)
            return {FileDragStatus::Failed, "Wayland drag context unavailable"};
        auto* surface = static_cast<wl_surface*>(SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr));
        Impl::Seat* seat = nullptr;
        for (auto& candidate : self.seats)
            if (surface && candidate->origin == surface && candidate->held && candidate->serial && candidate->device) {
                seat = candidate.get();
                break;
            }
        if (!seat)
            return {FileDragStatus::Cancelled, "No matching Wayland pointer grab"};
        self.payload.clear();
        self.provider = std::move(files);
        self.source = wl_data_device_manager_create_data_source(self.manager);
        if (!self.source)
            return {FileDragStatus::Failed, "Cannot create Wayland data source"};
        self.active = true;
        self.done = false;
        self.failed = false;
        self.provider_failed = false;
        self.dropped = false;
        struct Scope {
            Impl& s;
            ~Scope() {
                if (s.source)
                    wl_data_source_destroy(s.source);
                s.source = nullptr;
                s.destroyIcon();
                s.active = false;
                s.payload.clear();
                s.provider = {};
            }
        } scope{self};
        wl_data_source_add_listener(self.source, &Impl::source_listener, &self);
        wl_data_source_offer(self.source, "text/uri-list");
        wl_data_source_set_actions(self.source, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);
        self.createIcon();
        wl_data_device_start_drag(seat->device, self.source, surface, self.drag_icon, seat->serial);
        self.updateIcon(false);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        while (!self.done && std::chrono::steady_clock::now() < deadline) {
            while (wl_display_prepare_read_queue(self.display, self.queue) != 0) {
                self.pump();
                if (self.done)
                    break;
            }
            if (self.done)
                break;
            const int flushed = wl_display_flush(self.display);
            if (flushed < 0 && errno != EAGAIN) {
                wl_display_cancel_read(self.display);
                self.failed = true;
                break;
            }
            const bool output_pending = flushed < 0 && errno == EAGAIN;
            pollfd event{wl_display_get_fd(self.display), static_cast<short>(POLLIN | (output_pending ? POLLOUT : 0)), 0};
            const int ready = poll(&event, 1, 50);
            if (ready > 0 && (event.revents & POLLIN)) {
                if (wl_display_read_events(self.display) < 0) {
                    self.failed = true;
                    break;
                }
                self.pump();
            } else {
                wl_display_cancel_read(self.display);
                if ((ready < 0 && errno != EINTR) || (event.revents & (POLLERR | POLLHUP | POLLNVAL))) {
                    self.failed = true;
                    break;
                }
            }
        }
        if (self.provider_failed)
            return {FileDragStatus::Failed, "Deferred file preparation failed"};
        return {self.done && !self.failed ? FileDragStatus::Copied : FileDragStatus::Cancelled, {}};
#else
        (void)files;
        (void)window;
        return {FileDragStatus::Unsupported, "Wayland drag not built"};
#endif
    }
    bool NativeDragContext::dropPerformed() const {
#ifdef LFS_HAS_WAYLAND_DRAG
        return impl_->dropped;
#else
        return false;
#endif
    }
} // namespace lfs::vis::gui
