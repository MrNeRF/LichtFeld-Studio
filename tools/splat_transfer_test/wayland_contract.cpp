/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Real libwayland client/server contract, without a desktop or input injection.
#include "gui/utils/native_file_drag.hpp"
#include <SDL3/SDL.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <wayland-client.h>
#include <wayland-server.h>
using namespace lfs::vis::gui;
namespace {
    wl_display* client_display;
    wl_surface* client_surface;
    wl_compositor* client_compositor;
    std::string received;
    std::atomic<int> resources{0};
    std::atomic<int> mode{0};
    std::atomic<bool> serial_ok{false};
    std::atomic<bool> payload_ready{false};
    std::atomic<int> icon_commits{0};
    std::atomic<bool> icon_ok{false};
    std::atomic<uint32_t> first_icon_buffer{0};
    std::atomic<bool> distinct_icon_buffers{false};
    wl_resource* server_surface;
    void destroyed(wl_resource*) { --resources; }
    wl_resource* resource(wl_client* c, const wl_interface* i, int v, uint32_t id, const void* impl) {
        auto* r = wl_resource_create(c, i, v, id);
        ++resources;
        wl_resource_set_implementation(r, impl, nullptr, destroyed);
        return r;
    }
    void destroy(wl_client*, wl_resource* r) { wl_resource_destroy(r); }
    void attach(wl_client*, wl_resource* surface, wl_resource* buffer, int32_t, int32_t) {
        if (buffer && surface != server_surface) {
            const auto id = wl_resource_get_id(buffer);
            if (!first_icon_buffer)
                first_icon_buffer = id;
            else if (first_icon_buffer != id)
                distinct_icon_buffers = true;
        }
        if (buffer && !wl_shm_buffer_get(buffer))
            std::abort();
    }
    void damage(wl_client*, wl_resource*, int32_t, int32_t, int32_t, int32_t) {}
    void commit(wl_client*, wl_resource* r) {
        if (r != server_surface)
            ++icon_commits;
    }
    const struct wl_surface_interface surface_impl = {.destroy = destroy, .attach = attach, .damage = damage, .commit = commit};
    void createSurface(wl_client* c, wl_resource*, uint32_t id) {
        auto* surface = resource(c, &wl_surface_interface, 1, id, &surface_impl);
        if (!server_surface)
            server_surface = surface;
    }
    const struct wl_compositor_interface compositor_impl = {.create_surface = createSurface};
    void bindCompositor(wl_client* c, void*, uint32_t, uint32_t id) { resource(c, &wl_compositor_interface, 1, id, &compositor_impl); }
    const struct wl_pointer_interface pointer_impl = {.release = destroy};
    void getPointer(wl_client* c, wl_resource*, uint32_t id) {
        auto* r = resource(c, &wl_pointer_interface, 5, id, &pointer_impl);
        wl_pointer_send_enter(r, 6, server_surface, wl_fixed_from_int(150), wl_fixed_from_int(20));
        wl_pointer_send_button(r, 7, 0, 0x110, WL_POINTER_BUTTON_STATE_PRESSED);
        wl_pointer_send_frame(r);
    }
    const struct wl_seat_interface seat_impl = {.get_pointer = getPointer, .release = destroy};
    void bindSeat(wl_client* c, void*, uint32_t, uint32_t id) {
        auto* r = resource(c, &wl_seat_interface, 5, id, &seat_impl);
        wl_seat_send_capabilities(r, WL_SEAT_CAPABILITY_POINTER);
        wl_seat_send_name(r, "contract");
    }
    void offer(wl_client*, wl_resource*, const char*) {}
    void actions(wl_client*, wl_resource*, uint32_t) {}
    const struct wl_data_source_interface source_impl = {.offer = offer, .destroy = destroy, .set_actions = actions};
    void createSource(wl_client* c, wl_resource*, uint32_t id) { resource(c, &wl_data_source_interface, 3, id, &source_impl); }
    void startDrag(wl_client* c, wl_resource*, wl_resource* source, wl_resource* origin, wl_resource* icon, uint32_t serial) {
        icon_ok = icon && icon != origin;
        wl_data_source_send_target(source, "text/uri-list");
        wl_data_source_send_action(source, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);
        serial_ok = serial == 7 && origin == server_surface;
        if (mode == 1) {
            wl_data_source_send_cancelled(source);
            return;
        }
        int fd[2];
        if (pipe(fd))
            std::abort();
        if (mode == 2)
            close(fd[0]);
        wl_data_source_send_send(source, "text/uri-list", fd[1]);
        wl_display_flush_clients(wl_client_get_display(c));
        close(fd[1]);
        if (mode == 2)
            return;
        char buf[4096];
        ssize_t n;
        received.clear();
        while ((n = read(fd[0], buf, sizeof(buf))) > 0)
            received.append(buf, size_t(n));
        close(fd[0]);
        if (mode == 4) {
            int second[2];
            if (pipe(second))
                std::abort();
            wl_data_source_send_send(source, "text/uri-list", second[1]);
            wl_display_flush_clients(wl_client_get_display(c));
            close(second[1]);
            std::string duplicate;
            while ((n = read(second[0], buf, sizeof(buf))) > 0)
                duplicate.append(buf, size_t(n));
            close(second[0]);
            if (duplicate != received)
                std::abort();
        }
        payload_ready.store(true, std::memory_order_release);
        wl_data_source_send_action(source, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);
        wl_data_source_send_dnd_drop_performed(source);
        wl_data_source_send_dnd_finished(source);
    }
    const struct wl_data_device_interface device_impl = {.start_drag = startDrag, .release = destroy};
    void getDevice(wl_client* c, wl_resource*, uint32_t id, wl_resource*) { resource(c, &wl_data_device_interface, 3, id, &device_impl); }
    const struct wl_data_device_manager_interface manager_impl = {.create_data_source = createSource, .get_data_device = getDevice};
    void bindManager(wl_client* c, void*, uint32_t, uint32_t id) { resource(c, &wl_data_device_manager_interface, 3, id, &manager_impl); }
    void global(void*, wl_registry* registry, uint32_t name, const char* iface, uint32_t) {
        if (std::string_view(iface) == "wl_compositor")
            client_compositor = static_cast<wl_compositor*>(wl_registry_bind(registry, name, &wl_compositor_interface, 1));
    }
    void removed(void*, wl_registry*, uint32_t) {}
    const wl_registry_listener listener{global, removed};
    void check(bool ok, const char* message) {
        if (!ok)
            throw std::runtime_error(message);
    }
} // namespace
static bool drop_files = true, drop_text = false;
extern "C" bool SDL_EventEnabled(Uint32 type) { return type == SDL_EVENT_DROP_FILE ? drop_files : drop_text; }
extern "C" void SDL_SetEventEnabled(Uint32 type, bool enabled) {
    (type == SDL_EVENT_DROP_FILE ? drop_files : drop_text) = enabled;
}
extern "C" void SDL_FlushEvents(Uint32, Uint32) {}
extern "C" const char* SDL_GetCurrentVideoDriver() { return "wayland"; }
extern "C" SDL_PropertiesID SDL_GetGlobalProperties() { return 1; }
extern "C" SDL_PropertiesID SDL_GetWindowProperties(SDL_Window*) { return 2; }
extern "C" void* SDL_GetPointerProperty(SDL_PropertiesID id, const char*, void* fallback) { return id == 1 ? static_cast<void*>(client_display) : id == 2 ? static_cast<void*>(client_surface)
                                                                                                                                                          : fallback; }
extern "C" Sint64 SDL_GetNumberProperty(SDL_PropertiesID, const char*, Sint64 fallback) { return fallback; }
extern "C" SDL_MouseButtonFlags SDL_GetGlobalMouseState(float* x, float* y) {
    *x = *y = 0;
    return 0;
}
extern "C" bool SDL_GetWindowSize(SDL_Window*, int* w, int* h) {
    *w = *h = 100;
    return true;
}
extern "C" SDL_Window** SDL_GetWindows(int* n) {
    *n = 0;
    return nullptr;
}
extern "C" void SDL_free(void* p) { std::free(p); }
int main() {
    try {
        const auto path = std::filesystem::temp_directory_path() / "lichtfeld wayland contract.ply";
        std::ofstream(path) << "fixture";
        for (int iteration = 0; iteration < 50; ++iteration) {
            mode = iteration % 5;
            serial_ok = false;
            payload_ready = false;
            server_surface = nullptr;
            icon_commits = 0;
            icon_ok = false;
            first_icon_buffer = 0;
            distinct_icon_buffers = false;
            auto* server = wl_display_create();
            check(wl_display_init_shm(server) == 0, "server shm initialization failed");
            wl_global_create(server, &wl_compositor_interface, 1, nullptr, bindCompositor);
            wl_global_create(server, &wl_seat_interface, 5, nullptr, bindSeat);
            wl_global_create(server, &wl_data_device_manager_interface, 3, nullptr, bindManager);
            int sockets[2];
            check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair failed");
            wl_client_create(server, sockets[0]);
            client_display = wl_display_connect_to_fd(sockets[1]);
            std::jthread thread([server](std::stop_token stop) { while (!stop.stop_requested()) { wl_event_loop_dispatch(wl_display_get_event_loop(server), 10); wl_display_flush_clients(server); } });
            auto* registry = wl_display_get_registry(client_display);
            wl_registry_add_listener(registry, &listener, nullptr);
            check(wl_display_roundtrip(client_display) >= 0, "registry failed");
            client_surface = wl_compositor_create_surface(client_compositor);
            check(wl_display_roundtrip(client_display) >= 0, "surface failed");
            const int baseline_resources = resources.load();
            {
                NativeDragContext context;
                check(wl_display_roundtrip(client_display) >= 0, "pointer initialization failed");
                const auto pointer = context.pointer(reinterpret_cast<SDL_Window*>(1));
                check(pointer && pointer->held && pointer->outside, "pointer serial/focus not observed on own queue");
                int calls = 0;
                const auto result = dragFiles([&](FileDragTarget target) {
                    check(target == FileDragTarget::Unknown, "Wayland must not guess the receiving application");
                    check(!drop_files && !drop_text, "self-drop reception remained enabled during source transfer");
                    ++calls;
                    if (mode == 3)
                        throw std::runtime_error("synthetic export failure");
                    return std::vector<std::filesystem::path>{path};
                },
                                              reinterpret_cast<SDL_Window*>(1), &context);
                check(drop_files && !drop_text, "drop event state not restored after native gesture");
                check(calls == (mode == 1 ? 0 : 1), "provider ran before needed or more than once");
                check(icon_ok, "native drag icon surface missing");
                check(serial_ok, "source did not use the original surface and real press serial");
                check(result.status == ((mode == 0 || mode == 4) ? FileDragStatus::Copied : mode == 3 ? FileDragStatus::Failed
                                                                                                      : FileDragStatus::Cancelled),
                      "wrong drag outcome");
                if (mode == 0 || mode == 4) {
                    check(payload_ready.load(std::memory_order_acquire), "payload not completed");
                    check(received.find("lichtfeld%20wayland%20contract.ply\r\n") != std::string::npos, "URI bytes not encoded/transferred");
                }
                if (mode == 1) {
                    mode = 0;
                    const auto retry = dragFiles([&](FileDragTarget) { return std::vector<std::filesystem::path>{path}; }, reinterpret_cast<SDL_Window*>(1), &context);
                    check(retry.status == FileDragStatus::Copied, "context cannot be reused after cancellation");
                }
            }
            check(wl_display_roundtrip(client_display) >= 0, "context release roundtrip failed");
            // wl_data_device_manager has no protocol destructor: its bound
            // server resources live until disconnect. Every releasable resource
            // (seat, pointer, device, source) must already have been released.
            check(distinct_icon_buffers, "copy and forbidden feedback did not use distinct native images");
            check(icon_commits > 0, "native drag icon never committed a buffer");
            check(resources == baseline_resources + 2, "drag resources survived context destruction");
            wl_surface_destroy(client_surface);
            wl_compositor_destroy(client_compositor);
            wl_registry_destroy(registry);
            check(wl_display_roundtrip(client_display) >= 0, "destroy roundtrip failed");
            wl_display_disconnect(client_display);
            thread.request_stop();
            thread.join();
            wl_display_destroy_clients(server);
            wl_display_destroy(server);
            check(resources == 0, "server resources leaked across drag contexts");
        }
        std::filesystem::remove(path);
        std::cout << "PASS 50 Wayland cycles: original grab serial/surface, URI transfer, cancel, broken reader, provider failure/deferred invocation, repeated reads, resource release\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
