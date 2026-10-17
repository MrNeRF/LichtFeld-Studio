/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/utils/native_drag_icon.hpp"
#include "gui/utils/native_file_drag.hpp"
#include <SDL3/SDL.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
static const char* driver = nullptr;
static bool drop_files = true, drop_text = false;
extern "C" bool SDL_EventEnabled(Uint32 type) { return type == SDL_EVENT_DROP_FILE ? drop_files : drop_text; }
extern "C" void SDL_SetEventEnabled(Uint32 type, bool enabled) {
    (type == SDL_EVENT_DROP_FILE ? drop_files : drop_text) = enabled;
}
extern "C" void SDL_FlushEvents(Uint32, Uint32) {}
extern "C" const char* SDL_GetCurrentVideoDriver() { return driver; }
extern "C" SDL_MouseButtonFlags SDL_GetGlobalMouseState(float* x, float* y) {
    *x = *y = 0;
    return 0;
}
extern "C" SDL_PropertiesID SDL_GetWindowProperties(SDL_Window*) { return 0; }
extern "C" void* SDL_GetPointerProperty(SDL_PropertiesID, const char*, void* fallback) { return fallback; }
extern "C" Sint64 SDL_GetNumberProperty(SDL_PropertiesID, const char*, Sint64 fallback) { return fallback; }
extern "C" SDL_Window** SDL_GetWindows(int* count) {
    *count = 0;
    return nullptr;
}
extern "C" void SDL_free(void* data) { std::free(data); }
int main() {
    using namespace lfs::vis::gui;
    const auto check = [](bool ok) { if (!ok) throw std::runtime_error("Native drag guard failed"); };
    for (const auto& icon_path : {lfs::core::getAssetsDir() / "lichtfeld-icon.png", std::filesystem::path("/missing/lichtfeld-icon.png")}) {
        auto* icon = loadNativeDragIcon(64, icon_path);
        check(icon && gdk_pixbuf_get_width(icon) == 64 && gdk_pixbuf_get_height(icon) == 64);
        g_object_unref(icon);
    }
    check(dragReadyFiles({}).status == FileDragStatus::Failed);
    const auto path = std::filesystem::temp_directory_path() / "lichtfeld-native-drag-contract.ply";
    std::ofstream(path) << "fixture";
    const std::filesystem::path files[] = {path};
    auto* window = reinterpret_cast<SDL_Window*>(1);
    check(dragReadyFiles(files, window).status == FileDragStatus::Unsupported);
    driver = "wayland";
    check(dragReadyFiles(files, window).status == FileDragStatus::Unsupported);
    driver = "x11";
    check(dragReadyFiles(files, window).status == FileDragStatus::Cancelled);
    check(drop_files && !drop_text);
    check(mayShowTransferProgress(nullptr));
    std::filesystem::remove(path);
    check(dragReadyFiles(files, window).status == FileDragStatus::Failed);
    std::cout << "PASS: empty/missing payload, no video driver, Wayland gate, released gesture\n";
}
