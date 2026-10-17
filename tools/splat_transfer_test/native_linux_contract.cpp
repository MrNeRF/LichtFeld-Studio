/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/utils/native_file_drag.hpp"
#include <SDL3/SDL.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
static const char* driver = nullptr;
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
    std::filesystem::remove(path);
    check(dragReadyFiles(files, window).status == FileDragStatus::Failed);
    std::cout << "PASS: empty/missing payload, no video driver, Wayland gate, released gesture\n";
}
