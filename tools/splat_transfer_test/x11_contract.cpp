/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <SDL3/SDL.h>
#include <cstdlib>
#include <iostream>
// This contract probes GTK/X11 metadata only. Any attempt to pump or access
// the application's independent SDL connection is a contract failure.
extern "C" bool SDL_EventEnabled(Uint32) { std::abort(); }
extern "C" void SDL_SetEventEnabled(Uint32, bool) { std::abort(); }
extern "C" void SDL_FlushEvents(Uint32, Uint32) { std::abort(); }
extern "C" const char* SDL_GetCurrentVideoDriver() { std::abort(); }
extern "C" SDL_PropertiesID SDL_GetGlobalProperties() { std::abort(); }
extern "C" SDL_PropertiesID SDL_GetWindowProperties(SDL_Window*) { std::abort(); }
extern "C" void* SDL_GetPointerProperty(SDL_PropertiesID, const char*, void*) { std::abort(); }
extern "C" Sint64 SDL_GetNumberProperty(SDL_PropertiesID, const char*, Sint64) { std::abort(); }
extern "C" SDL_MouseButtonFlags SDL_GetGlobalMouseState(float*, float*) { std::abort(); }
extern "C" bool SDL_GetWindowSize(SDL_Window*, int*, int*) { std::abort(); }
extern "C" SDL_Window** SDL_GetWindows(int*) { std::abort(); }
extern "C" void SDL_free(void*) { std::abort(); }
namespace lfs::vis::gui {
    int nativeX11DragContract();
}
int main() {
    const auto result = lfs::vis::gui::nativeX11DragContract();
    if (result == 0)
        std::cout << "PASS X11 receiver classes, child windows, missing/destroyed windows, immutable URI variants\n";
    else if (result == 77)
        std::cout << "SKIP X11 display unavailable\n";
    else
        std::cerr << "FAIL X11 drag contract: " << result << '\n';
    return result;
}
