# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(HeicSystemProviderProbe LANGUAGES CXX)
# Qualification only: deliberately not linked into or installed with the app.
if(WIN32)
    add_executable(heic_provider_probe wic.cpp)
    target_compile_definitions(heic_provider_probe PRIVATE NOMINMAX UNICODE _UNICODE)
    target_link_libraries(heic_provider_probe PRIVATE windowscodecs ole32)
elseif(APPLE)
    add_executable(heic_provider_probe imageio.cpp)
    target_link_libraries(heic_provider_probe PRIVATE "-framework ImageIO" "-framework CoreGraphics" "-framework CoreFoundation")
else()
    message(FATAL_ERROR "No approved Linux system HEIC adapter yet; this probe must not bundle HEVC.")
endif()
target_compile_features(heic_provider_probe PRIVATE cxx_std_20)
