# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(MediaExternalConsumer LANGUAGES CXX)
find_package(LichtFeldMedia 0.1 CONFIG REQUIRED)
add_executable(media_consumer main.cpp)
target_link_libraries(media_consumer PRIVATE LichtFeldMedia::media)
