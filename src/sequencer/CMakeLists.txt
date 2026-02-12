# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(lfs_sequencer STATIC
        animation_clip.cpp
        animation_track.cpp
        animation_value.cpp
        interpolation.cpp
        timeline.cpp
)

target_include_directories(lfs_sequencer
        PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/rendering/include
)

target_link_libraries(lfs_sequencer
        PUBLIC
        glm::glm
        PRIVATE
        lfs_core
        nlohmann_json::nlohmann_json
)

target_compile_features(lfs_sequencer PUBLIC cxx_std_23)
