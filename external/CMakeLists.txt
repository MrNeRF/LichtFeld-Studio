# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# External dependencies (EXCLUDE_FROM_ALL to prevent third-party install rules)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/CMakeLists.txt")
    message(STATUS "Adding ImGuizmo")
    add_subdirectory(imguizmo EXCLUDE_FROM_ALL)
else()
    message(WARNING "ImGuizmo not found - gizmo controls unavailable")
endif()

# ImGuiColorTextEdit - Syntax highlighting text editor for ImGui (santaclose fork)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ImGuiColorTextEdit/TextEditor.cpp")
    message(STATUS "Adding ImGuiColorTextEdit from external/ImGuiColorTextEdit")
    find_package(Boost REQUIRED COMPONENTS regex)
    add_library(imgui_textedit STATIC
        ImGuiColorTextEdit/TextEditor.cpp
        ImGuiColorTextEdit/LanguageDefinitions.cpp
    )
    target_include_directories(imgui_textedit PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiColorTextEdit
    )
    target_link_libraries(imgui_textedit PUBLIC imgui::imgui Boost::regex)
    set_target_properties(imgui_textedit PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    message(WARNING "ImGuiColorTextEdit not found at external/ImGuiColorTextEdit")
endif()

# OpenMesh - Half-edge mesh data structure
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenMesh/CMakeLists.txt")
    message(STATUS "Adding OpenMesh")
    set(OPENMESH_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(OPENMESH_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(OpenMesh EXCLUDE_FROM_ALL)
else()
    message(WARNING "OpenMesh not found at external/OpenMesh")
endif()

# nvImageCodec - GPU-accelerated image codec library
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nvImageCodec/CMakeLists.txt")
    message(STATUS "Adding nvImageCodec")
    if(DEFINED DETECTED_COMPUTE_CAP)
        string(REPLACE "." "" _arch "${DETECTED_COMPUTE_CAP}")
        set(CUDA_TARGET_ARCHS "${_arch}" CACHE STRING "" FORCE)
    endif()
    set(BUILD_NVJPEG_EXT ON CACHE BOOL "" FORCE)
    set(BUILD_NVJPEG2K_EXT ON CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG OFF CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG2K OFF CACHE BOOL "" FORCE)
    add_subdirectory(nvImageCodec EXCLUDE_FROM_ALL)
else()
    message(WARNING "nvImageCodec not found - GPU image loading unavailable")
endif()
