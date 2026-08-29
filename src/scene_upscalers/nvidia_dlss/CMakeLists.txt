# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(lfs_scene_upscaler_nvidia_dlss SHARED
    nvidia_dlss_plugin.cpp)

target_compile_features(lfs_scene_upscaler_nvidia_dlss PRIVATE cxx_std_23)
target_compile_definitions(lfs_scene_upscaler_nvidia_dlss PRIVATE
    LFS_SCENE_UPSCALER_PLUGIN_BUILD=1)
target_include_directories(lfs_scene_upscaler_nvidia_dlss PRIVATE
    "${CMAKE_SOURCE_DIR}/src/visualizer")
target_link_libraries(lfs_scene_upscaler_nvidia_dlss PRIVATE
    NVIDIA::NGX
    Vulkan::Vulkan
    Threads::Threads
    ${CMAKE_DL_LIBS})

set_target_properties(lfs_scene_upscaler_nvidia_dlss PROPERTIES
    OUTPUT_NAME "lfs_scene_upscaler_nvidia_dlss"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/scene_upscalers/nvidia/$<0:>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/scene_upscalers/nvidia/$<0:>")

set(_lfs_dlss_runtime
    "$<IF:$<CONFIG:Debug>,${LFS_NVIDIA_DLSS_RUNTIME_DEBUG},${LFS_NVIDIA_DLSS_RUNTIME_RELEASE}>")
add_custom_command(TARGET lfs_scene_upscaler_nvidia_dlss POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_lfs_dlss_runtime}"
        "$<TARGET_FILE_DIR:lfs_scene_upscaler_nvidia_dlss>"
    COMMENT "Staging the NVIDIA DLSS runtime beside the optional plugin")

include(GNUInstallDirs)
if(WIN32)
    set(_lfs_dlss_install_dir "${CMAKE_INSTALL_BINDIR}/scene_upscalers/nvidia")
else()
    set(_lfs_dlss_install_dir "${CMAKE_INSTALL_LIBDIR}/scene_upscalers/nvidia")
endif()
install(TARGETS lfs_scene_upscaler_nvidia_dlss
    RUNTIME DESTINATION "${_lfs_dlss_install_dir}"
    LIBRARY DESTINATION "${_lfs_dlss_install_dir}")
install(FILES "${LFS_NVIDIA_DLSS_RUNTIME_DEBUG}"
    DESTINATION "${_lfs_dlss_install_dir}"
    CONFIGURATIONS Debug)
install(FILES "${LFS_NVIDIA_DLSS_RUNTIME_RELEASE}"
    DESTINATION "${_lfs_dlss_install_dir}"
    CONFIGURATIONS Release RelWithDebInfo MinSizeRel)
install(FILES "${LFS_NVIDIA_DLSS_ROOT_RESOLVED}/LICENSE.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/NVIDIA-DLSS"
    OPTIONAL)
