# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: MIT

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

# Stage next to the exe so the loader's <exe>/scene_upscalers/nvidia probe hits
# both single-config (build/) and multi-config (build/$<CONFIG>/) layouts.
# TARGET_FILE_DIR already includes the per-config dir; the genex also suppresses
# CMake's extra $<CONFIG> append on OUTPUT_DIRECTORY.
set_target_properties(lfs_scene_upscaler_nvidia_dlss PROPERTIES
    OUTPUT_NAME "lfs_scene_upscaler_nvidia_dlss"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:LichtFeld-Studio>/scene_upscalers/nvidia"
    LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:LichtFeld-Studio>/scene_upscalers/nvidia")

set(_lfs_dlss_runtime
    "$<IF:$<CONFIG:Debug>,${LFS_NVIDIA_DLSS_RUNTIME_DEBUG},${LFS_NVIDIA_DLSS_RUNTIME_RELEASE}>")
add_custom_command(TARGET lfs_scene_upscaler_nvidia_dlss POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${_lfs_dlss_runtime}"
        "${LFS_NVIDIA_DLSS_LICENSE}"
        "$<TARGET_FILE_DIR:lfs_scene_upscaler_nvidia_dlss>"
    COMMENT "Staging the NVIDIA DLSS runtime and license beside the optional plugin")

include(GNUInstallDirs)
if(WIN32)
    set(_lfs_dlss_install_dir "${CMAKE_INSTALL_BINDIR}/scene_upscalers/nvidia")
else()
    # Loader contract: probe <exe>/scene_upscalers/nvidia and <exe-adjacent lib>/scene_upscalers/nvidia (getLibDir() is hard-coded to lib, not GNUInstallDirs).
    set(_lfs_dlss_install_dir "lib/scene_upscalers/nvidia")
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
install(FILES "${LFS_NVIDIA_DLSS_LICENSE}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/NVIDIA-DLSS")
