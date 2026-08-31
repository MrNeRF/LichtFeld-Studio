# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: MIT

add_library(lfs_scene_upscaler_amd_fsr3 SHARED
    amd_fsr3_plugin.cpp)

target_compile_features(lfs_scene_upscaler_amd_fsr3 PRIVATE cxx_std_23)
target_compile_definitions(lfs_scene_upscaler_amd_fsr3 PRIVATE
    LFS_SCENE_UPSCALER_PLUGIN_BUILD=1)
target_include_directories(lfs_scene_upscaler_amd_fsr3 PRIVATE
    "${CMAKE_SOURCE_DIR}/src/visualizer")
target_link_libraries(lfs_scene_upscaler_amd_fsr3 PRIVATE
    AMD::FidelityFXFsr3Upscaler
    AMD::FidelityFXVkBackend
    Vulkan::Vulkan
    Threads::Threads
    ${CMAKE_DL_LIBS})
if(MSVC)
    # The isolated SDK build deliberately produces its static archives in
    # Release. Keep this plugin on the matching DLL CRT in every host config;
    # no C++/CRT-owned objects cross the plugin's C ABI boundary.
    set_property(TARGET lfs_scene_upscaler_amd_fsr3 PROPERTY
        MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
endif()
if(LFS_AMD_FSR3_BUILD_TARGET)
    add_dependencies(lfs_scene_upscaler_amd_fsr3
        "${LFS_AMD_FSR3_BUILD_TARGET}")
endif()

set_target_properties(lfs_scene_upscaler_amd_fsr3 PROPERTIES
    OUTPUT_NAME "lfs_scene_upscaler_amd_fsr3"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:LichtFeld-Studio>/scene_upscalers/amd"
    LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:LichtFeld-Studio>/scene_upscalers/amd")

add_custom_command(TARGET lfs_scene_upscaler_amd_fsr3 POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${LFS_AMD_FSR3_LICENSE}"
        "$<TARGET_FILE_DIR:lfs_scene_upscaler_amd_fsr3>"
    COMMENT "Staging the AMD FidelityFX SDK license beside the optional FSR plugin")

include(GNUInstallDirs)
if(WIN32)
    set(_lfs_fsr3_install_dir "${CMAKE_INSTALL_BINDIR}/scene_upscalers/amd")
else()
    set(_lfs_fsr3_install_dir "lib/scene_upscalers/amd")
endif()
install(TARGETS lfs_scene_upscaler_amd_fsr3
    RUNTIME DESTINATION "${_lfs_fsr3_install_dir}"
    LIBRARY DESTINATION "${_lfs_fsr3_install_dir}")
install(FILES "${LFS_AMD_FSR3_LICENSE}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/AMD-FidelityFX-SDK")
