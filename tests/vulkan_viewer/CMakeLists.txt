# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# GPU runners explicitly opt in. macOS CI has a Vulkan/MoltenVK device;
# Windows/Linux CI must not acquire a GPU requirement from these tests.
option(LFS_BUILD_VULKAN_DEPTH_TESTS "Build and run the Vulkan HiGS depth GPU contracts" ${APPLE})
if(APPLE)
    set(_DEPTH_RASTERIZER lfs_vulkan_rasterizer_macos_reference)
else()
    set(_DEPTH_RASTERIZER lfs_vulkan_rasterizer)
endif()
if(NOT LFS_BUILD_VULKAN_DEPTH_TESTS OR NOT TARGET ${_DEPTH_RASTERIZER})
    return()
endif()

add_executable(vulkan_depth_contracts depth_contracts.cpp)
target_link_libraries(vulkan_depth_contracts PRIVATE ${_DEPTH_RASTERIZER})
target_compile_features(vulkan_depth_contracts PRIVATE cxx_std_23)
add_test(NAME VulkanDepthContracts COMMAND vulkan_depth_contracts)
set_tests_properties(VulkanDepthContracts PROPERTIES LABELS "vulkan;gpu;depth" TIMEOUT 120)
