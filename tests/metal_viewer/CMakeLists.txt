# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(LichtFeldViewerContracts LANGUAGES CXX)
    enable_testing()
endif()

# Selection/default policy also runs on Windows and Linux.
add_executable(viewer_backend_contracts backend_contracts.cpp)
target_compile_features(viewer_backend_contracts PRIVATE cxx_std_20)
target_include_directories(viewer_backend_contracts PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/rendering/include")
add_test(NAME ViewerBackendSelectionContracts COMMAND viewer_backend_contracts)
set_tests_properties(ViewerBackendSelectionContracts PROPERTIES LABELS "viewer;cpu" TIMEOUT 30)

if(NOT APPLE)
    return()
endif()
add_executable(metal_frame_budget_contracts frame_budget_contracts.cpp)
target_compile_features(metal_frame_budget_contracts PRIVATE cxx_std_20)
target_include_directories(metal_frame_budget_contracts PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/rendering/rasterizer/metal")
add_test(NAME MetalViewerFrameBudgetContracts COMMAND metal_frame_budget_contracts)
set_tests_properties(MetalViewerFrameBudgetContracts PROPERTIES LABELS "viewer;metal;cpu" TIMEOUT 30)
enable_language(OBJCXX)
if(NOT TARGET lfs_metal_viewer)
    add_subdirectory(../../src/rendering/rasterizer/metal metal_viewer)
endif()
option(LFS_METAL_TEST_REQUIRE_DEVICE "Fail Metal contracts when no GPU is available" OFF)
function(lfs_add_metal_contract target source test_name)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE lfs_metal_viewer)
    target_compile_options(${target} PRIVATE -fobjc-arc)
    target_compile_definitions(${target} PRIVATE LFS_METAL_TEST_REQUIRE_DEVICE=$<BOOL:${LFS_METAL_TEST_REQUIRE_DEVICE}>)
    set_target_properties(${target} PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES)
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES LABELS "viewer;metal;gpu" TIMEOUT 120 SKIP_RETURN_CODE 77)
endfunction()
lfs_add_metal_contract(metal_viewer_contracts projection_contracts.mm MetalViewerProjectionContracts)
lfs_add_metal_contract(metal_raster_contracts raster_contracts.mm MetalViewerRasterContracts)
if(TARGET lfs_core)
    lfs_add_metal_contract(metal_tensor_reader_contracts tensor_reader_contracts.mm MetalViewerTensorReaderContracts)
    target_link_libraries(metal_tensor_reader_contracts PRIVATE lfs_core)
    set_target_properties(metal_tensor_reader_contracts PROPERTIES OBJCXX_STANDARD 23)
    set_property(TEST MetalViewerTensorReaderContracts APPEND PROPERTY LABELS metal4)
endif()
if(TARGET lfs_visualizer)
    find_package(Python REQUIRED COMPONENTS Development.Embed)
    lfs_add_metal_contract(metal_viewport_contracts viewport_contracts.mm MetalViewerViewportContracts)
    target_link_libraries(metal_viewport_contracts PRIVATE lfs_visualizer lfs_core Python::Python)
    target_include_directories(metal_viewport_contracts PRIVATE
        "${CMAKE_SOURCE_DIR}/src/visualizer/rendering"
        "${CMAKE_SOURCE_DIR}/src/visualizer"
        "${CMAKE_SOURCE_DIR}/src/io/include"
        "${CMAKE_SOURCE_DIR}/src/rendering/include"
        "${CMAKE_SOURCE_DIR}/src")
    set_target_properties(metal_viewport_contracts PROPERTIES OBJCXX_STANDARD 23)
    set_property(TEST MetalViewerViewportContracts APPEND PROPERTY LABELS metal4)
    # The comparison uses the real desktop adapters and is never built on iOS,
    # Windows or Linux. CI checks completion/report generation, not speed ratios.
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        option(LFS_BUILD_MAC_VIEWER_BENCHMARK "Build the native Metal versus Vulkan viewer benchmark" ON)
        if(LFS_BUILD_MAC_VIEWER_BENCHMARK)
            add_executable(mac_viewer_backend_benchmark backend_benchmark.mm)
            target_link_libraries(mac_viewer_backend_benchmark PRIVATE lfs_metal_viewer lfs_visualizer lfs_core Python::Python)
            target_compile_options(mac_viewer_backend_benchmark PRIVATE -fobjc-arc)
            target_include_directories(mac_viewer_backend_benchmark PRIVATE
                "${CMAKE_SOURCE_DIR}/src/visualizer/rendering"
                "${CMAKE_SOURCE_DIR}/src/visualizer"
                "${CMAKE_SOURCE_DIR}/src/io/include"
                "${CMAKE_SOURCE_DIR}/src/rendering/include"
                "${CMAKE_SOURCE_DIR}/src")
            set_target_properties(mac_viewer_backend_benchmark PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED YES)
            add_test(NAME MacViewerBackendBenchmarkSmoke COMMAND mac_viewer_backend_benchmark
                --smoke --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-benchmark-smoke.json")
            set_tests_properties(MacViewerBackendBenchmarkSmoke PROPERTIES
                LABELS "benchmark;metal;gpu;metal4" TIMEOUT 180 SKIP_RETURN_CODE 77)
            foreach(_mode IN ITEMS mip ortho depth export_scale gut equirect near)
                add_test(NAME MacViewerParity_${_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke --${_mode} --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_mode}.json")
                set_tests_properties(MacViewerParity_${_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_gut_mode IN ITEMS mip ortho depth export_scale near)
                add_test(NAME MacViewerParity_gut_${_gut_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke --gut --${_gut_mode}
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-gut-${_gut_mode}.json")
                set_tests_properties(MacViewerParity_gut_${_gut_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_gut_overlay IN ITEMS affine selection crop markers)
                add_test(NAME MacViewerParity_gut_${_gut_overlay} COMMAND mac_viewer_backend_benchmark
                    --smoke --gut --overlay ${_gut_overlay}
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-gut-${_gut_overlay}.json")
                set_tests_properties(MacViewerParity_gut_${_gut_overlay} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_panorama_mode IN ITEMS mip depth export_scale subregion)
                add_test(NAME MacViewerParity_equirect_${_panorama_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke --equirect --${_panorama_mode}
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-equirect-${_panorama_mode}.json")
                set_tests_properties(MacViewerParity_equirect_${_panorama_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_panorama_overlay IN ITEMS affine selection crop window markers)
                add_test(NAME MacViewerParity_equirect_${_panorama_overlay} COMMAND mac_viewer_backend_benchmark
                    --smoke --equirect --overlay ${_panorama_overlay}
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-equirect-${_panorama_overlay}.json")
                set_tests_properties(MacViewerParity_equirect_${_panorama_overlay} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_portal_mode IN ITEMS native mip ortho depth export_scale gut equirect near tone)
                set(_portal_args --portal)
                if(_portal_mode STREQUAL "tone")
                    list(APPEND _portal_args --portal_tone)
                elseif(NOT _portal_mode STREQUAL "native")
                    list(APPEND _portal_args --${_portal_mode})
                endif()
                add_test(NAME MacViewerParity_portal_${_portal_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke ${_portal_args} --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-portal-${_portal_mode}.json")
                set_tests_properties(MacViewerParity_portal_${_portal_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_portal_gut_mode IN ITEMS mip depth export_scale near portal_tone)
                add_test(NAME MacViewerParity_portal_gut_${_portal_gut_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke --portal --gut --${_portal_gut_mode}
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-portal-gut-${_portal_gut_mode}.json")
                set_tests_properties(MacViewerParity_portal_gut_${_portal_gut_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_portal_overlay IN ITEMS affine selection crop markers)
                add_test(NAME MacViewerParity_portal_${_portal_overlay} COMMAND mac_viewer_backend_benchmark
                    --smoke --portal --overlay ${_portal_overlay}
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-portal-${_portal_overlay}.json")
                set_tests_properties(MacViewerParity_portal_${_portal_overlay} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_lod_mode IN ITEMS lod lod_logical lod_weights lod_debug)
                foreach(_lod_raster IN ITEMS studio gut portal)
                    set(_lod_args --${_lod_mode})
                    if(_lod_raster STREQUAL "gut")
                        list(APPEND _lod_args --gut)
                    elseif(_lod_raster STREQUAL "portal")
                        list(APPEND _lod_args --portal)
                    endif()
                    add_test(NAME MacViewerParity_${_lod_raster}_${_lod_mode} COMMAND mac_viewer_backend_benchmark
                        --smoke ${_lod_args} --overlay selection
                        --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_lod_raster}-${_lod_mode}.json")
                    set_tests_properties(MacViewerParity_${_lod_raster}_${_lod_mode} PROPERTIES
                        LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
                endforeach()
            endforeach()
            foreach(_overlay IN ITEMS selection preview crop ellipsoid window markers flash affine)
                add_test(NAME MacViewerParity_${_overlay} COMMAND mac_viewer_backend_benchmark
                    --smoke --overlay ${_overlay}
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_overlay}.json")
                set_tests_properties(MacViewerParity_${_overlay} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
        endif()
    endif()
endif()

set(_interop_default OFF)
if(TARGET Vulkan::Vulkan AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_interop_default ON)
endif()
option(LFS_TEST_METAL_VULKAN_INTEROP "Test the installed desktop Vulkan ICD" ${_interop_default})
if(LFS_TEST_METAL_VULKAN_INTEROP)
    find_package(Vulkan REQUIRED)
    add_executable(metal_interop_contracts interop_contracts.mm)
    target_link_libraries(metal_interop_contracts PRIVATE Vulkan::Vulkan "-framework Metal" "-framework Foundation")
    target_compile_options(metal_interop_contracts PRIVATE -fobjc-arc)
    set_target_properties(metal_interop_contracts PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES)
    add_test(NAME MetalViewerInteropContracts COMMAND metal_interop_contracts)
    set_tests_properties(MetalViewerInteropContracts PROPERTIES LABELS "viewer;metal;gpu;interop" TIMEOUT 30)
endif()
