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
    add_test(NAME ${test_name} COMMAND ${target} ${ARGN})
    set_tests_properties(${test_name} PROPERTIES LABELS "viewer;metal;gpu" TIMEOUT 120 SKIP_RETURN_CODE 77)
endfunction()
lfs_add_metal_contract(metal_lod_contracts lod_contracts.mm MetalViewerLodContracts)
lfs_add_metal_contract(metal_rad_contracts rad_contracts.mm MetalViewerRadContracts)
lfs_add_metal_contract(metal_spark_gut_contracts spark_gut_contracts.mm MetalViewerSparkGutContracts)
lfs_add_metal_contract(metal_selection_contracts selection_contracts.mm MetalViewerSelectionContracts)
lfs_add_metal_contract(metal_viewer_contracts projection_contracts.mm MetalViewerProjectionContracts)
lfs_add_metal_contract(metal_raster_contracts raster_contracts.mm MetalViewerRasterContracts)
if(TARGET lfs_core)
    lfs_add_metal_contract(metal_rad_tensor_contracts rad_tensor_contracts.mm MetalViewerRadTensorContracts)
    target_link_libraries(metal_rad_tensor_contracts PRIVATE lfs_core)
    set_target_properties(metal_rad_tensor_contracts PROPERTIES OBJCXX_STANDARD 23)
    set_property(TEST MetalViewerRadTensorContracts APPEND PROPERTY LABELS metal4)
    lfs_add_metal_contract(metal_tensor_reader_contracts tensor_reader_contracts.mm MetalViewerTensorReaderContracts)
    target_link_libraries(metal_tensor_reader_contracts PRIVATE lfs_core)
    set_target_properties(metal_tensor_reader_contracts PROPERTIES OBJCXX_STANDARD 23)
    set_property(TEST MetalViewerTensorReaderContracts APPEND PROPERTY LABELS metal4)
endif()
if(TARGET lfs_visualizer)
    find_package(Python REQUIRED COMPONENTS Development.Embed)
    lfs_add_metal_contract(metal_rad_pager_contracts rad_pager_contracts.mm MetalViewerRadPagerContracts)
    target_link_libraries(metal_rad_pager_contracts PRIVATE lfs_visualizer lfs_core lfs_io Python::Python)
    target_include_directories(metal_rad_pager_contracts PRIVATE
        "${CMAKE_SOURCE_DIR}/src/visualizer/rendering"
        "${CMAKE_SOURCE_DIR}/src/visualizer"
        "${CMAKE_SOURCE_DIR}/src/io/include"
        "${CMAKE_SOURCE_DIR}/src/rendering/include"
        "${CMAKE_SOURCE_DIR}/src")
    set_target_properties(metal_rad_pager_contracts PROPERTIES OBJCXX_STANDARD 23)
    set_property(TEST MetalViewerRadPagerContracts APPEND PROPERTY LABELS metal4)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        lfs_add_metal_contract(metal_selection_viewport_contracts selection_viewport_contracts.mm MetalViewerSelectionViewportContracts --native-only)
        target_link_libraries(metal_selection_viewport_contracts PRIVATE lfs_visualizer lfs_core Python::Python)
        target_include_directories(metal_selection_viewport_contracts PRIVATE
            "${CMAKE_SOURCE_DIR}/src/visualizer/rendering"
            "${CMAKE_SOURCE_DIR}/src/visualizer"
            "${CMAKE_SOURCE_DIR}/src/io/include"
            "${CMAKE_SOURCE_DIR}/src/rendering/include"
            "${CMAKE_SOURCE_DIR}/src")
        set_target_properties(metal_selection_viewport_contracts PROPERTIES OBJCXX_STANDARD 23)
        set_property(TEST MetalViewerSelectionViewportContracts APPEND PROPERTY LABELS metal4)
        # MoltenVK's legacy polygon kernel exceeds the Metal validator's
        # threadgroup-memory limit. Native queries retain full validation;
        # the Vulkan comparison runs alongside the other parity checks.
        add_test(NAME MacViewerParity_selectionQuery COMMAND metal_selection_viewport_contracts)
        set_tests_properties(MacViewerParity_selectionQuery PROPERTIES
            LABELS "metal;gpu;metal4;parity" TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
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
            target_link_libraries(mac_viewer_backend_benchmark PRIVATE lfs_metal_viewer lfs_visualizer lfs_core lfs_io Python::Python)
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
            # Exercise the real shared loader and camera fit in macOS CI without
            # downloads or a user-provided dataset. Generated data stays in build.
            add_test(NAME MacViewerParity_input COMMAND mac_viewer_backend_benchmark
                --smoke --input-fixture "${CMAKE_CURRENT_BINARY_DIR}/metal-viewer-input-smoke.ply"
                --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-input.json")
            set_tests_properties(MacViewerParity_input PROPERTIES
                LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            # Import format parity is macOS/GPU-only. Each case uses an isolated
            # generated file, the shared loader, both SH0 and SH3/Q16, and real
            # tensor storage. CI checks correctness, never a timing ratio.
            foreach(_format IN ITEMS ply spz3 spz4 glb sog ssog rad)
                set(_extension "${_format}")
                if(_format MATCHES "^spz")
                    set(_extension spz)
                endif()
                foreach(_tensor IN ITEMS metal vulkan)
                    foreach(_raster IN ITEMS gs gut)
                        set(_format_test MacViewerParity_format_${_format}_${_tensor}_${_raster})
                        set(_format_args)
                        if(_raster STREQUAL "gut")
                            list(APPEND _format_args --gut)
                        endif()
                        add_test(NAME ${_format_test} COMMAND mac_viewer_backend_benchmark
                            --smoke --tensor-backend ${_tensor} ${_format_args}
                            --fixture-format ${_format}
                            --input-fixture "${CMAKE_CURRENT_BINARY_DIR}/${_format_test}.${_extension}"
                            --output "${CMAKE_CURRENT_BINARY_DIR}/${_format_test}.json")
                        set_tests_properties(${_format_test} PROPERTIES
                            LABELS "metal;gpu;metal4;parity;formats" TIMEOUT 180 SKIP_RETURN_CODE 77)
                    endforeach()
                endforeach()
            endforeach()
            add_test(NAME MacViewerGpuProfileSmoke COMMAND mac_viewer_backend_benchmark
                --smoke --profile --deterministic-reference
                --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-gpu-profile-smoke.json")
            set_tests_properties(MacViewerGpuProfileSmoke PROPERTIES
                LABELS "benchmark;metal;gpu;metal4" TIMEOUT 180 SKIP_RETURN_CODE 77)
            foreach(_frustum_mode IN ITEMS native mip ortho portal)
                set(_frustum_args --frustum)
                if(NOT _frustum_mode STREQUAL "native")
                    list(APPEND _frustum_args --${_frustum_mode})
                endif()
                foreach(_frustum_alpha IN ITEMS opaque transparent)
                    set(_frustum_test MacViewerParity_frustum_${_frustum_mode}_${_frustum_alpha})
                    set(_frustum_display)
                    if(_frustum_alpha STREQUAL "transparent")
                        list(APPEND _frustum_display --transparent)
                    endif()
                    add_test(NAME ${_frustum_test} COMMAND mac_viewer_backend_benchmark
                        --smoke ${_frustum_args} ${_frustum_display}
                        --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-frustum-${_frustum_mode}-${_frustum_alpha}.json")
                    set_tests_properties(${_frustum_test} PROPERTIES
                        LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
                endforeach()
            endforeach()
            # Exercise every display curve in both composition profiles and
            # raster algorithms. RGBA cases check alpha independently of RGB.
            foreach(_display_profile IN ITEMS studio studio_gut portal portal_gut)
                set(_display_args)
                if(_display_profile MATCHES "portal")
                    list(APPEND _display_args --portal)
                endif()
                if(_display_profile MATCHES "gut")
                    list(APPEND _display_args --gut)
                endif()
                foreach(_tone RANGE 0 6)
                    set(_display_test MacViewerParity_${_display_profile}_tone${_tone})
                    add_test(NAME ${_display_test} COMMAND mac_viewer_backend_benchmark
                        --smoke ${_display_args} --tone ${_tone} --exposure 1.6
                        --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_display_profile}-tone${_tone}.json")
                    set_tests_properties(${_display_test} PROPERTIES
                        LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
                endforeach()
                foreach(_transparent_overlay IN ITEMS selection preview markers affine)
                    set(_display_test MacViewerParity_${_display_profile}_transparent_${_transparent_overlay})
                    add_test(NAME ${_display_test} COMMAND mac_viewer_backend_benchmark
                        --smoke ${_display_args} --transparent --overlay ${_transparent_overlay}
                        --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_display_profile}-transparent-${_transparent_overlay}.json")
                    set_tests_properties(${_display_test} PROPERTIES
                        LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
                endforeach()
                add_test(NAME MacViewerParity_${_display_profile}_transparent_tone COMMAND mac_viewer_backend_benchmark
                    --smoke ${_display_args} --transparent --tone 1 --exposure 1.6
                    --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_display_profile}-transparent-tone.json")
                set_tests_properties(MacViewerParity_${_display_profile}_transparent_tone PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
                foreach(_display_mode IN ITEMS transparent depth-gray)
                    set(_display_test MacViewerParity_${_display_profile}_${_display_mode})
                    add_test(NAME ${_display_test} COMMAND mac_viewer_backend_benchmark
                        --smoke ${_display_args} --${_display_mode}
                        --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_display_profile}-${_display_mode}.json")
                    set_tests_properties(${_display_test} PROPERTIES
                        LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
                endforeach()
            endforeach()
            foreach(_mode IN ITEMS mip ortho depth export_scale gut equirect near saturation)
                add_test(NAME MacViewerParity_${_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke --${_mode} --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-${_mode}.json")
                set_tests_properties(MacViewerParity_${_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_gut_mode IN ITEMS mip ortho depth export_scale near saturation)
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
            foreach(_spark_mode IN ITEMS native mip ortho depth export_scale portal portal_tone lod_weights)
                set(_spark_args --spark)
                if(NOT _spark_mode STREQUAL "native")
                    list(APPEND _spark_args --${_spark_mode})
                endif()
                add_test(NAME MacViewerParity_spark_${_spark_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke ${_spark_args} --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-spark-${_spark_mode}.json")
                set_tests_properties(MacViewerParity_spark_${_spark_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            foreach(_gpu_lod_mode IN ITEMS native mip ortho depth export_scale portal portal_tone spark budget selection affine)
                set(_gpu_lod_args --gpu_lod)
                if(_gpu_lod_mode STREQUAL "budget")
                    list(APPEND _gpu_lod_args --gpu_lod_budget)
                elseif(_gpu_lod_mode STREQUAL "selection" OR _gpu_lod_mode STREQUAL "affine")
                    list(APPEND _gpu_lod_args --overlay ${_gpu_lod_mode})
                elseif(NOT _gpu_lod_mode STREQUAL "native")
                    list(APPEND _gpu_lod_args --${_gpu_lod_mode})
                endif()
                add_test(NAME MacViewerParity_gpu_lod_${_gpu_lod_mode} COMMAND mac_viewer_backend_benchmark
                    --smoke ${_gpu_lod_args} --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-gpu-lod-${_gpu_lod_mode}.json")
                set_tests_properties(MacViewerParity_gpu_lod_${_gpu_lod_mode} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            endforeach()
            add_test(NAME MacViewerParity_gpu_lod_multichunk COMMAND mac_viewer_backend_benchmark
                --smoke --gpu_lod --count 4099 --overlay selection
                --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-gpu-lod-multichunk.json")
            set_tests_properties(MacViewerParity_gpu_lod_multichunk PROPERTIES
                LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
            foreach(_tensor_case IN ITEMS gs gut gs_transparent gut_transparent gpu_lod)
                set(_tensor_args --tensor-backend vulkan)
                if(_tensor_case MATCHES "^gut")
                    list(APPEND _tensor_args --gut)
                endif()
                if(_tensor_case MATCHES "transparent$")
                    list(APPEND _tensor_args --transparent)
                endif()
                if(_tensor_case STREQUAL "gpu_lod")
                    list(APPEND _tensor_args --gpu_lod --overlay selection)
                endif()
                add_test(NAME MacViewerParity_tensorVulkan_${_tensor_case} COMMAND mac_viewer_backend_benchmark
                    --smoke ${_tensor_args} --output "${CMAKE_CURRENT_BINARY_DIR}/mac-viewer-parity-tensor-vulkan-${_tensor_case}.json")
                set_tests_properties(MacViewerParity_tensorVulkan_${_tensor_case} PROPERTIES
                    LABELS "metal;gpu;metal4;parity" TIMEOUT 180 SKIP_RETURN_CODE 77)
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
