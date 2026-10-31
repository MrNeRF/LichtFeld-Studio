# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Testing module - comprehensive tests for all components

# Every runner built from test_main.cpp installs the tensor backend trace listener,
# including the visualizer-only configuration that never reaches the full suite.
set_property(SOURCE test_main.cpp APPEND PROPERTY COMPILE_DEFINITIONS LFS_TENSOR_PRIVATE_ACCESS)

# Standalone policy tests may include public Core descriptors without linking Core.
link_libraries(lfs_backend_configuration)
if(LFS_HAS_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Independent of the full/visualizer suites: CPU CI enables only format tests.
    # This executable supplies fake CUDART lookup functions and needs no driver.
    find_package(GTest CONFIG REQUIRED)
    add_executable(cuda_driver_entry_tests test_cuda_driver_entry.cpp)
    target_include_directories(cuda_driver_entry_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core/include)
    target_include_directories(cuda_driver_entry_tests SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_features(cuda_driver_entry_tests PRIVATE cxx_std_23)
    target_compile_definitions(cuda_driver_entry_tests PRIVATE LFS_HAS_CUDA=1)
    target_link_libraries(cuda_driver_entry_tests PRIVATE GTest::gtest_main)
    add_test(NAME CudaDriverEntryContracts COMMAND cuda_driver_entry_tests)
endif()

if(BUILD_TESTS OR BUILD_VISUALIZER_TESTS)
    find_package(GTest CONFIG REQUIRED)
    add_executable(gpu_configuration_policy_tests test_gpu_configuration_policy.cpp)
    target_include_directories(gpu_configuration_policy_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core/include)
    target_compile_features(gpu_configuration_policy_tests PRIVATE cxx_std_23)
    target_compile_definitions(gpu_configuration_policy_tests PRIVATE LFS_STATIC_BUILD)
    target_link_libraries(gpu_configuration_policy_tests PRIVATE GTest::gtest_main)
    if(LFS_HAS_CUDA)
        target_link_libraries(gpu_configuration_policy_tests PRIVATE CUDA::cudart)
        add_executable(cuda_device_selection_tests gpu_contract_main.cpp test_cuda_device_selection.cpp)
        target_include_directories(cuda_device_selection_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_compile_features(cuda_device_selection_tests PRIVATE cxx_std_23)
        target_compile_definitions(cuda_device_selection_tests PRIVATE LFS_TENSOR_PRIVATE_ACCESS)
        target_link_libraries(cuda_device_selection_tests PRIVATE lfs_core CUDA::cudart GTest::gtest)
        add_test(NAME CudaDeviceSelection COMMAND cuda_device_selection_tests)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_executable(cuda_driver_entry_live_tests test_cuda_driver_entry_live.cpp)
            target_include_directories(cuda_driver_entry_live_tests PRIVATE
                ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core/include)
            target_compile_features(cuda_driver_entry_live_tests PRIVATE cxx_std_23)
            target_compile_definitions(cuda_driver_entry_live_tests PRIVATE LFS_HAS_CUDA=1)
            target_link_libraries(cuda_driver_entry_live_tests PRIVATE CUDA::cudart GTest::gtest_main)
            add_test(NAME CudaDriverEntryLive COMMAND cuda_driver_entry_live_tests)
            set_tests_properties(CudaDriverEntryLive PROPERTIES LABELS "gpu" TIMEOUT 30)
        endif()
    endif()
    add_test(NAME GpuConfigurationPolicy COMMAND gpu_configuration_policy_tests)
    if(LFS_TENSOR_VULKAN)
        add_executable(gpu_cross_device_interop_tests gpu_contract_main.cpp test_gpu_cross_device_interop.cpp)
        target_compile_features(gpu_cross_device_interop_tests PRIVATE cxx_std_23)
        target_link_libraries(gpu_cross_device_interop_tests PRIVATE lfs_core GTest::gtest)
        add_test(NAME GpuCrossDeviceInterop COMMAND gpu_cross_device_interop_tests)
        set_tests_properties(GpuCrossDeviceInterop PROPERTIES LABELS "gpu;interop" TIMEOUT 120)
    endif()
endif()
if(BUILD_TESTS OR BUILD_FORMAT_TESTS OR BUILD_VISUALIZER_TESTS)
    add_subdirectory(media)
endif()

if(BUILD_TESTS OR BUILD_FORMAT_TESTS)
    add_executable(apple_reframe_conversion_tests test_apple_reframe_conversion.cpp)
    target_include_directories(apple_reframe_conversion_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_features(apple_reframe_conversion_tests PRIVATE cxx_std_23)
    add_test(NAME AppleReframeConversion COMMAND apple_reframe_conversion_tests)
endif()

if((BUILD_TESTS OR BUILD_FORMAT_TESTS OR BUILD_VISUALIZER_TESTS) AND
   (NOT APPLE OR (LFS_GRAPHICS_VULKAN_BUILD AND LFS_TENSOR_METAL)))
    add_subdirectory(metal_viewer)
endif()

if(BUILD_TESTS OR BUILD_FORMAT_TESTS)
    add_subdirectory(image_codecs)
endif()

# =============================================================================
# Headless localization contracts (no LibTorch/CUDA)
# =============================================================================
if(BUILD_LOCALIZATION_TESTS)
    find_package(Python REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME LocalizationContracts
        COMMAND ${Python_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/python/test_localization_contracts.py
    )
    set_tests_properties(LocalizationContracts PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TIMEOUT 30
    )
    add_test(NAME MacosPortableInventory
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_macos_portable_inventory.py)
    set_tests_properties(MacosPortableInventory PROPERTIES TIMEOUT 30)
    if(APPLE)
        add_test(NAME MacosPortableResourcePaths
            COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_macos_portable_resource_paths.py)
        set_tests_properties(MacosPortableResourcePaths PROPERTIES TIMEOUT 30)
    endif()
    add_custom_target(test_localization_contracts
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R LocalizationContracts
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake
        USES_TERMINAL
    )
    message(STATUS "Registered headless localization contract tests (no LibTorch)")
    if(NOT BUILD_TESTS AND NOT BUILD_UNICODE_TEST_ONLY AND NOT BUILD_FORMAT_TESTS AND NOT BUILD_TEMPORAL_CPU_TESTS AND NOT BUILD_PANEL_LAYOUT_TESTS AND NOT BUILD_VISUALIZER_TESTS)
        return()
    endif()
endif()

if(BUILD_TESTS OR BUILD_VISUALIZER_TESTS)
    add_subdirectory(vulkan_viewer)
endif()

# Find GTest
find_package(GTest CONFIG REQUIRED)

# Runtime shader effect compiler contracts; no GPU. Validation builds also run
# spirv-val on every SPIR-V module the tests emit.
if((BUILD_TESTS OR BUILD_VISUALIZER_TESTS) AND LFS_VKSPLAT_VIEWPORT_BUILD AND TARGET lfs_visualizer
        AND TARGET lfs_user_shader_ir)
    get_target_property(_lfs_user_shader_dir lfs_user_shader_ir LFS_USER_SHADER_PACKAGE_DIR)
    get_target_property(_lfs_user_shader_test_dir lfs_user_shader_ir LFS_USER_SHADER_TEST_PACKAGE_DIR)
    get_target_property(_lfs_spirv_val lfs_user_shader_ir LFS_SPIRV_VAL)
    set(_lfs_slang_test_spv "${CMAKE_CURRENT_BINARY_DIR}/slang_compiler_spv")
    add_executable(lichtfeld_slang_compiler_tests test_slang_compiler.cpp)
    target_compile_features(lichtfeld_slang_compiler_tests PRIVATE cxx_std_23)
    target_include_directories(lichtfeld_slang_compiler_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/visualizer ${CMAKE_SOURCE_DIR}/src/core/include)
    target_compile_definitions(lichtfeld_slang_compiler_tests PRIVATE
        LFS_SHADER_PACKAGE_DIR="${_lfs_user_shader_dir}"
        LFS_SHADER_TEST_PACKAGE_DIR="${_lfs_user_shader_test_dir}"
        LFS_SHADER_TEST_SPIRV_DIR="${_lfs_slang_test_spv}")
    target_link_libraries(lichtfeld_slang_compiler_tests PRIVATE
        lfs_vulkan_rasterizer lfs_visualizer lfs_core GTest::gtest_main)
    add_dependencies(lichtfeld_slang_compiler_tests lfs_user_shader_ir lfs_user_shader_test_package)
    if(MSVC)
        target_link_options(lichtfeld_slang_compiler_tests PRIVATE "/DELAYLOAD:nvcuda.dll")
        target_link_libraries(lichtfeld_slang_compiler_tests PRIVATE delayimp)
    endif()
    add_test(NAME lichtfeld.slang_compiler
        COMMAND ${CMAKE_COMMAND}
            "-DTEST_EXECUTABLE=$<TARGET_FILE:lichtfeld_slang_compiler_tests>"
            "-DSPIRV_DIR=${_lfs_slang_test_spv}"
            "-DSPIRV_VAL=${_lfs_spirv_val}"
            -P "${CMAKE_SOURCE_DIR}/cmake/RunSlangCompilerTests.cmake")
    set_tests_properties(lichtfeld.slang_compiler PROPERTIES LABELS "fast;cpu;shaders" TIMEOUT 120
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:slang::slang>;PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_visualizer>")
endif()

add_executable(lichtfeld_batch_process_tests test_batch_process.cpp)
target_compile_features(lichtfeld_batch_process_tests PRIVATE cxx_std_23)
target_include_directories(lichtfeld_batch_process_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(lichtfeld_batch_process_tests PRIVATE GTest::gtest_main)
add_test(NAME lichtfeld.batch_process COMMAND lichtfeld_batch_process_tests)
set_tests_properties(lichtfeld.batch_process PROPERTIES LABELS "cpu;batch" TIMEOUT 60)

if(LFS_HAS_CUDA)
    add_executable(lichtfeld_collision_tests test_collision.cpp
        ../src/app/converter.cpp ../src/app/converter_overwrite.cpp ../src/app/converter_output_paths.cpp)
    target_compile_features(lichtfeld_collision_tests PRIVATE cxx_std_23)
    target_compile_definitions(lichtfeld_collision_tests PRIVATE PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}")
    target_include_directories(lichtfeld_collision_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/app/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/external)
    target_link_libraries(lichtfeld_collision_tests PRIVATE lfs_io lfs_core lfs_rendering GTest::gtest_main CUDA::cudart CUDA::cuda_driver LibArchive::LibArchive)
    add_test(NAME lichtfeld.collision COMMAND lichtfeld_collision_tests --gtest_filter=-CollisionGpu.*)
    add_test(NAME lichtfeld.collision_gpu COMMAND lichtfeld_collision_tests --gtest_filter=CollisionGpu.*)
    set_tests_properties(lichtfeld.collision PROPERTIES LABELS "cpu;collision")
    set_tests_properties(lichtfeld.collision_gpu PROPERTIES LABELS "gpu;collision" TIMEOUT 120)
endif()

# Shared tensor contracts are available without the full LibTorch comparison suite.
if(BUILD_TESTS OR BUILD_FORMAT_TESTS OR BUILD_VISUALIZER_TESTS)
    include(${CMAKE_CURRENT_LIST_DIR}/TensorOperationContracts.cmake)
endif()

# Deliberately no executable opt-in: exercise extended-path I/O independently
# of the user's/runner's LongPathsEnabled registry policy.
if(WIN32 AND (BUILD_TESTS OR BUILD_UNICODE_TEST_ONLY OR BUILD_FORMAT_TESTS OR BUILD_VISUALIZER_TESTS))
    add_executable(lichtfeld_native_filesystem_tests
        test_native_filesystem.cpp test_native_logging.cpp native_filesystem_test.manifest)
    target_compile_features(lichtfeld_native_filesystem_tests PRIVATE cxx_std_23)
    target_compile_definitions(lichtfeld_native_filesystem_tests PRIVATE
        LFS_STATIC_BUILD LFS_REQUIRE_LONG_PATHS_DISABLED)
    target_include_directories(lichtfeld_native_filesystem_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core/include)
    target_link_libraries(lichtfeld_native_filesystem_tests PRIVATE
        GTest::gtest_main spdlog::spdlog xxHash::xxhash shell32)
    add_test(NAME lichtfeld.native_filesystem COMMAND lichtfeld_native_filesystem_tests)
    set_tests_properties(lichtfeld.native_filesystem PROPERTIES LABELS "cpu;windows;filesystem" TIMEOUT 30)
endif()

# Exercise a real EXE/DLL boundary: linking the same static runner into both
# modules would pass same-module tests but split callback and initialization state.
# Windows CI enables the focused visualizer tests without the full test suite.
if(WIN32 AND (BUILD_TESTS OR BUILD_VISUALIZER_TESTS) AND TARGET lfs_python_utils)
    add_library(lichtfeld_python_state_helper SHARED python_runtime_state_helper.cpp)
    target_compile_features(lichtfeld_python_state_helper PRIVATE cxx_std_23)
    target_compile_definitions(lichtfeld_python_state_helper PRIVATE LFS_TEST_PYTHON_STATE_EXPORTS)
    target_link_libraries(lichtfeld_python_state_helper PRIVATE lfs_python_utils)

    add_executable(lichtfeld_python_state_tests test_python_runtime_state.cpp)
    target_compile_features(lichtfeld_python_state_tests PRIVATE cxx_std_23)
    target_link_libraries(lichtfeld_python_state_tests PRIVATE
        lfs_python_utils lichtfeld_python_state_helper GTest::gtest_main)
    add_test(NAME lichtfeld.python_runtime_state COMMAND lichtfeld_python_state_tests)
    set_tests_properties(lichtfeld.python_runtime_state PROPERTIES
        LABELS "python;windows" TIMEOUT 30
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_python_runtime>;PATH=path_list_prepend:${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/$<IF:$<CONFIG:Debug>,debug/bin,bin>")
endif()


if((BUILD_TESTS OR BUILD_VISUALIZER_TESTS) AND LFS_TENSOR_METAL AND NOT LFS_GRAPHICS_VULKAN_BUILD)
    add_executable(metal_scene_upscaler_contracts
        test_metal_scene_upscaler.cpp
        test_temporal_frame_tracker.cpp
        test_scene_temporal_coordinator.cpp
        test_scene_upscaler_registry.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_upscaler_registry.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_upscaler_plugin_unavailable.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/tensor_scene_temporal_pipeline.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/metal_scene_upscaler.mm
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_coordinator.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_plan.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_resolve.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_frame_setup.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/temporal_frame_tracker.cpp)
    target_include_directories(metal_scene_upscaler_contracts PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/visualizer
        ${CMAKE_SOURCE_DIR}/src/rendering/include
        ${CMAKE_SOURCE_DIR}/src/rendering/rasterizer/tensor)
    target_compile_features(metal_scene_upscaler_contracts PRIVATE cxx_std_23)
    set_target_properties(metal_scene_upscaler_contracts PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED YES)
    target_compile_options(metal_scene_upscaler_contracts PRIVATE
        "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
    target_link_libraries(metal_scene_upscaler_contracts PRIVATE
        lfs_tensor_rasterizer GTest::gtest_main "-framework MetalFX" "-framework Metal" "-framework Foundation")
    add_test(NAME MetalSceneUpscalerContracts COMMAND metal_scene_upscaler_contracts)
    set_tests_properties(MetalSceneUpscalerContracts PROPERTIES LABELS "viewer;metal;gpu;upscaler" TIMEOUT 180
        ENVIRONMENT "MTL_DEBUG_LAYER=1")
endif()

# The tensor HDR tonemapper against libplacebo, on clips ffmpeg encodes.
if(LFS_TENSOR_METAL AND BUILD_TESTS)
    find_package(PkgConfig REQUIRED)
    find_program(LFS_HDR_TEST_FFMPEG NAMES ffmpeg)
    add_executable(hdr_tonemap_contracts test_hdr_tonemap_tensor.cpp)
    target_compile_features(hdr_tonemap_contracts PRIVATE cxx_std_23)
    # Fixture construction uses a private FFmpeg link; renderer input is media-owned.
    target_include_directories(hdr_tonemap_contracts PRIVATE ${CMAKE_SOURCE_DIR}/src/io ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(hdr_tonemap_contracts PRIVATE LFS_HDR_TEST_FFMPEG="${LFS_HDR_TEST_FFMPEG}"
        $<$<BOOL:${LFS_USE_VULKAN}>:LFS_USE_VULKAN>)
    target_link_libraries(hdr_tonemap_contracts PRIVATE lfs_video lfs_core GTest::gtest_main ${FFMPEG_LIBRARIES})
    target_include_directories(hdr_tonemap_contracts SYSTEM PRIVATE ${FFMPEG_INCLUDE_DIRS})
    add_test(NAME HdrTonemapContracts COMMAND hdr_tonemap_contracts)
    # libplacebo is the precise-math reference; MoltenVK defaults to fast math.
    set_tests_properties(HdrTonemapContracts PROPERTIES ENVIRONMENT "MVK_CONFIG_FAST_MATH_ENABLED=0"
        LABELS "tensor;gpu;video" TIMEOUT 300)
endif()

# Standalone tensor/training proofs must not pull in the Vulkan window or its
# test harness when the core Vulkan backend is disabled.
if(LFS_TENSOR_METAL AND BUILD_TESTS AND NOT LFS_TENSOR_VULKAN)
    add_executable(metal_tensor_contracts test_tensor_metal.cpp test_tensor_restored.cpp test_tensor_image_regions.cpp)
    target_compile_features(metal_tensor_contracts PRIVATE cxx_std_23)
    target_compile_definitions(metal_tensor_contracts PRIVATE LFS_TENSOR_PRIVATE_ACCESS)
    target_include_directories(metal_tensor_contracts PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(metal_tensor_contracts PRIVATE lfs_core GTest::gtest_main)
    add_test(NAME MetalTensorNoVulkanContracts COMMAND metal_tensor_contracts)
    if(LFS_BUILD_TRAINER)
        add_executable(metal_training_contracts
            test_training_ops_metal_losses.cpp
            test_training_ops_metal_densify.cpp
            test_training_ops_metal_appearance.cpp
            test_training_ops_metal_gsplat.cpp)
        target_compile_features(metal_training_contracts PRIVATE cxx_std_23)
        target_link_libraries(metal_training_contracts PRIVATE lfs_training lfs_python_embed GTest::gtest_main)
        find_package(Python REQUIRED COMPONENTS Development.Embed)
        target_link_options(metal_training_contracts PRIVATE "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
        add_test(NAME MetalTrainingNoVulkanContracts COMMAND metal_training_contracts)
    endif()
endif()

if(BUILD_TESTS)
    if(LFS_TENSOR_METAL)
        add_executable(tensor_mesh_offscreen_contracts test_tensor_mesh_offscreen.cpp)
        target_compile_features(tensor_mesh_offscreen_contracts PRIVATE cxx_std_23)
        target_compile_definitions(tensor_mesh_offscreen_contracts PRIVATE
            LFS_TEST_VULKAN_MESH_REFERENCE=$<BOOL:${LFS_GRAPHICS_VULKAN_BUILD}>)
        target_include_directories(tensor_mesh_offscreen_contracts PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/visualizer
            ${CMAKE_SOURCE_DIR}/src/rendering/include)
        target_link_libraries(tensor_mesh_offscreen_contracts PRIVATE
            lfs_visualizer lfs_core GTest::gtest_main)
        add_test(NAME TensorMeshOffscreenContracts COMMAND tensor_mesh_offscreen_contracts)
        set_tests_properties(TensorMeshOffscreenContracts PROPERTIES
            LABELS "tensor;gpu;rendering" TIMEOUT 180)

        add_executable(tensor_frosted_glass_contracts test_tensor_frosted_glass.cpp)
        target_compile_features(tensor_frosted_glass_contracts PRIVATE cxx_std_23)
        target_include_directories(tensor_frosted_glass_contracts PRIVATE
            ${CMAKE_SOURCE_DIR}/src/visualizer)
        target_link_libraries(tensor_frosted_glass_contracts PRIVATE
            lfs_visualizer lfs_core GTest::gtest_main)
        add_test(NAME TensorFrostedGlassContracts COMMAND tensor_frosted_glass_contracts)
        set_tests_properties(TensorFrostedGlassContracts PROPERTIES
            LABELS "tensor;gpu;ui" TIMEOUT 120)
        if(APPLE)
            find_package(Python REQUIRED COMPONENTS Development.Embed)
            target_link_options(tensor_mesh_offscreen_contracts PRIVATE
                "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
            target_link_options(tensor_frosted_glass_contracts PRIVATE
                "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
            target_link_libraries(tensor_mesh_offscreen_contracts PRIVATE lfs_python_embed)
            target_link_libraries(tensor_frosted_glass_contracts PRIVATE lfs_python_embed)
        endif()
    endif()
    add_executable(mesh2splat_tensor_contracts test_mesh2splat_tensor.cpp)
    target_compile_features(mesh2splat_tensor_contracts PRIVATE cxx_std_23)
    target_link_libraries(mesh2splat_tensor_contracts PRIVATE
        lfs_rendering_tensor lfs_core GTest::gtest_main)
    if(LFS_USE_VULKAN AND NOT APPLE)
        target_compile_definitions(mesh2splat_tensor_contracts PRIVATE
            LFS_MESH2SPLAT_VULKAN_REFERENCE)
    endif()
    add_test(NAME Mesh2SplatTensorContracts COMMAND mesh2splat_tensor_contracts)
    set_tests_properties(Mesh2SplatTensorContracts PROPERTIES
        LABELS "tensor;gpu" TIMEOUT 180
        ENVIRONMENT "MVK_CONFIG_FAST_MATH_ENABLED=0")
    if(LFS_TENSOR_VULKAN)
        add_executable(ui_texture_tensor_contracts test_ui_texture_tensor.cpp)
        target_compile_features(ui_texture_tensor_contracts PRIVATE cxx_std_23)
        target_include_directories(ui_texture_tensor_contracts PRIVATE ${CMAKE_SOURCE_DIR}/src/visualizer)
        target_link_libraries(ui_texture_tensor_contracts PRIVATE lfs_visualizer lfs_core GTest::gtest_main lfs_python_embed)
        if(APPLE)
            target_link_options(ui_texture_tensor_contracts PRIVATE "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
        endif()
        add_test(NAME UiTextureTensorContracts COMMAND ui_texture_tensor_contracts)
        set_tests_properties(UiTextureTensorContracts PROPERTIES LABELS "gpu;ui" TIMEOUT 120)
    endif()
    add_executable(gpu_program_contracts gpu_contract_main.cpp test_gpu_program.cpp test_gpu_bookkeeping.cpp)
    target_include_directories(gpu_program_contracts PRIVATE ${CMAKE_SOURCE_DIR}/src/visualizer)
    target_link_libraries(gpu_program_contracts PRIVATE lfs_core GTest::gtest)
    target_compile_features(gpu_program_contracts PRIVATE cxx_std_23)
    lfs_add_gpu_program(gpu_program_contracts spark_alpha_contract SOURCE shaders/spark_alpha_contract.slang
        DEPENDS "${CMAKE_SOURCE_DIR}/src/rendering/rasterizer/slang/spark_alpha.inc"
        COMPUTE evaluate)
    lfs_add_gpu_program(gpu_program_contracts split_view_contract
        SOURCE "${CMAKE_SOURCE_DIR}/src/visualizer/rendering/split_view_tensor.slang" COMPUTE composeSplitView)
    lfs_add_gpu_program(gpu_program_contracts program_contract SOURCE shaders/program_contract.slang
        COMPUTE transform VERTEX vertexMain FRAGMENT fragmentMain)
    lfs_add_gpu_program(gpu_program_contracts program_features SOURCE shaders/program_features.slang
        COMPUTE waveOps sharedReduce atomics narrowAndWide prepareIndirect markIndirect optionalInput variant)
    lfs_add_gpu_program(gpu_program_contracts program_features_variant SOURCE shaders/program_features.slang
        COMPUTE variant DEFINES LFS_FEATURE_VARIANT=5)
    add_test(NAME GpuProgramCudaArchitectures COMMAND ${CMAKE_COMMAND}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_gpu_program_cuda.cmake")
    add_test(NAME GpuProgramContracts COMMAND gpu_program_contracts)
    if(LFS_HAS_CUDA)
        lfs_gpu_program_cuda_flags(_program_cuda_flags)
    endif()
    if(LFS_HAS_CUDA AND "${_program_cuda_flags}" MATCHES "sm_[0-9]")
        # A driver capable of JIT-compiling PTX would otherwise hide missing SASS.
        add_test(NAME GpuProgramNativeCodeContracts COMMAND gpu_program_contracts)
        set_tests_properties(GpuProgramNativeCodeContracts PROPERTIES
            ENVIRONMENT "CUDA_DISABLE_PTX_JIT=1" LABELS "tensor;gpu" TIMEOUT 120)
    endif()
    set_tests_properties(GpuProgramContracts PROPERTIES LABELS "tensor;gpu" TIMEOUT 120)
    if(TARGET lfs_tensor_rasterizer)
        add_executable(tensor_rasterizer_contracts
            gpu_contract_main.cpp
            test_splat_point_renderer.cpp
            test_splat_tile_binner.cpp
            test_splat_rasterizer.cpp
            test_splat_lod_selector.cpp
            test_splat_raster_contracts.cpp
            test_splat_projection_contracts.cpp
            test_splat_spark_gut_contracts.cpp
            test_splat_rad_contracts.cpp
            test_tensor_scene_temporal.cpp
            ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_resolve.cpp)
        target_include_directories(tensor_rasterizer_contracts PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/visualizer)
        target_link_libraries(tensor_rasterizer_contracts PRIVATE lfs_tensor_rasterizer GTest::gtest)
        target_compile_features(tensor_rasterizer_contracts PRIVATE cxx_std_23)
        add_test(NAME TensorRasterizerContracts COMMAND tensor_rasterizer_contracts)
        # The contracts check precise-math oracles; MoltenVK compiles its MSL
        # with fast math unless told otherwise.
        set_tests_properties(TensorRasterizerContracts PROPERTIES ENVIRONMENT "MVK_CONFIG_FAST_MATH_ENABLED=0")
        set_tests_properties(TensorRasterizerContracts PROPERTIES LABELS "tensor;gpu" TIMEOUT 180)
    endif()
endif()

if(APPLE AND (BUILD_VISUALIZER_TESTS OR BUILD_TESTS))
    # Cocoa delivers SDL drag events; this contract exercises the bridge without a GPU.
    add_executable(mac_native_drag_drop_contracts
        test_mac_native_drag_drop.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/gui/utils/drag_drop_native.cpp)
    target_compile_features(mac_native_drag_drop_contracts PRIVATE cxx_std_23)
    target_include_directories(mac_native_drag_drop_contracts PRIVATE
        ${CMAKE_SOURCE_DIR}/src/visualizer)
    target_link_libraries(mac_native_drag_drop_contracts PRIVATE
        SDL3::SDL3 lfs_logger GTest::gtest_main)
    add_test(NAME MacNativeDragDropContracts COMMAND mac_native_drag_drop_contracts)
    set_tests_properties(MacNativeDragDropContracts PROPERTIES
        LABELS "cpu;macos" TIMEOUT 30 ENVIRONMENT "SDL_VIDEODRIVER=dummy")
endif()

# Save-path policy and the production native-dialog adapter, without opening an OS picker.
add_executable(lichtfeld_file_dialog_tests test_file_dialog_path.cpp)
target_compile_features(lichtfeld_file_dialog_tests PRIVATE cxx_std_23)
target_include_directories(lichtfeld_file_dialog_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/geometry/include
    ${CMAKE_SOURCE_DIR}/src/visualizer
)
# scene_export_path.hpp includes the core event declarations and geometry headers.
# Declare their header dependencies here, independently of the Linux GTK adapter.
target_link_libraries(lichtfeld_file_dialog_tests PRIVATE
    GTest::gtest_main glm::glm nlohmann_json::nlohmann_json
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET lfs_visualizer AND TARGET PkgConfig::LFS_GTK3)
    target_sources(lichtfeld_file_dialog_tests PRIVATE
        test_native_save_dialog.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/gui/utils/native_file_dialog.cpp
    )
    target_link_libraries(lichtfeld_file_dialog_tests PRIVATE
        lfs_io lfs_logger SDL3::SDL3 nfd::nfd PkgConfig::LFS_GTK3
    )
    target_link_options(lichtfeld_file_dialog_tests PRIVATE
        "-Wl,--wrap=NFD_Init" "-Wl,--wrap=NFD_Quit" "-Wl,--wrap=NFD_SaveDialogU8_With_Impl"
    )
endif()
add_test(NAME lichtfeld.file_dialog_paths COMMAND lichtfeld_file_dialog_tests)
set_tests_properties(lichtfeld.file_dialog_paths PROPERTIES LABELS "fast;cpu;io" TIMEOUT 30)

add_executable(lichtfeld_apriltag_scaling_tests test_apriltag_scaling.cpp)
target_compile_features(lichtfeld_apriltag_scaling_tests PRIVATE cxx_std_23)
target_link_libraries(lichtfeld_apriltag_scaling_tests PRIVATE
    lfs_geometry
    GTest::gtest_main
    glm::glm
    nlohmann_json::nlohmann_json
)
add_test(NAME lichtfeld.apriltag_scaling COMMAND lichtfeld_apriltag_scaling_tests)
set_tests_properties(lichtfeld.apriltag_scaling PROPERTIES LABELS "fast;cpu;geometry" TIMEOUT 30)

# =============================================================================
# CPU-only visualizer tests (no LibTorch)
# =============================================================================
if(BUILD_VISUALIZER_TESTS OR BUILD_TESTS)
    # The focused menu-bar suite exercises no GPU code. Windows CI installs
    # the CUDA toolkit but has no display driver, so defer loading nvcuda.dll
    # until a driver API is actually called. Application builds load the driver eagerly.
    # lfs_mcp and the Python module call the driver API directly too, and CI imports
    # the module from a plain Python process where nothing has loaded nvcuda.dll first.
    if(MSVC AND BUILD_VISUALIZER_TESTS)
        foreach(_driver_user IN ITEMS lfs_core lfs_visualizer lfs_mcp lfs_py)
            if(TARGET ${_driver_user})
                target_link_options(${_driver_user} PRIVATE "/DELAYLOAD:nvcuda.dll")
                target_link_libraries(${_driver_user} PRIVATE delayimp)
            endif()
        endforeach()
    endif()

    if(WIN32)
        find_package(PkgConfig REQUIRED)
        find_package(LibArchive REQUIRED)
        pkg_check_modules(LFS_PATH_TEST_FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
        add_executable(lichtfeld_native_third_party_tests
            test_native_third_party_paths.cpp native_filesystem_test.manifest)
        target_compile_features(lichtfeld_native_third_party_tests PRIVATE cxx_std_23)
        target_compile_definitions(lichtfeld_native_third_party_tests PRIVATE LFS_STATIC_BUILD)
        target_include_directories(lichtfeld_native_third_party_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/core/include ${CMAKE_SOURCE_DIR}/src/visualizer)
        target_link_libraries(lichtfeld_native_third_party_tests PRIVATE
            GTest::gtest_main RmlUi::RmlUi assimp::assimp LibArchive::LibArchive PkgConfig::LFS_PATH_TEST_FFMPEG shell32)
        add_test(NAME lichtfeld.native_third_party_paths COMMAND lichtfeld_native_third_party_tests)
        set_tests_properties(lichtfeld.native_third_party_paths PROPERTIES LABELS "cpu;windows;filesystem" TIMEOUT 30)
    endif()

    set(LICHTFELD_VISUALIZER_TEST_SOURCES
        test_analog_navigation.cpp
        test_gamepad_service.cpp
        test_gamepad_navigation.cpp
        test_gamepad_sequencer.cpp
        test_sequencer_keyframe_gesture.cpp
        licht_test_support.cpp
        test_json_chapter_dom.cpp
        test_visualizer_post_work.cpp
        test_project_document.cpp
        test_keymap_upgrade.cpp
        test_keymap_default_contract.cpp
        test_keymap_layout_contract.cpp
        test_align_transform.cpp
        test_scale_distance.cpp
        test_orthographic_zoom.cpp
        test_zoom_toward_cursor.cpp
        test_ui_visibility_transition.cpp
        test_summary_labels.cpp
        test_menu_bar_title.cpp
        test_point_cloud_updates.cpp
        test_preferences_tooltips.cpp
        test_guide_segment_clipping.cpp

        test_viewport_depth_layout.cpp
        test_properties_tab_rail.cpp
        test_preferences_migration.cpp
        test_scene_upscaler_registry.cpp
        test_screen_layout.cpp
        test_screen.cpp
        test_view_render_state.cpp
        test_python_editor_input.cpp
        test_python_runtime_callbacks.cpp
        test_startup_overlay_geometry.cpp
        test_status_bar_fit.cpp
        test_viewer_backend_metadata.cpp
        test_trackpad_navigation.cpp
        test_training_manager_capacity.cpp
        test_vram_hud_geometry.cpp
        test_vram_hud_header_click.cpp
        test_input_key_repeat.cpp
        test_input_controller_focus.cpp
    )
    # Match the renderer sources in src/visualizer: Metal builds (including
    # Vulkan-graphics/Metal-tensor hybrids) use the Metal scene renderer and
    # do not compile PointCloudVulkanRenderer.
    if(NOT LFS_TENSOR_METAL)
        list(APPEND LICHTFELD_VISUALIZER_TEST_SOURCES test_point_cloud_vulkan_projection.cpp)
    endif()
    if(LFS_GRAPHICS_VULKAN_BUILD)
        list(APPEND LICHTFELD_VISUALIZER_TEST_SOURCES test_viewport_grid.cpp)
    endif()
    if(NOT LFS_HAS_CUDA)
        # CUDA builds run these in lichtfeld_tests.
        list(APPEND LICHTFELD_VISUALIZER_TEST_SOURCES
            test_gpu_memory_query.cpp
            test_node_canvas_widgets.cpp
    test_panel_layout_render_demand.cpp
            test_rml_static_style_boundaries.cpp
            test_viewer_gpu_memory.cpp)
    endif()
    add_executable(lichtfeld_visualizer_tests
        test_main.cpp
        ${LICHTFELD_VISUALIZER_TEST_SOURCES}
    )
    # Focused RmlUi migration contracts for the retained master interface.
    add_executable(lichtfeld_rmlui_tests
        test_rml_preview_texture.cpp
        test_rml_text_input_handler.cpp
        test_rml_text_input_native_keys.cpp
        test_rml_path_utils.cpp
        test_rml_static_style_boundaries.cpp
        test_rml_drag_payload.cpp
        test_sdl_rml_key_mapping.cpp
        test_rml_form_control_layout.cpp
        test_histogram_layout.cpp
    )
    target_compile_features(lichtfeld_rmlui_tests PRIVATE cxx_std_23)
    target_include_directories(lichtfeld_rmlui_tests PRIVATE
        $<TARGET_PROPERTY:lfs_visualizer,INCLUDE_DIRECTORIES>)
    target_compile_definitions(lichtfeld_rmlui_tests PRIVATE PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}")
    target_link_libraries(lichtfeld_rmlui_tests PRIVATE
        lfs_visualizer lfs_core lfs_python_runtime GTest::gtest_main Zep::Zep SDL3::SDL3)
    if(MSVC)
        target_link_options(lichtfeld_rmlui_tests PRIVATE "/DELAYLOAD:nvcuda.dll")
        target_link_libraries(lichtfeld_rmlui_tests PRIVATE delayimp)
    endif()
    if(APPLE)
        # The visualizer resolves CPython from the host process; see lichtfeld_visualizer_tests.
        find_package(Python REQUIRED COMPONENTS Development.Embed)
        target_link_options(lichtfeld_rmlui_tests PRIVATE
            "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
        target_link_libraries(lichtfeld_rmlui_tests PRIVATE lfs_python_embed)
    endif()
    add_test(NAME lichtfeld.rmlui COMMAND lichtfeld_rmlui_tests)
    set_tests_properties(lichtfeld.rmlui PROPERTIES LABELS "fast;cpu;ui" TIMEOUT 120
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    target_compile_features(lichtfeld_visualizer_tests PRIVATE cxx_std_23)
    target_include_directories(lichtfeld_visualizer_tests PRIVATE
        $<TARGET_PROPERTY:lfs_visualizer,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(lichtfeld_visualizer_tests PRIVATE
        LFS_TEST_SKIP_GPU_PREWARM
        PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
    )
    target_link_libraries(lichtfeld_visualizer_tests PRIVATE
        lfs_training
        lfs_io
        lfs_sequencer
        lfs_visualizer
        lfs_core
        lfs_python_runtime
        GTest::gtest
        Zep::Zep
        SDL3::SDL3
        ZLIB::ZLIB
    )
    if(MSVC)
        # Static training/I/O objects also bring driver imports into this EXE.
        # CPU-only CI has the toolkit, but no NVIDIA display driver.
        target_link_options(lichtfeld_visualizer_tests PRIVATE "/DELAYLOAD:nvcuda.dll")
        target_link_libraries(lichtfeld_visualizer_tests PRIVATE delayimp)
    endif()
    if(APPLE)
        # On macOS the visualizer resolves CPython from the host process. The app
        # pulls libpython in by embedding it; both test hosts must load it whole.
        find_package(Python REQUIRED COMPONENTS Development.Embed)
        foreach(_python_visualizer_test IN ITEMS lichtfeld_visualizer_tests lichtfeld_rmlui_tests)
            target_link_options(${_python_visualizer_test} PRIVATE
                "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
            target_link_libraries(${_python_visualizer_test} PRIVATE lfs_python_embed)
        endforeach()
    endif()
    # The trackpad tests set SDL's modifier state. A static SDL is reached through
    # lfs_visualizer; a shared one (Windows) must be linked so both see one state.
    if(TARGET SDL3::SDL3-shared)
        target_link_libraries(lichtfeld_visualizer_tests PRIVATE SDL3::SDL3-shared)
    endif()
    add_test(
        NAME lichtfeld_visualizer_tests
        COMMAND lichtfeld_visualizer_tests --gtest_filter=-CollisionJobsTest.*:AsyncSplatPayloadTest.*:CropVolumeSourceTest.*:DatasetEmbedIntegrationTest.*:ImportComparisonTest.*:ImportErrorTest.*:MainLoopSignalTest.*:PayloadRecaptureTest.*:ProjectSwitchErrorTest.*:SceneGenerationRenderPublicationTest.*:SelectionSubmodeTest.*:SequencerFrameDemandTest.*:SplatSimplifyCompletionTest.*:SplatSimplifyIdentityTest.*:StoredSessionRestoreTest.*:TrainerProjectWriterTest.*:ViewportAppearanceCorrectionTest.*:VisualizerImplResetTest.*:VisualizerPostWorkTest.*:VisualizerPostedWorkTest.*:CameraCreationSchemaTest.*:LoaderGeoreferenceTest.*:ProjectDocumentPerformanceGate.*:ProjectDocumentTest.*:SceneChapterAdapterTest.*:SfmObservationChapterTest.*:SpanStreambufTest.*:SplatChapterTest.*
    )
    set_tests_properties(lichtfeld_visualizer_tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        # This aggregate runs hundreds of cases, including repeated real-time
        # tooltip reveal delays across locales and scales. Allow slower CI hosts
        # and the assert-enabled Windows Debug build to finish without changing
        # individual contracts or disabling coverage.
        TIMEOUT 300
    )
    add_test(NAME lichtfeld.collision_jobs_gpu COMMAND lichtfeld_visualizer_tests --gtest_filter=CollisionJobsTest.*:VisualizerImplResetTest.StartupRecoveryScansRootAndProjectLocationScratch)
    add_test(NAME lichtfeld.collision_archive COMMAND lichtfeld_visualizer_tests --gtest_filter=ProjectDocumentTest.SaveAsAtomicallyRebindsAndPreservesUnloadedPayloads:ProjectDocumentTest.PhaseAFailureInjectionLeavesEveryLiveSceneByteUntouched:ProjectDocumentTest.TrainingAutosaveRecoveryPreservesInactiveParameterSlotsAndBindings:VisualizerImplResetTest.ScratchDirectoryFallsBackToRootWhenProjectLocationUnusable)
    set_tests_properties(lichtfeld.collision_archive PROPERTIES LABELS "cpu;collision" TIMEOUT 120 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(lichtfeld.collision_jobs_gpu PROPERTIES LABELS "gpu;collision" TIMEOUT 120
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    if(NOT BUILD_TESTS AND NOT BUILD_FORMAT_TESTS AND NOT BUILD_UNICODE_TEST_ONLY AND NOT BUILD_PANEL_LAYOUT_TESTS AND NOT BUILD_TEMPORAL_CPU_TESTS)
        return()
    endif()
endif()

if(BUILD_PANEL_LAYOUT_TESTS)
    add_executable(panel_layout_tests EXCLUDE_FROM_ALL

        test_panel_registry_animation_demand.cpp
        test_panel_registry_default_closed.cpp
    )
    target_compile_features(panel_layout_tests PRIVATE cxx_std_23)
    target_include_directories(panel_layout_tests PRIVATE
        $<TARGET_PROPERTY:${PROJECT_NAME},INCLUDE_DIRECTORIES>
    )
    target_link_libraries(panel_layout_tests PRIVATE
        $<TARGET_PROPERTY:${PROJECT_NAME},LINK_LIBRARIES>
        GTest::gtest_main
    )
    if(NOT BUILD_TESTS AND NOT BUILD_FORMAT_TESTS AND NOT BUILD_UNICODE_TEST_ONLY AND NOT BUILD_TEMPORAL_CPU_TESTS AND NOT BUILD_VISUALIZER_TESTS)
        return()
    endif()
endif()

# =============================================================================
# Unicode Path Test Only (no CUDA runtime linkage)
# =============================================================================
if(BUILD_UNICODE_TEST_ONLY)
    # Standalone Unicode path tests without CUDA runtime linkage
    add_executable(unicode_path_test
        test_unicode_paths_windows.cpp
        test_native_filesystem.cpp
        test_project_save_as_paths.cpp
        test_weight_file_paths.cpp
        test_crash_handler.cpp
        test_converter_output_paths.cpp
        test_converter_overwrite.cpp
        licht_format_test_runtime.cpp
        ${CMAKE_SOURCE_DIR}/src/app/converter_output_paths.cpp
        ${CMAKE_SOURCE_DIR}/src/app/converter_overwrite.cpp
        ${CMAKE_SOURCE_DIR}/src/core/crash_handler.cpp
        ${CMAKE_SOURCE_DIR}/src/core/error.cpp
        ${CMAKE_SOURCE_DIR}/src/core/environment.cpp
        ${CMAKE_SOURCE_DIR}/src/core/failure_report.cpp
        ${CMAKE_SOURCE_DIR}/src/core/mapped_file.cpp
        ${CMAKE_SOURCE_DIR}/src/core/memory_domain.cpp
        ${CMAKE_SOURCE_DIR}/src/core/nn/weight_file_open.cpp
        ${CMAKE_SOURCE_DIR}/src/core/user_paths.cpp
        ${CMAKE_SOURCE_DIR}/src/core/atomic_file.cpp
        ${CMAKE_SOURCE_DIR}/src/io/stb_image_impl.cpp
    )

    target_include_directories(unicode_path_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/io/include
        ${CMAKE_SOURCE_DIR}/external
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(unicode_path_test PRIVATE
        GTest::gtest
        GTest::gtest_main
        nlohmann_json::nlohmann_json
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
        target_link_libraries(unicode_path_test PRIVATE stdc++exp)
    endif()
    target_compile_definitions(unicode_path_test PRIVATE
        LFS_STATIC_BUILD
        LFS_UNICODE_TEST_STANDALONE)

    # C++23 standard
    set_target_properties(unicode_path_test PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
    )
    if(COMMAND lfs_enable_fast_cxx_compile)
        lfs_enable_fast_cxx_compile(unicode_path_test)
    endif()

    # Enable Windows long path support (paths > MAX_PATH/260 chars)
    if(WIN32 AND MSVC)
        # Add manifest file as a source to enable long path awareness
        target_sources(unicode_path_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/unicode_path_test.manifest
        )
    endif()

    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(unicode_path_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES TIMEOUT 30
    )

    message(STATUS "Building standalone Unicode path tests (no CUDA runtime linkage)")
    if(NOT BUILD_TESTS AND NOT BUILD_FORMAT_TESTS AND NOT BUILD_TEMPORAL_CPU_TESTS)
        return()  # Skip the remaining targets when only standalone coverage was requested.
    endif()
endif()

# =============================================================================
# Standalone temporal reconstruction contracts (CPU execution only)
# =============================================================================
if(BUILD_TEMPORAL_CPU_TESTS)
    add_executable(lichtfeld_temporal_cpu_tests
        test_scene_upscaler_plugin_api.cpp
        test_scene_motion_contract.cpp
        test_scene_temporal_coordinator.cpp
        test_scene_temporal_plan.cpp
        test_scene_temporal_resolve.cpp
        test_temporal_frame_tracker.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_coordinator.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_plan.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_resolve.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/scene_temporal_frame_setup.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/temporal_frame_tracker.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/passes/vulkan_scene_plugin_pipeline_contract.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/passes/vulkan_scene_temporal_pipeline_contract.cpp
    )
    target_include_directories(lichtfeld_temporal_cpu_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_SOURCE_DIR}/src/rendering/include
        ${CMAKE_SOURCE_DIR}/src/visualizer
        ${CMAKE_SOURCE_DIR}/src/visualizer/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}/include
        ${Vulkan_INCLUDE_DIRS}
    )
    target_compile_definitions(lichtfeld_temporal_cpu_tests PRIVATE
        LFS_VIS_EXPORTS
    )
    target_link_libraries(lichtfeld_temporal_cpu_tests PRIVATE
        GTest::gtest_main
        glm::glm
    )
    set_target_properties(lichtfeld_temporal_cpu_tests PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
    )
    if(COMMAND lfs_enable_fast_cxx_compile)
        lfs_enable_fast_cxx_compile(lichtfeld_temporal_cpu_tests)
    endif()
    add_test(NAME lichtfeld.temporal.cpu
        COMMAND $<TARGET_FILE:lichtfeld_temporal_cpu_tests> --gtest_color=no
    )
    set_tests_properties(lichtfeld.temporal.cpu PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        LABELS "fast;cpu;temporal"
        TIMEOUT 120
    )
endif()

if(NOT BUILD_FORMAT_TESTS AND NOT BUILD_TESTS)
    message(STATUS "Building standalone temporal reconstruction CPU tests")
    return()
endif()

# Headless SDL boundary regression tests, also built in the CPU-only CI configuration.
# Do not link SDL itself: this isolated executable provides the four SDL state
# queries used by the production coordinate helpers and frame finalization, and
# reports no injected MCP pointer.
add_executable(lichtfeld_sdl_coordinate_tests test_sdl_pixel_coordinates.cpp)
target_compile_features(lichtfeld_sdl_coordinate_tests PRIVATE cxx_std_23)
target_compile_definitions(lichtfeld_sdl_coordinate_tests PRIVATE LFS_STATIC_BUILD)
target_include_directories(lichtfeld_sdl_coordinate_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/visualizer
    ${CMAKE_SOURCE_DIR}/src/core/include
)
target_link_libraries(lichtfeld_sdl_coordinate_tests PRIVATE
    GTest::gtest_main
    SDL3::Headers
    glm::glm
)
add_test(NAME lichtfeld.sdl_coordinates COMMAND lichtfeld_sdl_coordinate_tests)
set_tests_properties(lichtfeld.sdl_coordinates PROPERTIES LABELS "fast;cpu;input" TIMEOUT 30)

# MSVC resolves every object reference before /OPT:REF, while GNU ld can discard
# unreferenced function sections before resolution. Keep the dependency-heavy
# chapter coverage in the non-Windows composition.
set(LICHT_FORMAT_WINDOWS_FILES
    test_property_defaults.cpp
    test_atomic_output.cpp
    test_project_save_as_paths.cpp
    test_native_fs.cpp
    test_json_chapter_dom.cpp
    test_project_chapters.cpp
    test_project_container.cpp
    licht_format_test_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/core/collision_volume.cpp
    ${CMAKE_SOURCE_DIR}/src/core/error.cpp
    ${CMAKE_SOURCE_DIR}/src/core/memory_domain.cpp
    ${CMAKE_SOURCE_DIR}/src/core/optimization_properties.cpp
    ${CMAKE_SOURCE_DIR}/src/core/parameters.cpp
    ${CMAKE_SOURCE_DIR}/src/core/property_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/core/uuid.cpp
    ${CMAKE_SOURCE_DIR}/src/io/json_chapter_dom.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_chapters.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/crc32c.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_file.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_container.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_writer.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_recovery.cpp
)

set(LICHT_FORMAT_NON_WINDOWS_FILES
    ${LICHT_FORMAT_WINDOWS_FILES}
    test_project_inspector.cpp
    test_project_operations.cpp
    ${CMAKE_SOURCE_DIR}/src/core/user_paths.cpp
        ${CMAKE_SOURCE_DIR}/src/core/atomic_file.cpp
    ${CMAKE_SOURCE_DIR}/src/core/environment.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_inspector.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_operations.cpp
    test_selection_chapter.cpp
    test_p8_compatibility.cpp
    licht_format_tensor_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/core/checkpoint_format.cpp
    ${CMAKE_SOURCE_DIR}/src/core/tensor/tensor_serialization.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/geometry_payload.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/project_document.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/selection_chapter.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/session_chapters.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/sfm_observation_chapter.cpp
    ${CMAKE_SOURCE_DIR}/src/io/project/splat_chapter.cpp
    ${CMAKE_SOURCE_DIR}/src/visualizer/gui/layout_state.cpp
)
# test_project_document.cpp is excluded: it links headless recovery, training strategies, and visualizer parameter state.
# test_p5_session_chapters.cpp is excluded: it links the Python editor and TrainingManager runtime.

if(WIN32)
    set(LICHT_FORMAT_TARGET_FILES ${LICHT_FORMAT_WINDOWS_FILES})
else()
    set(LICHT_FORMAT_TARGET_FILES ${LICHT_FORMAT_NON_WINDOWS_FILES})
endif()

add_executable(lichtfeld_format_tests ${LICHT_FORMAT_TARGET_FILES})
target_link_libraries(lichtfeld_format_tests PRIVATE ${FFMPEG_LIBRARIES})
target_include_directories(lichtfeld_format_tests SYSTEM PRIVATE ${FFMPEG_INCLUDE_DIRS})
if(WIN32 AND MSVC)
    target_sources(lichtfeld_format_tests PRIVATE native_filesystem_test.manifest)
endif()
target_include_directories(lichtfeld_format_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/io/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/rendering/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_BINARY_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
if(NOT WIN32)
    target_include_directories(lichtfeld_format_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/diagnostics/include
        ${CMAKE_SOURCE_DIR}/src/training
        ${CMAKE_SOURCE_DIR}/src/training/include
        ${CMAKE_SOURCE_DIR}/src/visualizer
    )
endif()
target_compile_definitions(lichtfeld_format_tests PRIVATE
    LFS_TENSOR_PRIVATE_ACCESS
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)
if(WIN32)
    target_compile_definitions(lichtfeld_format_tests PRIVATE
        LFS_CORE_EXPORTS
        LFS_ERROR_EXPORTS
        LFS_LOGGER_EXPORTS
    )
else()
    target_compile_definitions(lichtfeld_format_tests PRIVATE
        LFS_FORMAT_TEST_TARGET
    )
endif()
target_link_libraries(lichtfeld_format_tests PRIVATE
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    xxHash::xxhash
    zstd::libzstd
    Threads::Threads
)
if(NOT WIN32)
    target_link_libraries(lichtfeld_format_tests PRIVATE OpenSSL::Crypto)
endif()
if(MSVC)
    target_compile_options(lichtfeld_format_tests PRIVATE /Gy)
    target_link_options(lichtfeld_format_tests PRIVATE /OPT:REF)
else()
    target_compile_options(lichtfeld_format_tests PRIVATE
        -ffunction-sections -fdata-sections
    )
    target_link_options(lichtfeld_format_tests PRIVATE
        $<$<PLATFORM_ID:Darwin>:-Wl,-dead_strip>
        $<$<NOT:$<PLATFORM_ID:Darwin>>:-Wl,--no-export-dynamic>
        $<$<NOT:$<PLATFORM_ID:Darwin>>:-Wl,--gc-sections>
    )
endif()
if(WIN32)
    target_link_libraries(lichtfeld_format_tests PRIVATE bcrypt)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_link_libraries(lichtfeld_format_tests PRIVATE stdc++exp)
endif()
set_target_properties(lichtfeld_format_tests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(lichtfeld_format_tests)
endif()
add_test(NAME lichtfeld_format_tests
    COMMAND $<TARGET_FILE:lichtfeld_format_tests> --gtest_color=no
)
set_tests_properties(lichtfeld_format_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast;format"
    TIMEOUT 300
)

add_executable(lichtfeld_frame_tests
    test_frame_demand_ledger.cpp
    test_frustum_depth_coverage.cpp
    test_framerate_controller.cpp
    ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/framerate_controller.cpp)

target_compile_definitions(lichtfeld_frame_tests PRIVATE LFS_STATIC_BUILD)

target_include_directories(lichtfeld_frame_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/visualizer/rendering
    ${CMAKE_SOURCE_DIR}/src/visualizer
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(lichtfeld_frame_tests PRIVATE GTest::gtest_main Threads::Threads)
set_target_properties(lichtfeld_frame_tests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(lichtfeld_frame_tests)
endif()
add_test(NAME lichtfeld_frame_tests
    COMMAND $<TARGET_FILE:lichtfeld_frame_tests> --gtest_color=no
)
set_tests_properties(lichtfeld_frame_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast;frame"
    TIMEOUT 30
)

add_executable(lichtfeld_tls_distrust_tests test_tls_distrust.cpp)
target_include_directories(lichtfeld_tls_distrust_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/core/include)
target_link_libraries(lichtfeld_tls_distrust_tests PRIVATE GTest::gtest_main OpenSSL::SSL OpenSSL::Crypto)
set_target_properties(lichtfeld_tls_distrust_tests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME lichtfeld_tls_distrust_tests
    COMMAND $<TARGET_FILE:lichtfeld_tls_distrust_tests> --gtest_color=no
)
set_tests_properties(lichtfeld_tls_distrust_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast;security"
    TIMEOUT 30
)

if(NOT BUILD_TESTS)
    message(STATUS "Building standalone CPU-only tests; full LibTorch/CUDA suite disabled")
    return()
endif()

# Reuse nanobind headers for Python UI regression tests
if(NOT DEFINED NB_DIR)
    find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.SABIModule)
    find_package(nanobind CONFIG REQUIRED)
endif()

if(COMMAND nanobind_build_library AND NOT TARGET nanobind-static)
    nanobind_build_library(nanobind-static)
endif()

# Find LibTorch for tensor comparison tests
# Set Torch_DIR based on platform
get_filename_component(PROJ_ROOT_DIR "${CMAKE_SOURCE_DIR}" ABSOLUTE)
if(WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/release/libtorch/share/cmake/Torch")
    else()
        set(Torch_DIR "${PROJ_ROOT_DIR}/external/debug/libtorch/share/cmake/Torch")
    endif()
else()
    set(Torch_DIR "${PROJ_ROOT_DIR}/external/libtorch/share/cmake/Torch")
endif()
# The native reference kernel must use the Release tensor-kernel flags it is compared against.
set_source_files_properties(test_tensor_expression_cuda.cu PROPERTIES
    COMPILE_OPTIONS "$<TARGET_PROPERTY:lfs_tensor_kernels,COMPILE_OPTIONS>")

# The shared harness records tensor facade traces, and tensor-library tests may
# include private tensor headers. Other tests do not get the definition, so a stray
# private include fails to compile.
function(lfs_tensor_private_access)
    set(_prefixes
        test_tensor_ tensor_ bench_tensor_ test_allocator_ test_cuda_ test_stale_stream_
        test_zero_stride_ test_strided_reduce_ expression_reference cuda_backend_test test_alloc_counter)
    foreach(_src IN LISTS ARGN)
        get_filename_component(_name "${_src}" NAME)
        foreach(_prefix IN LISTS _prefixes)
            if(_name MATCHES "^${_prefix}")
                set_property(SOURCE "${_src}" APPEND PROPERTY COMPILE_DEFINITIONS LFS_TENSOR_PRIVATE_ACCESS)
                break()
            endif()
        endforeach()
    endforeach()
endfunction()

# Shared by the full suite and the focused POPSpa target.
set(POPSPA_TEST_FILES
    test_pop_score_backends.cpp
    test_popspa_controller.cpp
    test_popspa_regularization.cpp
    test_popspa_compaction.cpp
    test_popspa_pipeline.cpp
    test_popspa_ops.cpp
)

if(NOT LFS_HAS_CUDA)
    set(TEST_SOURCES
        test_main.cpp
        test_tensor_backend_semantics.cpp
        test_error_bus.cpp
        test_tensor_restored.cpp
        test_tensor_image_regions.cpp
        test_nodes_core.cpp
        test_nodes_scene.cpp
        test_nodes_modifier_manager.cpp
        test_splat_simplify.cpp
        test_tiles3d.cpp
        test_ssog_tiles.cpp
        test_node_canvas_interaction.cpp
        test_mcp_node_tools.cpp
        test_mcp_screen_tools.cpp
        test_viewport.cpp
        test_tensor_metal.cpp
        test_camera_priors_portable.cpp
        test_undistort_portable.cpp
        test_pipelined_loader_backends.cpp
        test_viewer_no_cuda.cpp
        test_lod_page_cache.cpp
        test_colmap_text_images.cpp
    )
    if(LFS_TENSOR_VULKAN)
        list(APPEND TEST_SOURCES
            test_tensor_queue_readback.cpp
            test_tensor_vulkan_runtime.cpp
            test_tensor_vulkan_allocation_padding.cpp
            test_tensor_vulkan_adoption.cpp
            test_tensor_vulkan_pointwise.cpp
            test_tensor_vulkan_reduce.cpp
            test_tensor_vulkan_index.cpp
            test_tensor_vulkan_matrix_nn.cpp
            test_tensor_vulkan_random.cpp
            test_tensor_vulkan_buffer.cpp)
    endif()
    if(LFS_BUILD_TRAINER)
        list(APPEND TEST_SOURCES
            test_session_loader_storage.cpp
            test_training_command_masks.cpp
            test_mesh_mask_kernels.cpp
            test_training_ops_parity.cpp
            test_flip.cpp
            test_masked_ssim_metrics.cpp
            test_training_ops_metal_densify.cpp
            test_training_ops_metal_appearance.cpp
            test_training_ops_metal_gsplat.cpp
            test_training_ops_metal_losses.cpp
            test_training_ops_metal_adam_sh.cpp
            test_training_ops_metal_fast.cpp
            test_roi_masks.cpp
            test_roi_polygon_weight_map.cpp
            test_training_snapshot_service.cpp
            test_ppisp_regularizer_portable.cpp
            ${POPSPA_TEST_FILES})
        if(LFS_TENSOR_VULKAN)
            list(APPEND TEST_SOURCES
                test_vulkan_trainer_start.cpp
                test_training_ops_vulkan_gsplat.cpp
                test_tensor_vulkan_training_fault.cpp
                test_training_pair_sort.cpp)
        endif()
    endif()
    if(APPLE)
        list(APPEND TEST_SOURCES test_video_videotoolbox.cpp test_mcp_render_capture_utils.cpp)
    endif()
    add_executable(lichtfeld_tests ${TEST_SOURCES})
    target_compile_definitions(lichtfeld_tests PRIVATE LFS_JXL_FIXTURES="${CMAKE_SOURCE_DIR}/tests/data/jxl")
    target_link_libraries(lichtfeld_tests PRIVATE ${FFMPEG_LIBRARIES})
    target_include_directories(lichtfeld_tests SYSTEM PRIVATE ${FFMPEG_INCLUDE_DIRS})
    target_compile_definitions(lichtfeld_tests PRIVATE PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}" TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
    if(APPLE)
        target_sources(lichtfeld_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/io/stb_image_impl.cpp)
        # lfs_video uses opaque media frames. Fixture-only FFmpeg libraries
        # are linked privately above, preserving package framework qualifiers.
        target_include_directories(lichtfeld_tests SYSTEM PRIVATE
            ${CMAKE_SOURCE_DIR}/external)
        target_link_libraries(lichtfeld_tests PRIVATE lfs_video nlohmann_json::nlohmann_json)
    endif()
    target_compile_features(lichtfeld_tests PRIVATE cxx_std_23)
    target_include_directories(lichtfeld_tests PRIVATE
        ${LFS_CORE_ABI_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/visualizer
        ${CMAKE_SOURCE_DIR}/src/training
        ${CMAKE_SOURCE_DIR}/src/app/include
    )
    lfs_tensor_private_access(${TEST_SOURCES})
    target_sources(lichtfeld_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/app/mcp_node_tools.cpp
        ${CMAKE_SOURCE_DIR}/src/app/mcp_node_resources.cpp
        ${CMAKE_SOURCE_DIR}/src/app/mcp_node_editor_tools.cpp
        ${CMAKE_SOURCE_DIR}/src/app/mcp_screen_tools.cpp
        ${CMAKE_SOURCE_DIR}/src/app/mcp_pointer_tool.cpp)
    target_link_libraries(lichtfeld_tests PRIVATE lfs_mcp)
    target_link_libraries(lichtfeld_tests PRIVATE
        lfs_rendering_tensor
        lfs_core
        lfs_io
        lfs_visualizer
        GTest::gtest
    )
    if(LFS_TENSOR_VULKAN)
        target_link_libraries(lichtfeld_tests PRIVATE Vulkan::Vulkan)
    endif()
    if(LFS_BUILD_TRAINER)
        target_link_libraries(lichtfeld_tests PRIVATE lfs_training lfs_roi_masks)
        if(APPLE)
            # lfs_training resolves CPython from the host process, which the app
            # embeds; the test binary must load it whole.
            find_package(Python REQUIRED COMPONENTS Development.Embed)
            target_link_options(lichtfeld_tests PRIVATE
                "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
            target_link_libraries(lichtfeld_tests PRIVATE lfs_python_embed)
        endif()
    endif()
    if(LFS_TENSOR_VULKAN)
        add_test(NAME lichtfeld_tests COMMAND lichtfeld_tests --tensor-backend=vulkan)
    else()
        add_test(NAME lichtfeld_tests COMMAND lichtfeld_tests --tensor-backend=metal)
    endif()
    if(APPLE)
        # This timeout covers the entire suite, including throughput tests
        # that take longer on hosted macOS paravirtual GPUs.
        set_tests_properties(lichtfeld_tests PROPERTIES TIMEOUT 600)
    else()
        set_tests_properties(lichtfeld_tests PROPERTIES TIMEOUT 300)
    endif()

    # Metal vs Vulkan tensor benchmark, built on demand and not registered with ctest.
    if(LFS_TENSOR_METAL AND LFS_TENSOR_VULKAN)
        add_executable(bench_tensor_metal_vulkan EXCLUDE_FROM_ALL bench_tensor_metal_vulkan.cpp)
        target_compile_features(bench_tensor_metal_vulkan PRIVATE cxx_std_23)
        target_include_directories(bench_tensor_metal_vulkan PRIVATE ${LFS_CORE_ABI_INCLUDE_DIR})
        target_link_libraries(bench_tensor_metal_vulkan PRIVATE lfs_core)
    endif()
    return()
elseif(NOT LFS_BUILD_TRAINER)
    include(${CMAKE_CURRENT_SOURCE_DIR}/ViewerTests.cmake)
    return()
endif()

find_package(Torch REQUIRED)
message(STATUS "Tests: LibTorch ${Torch_VERSION} found for tensor comparison tests")

# Regular test sources (unit tests, integration tests)

set(TEST_FILES
    licht_test_support.cpp
    test_nodes_core.cpp
    test_nodes_scene.cpp
    test_nodes_modifier_manager.cpp
    test_node_canvas_interaction.cpp
    test_assert_hardening.cpp
    test_cuda_error.cpp
    test_exportable_storage.cpp
    test_session_loader_storage.cpp
    test_shareable_allocation_limit.cpp
    test_ply_q16_streamed_encode.cpp
    test_vram_leak_regression.cpp
    test_cuda_error_typed.cpp
    test_cuda_launch_check.cpp
    test_tensor_launch_check.cpp
    test_device_fault.cpp
    device_fault_cuda_utils.cu
    test_selection_ops.cpp
    test_tensor_point_ops.cpp
    test_tensor_restored.cpp
    test_tensor_image_regions.cpp
    test_tensor_expression.cpp
    test_tensor_expression_cuda.cu
    test_tensor_warp_reduce.cpp
    test_tensor_warp_reduce_cuda.cu
    test_splat_codec_edits.cpp
    test_tensor_splat_affine.cpp
    test_tensor_sh_codec.cpp
    test_tensor_image_ops.cpp
    test_selection_cursor.cpp
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_memory.cpp
    test_allocator_policy.cpp
    test_alloc_counter.cpp
    test_joint_bounds_grow_only.cpp
    test_steady_alloc_invariant.cpp
    test_allocator_hygiene.cpp
    test_pinned_stream_teardown.cpp
    test_cuda_pool_stream_teardown.cpp
    test_stale_stream_teardown.cpp
    test_fastgs_sort_buffers.cpp
    test_warp_cull_blend.cpp
    test_warp_cull_bwd.cpp
    test_thin_structure.cpp
    test_mask_thin_structure_guidance.cpp
    test_mask_edge_guidance.cpp
    test_blob_seeding.cpp
    test_rendered_structure.cpp
    test_gradient_residual.cpp
    test_fused_reg_loss.cpp
    test_opacity_gradients.cpp
    test_fused_bg_blend.cpp
    test_photometric_hygiene.cpp
    test_photometric_ops.cpp
    test_training_ops_parity.cpp
    test_training_ops_vulkan_gsplat.cpp
    test_tensor_vulkan_training_fault.cpp
    test_training_pair_sort.cpp
    test_gsplat_ops.cpp
    test_fast_ops.cpp
    test_session_ops.cpp
    fast_raster_test_helpers.cpp
    test_training_image_ops.cpp
    test_shared_image_ops.cpp
    test_adam_ops.cpp
    test_lpips_ops.cpp
    test_ppisp_ops.cpp
    test_geometry_ops.cpp
    test_bilateral_ops.cpp
    test_mcmc_ops.cpp
    test_mrnf_ops.cpp
    test_morton_ops.cpp
    test_mask_ops.cpp
    test_extra_loss_ops.cpp
    test_sh_ops.cpp
    test_refine_ops.cpp
    test_persistent_masks.cpp
    test_fused_noise_injection.cpp
    test_densification_info_zero.cpp
    test_densify_events_4x.cpp
    test_median_selection_perf.cu
    test_training_state_ledger.cpp
    test_vram_ledger_model.cpp
    test_vram_owner_timeline.cpp
    test_gpu_memory_query.cpp
    test_host_metrics.cpp
    test_joint_adam_codec.cpp
    test_sh_value_codec.cpp
    test_sh_value_storage.cpp
    test_morton_reorder.cpp
    test_transform_bake_q16.cpp
    test_splat_affine_transform.cpp
    test_splat_sh_affine.cpp
    test_splat_data_clone.cpp
    test_splat_data_mirror.cpp
    test_tensor_reduction.cpp
    test_tensor_reduction_alignment.cpp
    test_tensor_integer_and_reduce_policy.cpp
    test_selection_attribute_ops.cpp
    test_selection_screen_ops.cpp
    test_selection_tensor_projection.cpp
    test_export_band_pack.cpp
    test_viewer_appearance.cpp
    test_point_cloud_merge.cpp
    test_display_tensors.cpp
    test_viewer_gpu_memory.cpp
    test_tensor_views.cpp
    test_tensor_advanced.cpp
    test_tensor_async_consts.cpp
    test_tensor_stress.cpp
    test_tensor_matrix.cpp
    test_slice_mm_bug.cpp
    test_tensor_random.cpp
    test_tensor_broadcast.cpp
    test_zero_stride_expand.cpp
    test_strided_reduce_w2.cpp
    test_tensor_masking.cpp
    test_strided_tensor_ops.cpp
    test_tensor_lazy_gather.cpp
    test_tensor_ordering_distance.cpp
    test_tensor_kmeans_workflow.cpp
    test_tensor_masked_select_rows.cpp
    test_tensor_bitwise.cpp
    test_tensor_accessor.cpp
    test_tensor_random_advanced.cpp
    test_tensor_indexing_advanced.cpp
    test_tensor_clamping.cpp
    test_tensor_cumsum_convenience.cpp
    test_tensor_conversions_shape.cpp
    test_tensor_reserve_inplace_cat.cpp
    test_mcmc.cpp
    test_expression_templates.cpp
    test_lfs_scheduler.cpp
    test_uncapped_compaction.cpp
    test_mrnf_strategy.cpp
    test_compact_splats_peak.cpp
    test_adam_capacity_invariant.cpp
    test_dual_rep_optimizer.cpp
    test_gut_shN_joint_adam.cpp
    test_gut_undistort_parity.cpp
    ${POPSPA_TEST_FILES}
    test_mcmc_tensor_ops.cpp
    test_training_cropbox_mask.cpp
    test_training_parameters.cpp
    test_property_defaults.cpp
    test_tensor_index_select_uint8.cpp
    test_tensor_cuda_byte_image.cpp
    test_image_tensor.cpp
    test_tensor_uint8_inversion.cpp
    test_tensor_uint8_masking.cpp
    test_tensor_serialization.cpp
    test_tensor_cat_reduction_bug.cpp
    test_percentile_bounds.cpp
    test_tensor_inplace_capacity.cpp
    test_sog_format.cpp
    test_ssog_format.cpp
    test_ssog_tiles.cpp
    test_tiles3d.cpp
    test_splat_decimate.cpp
    test_ply_error_taxonomy.cpp
    test_ply_io_throughput.cpp
    test_gsplat_rasterizer.cpp
    test_spz_format.cpp
    test_provenance.cpp
    test_converter_usd.cpp
    test_converter_licht.cpp
    test_converter_output_paths.cpp
    test_converter_overwrite.cpp
    test_usd_format.cpp
    test_nurec_usdz_format.cpp
    test_unicode_paths_windows.cpp
    test_native_filesystem.cpp
    test_interleaved_slice_copy.cpp
    test_densification_tensor_ops.cpp
    test_tensor_zero_dimension.cpp
    test_mask_loss.cpp
    test_mask_preprocess_fusion.cpp
    test_mask_memory_alignment.cpp
    test_mask_tiff_consistency.cpp
    test_roi_weight_map.cpp
    test_mesh_mask_kernels.cpp
    test_flip.cpp
    test_compare_images.cpp
    test_roi_weighted_losses.cpp
    test_checkpoint_resume.cpp
    test_evaluation_split.cpp
    test_component_config_serialization.cpp
    test_training_snapshot_service.cpp
    test_atomic_output.cpp
    test_geometry_payload.cpp
    test_selection_chapter.cpp
    test_project_document.cpp
    test_project_thumbnail_sources.cpp
    test_project_training_copy.cpp
    test_gallery_scene_publication.cpp
    test_project_lifecycle.cpp
    test_p8_compatibility.cpp
    test_job_registry.cpp
    test_p5_session_chapters.cpp
    test_training_terminal_state.cpp
    test_training_camera_errors.cpp
    test_training_controller_freeze.cpp
    test_headless_run_coordinator.cpp
    test_argument_parser.cpp
    test_licht_command.cpp
    test_user_paths.cpp
    test_window_state.cpp
    test_crash_handler.cpp
    test_gpu_teardown_order.cpp
    test_logger.cpp
    test_exif.cpp
    test_nanoflann_build.cpp
    test_initial_scales.cpp
    test_system_info.cpp
    test_parameter_manager.cpp
    test_training_manager_events.cpp
    test_mcp_protocol.cpp
    test_mcp_training_state.cpp
    test_training_command_masks.cpp
    test_mcp_app_utils.cpp
    test_mcp_operator_schema.cpp
    test_mcp_shared_scene_tools.cpp
    test_mcp_sequencer_tools.cpp
    test_mcp_screen_tools.cpp
    test_mcp_node_tools.cpp
    test_mcp_render_capture_utils.cpp
    test_mcp_shader_tools.cpp
    test_tcp_publisher.cpp
    test_view_info_json.cpp
    test_video_reconstruction.cpp
    test_video_export_utils.cpp
    test_viewport_point_cloud_export.cpp
    test_video_export_frame_schedule.cpp
    test_video_resolution.cpp
    test_video_frame_extractor_naming.cpp
    test_video_cpu_rgb_conversion.cpp
    test_video_cuda_frame_handoff.cpp
    test_video_hdr_logic.cpp
    test_visualizer_post_work.cpp
    test_file_association.cpp
    test_panel_registry_animation_demand.cpp
    test_panel_registry_default_closed.cpp

    test_number_format.cpp
    test_reactive_store.cpp
    test_video_color_convert.cpp
    test_video_reframe.cpp
    test_python_integration.cpp
    test_python_lsp_client.cpp
    test_uv_runner.cpp
    test_nan_inf_gpu_check.cpp
    test_mcmc_nan_fix.cpp
    test_stride_bugs.cpp
    test_strided_fill.cpp
    test_expanded_tensor_ops.cpp
    test_bool_any_all.cpp
    test_fused_l1_ssim.cpp
    test_masked_ssim_window.cpp
    test_densify_error_map.cpp
    test_screen_share.cpp
    test_loss_workspace_union.cpp
    test_image_kernels.cpp
    test_add_new_gs_tensor_ops.cpp
    test_cpu_dtype_conversions.cpp
    test_gradient_accumulation.cpp
    test_image_resize_parallel.cpp
    test_tensor_slicing_safety.cpp
    test_tensor_deferred_d1_copy_registry.cpp
    test_tensor_deferred_d3_pin_operands.cpp
    test_tensor_lazy_ir.cpp
    test_tensor_dispatch_6a.cpp
    test_tensor_kernels_6c.cpp
    test_tensor_kernels_6c_elem.cpp
    test_tensor_lazy_stateful_ops.cpp
    test_fastgs_kernels.cpp
    test_color_variance.cpp
    fastgs_visibility_readback_cuda.cu
    test_fastgs_fuzz.cpp
    test_fastgs_tile_culling_boundary.cpp
    fastgs_tile_culling_boundary_cuda.cu
    test_fastgs_forward_mask.cpp
    test_raster_evaluation.cu
    test_raster_reduction.cu
    test_raster_tile_keys.cu
    test_sh_swizzle_layout.cpp
    test_shn_bda_index_math.cpp
    test_python_io.cpp
    test_extended_unary_ops_vs_torch.cpp
    test_scene_validity.cpp
    test_scene_bounds_cache.cpp
    test_training_scene_init_concurrency.cpp
    test_training_init_centering.cpp
    test_scene_consolidation.cpp
    test_scene_combining_backends.cpp
    test_scene_snapshot_backend.cpp
    test_scene_graph_fuzz.cpp
    test_uuid.cpp
    test_render_settings_defaults.cpp
    test_portal_render_profile.cpp
    test_scene_upscaler_registry.cpp
    test_scene_upscaler_plugin_api.cpp
    test_scene_motion_contract.cpp
    test_scene_temporal_coordinator.cpp
    test_scene_temporal_plan.cpp
    test_scene_temporal_resolve.cpp
    test_temporal_frame_tracker.cpp
    test_selection_rasterization_ops.cpp
    test_selection_screen_window.cpp
    test_rendering_refactor_services.cpp
    test_view_render_state.cpp
    test_rml_pointer_routing.cpp
    test_crop_polygon.cpp
    test_higs_deterministic_export.cpp
    test_point_cloud_raster_orientation.cpp
    test_point_cloud_vulkan_projection.cpp
    test_point_cloud_selection_overlay.cpp
    test_viewport_grid.cpp
    test_theme_registry.cpp
    test_viewport.cpp
    test_navigation_up_axis.cpp
    test_analog_navigation.cpp
    test_gamepad_service.cpp
    test_gamepad_navigation.cpp
    test_gamepad_sequencer.cpp
    test_camera_browse_repeat.cpp
    test_input_controller_focus.cpp
    test_input_key_repeat.cpp
    test_input_layout_chords.cpp
    test_keymap_layout_contract.cpp
    test_keymap_profile_io.cpp
    test_keymap_upgrade.cpp
    test_keymap_default_contract.cpp
    test_ordered_input_dispatch.cpp
    test_python_editor_input.cpp
    test_input_controller_dataset_load.cpp
    test_selection_service_interactions.cpp
    test_selection_group_mask.cpp
    test_scene_graph_identity.cpp
    test_undo_history.cpp
    test_selection_command_dispatch.cpp
    test_selection_operator_modal.cpp
    test_operator_registry_props.cpp
    test_fastgs_analytical_gradients.cpp
    test_normal_prior_decode.cpp
    test_normal_auto_generate.cpp
    test_lens_priors.cpp
    test_metrics_evaluator.cpp
    test_metrics_eval_mask.cpp
    test_masked_ssim_metrics.cpp
    test_background_image.cpp
    test_nvcodec_jpeg2k_16bit.cpp
    test_nvcodec_jpeg2k_chw_layout.cpp
    test_nvcodec_sentinel_validation.cpp
    test_compressed_blob_tier.cpp
    test_pipelined_loader.cpp
    test_pipelined_loader_backends.cpp
    test_pipelined_loader_ledger.cpp
    test_pipelined_loader_sidecar_jpeg2k.cpp
    test_image_sharpen.cpp
    test_operator_return_value.cpp
    test_sdl_rml_key_mapping.cpp
    test_rml_path_utils.cpp
    test_rml_system_interface.cpp
    test_rml_drag_payload.cpp
    test_context_menu_placement.cpp
    test_rml_static_style_boundaries.cpp
    test_node_canvas_widgets.cpp
    test_panel_layout_render_demand.cpp
    test_preferences_hint_layout.cpp
    test_preferences_tooltips.cpp
    test_viewport_depth_layout.cpp
    test_viewport_select.cpp
    test_sequencer_regressions.cpp
    test_sequencer_export_settings.cpp
    test_sequencer_export_workflow.cpp
    test_timeline_save_budget.cpp
    test_tensor_storage_version.cpp
    test_tensor_backend_selection.cpp
    test_tensor_backend_selftest.cpp
    test_tensor_seam_regressions.cpp
    test_tensor_backend_facade.cpp
    test_tensor_backend_facade_c.cpp
    test_modal_form_behavior.cpp
    test_modal_registry_regression.cpp
    test_training_notifications.cpp
    test_camera_training_notice.cpp
    test_tensor_nn_ops.cpp
    test_nn_ops.cpp
    test_weight_file_paths.cpp
    test_moge2.cpp
    test_romav1.cpp
    test_sam2.cpp
    test_lpips.cpp
    test_video_masking.cpp
    test_ppisp_regularization.cpp
    test_ppisp_sidecar.cpp
    test_ppisp_checkpoint_roundtrip.cpp
    test_bilateral_grid_projection.cpp
    test_bilateral_grid_exposure_chroma.cpp
    test_appearance_in_place.cpp
    test_ppisp_projection.cpp
    test_ppisp_cuda_vs_torch.cpp
    test_ppisp_banded_region.cpp
    test_ppisp_exposure_seed.cpp
    test_metrics_evaluator_appearance.cpp
    test_export_env_composite.cpp
    test_image_io.cpp
    test_project_thumbnail_image.cpp
    test_mask_resized_cache.cpp
    test_mask_rgb_cache.cpp
    test_mask_cache_endpoints.cpp
    test_mask_color_consistency.cpp
    test_camera_mask_threshold.cpp
    test_camera_thumbnail_policy.cpp
    test_apriltag_scene_scale.cpp
    test_undistort.cpp
    test_undistort_portable.cpp
    gut_camera_model_cuda.cu
    test_viewport_region_utils.cpp
    test_viewport_interop_decision.cpp
    test_mesh_data.cpp
    test_mesh_offscreen_renderer.cpp
    test_align_transform.cpp
    test_scale_distance.cpp
    test_mesh_loader.cpp
    test_colmap_image_layout.cpp
    test_missing_dataset_images.cpp
    test_colmap_points3d_text.cpp
    test_colmap_binary_error_taxonomy.cpp
    test_colmap_binary_parallel_parse.cpp
    test_colmap_camera_import.cpp
    test_colmap_text_images.cpp
    test_splat_simplify.cpp
    test_rml_text_input_handler.cpp
    test_rml_text_input_native_keys.cpp
    test_x11_input_modifier.cpp
    test_scene_graph_dnd.cpp
    test_rml_im_mode_layout.cpp
    test_rml_python_listener_lifetime.cpp
    test_rendering_environment_select.cpp
    test_summary_labels.cpp
    test_status_bar_fit.cpp
    test_gallery_file_panel_layout.cpp
    test_histogram_layout.cpp
    test_setting_row_layout.cpp
    test_rmlui_preview_decode.cpp
    test_training_progress_publisher.cpp
    test_startup_overlay_geometry.cpp
    test_vram_hud_geometry.cpp
    test_vram_hud_header_click.cpp
    test_ui_visibility_transition.cpp
    test_menu_bar_title.cpp
    test_guide_segment_clipping.cpp
    test_status_bar_mining.cpp
    test_status_bar_tessle.cpp
    test_properties_tab_rail.cpp
    test_rotated_sh_correctness.cpp
    test_splat_raw_payload.cpp
    test_tensor_stream.cpp
    test_tensor_queue_readback.cpp
    test_vulkan_trainer_start.cpp
    test_tensor_completion.cpp
    test_tensor_d4_from_blob_stream.cpp
    tensor_hardening_cuda_utils.cu
    test_cuda_event_pool.cpp
    test_tensor_multistream.cpp
    test_tensor_discovery.cpp
    test_tensor_discovery_fuzz.cpp
    test_mcmc_multinomial.cpp
    test_arena_metrics_contention.cpp
    test_memory_arena_shutdown.cpp
    test_tensor_memory_planner.cpp
    test_vram_profiler_metrics.cpp
    test_memory_pressure.cpp
    test_error_core.cpp
    test_error_reporter.cpp
    test_error_bus.cpp
    test_run_guarded.cpp
    test_tcp_responder.cpp
    test_error_envelope.cpp
    test_plugin_localization.cpp
)
if(LFS_TENSOR_VULKAN)
    list(APPEND TEST_FILES
        test_tensor_backend_validators.cpp
        test_tensor_vulkan_runtime.cpp
        test_tensor_vulkan_allocation_padding.cpp
        test_tensor_vulkan_adoption.cpp
        test_tensor_vulkan_pointwise.cpp
        test_tensor_vulkan_reduce.cpp
        test_tensor_vulkan_index.cpp
        test_tensor_vulkan_matrix_nn.cpp
        test_tensor_vulkan_random.cpp
        test_tensor_vulkan_buffer.cpp
        test_tensor_vulkan_cuda_bridge.cpp
        test_tensor_backend_parity.cpp
        test_tensor_backend_semantics.cpp
        test_tensor_restored.cpp
        test_tensor_image_regions.cpp
        test_ply_vulkan_q16_bands.cpp
        test_viewer_vulkan_load.cpp
        test_headless_vulkan_device_selection.cpp
        test_viewer_no_cuda.cpp
        test_selection_mask_ops.cpp
        test_sh_quant_tensor_program.cpp
    )
endif()

# Preference tests are part of the focused CPU visualizer suite when that
# target is enabled (notably on Windows CI), and remain in the full suite on
# configurations that do not build the focused target.
if(NOT BUILD_VISUALIZER_TESTS)
    list(APPEND TEST_FILES test_preferences_migration.cpp)
endif()
list(APPEND TEST_FILES test_vksplat_input_packer.cpp)
list(APPEND TEST_FILES test_viewer_sh_zerocopy_triad.cpp)
# Phase 7A: bounded-wait machine + SubmissionState + QW-6 (GPU-free).
list(APPEND TEST_FILES test_vulkan_wait_bounded.cpp)
list(APPEND TEST_FILES test_submission_state_transitions.cpp)
list(APPEND TEST_FILES test_vksplat_failed_submit_no_publish.cpp)
# Epic #1496: computed barriers (GPU-free planner/tracker suites).
list(APPEND TEST_FILES test_vksplat_barrier_planner.cpp)
list(APPEND TEST_FILES test_vksplat_tagged_dispatch.cpp)
list(APPEND TEST_FILES test_vksplat_buffer_retire.cpp)
list(APPEND TEST_FILES test_vulkan_image_tracker.cpp)
list(APPEND TEST_FILES test_vulkan_gpu_census.cpp)
# Epic #1497: output image pool (GPU-free host bookkeeping).
list(APPEND TEST_FILES test_output_image_pool.cpp)
# Epic #1568 / #1567: output slot ring (GPU-free host bookkeeping).
list(APPEND TEST_FILES test_output_slot_ring.cpp)
list(APPEND TEST_FILES test_vr_math.cpp)
list(APPEND TEST_FILES test_split_output_lifetime.cpp)
# Epic #1577 / #1574: readback ticket ring (GPU-free host bookkeeping).
list(APPEND TEST_FILES test_readback_ticket_ring.cpp)
# Epic #1568 / #1566: payload-generic GpuResourcePool + subrect fit helper.
list(APPEND TEST_FILES test_gpu_resource_pool.cpp)
list(APPEND TEST_FILES test_viewport_scratch_bucket.cpp)
list(APPEND TEST_FILES test_lod_page_cache.cpp)
list(APPEND TEST_FILES test_vulkan_lod_upload_engine.cpp)
list(APPEND TEST_FILES test_splat_sh_tensor_layout.cpp)
list(APPEND TEST_FILES test_tensor_device_runtime.cpp)
list(APPEND TEST_FILES test_octree_lod.cpp)
list(APPEND TEST_FILES test_ply_to_rad_lod.cpp)
list(APPEND TEST_FILES test_png_parallel_writer.cpp)
list(APPEND TEST_FILES test_rad_gpu_encode.cpp)
list(APPEND TEST_FILES test_rad_meta_sidecar.cpp)
list(APPEND TEST_FILES test_rad_spark_paging.cpp)
list(APPEND TEST_FILES test_roi_masks.cpp test_roi_polygon_weight_map.cpp)
list(APPEND TEST_FILES test_batch_trainer.cpp)

# Keep the source manifest strict: a stale entry should fail configuration
# instead of silently dropping coverage from the executable.
set(TEST_SOURCES test_main.cpp ${TEST_FILES})

# Also exercise the portable backend references in Linux CUDA/Vulkan builds.
# Their _Float16 CPU oracle requires GCC or Clang.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND TEST_SOURCES
        test_training_ops_metal_appearance.cpp
        test_training_ops_metal_adam_sh.cpp
        test_training_ops_metal_densify.cpp
        test_training_ops_metal_fast.cpp
        test_training_ops_metal_gsplat.cpp)
endif()

# PyModalRegistry and its native-modal dispatch path for regression tests.
list(APPEND TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_error.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_modal_registry.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_operator_properties.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_ui_modals.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/notification_bridge.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/rml_im_mode_layout.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_sequencer_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_screen_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_node_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_node_resources.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_node_editor_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_pointer_tool.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_shader_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_editor_tools.cpp
    ${CMAKE_SOURCE_DIR}/src/app/mcp_operator_schema.cpp
    ${CMAKE_SOURCE_DIR}/src/app/converter.cpp
    ${CMAKE_SOURCE_DIR}/src/app/licht_command.cpp
    ${CMAKE_SOURCE_DIR}/src/app/converter_output_paths.cpp
    ${CMAKE_SOURCE_DIR}/src/app/converter_overwrite.cpp
    ${CMAKE_SOURCE_DIR}/src/app/batch_trainer.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_params.cpp
)

# Create test executable
add_executable(lichtfeld_tests ${TEST_SOURCES})
target_compile_definitions(lichtfeld_tests PRIVATE LFS_JXL_FIXTURES="${CMAKE_SOURCE_DIR}/tests/data/jxl")
target_link_libraries(lichtfeld_tests PRIVATE ${FFMPEG_LIBRARIES})
target_include_directories(lichtfeld_tests SYSTEM PRIVATE ${FFMPEG_INCLUDE_DIRS})

# test_assert_hardening.cpp includes the generated ABI header directly. Ensure
# the build-time tripwire has run before this target starts compiling.
add_dependencies(lichtfeld_tests lfs_core_abi_stamp)

# Python integration tests import the in-tree extension at runtime.
add_dependencies(lichtfeld_tests lfs_py)

# Set CUDA architectures
set_target_properties(lichtfeld_tests PROPERTIES
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(lichtfeld_tests PRIVATE
    ${LFS_CORE_ABI_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src/training   # For training headers
    ${CMAKE_SOURCE_DIR}/src            # For module headers
    ${CMAKE_SOURCE_DIR}/src/preprocessing/include
    ${CMAKE_SOURCE_DIR}/src/app/include
    ${CMAKE_SOURCE_DIR}/src/visualizer # For visualizer-relative headers (e.g. gui/ui_widgets.hpp)
    ${CMAKE_SOURCE_DIR}/external/spz   # SPZ library headers
    ${CMAKE_SOURCE_DIR}/external       # Vendored stb headers
    ${NB_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Define TEST_DATA_DIR and PROJECT_ROOT_PATH for tests
target_compile_definitions(lichtfeld_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)
if(LFS_TENSOR_VULKAN)
    target_compile_definitions(lichtfeld_tests PRIVATE LFS_TEST_TENSOR_VULKAN=1)
endif()

# Large test assets are hosted in the public Cloudflare R2 bucket
# 'lichtfeld-test-assets' and fetched at configure time. A failed download only
# warns: tests guard on the file's existence and GTEST_SKIP when it is missing.
set(LFS_SIMPLIFY_TEST_PLY "${CMAKE_SOURCE_DIR}/data/kerstbol-isolated-rotated_137502.ply")
set(LFS_SIMPLIFY_TEST_PLY_SHA256 "7fc85d1b0f4b7ab26acaa988b397b41b4ecf83f90e78cfb91b65ed474cb671fc")
set(LFS_SIMPLIFY_TEST_PLY_OK FALSE)
if(EXISTS "${LFS_SIMPLIFY_TEST_PLY}")
    file(SHA256 "${LFS_SIMPLIFY_TEST_PLY}" LFS_SIMPLIFY_TEST_PLY_HASH)
    if(LFS_SIMPLIFY_TEST_PLY_HASH STREQUAL LFS_SIMPLIFY_TEST_PLY_SHA256)
        set(LFS_SIMPLIFY_TEST_PLY_OK TRUE)
    endif()
endif()
if(NOT LFS_SIMPLIFY_TEST_PLY_OK)
    message(STATUS "Downloading simplify regression asset kerstbol-isolated-rotated_137502.ply (33 MB)")
    file(DOWNLOAD
        "https://pub-c3fee24842994e5d866dbd9cc41c3f66.r2.dev/kerstbol-isolated-rotated_137502.ply"
        "${LFS_SIMPLIFY_TEST_PLY}.tmp"
        INACTIVITY_TIMEOUT 60
        TIMEOUT 300
        STATUS LFS_SIMPLIFY_TEST_PLY_STATUS
    )
    list(GET LFS_SIMPLIFY_TEST_PLY_STATUS 0 LFS_SIMPLIFY_TEST_PLY_RC)
    list(GET LFS_SIMPLIFY_TEST_PLY_STATUS 1 LFS_SIMPLIFY_TEST_PLY_ERR)
    if(LFS_SIMPLIFY_TEST_PLY_RC EQUAL 0)
        file(SHA256 "${LFS_SIMPLIFY_TEST_PLY}.tmp" LFS_SIMPLIFY_TEST_PLY_HASH)
        if(LFS_SIMPLIFY_TEST_PLY_HASH STREQUAL LFS_SIMPLIFY_TEST_PLY_SHA256)
            file(RENAME "${LFS_SIMPLIFY_TEST_PLY}.tmp" "${LFS_SIMPLIFY_TEST_PLY}")
            set(LFS_SIMPLIFY_TEST_PLY_OK TRUE)
        else()
            set(LFS_SIMPLIFY_TEST_PLY_ERR "SHA256 mismatch: got ${LFS_SIMPLIFY_TEST_PLY_HASH}")
        endif()
    endif()
    if(NOT LFS_SIMPLIFY_TEST_PLY_OK)
        file(REMOVE "${LFS_SIMPLIFY_TEST_PLY}.tmp")
        message(WARNING "Failed to download simplify regression asset "
            "kerstbol-isolated-rotated_137502.ply (${LFS_SIMPLIFY_TEST_PLY_ERR}); "
            "SplatSimplify.RealPlySimplifyDoesNotUndershootRequestedTarget will be skipped.")
    endif()
endif()
target_compile_definitions(lichtfeld_tests PRIVATE LFS_SIMPLIFY_TEST_PLY="${LFS_SIMPLIFY_TEST_PLY}")

if(CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(LFS_TEST_INTERNAL_LINK_LIBS
        "$<LINK_GROUP:RESCAN,lfs_mcp,lfs_tcp,lfs_visualizer,lfs_sequencer,lfs_rendering,lfs_training,lfs_core,lfs_geometry,lfs_io,lfs_video,fastlfs_backend,spz_lib,lfs_python_utils>"
    )
else()
    set(LFS_TEST_INTERNAL_LINK_LIBS
        lfs_mcp
        lfs_tcp
        lfs_visualizer
        lfs_sequencer
        lfs_rendering
        lfs_training
        lfs_core
        lfs_geometry
        lfs_io
        lfs_video
        fastlfs_backend
        spz_lib
        lfs_python_utils
    )
endif()

# Link libraries
target_link_libraries(lichtfeld_tests PRIVATE
    lfs_shared_image_cuda
    lfs_roi_masks
    lfs_preprocessing
    ${LFS_TEST_INTERNAL_LINK_LIBS}

    # External libraries
    Zep::Zep
    vterm
    GTest::gtest
    nanobind-static
    ZLIB::ZLIB
    ${TORCH_LIBRARIES}
    OpenMeshCore
    assimp::assimp

    # CUDA libraries
    CUDA::cudart
    CUDA::curand
    CUDA::cupti

    # System libraries
    spdlog::spdlog
    SDL3::SDL3
)

target_link_libraries(lichtfeld_tests PRIVATE lfs_vulkan_rasterizer Vulkan::Vulkan)

# Focused post-training compaction loop, including parameter and checkpoint
# compatibility contracts, without compiling the unrelated tensor oracle suite.
add_executable(popspa_tests
    test_main.cpp
    licht_test_support.cpp
    ${POPSPA_TEST_FILES}
    test_training_parameters.cpp
    test_checkpoint_resume.cpp
    test_training_snapshot_service.cpp
    ${CMAKE_SOURCE_DIR}/src/python/lfs/py_params.cpp
)
target_include_directories(popspa_tests PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/training
    ${CMAKE_SOURCE_DIR}/src/app/include
    ${CMAKE_SOURCE_DIR}/src/visualizer
    ${NB_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(popspa_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)
target_link_libraries(popspa_tests PRIVATE
    ${LFS_TEST_INTERNAL_LINK_LIBS}
    GTest::gtest nanobind-static
    lfs_vulkan_rasterizer Vulkan::Vulkan CUDA::cudart CUDA::curand CUDA::cupti
)
set_target_properties(popspa_tests PROPERTIES
    CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON
)
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(popspa_tests)
endif()
add_test(NAME POPSpaContracts COMMAND popspa_tests
    "--gtest_filter=*POPSpa*:PopScore*:CheckpointSparsityRoundTripTest.*:CheckpointVersionTest.*:TrainingParameters*"
)
set_tests_properties(POPSpaContracts PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "fast;gpu"
    TIMEOUT 180
)

# Compiler options for CUDA debugging
target_compile_options(lichtfeld_tests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo>
)
if(MSVC)
    target_compile_options(lichtfeld_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /DNDEBUG>
    )
    # The stack diagnostic regression checks its caller name in Release too.
    # Keep symbols for that source and a matching executable PDB.
    set_property(SOURCE test_cuda_error.cpp APPEND PROPERTY COMPILE_OPTIONS /Z7)
    target_link_options(lichtfeld_tests PRIVATE /DEBUG:FULL)
else()
    target_compile_options(lichtfeld_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
    )
endif()
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(lichtfeld_tests)
endif()

# Exercise the production SSIM translation unit with Debug's subnormal semantics,
# independently of the application's Release fast-math flags.
add_executable(masked_ssim_noftz_tests
    test_main.cpp
    test_masked_ssim_window.cpp
    ${CMAKE_SOURCE_DIR}/src/training/kernels/ssim.cu
)
set_target_properties(masked_ssim_noftz_tests PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON
)
target_include_directories(masked_ssim_noftz_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/training ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(masked_ssim_noftz_tests PRIVATE LFS_TENSOR_PRIVATE_ACCESS)
target_compile_options(masked_ssim_noftz_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false --prec-div=true --prec-sqrt=true --extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(masked_ssim_noftz_tests PRIVATE
    lfs_training lfs_core GTest::gtest CUDA::cudart)
if(MSVC)
    target_compile_definitions(masked_ssim_noftz_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>)
    target_compile_options(masked_ssim_noftz_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
add_test(NAME masked_ssim_noftz COMMAND masked_ssim_noftz_tests
    --gtest_filter=MaskedSSIMWindowTest.SparseSupportRemainsFinite:MaskedSSIMWindowTest.TinyWindowSupportApproachesUnsupported)
set_tests_properties(masked_ssim_noftz PROPERTIES LABELS "gpu")

# Keep both math modes in separate whole-program CUDA translation units, as in
# the application's static libraries. RDC would hide the host-stub collision.
foreach(mode precise fast)
    add_library(cuda_stub_${mode} STATIC cuda_template_stub_fixture.cu)
    target_compile_features(cuda_stub_${mode} PRIVATE cuda_std_20)
    target_include_directories(cuda_stub_${mode} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(cuda_stub_${mode} PRIVATE LFS_STUB_TEST_VARIANT=${mode} LFS_TENSOR_PRIVATE_ACCESS)
    target_link_libraries(cuda_stub_${mode} PRIVATE lfs_core)
    set_target_properties(cuda_stub_${mode} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endforeach()
target_compile_options(cuda_stub_fast PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

add_executable(cuda_template_stub_tests test_main.cpp test_cuda_template_stub.cpp)
target_compile_features(cuda_template_stub_tests PRIVATE cxx_std_23)
target_include_directories(cuda_template_stub_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(cuda_template_stub_tests PRIVATE PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}")
target_link_libraries(cuda_template_stub_tests PRIVATE
    cuda_stub_precise cuda_stub_fast lfs_core GTest::gtest CUDA::cudart)
add_test(NAME tensor.cuda_template_stub COMMAND cuda_template_stub_tests --gtest_color=no)
set_tests_properties(tensor.cuda_template_stub PROPERTIES LABELS "tensor;gpu" TIMEOUT 30)
if(WIN32)
    set_tests_properties(tensor.cuda_template_stub PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_core>;PATH=path_list_prepend:${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/$<IF:$<CONFIG:Debug>,debug/bin,bin>")
endif()

# Tensor-library hardening is intentionally isolated from the default test
# tiers.  During the hardening phases these oracle tests are expected to expose
# known failures, and crash-prone reproducers must not terminate the monolithic
# test process before unrelated tests run.
add_executable(tensor_hardening_tests
    test_main.cpp
    tensor_hardening_cuda_utils.cu
    test_tensor_hardening_theme_a.cpp
    test_tensor_hardening_theme_b.cpp
    test_tensor_hardening_theme_c.cpp
    test_tensor_hardening_theme_d.cpp
    test_tensor_hardening_theme_e.cpp
    test_tensor_hardening_theme_f.cpp
    test_tensor_sort_stream.cpp
    test_pinned_memory_allocator.cpp
)

set_target_properties(tensor_hardening_tests PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(tensor_hardening_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(tensor_hardening_tests PRIVATE
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(tensor_hardening_tests PRIVATE
    lfs_core
    GTest::gtest
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::curand
)

if(MSVC)
    target_compile_options(tensor_hardening_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /Z7>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(tensor_hardening_tests PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g -fno-omit-frame-pointer>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3 -DNDEBUG>
    )
endif()
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(tensor_hardening_tests)
endif()

# =============================================================================
# CTest registration
# =============================================================================
# The monolithic test executable still mixes quick contracts with real-data,
# training, stress, and optional-codec cases. Register each tier as one filtered
# invocation: per-case discovery would pay process startup for thousands of
# sub-second unit tests and make the "fast" CTest label needlessly slow.
string(CONCAT LFS_NIGHTLY_TEST_FILTER
    "*CheckpointResumeTest.TrainSaveLoadResume/nightly_*:"
    "TensorStressTest.*:"
    "TensorMemoryTest.VeryLargeTensor:"
    "TensorMemoryTest.StressTest*:"
    "TensorMatrixTest.VeryLargeMatMul:"
    "TensorMaskingTest.StressTest*:"
    "BoolAnyAllTest.LargeBoolSumPreservesDensificationCounts:"
    "TensorZeroDimensionTest.StressTest*:"
    "TensorCatReductionBugTest.*VeryLarge*:"
    "MaskedSelectRowsTest.*LargeScale*:"
    "CPUDtypeConversionTest.ProductionMuralPointCloudWorkflow:"
    "FastGSFuzzTest.RandomStress_*:"
    "FastGSFuzzTest.VeryLargeBatch_100k:"
    "NaNInfGPUCheckTest.*5M*:"
    "NaNInfGPUCheckTest.RandomStress_*:"
    "NaNInfGPUCheckTest.InfDetection_LargeTensors:"
    "NaNInfGPUCheckTest.MultiDim_LargeTensors:"
    "StridedFillTest.VeryLargeGrid_3D_Fill:"
    "TensorLazyRuntimeTest.ErankExpressionMatchesTorchOnInheritedStream:"
    "TensorLazyRuntimeTest.FusedSegmentedSquareSumMatchesTorchOnInheritedStream:"
    "PlyToRadLod.OutOfCore*:"
    "SpzFormatTest.RealPlyRoundtrip:"
    "PythonIOTest.LoadRealPlyFile:"
    "ShSwizzleLayout.CanonicalRoundTrip_RealData:"
    "RotatedShCorrectnessTest.*:"
    "NvCodecImageLoaderJpeg2k16Bit.*:"
    "PipelinedImageLoaderSidecarJpeg2k.*"
)

string(CONCAT LFS_SLOW_TEST_FILTER
    "TrainingVulkanPairSort.*:"
    "TrainingVulkanOps.*:"
    "VideoMaskingPipeline.*:VideoMaskExtraction.*:"
    "*CheckpointResumeTest.TrainSaveLoadResume/tiny_*:"
    "TrainingSetupRegressionTest.*:"
    "EvaluationSplitTest.*:"
    "PipelinedImageLoaderTest.*:"
    "ImageResizeParallelTest.*:"
    "PythonIOTest.LoadCOLMAPDataset:"
    "PythonIOTest.PipelinedLoader*:"
    "PythonIntegrationTest.SceneCameraExposesVisualizerRenderContract:"
    "PythonLspClientTest.*:"
    "UvRunnerTest.*:"
    "ViewportFrameLifecycleServiceTest.*:"
    "ArenaMetricsContentionTest.*:"
    "TensorMultiStreamTest.TrainerViewerHandshake*:"
    "CheckpointAtomicReplaceFaultTest.*:"
    "MemoryPressureTest.InjectedFailure*:"
    "UndistortCameraTest.BicyclePinholeNoDist:"
    "VideoColorConvertTest.*FullHD*:"
    "PlyToRadLod.*:"
    "TensorBackends/LodUpload.*:"
    "RadMetaSidecar.*:"
    "OctreeLod.*:"
    "NurecUsdzFormatTest.*:"
    "UsdFormatTest.RoundtripPreservesValues:"
    "ShareableAllocationLimitTest.Migrate*"
)

set(LFS_NON_FAST_TEST_FILTER
    "${LFS_SLOW_TEST_FILTER}:${LFS_NIGHTLY_TEST_FILTER}:DiscoveryFuzz.*:DiscoverySweep.*")

add_test(NAME lichtfeld.fast
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=*-${LFS_NON_FAST_TEST_FILTER}"
        --gtest_color=no
)
set_tests_properties(lichtfeld.fast PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast;gpu"
    TIMEOUT 60
)

if(NOT BUILD_TEMPORAL_CPU_TESTS)
    string(CONCAT LFS_TEMPORAL_CPU_TEST_FILTER
        "TemporalFrameTracker.*:"
        "SceneTemporalPlan.*:"
        "SceneHistoryContract.*:"
        "SceneHistoryTracker.*:"
        "SceneTemporalCoordinator.*:"
        "SceneMotionReprojection.*:"
        "VulkanSceneMotionContract.*:"
        "SceneTemporalResolve.*:"
        "SceneTemporalQualityRegression.*:"
        "VulkanSceneTemporalResolveContract.*:"
        "VulkanSceneTemporalPipelineContract.*"
    )
    add_test(NAME lichtfeld.temporal.cpu
        COMMAND $<TARGET_FILE:lichtfeld_tests>
            "--gtest_filter=${LFS_TEMPORAL_CPU_TEST_FILTER}"
            --gtest_color=no
    )
    set_tests_properties(lichtfeld.temporal.cpu PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        LABELS "fast;cpu"
        TIMEOUT 120
    )
endif()

add_test(NAME lichtfeld.slow
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=${LFS_SLOW_TEST_FILTER}-${LFS_NIGHTLY_TEST_FILTER}"
        --gtest_color=no
)
set_tests_properties(lichtfeld.slow PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "slow;gpu"
    TIMEOUT 300
)

add_test(NAME lichtfeld.nightly
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=${LFS_NIGHTLY_TEST_FILTER}"
        --gtest_color=no
)
set_tests_properties(lichtfeld.nightly PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "nightly;gpu"
    TIMEOUT 1800
)

add_test(NAME tensor.hardening
    COMMAND $<TARGET_FILE:tensor_hardening_tests>
        --gtest_color=no
        --gtest_death_test_style=threadsafe
)
set_tests_properties(tensor.hardening PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "hardening;gpu"
    TIMEOUT 900
)

add_test(NAME lichtfeld.discovery
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=DiscoveryFuzz.*:DiscoverySweep.*"
        --gtest_color=no
)

add_test(NAME system_info
    COMMAND $<TARGET_FILE:lichtfeld_tests>
        "--gtest_filter=SystemInfoTest.*:SessionBreadcrumbTest.*:LoggerTailTest.*"
        --gtest_color=no
)
set_tests_properties(system_info PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "fast"
    TIMEOUT 30
)
set_tests_properties(lichtfeld.discovery PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    LABELS "discovery;gpu"
    TIMEOUT 1800
)

# The pixel regression uses an independent decoder, not our video player.
# Set LFS_VIDEO_TEST_FFMPEG explicitly when ffmpeg is outside PATH.
find_program(LFS_VIDEO_TEST_FFMPEG NAMES ffmpeg)
if(LFS_VIDEO_TEST_FFMPEG)
    add_test(NAME video.orientation
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/verify_video_orientation.py
            --executable $<TARGET_FILE:${PROJECT_NAME}>
            --ffmpeg ${LFS_VIDEO_TEST_FFMPEG}
            --output ${CMAKE_CURRENT_BINARY_DIR}/video-orientation
    )
    set_tests_properties(video.orientation PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        LABELS "slow;gpu;video"
        TIMEOUT 600
        RUN_SERIAL TRUE
    )
else()
    message(STATUS "Skipping video.orientation: set LFS_VIDEO_TEST_FFMPEG to enable decoded-video regression")
endif()

# Keep every Python tier explicit so new modules cannot silently drift out of
# the curated suite.
set(LFS_PYTHON_FAST_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_default_reset.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_ci_format_push.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_ci_retry_configure.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gallery_storage.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_icon_export.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_asset_identity.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_bug_report.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_core_abi_stamp.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hip_runtime_configuration.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hip_native_ci_filter.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hip_pointwise_catalog.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_dense_matching.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_export_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gizmo_context.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gt_compare_controls.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_histogram_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_http.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_import_dialog_panels.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_settings_version_safety.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_io_hygiene_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_keymap_bindings.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_new_project_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_verified_dead_code_cleanup.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_viewer_spirv.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_layout_composition.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_manage_stubs.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh2splat_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_properties.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_rich_returns.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_project_operations.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_api_surface.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_marketplace_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_localization.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_templates.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rml_section_toggle.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_validator.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_preferences_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_registry.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_reveal_license.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_reveal_ui_contract.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_depth_view_scrub.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rendering_panel_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rendering_panel_simplify.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_preview_layout.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_preview_pan_scale.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_preview_camera_identity.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_preview_file_revision.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_rmlui_image_sources.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_runtime_state.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scale_distance_contract.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scripts_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_selection_controls_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_selection_groups_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_shader_header_target.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_splat_lod_hierarchy.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_store.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_store_subscriptions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_float16_import.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_basic_indexing.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_arg_extrema_nan.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_empty_slices.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_empty_batch_matmul.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_float16_extraction.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_sign_nan.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_inverse_trig_domain.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_global_keepdim.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_row_mask_assignment.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_repeat_regression.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_property.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tool_registry_activation.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_training_manager_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tools_new.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_training_panel_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_transform_controls_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_undo.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_viewport_overlay_hooks.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_windows_build_preflight.py
)

set(LFS_PYTHON_SLOW_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_asset_manager_panel.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_async_plugin_loading.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_capability_thread_safety.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_errors.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_histogram_localization_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hooks.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hot_reload_edge_cases.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_hot_reload_fixes.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_menu_schemas.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_menubar_resources.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_hot_reload.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_operator_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_pano_coordinate_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_callbacks.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_concurrency.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_docs.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_error_recovery.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_property_schema.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_property_system.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_signals.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_toolbar_hooks.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_vksplat_orthographic_regressions.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_vksplat_overlay_storage.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_vksplat_output_lifetime_regressions.py
)

set(LFS_PYTHON_NIGHTLY_TESTS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_io_load.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_io_save.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_packages.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scene.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scene_validity.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_pytorch_interop.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/test_unicode_path_bindings.py
)

add_test(NAME lichtfeld.cmake.vcpkg_profiles
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_vcpkg_profiles.cmake"
)
set_tests_properties(lichtfeld.cmake.vcpkg_profiles PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    LABELS "fast"
    TIMEOUT 60
)

add_test(NAME lichtfeld.cmake.native_file_dialog_port
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_native_file_dialog_port.cmake"
)
set_tests_properties(lichtfeld.cmake.native_file_dialog_port PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    LABELS "fast"
    TIMEOUT 10
)

add_test(NAME lichtfeld.python.fast
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        -m "not gpu and not slow and not integration"
        ${LFS_PYTHON_FAST_TESTS}
)
set_tests_properties(lichtfeld.python.fast PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "fast"
    TIMEOUT 60
    ENVIRONMENT_MODIFICATION "CMAKE_COMMAND=set:${CMAKE_COMMAND}"
)

add_test(NAME lichtfeld.python.slow
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        ${LFS_PYTHON_SLOW_TESTS}
)
set_tests_properties(lichtfeld.python.slow PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "slow;gpu"
    TIMEOUT 300
)

add_test(NAME lichtfeld.python.nightly
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        ${LFS_PYTHON_NIGHTLY_TESTS}
)
set_tests_properties(lichtfeld.python.nightly PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "nightly;gpu"
    TIMEOUT 1800
)

# Leave the baseline environment intact unless a directory is configured explicitly.
set(LFS_SHADER_BASE_DIR "" CACHE PATH "PR base Slang source directory for reveal shader comparison")
add_test(NAME lichtfeld.python.reveal_shader_default
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_reveal_shader_default_runtime.py
)
set_tests_properties(lichtfeld.python.reveal_shader_default PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "slow;integration"
    TIMEOUT 300
    ENVIRONMENT "LFS_SLANGC=${SLANGC}"
)
if(TARGET lfs_vulkan_rasterizer)
    set_property(TEST lichtfeld.python.reveal_shader_default APPEND PROPERTY
        ENVIRONMENT "LFS_SHADER_RECIPES=$<TARGET_PROPERTY:lfs_vulkan_rasterizer,LFS_SLANG_RECIPES>")
endif()
if(LFS_SHADER_BASE_DIR)
    set_property(TEST lichtfeld.python.reveal_shader_default APPEND PROPERTY
        ENVIRONMENT "LFS_SHADER_BASE_DIR=${LFS_SHADER_BASE_DIR}")
endif()

# App-backed lifetime coverage requires a real renderer and its own process.
add_test(NAME lichtfeld.python.point_cloud_lifetime
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_point_cloud_lifetime_runtime.py
)

if(UNIX AND NOT APPLE AND TARGET LichtFeld-Studio)
    # LFS_REVEAL_TEST_PROJECT is inherited; project coverage skips when it is absent.
    add_test(NAME lichtfeld.python.reveal_mcp
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_reveal_mcp_runtime.py
    )
    set_tests_properties(lichtfeld.python.reveal_mcp PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 600
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )

    add_test(NAME lichtfeld.python.camera_set_view_validation
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_camera_set_view_validation.py
    )
    set_tests_properties(lichtfeld.python.camera_set_view_validation PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>"
    )

    add_test(NAME lichtfeld.python.scene_export_path
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_scene_export_path_runtime.py
    )
    set_tests_properties(lichtfeld.python.scene_export_path PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration;io"
        TIMEOUT 600
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>"
    )
endif()
set_tests_properties(lichtfeld.python.point_cloud_lifetime PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "nightly;gpu;integration"
    TIMEOUT 300
    RUN_SERIAL TRUE
    ENVIRONMENT_MODIFICATION "LFS_EXECUTABLE=set:$<TARGET_FILE:${PROJECT_NAME}>"
)

add_test(NAME lichtfeld.python.point_cloud_first_open
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_project_point_cloud_first_open_runtime.py
)
set_tests_properties(lichtfeld.python.point_cloud_first_open PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "nightly;gpu;integration"
    TIMEOUT 300
    RUN_SERIAL TRUE
    ENVIRONMENT_MODIFICATION "LFS_EXECUTABLE=set:$<TARGET_FILE:${PROJECT_NAME}>"
)

add_test(NAME lichtfeld.python.gpu
    COMMAND ${Python_EXECUTABLE} -m pytest -q
        -m gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_basic_indexing.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_arg_extrema_nan.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_empty_slices.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_empty_batch_matmul.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_float16_extraction.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_sign_nan.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_inverse_trig_domain.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_global_keepdim.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_row_mask_assignment.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_repeat_regression.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_tensor_property.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_nn_sam2.py
)
set_tests_properties(lichtfeld.python.gpu PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "fast;gpu"
    TIMEOUT 60
)


# Owns its Xvfb display, GUI process and generated inputs; requires a Linux GPU.
if(UNIX AND NOT APPLE AND TARGET LichtFeld-Studio)
    add_test(NAME lichtfeld.python.gt_compare_reconstruction
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gt_compare_reconstruction.py
    )
    set_tests_properties(lichtfeld.python.gt_compare_reconstruction PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.lod_budget_recovery
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_lod_budget_recovery_runtime.py
    )
    set_tests_properties(lichtfeld.python.lod_budget_recovery PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.higs_deterministic_export
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_higs_deterministic_export_runtime.py
    )
    set_tests_properties(lichtfeld.python.higs_deterministic_export PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.mesh_headlight
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh_headlight.py
    )
    set_tests_properties(lichtfeld.python.mesh_headlight PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )

    add_test(NAME lichtfeld.python.capture_mesh
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_render_capture_mesh.py
    )
    set_tests_properties(lichtfeld.python.capture_mesh PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.capture_environment
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_render_capture_environment.py
    )
    set_tests_properties(lichtfeld.python.capture_environment PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.mesh_depth_view
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh_depth_view_runtime.py
    )
    set_tests_properties(lichtfeld.python.mesh_depth_view PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.mesh_empty_split
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh_empty_split_runtime.py
    )
    set_tests_properties(lichtfeld.python.mesh_empty_split PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.mesh_removal_keeps_viewport
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh_removal_keeps_viewport_runtime.py
    )
    set_tests_properties(lichtfeld.python.mesh_removal_keeps_viewport PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.mesh_panorama
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh_panorama_runtime.py
    )
    set_tests_properties(lichtfeld.python.mesh_panorama PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.mesh_split_depth
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mesh_split_depth_runtime.py
    )
    set_tests_properties(lichtfeld.python.mesh_split_depth PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
    add_test(NAME lichtfeld.python.gui_evaluation_download_policy
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_gui_evaluation_download_policy.py
    )
    set_tests_properties(lichtfeld.python.gui_evaluation_download_policy PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>;LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
endif()

# Owns its Xvfb display, GUI process and generated dataset; requires a Linux GPU.
if(UNIX AND NOT APPLE AND TARGET LichtFeld-Studio)
    add_test(NAME lichtfeld.python.mcp_ply_sh_degree
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mcp_ply_sh_degree.py
    )
    set_tests_properties(lichtfeld.python.mcp_ply_sh_degree PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RESOURCE_LOCK lfs_gui_gpu
        ENVIRONMENT "LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR};LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>"
    )
    add_test(NAME lichtfeld.python.dataset_load_training
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_mcp_dataset_load_training.py
    )
    set_tests_properties(lichtfeld.python.dataset_load_training PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 240
        RUN_SERIAL TRUE
        ENVIRONMENT "LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR};LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>"
    )
    add_test(NAME lichtfeld.python.training_strategy_update
        COMMAND ${Python_EXECUTABLE} -m pytest -q
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_training_strategy_update.py
    )
    set_tests_properties(lichtfeld.python.training_strategy_update PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "gpu;integration"
        TIMEOUT 300
        RUN_SERIAL TRUE
        ENVIRONMENT "LFS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR};LFS_EXECUTABLE=$<TARGET_FILE:LichtFeld-Studio>"
    )
endif()

# Custom targets for convenience
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L fast
    DEPENDS lichtfeld_tests lichtfeld_format_tests popspa_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the fast developer test tier"
)

add_custom_target(run_full_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -LE nightly
    DEPENDS lichtfeld_tests lichtfeld_format_tests popspa_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running fast and slow tests (excluding nightly)"
)

foreach(TEST_TARGET lichtfeld_tests popspa_tests)
    add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TEST_TARGET}>/resources/viewer"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${PROJECT_SOURCE_DIR}/src/visualizer/gui/resources/viewer"
            "$<TARGET_FILE_DIR:${TEST_TARGET}>/resources/viewer"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${PROJECT_SOURCE_DIR}/src/visualizer/gui/rmlui/resources"
            "$<TARGET_FILE_DIR:${TEST_TARGET}>/resources/assets/rmlui"
    )
endforeach()

# Platform-specific settings
if(WIN32)
    # Copy Torch DLLs for testing on Windows
    file(GLOB TORCH_DLLS "${Torch_DIR}/../../../lib/*.dll")
    foreach(TEST_TARGET lichtfeld_tests popspa_tests)
        foreach(TORCH_DLL ${TORCH_DLLS})
            add_custom_command(
                TARGET ${TEST_TARGET}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}"
                "$<TARGET_FILE_DIR:${TEST_TARGET}>")
        endforeach()
        # Torch may bundle an older CUDA runtime with the same DLL filename.
        # Match the toolkit used to build core (including diagnostics entry points).
        file(GLOB _lfs_test_cuda_runtime_dlls
            "${CUDAToolkit_BIN_DIR}/cudart64*.dll"
            "${CUDAToolkit_BIN_DIR}/curand64*.dll"
            "${CUDAToolkit_BIN_DIR}/x64/cudart64*.dll"
            "${CUDAToolkit_BIN_DIR}/x64/curand64*.dll")
        foreach(CUDA_DLL ${_lfs_test_cuda_runtime_dlls})
            add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_DLL}"
                    "$<TARGET_FILE_DIR:${TEST_TARGET}>")
        endforeach()
        if(TARGET nvimgcodec)
            add_dependencies(${TEST_TARGET} nvimgcodec)
            add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:nvimgcodec>"
                    "$<TARGET_FILE_DIR:${TEST_TARGET}>"
                COMMENT "Copying nvImageCodec library for ${TEST_TARGET}"
            )
        endif()
        if(TARGET nvjpeg_ext)
            add_dependencies(${TEST_TARGET} nvjpeg_ext)
            add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:nvjpeg_ext>"
                    "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions/"
                COMMENT "Copying nvjpeg extension for ${TEST_TARGET}"
            )
        endif()
        if(TARGET nvjpeg2k_ext)
            add_dependencies(${TEST_TARGET} nvjpeg2k_ext)
            add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:nvjpeg2k_ext>"
                    "$<TARGET_FILE_DIR:${TEST_TARGET}>/extensions/"
                COMMENT "Copying nvjpeg2k extension for ${TEST_TARGET}"
            )
            if(WITH_DYNAMIC_NVJPEG2K AND DEFINED _lfs_nvjpeg2k_runtime_file AND EXISTS "${_lfs_nvjpeg2k_runtime_file}")
                add_custom_command(TARGET ${TEST_TARGET} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_lfs_nvjpeg2k_runtime_file}"
                        "$<TARGET_FILE_DIR:${TEST_TARGET}>"
                    COMMENT "Copying nvjpeg2k runtime library for ${TEST_TARGET}"
                )
            endif()
        endif()
    endforeach()
endif()

# Standalone tensor launch-overhead bench (Phase 6B-2 P2 §9 pin).
# Built with the test target; NOT registered with ctest; not part of any
# commit list. Depends only on the public tensor API.
add_executable(bench_tensor_launch_overhead
    bench_tensor_launch_overhead.cpp
)
set_target_properties(bench_tensor_launch_overhead PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(bench_tensor_launch_overhead PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(bench_tensor_launch_overhead PRIVATE
    lfs_core
    CUDA::cudart
)

# Standalone GUI-workload tensor bench. Built with tests; NOT registered with
# ctest. Public tensor API plus tensor_backend selection/sync.
add_executable(bench_tensor_gui
    bench_tensor_gui.cpp
)
set_target_properties(bench_tensor_gui PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(bench_tensor_gui PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(bench_tensor_gui PRIVATE
    lfs_core
    CUDA::cudart
)

add_executable(bench_selection_tensor EXCLUDE_FROM_ALL bench_selection_tensor.cpp)
set_target_properties(bench_selection_tensor PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(bench_selection_tensor PRIVATE ${CMAKE_SOURCE_DIR}/external)
target_link_libraries(bench_selection_tensor PRIVATE lfs_rendering CUDA::cudart)

add_executable(tensor_backend_corpus
    tensor_backend_corpus.cpp
)
set_target_properties(tensor_backend_corpus PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(tensor_backend_corpus PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(tensor_backend_corpus PRIVATE
    lfs_core
    CUDA::cudart
)

add_executable(tensor_interface_freeze
    test_tensor_interface_freeze.cpp
)
set_target_properties(tensor_interface_freeze PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(tensor_interface_freeze PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(tensor_interface_freeze PRIVATE
    lfs_core
    CUDA::cudart
)
add_test(NAME tensor_interface_freeze COMMAND tensor_interface_freeze)

add_executable(bench_nn_ops EXCLUDE_FROM_ALL
    bench_nn_ops.cpp
)
set_target_properties(bench_nn_ops PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(bench_nn_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(bench_nn_ops PRIVATE
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)
target_link_libraries(bench_nn_ops PRIVATE
    lfs_core
    GTest::gtest_main
    CUDA::cudart
)

# Phase 6C-P3 negative nvcc TU fixtures (Ruling 1 / §1.11) live under
# tests/cuda_fixtures/ and are intentionally NOT listed in any target SOURCES.
# Positive nvcc TU: device_fault_cuda_utils.cu (compiled into lichtfeld_tests;
# includes the FOUR allowed headers and exercises DeviceFaultRecord +
# device_fault_try_record_first). A host gtest asserts the negative fixtures
# exist and contain the forbidden includes (see test_device_fault.cpp).

# Status messages
list(LENGTH TEST_FILES TEST_COUNT)

message(STATUS "Tests configured with ${CMAKE_BUILD_TYPE} build type")
message(STATUS "  Unit tests: ${TEST_COUNT} files -> lichtfeld_tests")
message(STATUS "  Build tests: ninja lichtfeld_tests")
message(STATUS "  Run tests: ninja run_tests")
message(STATUS "  Run full tests: ninja run_full_tests")
message(STATUS "  Bench (not ctest): bench_tensor_launch_overhead bench_tensor_gui")
message(STATUS "  Phase 6C device-fault tests: test_device_fault.cpp + device_fault_cuda_utils.cu")

add_test(NAME lichtfeld.decimate.perf
    COMMAND lichtfeld_tests --gtest_filter=SplatDecimate.DecimatePerfSmoke)
set_tests_properties(lichtfeld.decimate.perf PROPERTIES
    LABELS "nightly;gpu" ENVIRONMENT "LFS_DECIMATE_PERF=1" TIMEOUT 60)

# Focused Vulkan stochastic renderer contracts.
if(TARGET lfs_vulkan_rasterizer)
    add_executable(stochastic_tests test_stochastic_resampling.cpp)
    target_compile_features(stochastic_tests PRIVATE cxx_std_23)
    target_link_libraries(stochastic_tests PRIVATE lfs_vulkan_rasterizer lfs_core GTest::gtest_main)
    get_target_property(_stochastic_shader_dir lfs_vulkan_rasterizer LFS_VULKAN_RASTERIZER_SPV_DIR)
    target_compile_definitions(stochastic_tests PRIVATE
        LFS_STOCHASTIC_TEST_SPIRV_DIR="${_stochastic_shader_dir}/generated")
    add_test(NAME lichtfeld.stochastic COMMAND stochastic_tests)
    set_tests_properties(lichtfeld.stochastic PROPERTIES LABELS "gpu" TIMEOUT 120)
    # Test private history state without widening the Windows DLL exports.
    add_executable(stochastic_viewport_tests
        test_stochastic_viewport.cpp
        ${CMAKE_SOURCE_DIR}/src/visualizer/rendering/stochastic_viewport.cpp)
    target_compile_features(stochastic_viewport_tests PRIVATE cxx_std_23)
    target_include_directories(stochastic_viewport_tests PRIVATE
        $<TARGET_PROPERTY:lfs_visualizer,INCLUDE_DIRECTORIES>)
    target_link_libraries(stochastic_viewport_tests PRIVATE
        lfs_visualizer lfs_vulkan_rasterizer lfs_core GTest::gtest_main)
    add_test(NAME lichtfeld.stochastic_viewport COMMAND stochastic_viewport_tests)
endif()

# Compute programs from in-memory SPIR-V and user program sets on a real device.
if(LFS_VKSPLAT_VIEWPORT_BUILD AND TARGET lfs_visualizer AND TARGET lfs_user_shader_ir)
    add_executable(vksplat_program_set_tests test_vksplat_program_sets.cpp test_shader_variants.cpp)
    target_compile_features(vksplat_program_set_tests PRIVATE cxx_std_23)
    target_include_directories(vksplat_program_set_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/visualizer ${CMAKE_SOURCE_DIR}/src/core/include
        ${CMAKE_BINARY_DIR}/src/rendering/rasterizer/vulkan)
    get_target_property(_program_set_spv_dir lfs_vulkan_rasterizer LFS_VULKAN_RASTERIZER_SPV_DIR)
    get_target_property(_program_set_package_dir lfs_user_shader_ir LFS_USER_SHADER_PACKAGE_DIR)
    get_target_property(_program_set_test_package_dir lfs_user_shader_ir LFS_USER_SHADER_TEST_PACKAGE_DIR)
    target_compile_definitions(vksplat_program_set_tests PRIVATE
        LFS_PROGRAM_SET_TEST_SPIRV_DIR="${_program_set_spv_dir}"
        LFS_SHADER_PACKAGE_DIR="${_program_set_package_dir}"
        LFS_SHADER_TEST_PACKAGE_DIR="${_program_set_test_package_dir}")
    target_link_libraries(vksplat_program_set_tests PRIVATE
        lfs_vulkan_rasterizer lfs_visualizer lfs_core GTest::gtest_main)
    add_dependencies(vksplat_program_set_tests lfs_user_shader_ir lfs_user_shader_test_package)
    if(MSVC)
        target_link_options(vksplat_program_set_tests PRIVATE "/DELAYLOAD:nvcuda.dll")
        target_link_libraries(vksplat_program_set_tests PRIVATE delayimp)
    endif()
    add_test(NAME lichtfeld.vksplat_program_sets COMMAND vksplat_program_set_tests)
    set_tests_properties(lichtfeld.vksplat_program_sets PROPERTIES LABELS "gpu;shaders" TIMEOUT 180
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:slang::slang>;PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_visualizer>")
endif()

# Custom effects and the shader service share the viewer's test dependencies.
if(LFS_VKSPLAT_VIEWPORT_BUILD AND TARGET lfs_visualizer AND TARGET lfs_user_shader_ir)
    add_executable(vksplat_custom_effect_tests test_vksplat_custom_effect.cpp)
    add_executable(shader_service_tests test_shader_service.cpp test_shader_service_rendering.cpp)
    # Measurement subprocess isolation uses Linux fork/exec and /proc/self/exe.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(bench_user_shaders EXCLUDE_FROM_ALL bench_user_shaders.cpp)
        add_executable(user_shader_fault_tests test_user_shader_faults.cpp)
        target_compile_definitions(lfs_visualizer PRIVATE LFS_USER_SHADER_MEASURE)
        target_include_directories(lfs_visualizer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        set(_shader_measure_targets bench_user_shaders user_shader_fault_tests)
    endif()
    get_target_property(_custom_effect_package_dir lfs_user_shader_ir LFS_USER_SHADER_PACKAGE_DIR)
    foreach(_target IN ITEMS vksplat_custom_effect_tests shader_service_tests ${_shader_measure_targets})
        target_compile_features(${_target} PRIVATE cxx_std_23)
        target_include_directories(${_target} PRIVATE
            $<TARGET_PROPERTY:lfs_visualizer,INCLUDE_DIRECTORIES>)
        target_compile_definitions(${_target} PRIVATE LFS_SHADER_PACKAGE_DIR="${_custom_effect_package_dir}")
        target_link_libraries(${_target} PRIVATE
            lfs_vulkan_rasterizer lfs_visualizer lfs_core SDL3::SDL3 GTest::gtest)
        add_dependencies(${_target} lfs_user_shader_ir)
        if(MSVC)
            target_link_options(${_target} PRIVATE "/DELAYLOAD:nvcuda.dll")
            target_link_libraries(${_target} PRIVATE delayimp)
        endif()
    endforeach()
    target_link_libraries(vksplat_custom_effect_tests PRIVATE GTest::gtest_main)
    # lfs_visualizer links these static libraries privately; on Windows it is a DLL and
    # does not re-export their symbols, so the test has to link them itself.
    target_link_libraries(shader_service_tests PRIVATE lfs_io lfs_sequencer GTest::gtest_main)
    # The standalone benchmark has its own main and is deliberately absent from ctest.
    if(TARGET bench_user_shaders)
        target_compile_definitions(bench_user_shaders PRIVATE
            LFS_PROGRAM_SET_TEST_SPIRV_DIR="${_program_set_spv_dir}"
            LFS_SHADER_TEST_PACKAGE_DIR="${_program_set_test_package_dir}")
        add_dependencies(bench_user_shaders lfs_user_shader_test_package)
        add_test(NAME lichtfeld.user_shader_faults COMMAND user_shader_fault_tests)
        set_tests_properties(lichtfeld.user_shader_faults PROPERTIES LABELS "cpu;shaders" TIMEOUT 180)
    endif()
    add_test(NAME lichtfeld.vksplat_custom_effect COMMAND vksplat_custom_effect_tests)
    add_test(NAME lichtfeld.shader_service COMMAND shader_service_tests)
    set_tests_properties(lichtfeld.vksplat_custom_effect PROPERTIES TIMEOUT 600)
    set_tests_properties(lichtfeld.shader_service PROPERTIES TIMEOUT 300)
    set_tests_properties(lichtfeld.vksplat_custom_effect lichtfeld.shader_service PROPERTIES
        LABELS "gpu;shaders"
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:slang::slang>;PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_visualizer>")
endif()

# Focused CUDA renderer contracts.
add_executable(color_variance_tests test_main.cpp test_color_variance.cpp fast_raster_test_helpers.cpp)
target_compile_features(color_variance_tests PRIVATE cxx_std_23)
target_include_directories(color_variance_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/training ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(color_variance_tests PRIVATE lfs_training lfs_core GTest::gtest CUDA::cudart)
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(color_variance_tests)
endif()
add_test(NAME ColorVarianceContracts COMMAND color_variance_tests)
set_tests_properties(ColorVarianceContracts PROPERTIES LABELS "fast;gpu" TIMEOUT 120)

add_executable(gsplat_tests test_main.cpp test_gsplat_rasterizer.cpp fast_raster_test_helpers.cpp tensor_hardening_cuda_utils.cu)
target_compile_features(gsplat_tests PRIVATE cxx_std_23)
target_include_directories(gsplat_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/training ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gsplat_tests PRIVATE lfs_training lfs_core GTest::gtest CUDA::cudart)
if(COMMAND lfs_enable_fast_cxx_compile)
    lfs_enable_fast_cxx_compile(gsplat_tests)
endif()

lfs_tensor_private_access(${TEST_SOURCES})
# The controller parity fixture calls the direct SGEMM launcher.
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/test_ppisp_ops.cpp" APPEND
    PROPERTY COMPILE_DEFINITIONS LFS_TENSOR_PRIVATE_ACCESS)
foreach(_lfs_tensor_test_target IN ITEMS
        tensor_hardening_tests tensor_interface_freeze tensor_backend_corpus
        bench_tensor_launch_overhead bench_tensor_gui)
    if(TARGET ${_lfs_tensor_test_target})
        target_compile_definitions(${_lfs_tensor_test_target} PRIVATE LFS_TENSOR_PRIVATE_ACCESS)
    endif()
endforeach()

if(LFS_TENSOR_VULKAN AND BUILD_TESTS)
    function(lfs_write_program_inventory)
        get_property(reflections GLOBAL PROPERTY LFS_GPU_PROGRAM_REFLECTIONS)
        string(REPLACE ";" "\n" reflections "${reflections}")
        file(WRITE "${CMAKE_BINARY_DIR}/gpu_program_reflections.txt" "${reflections}\n")
    endfunction()
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL lfs_write_program_inventory)
    add_executable(gpu_push_constant_contracts test_tensor_gpu_push_constants.cpp)
    target_include_directories(gpu_push_constant_contracts PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(gpu_push_constant_contracts PRIVATE LFS_TENSOR_PRIVATE_ACCESS=1
        LFS_PROGRAM_INVENTORY="${CMAKE_BINARY_DIR}/gpu_program_reflections.txt")
    target_link_libraries(gpu_push_constant_contracts PRIVATE lfs_core GTest::gtest_main Vulkan::Vulkan)
    target_compile_features(gpu_push_constant_contracts PRIVATE cxx_std_23)
    add_dependencies(gpu_push_constant_contracts lfs_shader_headers)

    if(TARGET lfs_training_vulkan_shaders)
        add_dependencies(gpu_push_constant_contracts lfs_training_vulkan_shaders)
        target_compile_definitions(gpu_push_constant_contracts PRIVATE
            LFS_TRAINING_MANIFEST_DIR="${CMAKE_BINARY_DIR}/lfs_training_spv")
    endif()
    add_test(NAME GpuPushConstantContracts COMMAND gpu_push_constant_contracts)
endif()
