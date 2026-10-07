# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
add_executable(media_mcp_runner mcp_runner.cpp cpu_color_contracts.cpp ${CMAKE_SOURCE_DIR}/src/app/mcp_media_tools.cpp)
set_target_properties(media_mcp_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_include_directories(media_mcp_runner PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/app/include)
target_compile_features(media_mcp_runner PRIVATE cxx_std_23)
target_link_libraries(media_mcp_runner PRIVATE lfs_mcp lfs_media lfs_core)
if(LFS_HAS_CUDA)
    # Resolve the driver dependency explicitly, including toolkit stubs on
    # build-only Linux hosts where no NVIDIA driver is installed.
    target_link_libraries(media_mcp_runner PRIVATE CUDA::cuda_driver)
    if(MSVC)
        # CPU contract hosts must also load without the Windows display driver.
        # Match the existing CPU visualizer test policy; native GPU paths load
        # nvcuda.dll when they actually call a driver API.
        foreach(_media_cpu_host IN ITEMS lfs_core lfs_mcp)
            target_link_options(${_media_cpu_host} PRIVATE "/DELAYLOAD:nvcuda.dll")
            target_link_libraries(${_media_cpu_host} PRIVATE delayimp)
        endforeach()
    endif()
endif()
if(APPLE)
    # Studio's shared libraries resolve CPython from their host process on macOS.
    # This native runner must supply the complete embedding runtime, just like
    # the other Studio contract executables; ordinary archive linking drops it.
    find_package(Python REQUIRED COMPONENTS Development.Embed)
    target_link_options(media_mcp_runner PRIVATE
        "LINKER:-force_load,$<TARGET_LINKER_FILE:Python::Python>")
    target_link_libraries(media_mcp_runner PRIVATE lfs_python_embed)
endif()
add_test(NAME MediaMcpContracts COMMAND ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../test_mcp_media.py --runner $<TARGET_FILE:media_mcp_runner>)
set_tests_properties(MediaMcpContracts PROPERTIES LABELS "media;cpu;integration" TIMEOUT 300
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG}")
if(WIN32)
    set_tests_properties(MediaMcpContracts PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()
if(NOT LFS_HAS_CUDA)
    return()
endif()
find_package(GTest CONFIG REQUIRED)
add_executable(media_studio_runner runner.cpp native_video_contracts.cpp
    ${CMAKE_SOURCE_DIR}/tests/test_video_cuda_frame_handoff.cpp)
set_target_properties(media_studio_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_include_directories(media_studio_runner PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_compile_features(media_studio_runner PRIVATE cxx_std_23)
target_link_libraries(media_studio_runner PRIVATE lfs_video lfs_io lfs_core lfs_media CUDA::cudart GTest::gtest)
if(TARGET nvimgcodec)
    add_custom_command(TARGET media_studio_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:nvimgcodec>" "$<TARGET_FILE_DIR:media_studio_runner>")
endif()
if(TARGET nvjpeg_ext)
    add_dependencies(media_studio_runner nvjpeg_ext)
    add_custom_command(TARGET media_studio_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:media_studio_runner>/extensions"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:nvjpeg_ext>" "$<TARGET_FILE_DIR:media_studio_runner>/extensions")
endif()
add_test(NAME MediaStudioBackendContracts COMMAND ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../test_studio_backends.py --runner $<TARGET_FILE:media_studio_runner>)
set_tests_properties(MediaStudioBackendContracts PROPERTIES LABELS "media;gpu;integration" TIMEOUT 300
    SKIP_RETURN_CODE 77 ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
if(WIN32)
    set_tests_properties(MediaStudioBackendContracts PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR};PATH=path_list_prepend:${CUDAToolkit_BIN_DIR};PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}/x64")
endif()
