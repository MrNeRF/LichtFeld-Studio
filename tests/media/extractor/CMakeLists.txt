# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# CPU reference tests share the root production modules. Preview selects
# software decoding at runtime, using the same player implementation as Studio.
get_filename_component(MEDIA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
get_directory_property(_media_test_definitions COMPILE_DEFINITIONS)
list(FILTER _media_test_definitions EXCLUDE REGEX "^LFS_HAS_CUDA=")
list(APPEND _media_test_definitions LFS_HAS_CUDA=0)
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${_media_test_definitions}")

add_library(media_shared_core_bridge SHARED shared_core_bridge.cpp)
target_link_libraries(media_shared_core_bridge PRIVATE lfs_error)
set_target_properties(media_shared_core_bridge PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(media_extraction_runner runner.cpp probe_unit_contracts.cpp frame_sink_contracts.cpp shared_core_contracts.cpp jpeg_backend_contracts.cpp encode_session_contracts.cpp)
set_target_properties(media_extraction_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(media_extraction_runner PRIVATE cxx_std_23)
target_compile_definitions(media_extraction_runner PRIVATE LFS_HAS_CUDA=0)
target_include_directories(media_extraction_runner PRIVATE
    ${MEDIA_ROOT}/src ${MEDIA_ROOT}/src/io)
target_link_libraries(media_extraction_runner PRIVATE lfs_media lfs_image_codecs lfs_logger lfs_diagnostics media_shared_core_bridge)
if(MSVC)
    target_compile_options(media_extraction_runner PRIVATE /utf-8 /W4)
endif()
add_test(NAME MediaExtractionContracts
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_extraction_contracts.py
        --runner $<TARGET_FILE:media_extraction_runner>)
set_tests_properties(MediaExtractionContracts PROPERTIES LABELS "media;cpu;offline;integration"
    TIMEOUT 180
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
if(WIN32)
    set_tests_properties(MediaExtractionContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

add_test(NAME MediaProbeUnitContracts COMMAND media_extraction_runner --probe-unit)
set_tests_properties(MediaProbeUnitContracts PROPERTIES LABELS "media;cpu;offline" TIMEOUT 30)
if(WIN32)
    set_tests_properties(MediaProbeUnitContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

add_test(NAME MediaFrameSinkUnitContracts COMMAND media_extraction_runner --sink-unit)
set_tests_properties(MediaFrameSinkUnitContracts PROPERTIES LABELS "media;cpu;offline" TIMEOUT 30)
if(WIN32)
    set_tests_properties(MediaFrameSinkUnitContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

add_test(NAME MediaSharedCoreContracts COMMAND media_extraction_runner --shared-core-unit)
set_tests_properties(MediaSharedCoreContracts PROPERTIES LABELS "media;cpu;offline" TIMEOUT 30)
if(WIN32)
    set_tests_properties(MediaSharedCoreContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

add_test(NAME MediaIngestCLIContracts
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_ingest_cli.py
        --cli $<TARGET_FILE:media-ingest> --runner $<TARGET_FILE:media_extraction_runner>
        --version-header ${CMAKE_BINARY_DIR}/include/git_version.h
        --application $<TARGET_FILE:LichtFeld-Studio>
        --visualizer $<TARGET_FILE:lfs_visualizer>
        --python-module $<TARGET_FILE:lfs_py>
        --build-dir ${CMAKE_BINARY_DIR} --config $<CONFIG>
        --symbols-tool "$<IF:$<BOOL:${WIN32}>,${CMAKE_LINKER},${CMAKE_NM}>")
set_tests_properties(MediaIngestCLIContracts PROPERTIES
    LABELS "media;cpu;offline;integration" TIMEOUT 300
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
if(WIN32)
    set_tests_properties(MediaIngestCLIContracts PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${MEDIA_RUNTIME_LIBRARY_DIR}")
endif()

# Compile the actual binding group against the root production media target.
nanobind_add_module(media_test_bindings python_bindings.cpp
    ${MEDIA_ROOT}/src/python/lfs/py_media.cpp)
target_compile_features(media_test_bindings PRIVATE cxx_std_23)
target_include_directories(media_test_bindings PRIVATE ${MEDIA_ROOT}/src)
target_link_libraries(media_test_bindings PRIVATE lfs_media)
if(MSVC)
    target_compile_options(media_test_bindings PRIVATE /utf-8)
endif()
add_test(NAME MediaPythonBindingsContracts
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../test_media_bindings.py
        --module-dir $<TARGET_FILE_DIR:media_test_bindings> --runtime-dir ${CMAKE_BINARY_DIR}
        --dependency-dir "${MEDIA_RUNTIME_LIBRARY_DIR}")
set_tests_properties(MediaPythonBindingsContracts PROPERTIES LABELS "media;cpu;offline;integration"
    TIMEOUT 180
    ENVIRONMENT "LFS_MEDIA_TEST_FFMPEG=${LFS_MEDIA_TEST_FFMPEG};LFS_MEDIA_TEST_FFPROBE=${LFS_MEDIA_TEST_FFPROBE}")
