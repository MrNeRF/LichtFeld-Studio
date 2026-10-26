# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# External dependencies (EXCLUDE_FROM_ALL to prevent third-party install rules)

# Zep - embeddable editor core
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zep/CMakeLists.txt")
    message(STATUS "Adding Zep from external/zep")
    add_subdirectory(zep EXCLUDE_FROM_ALL)
    if(TARGET Zep)
        # Treat vendored headers as system so -Wreorder / -Wdeprecated from
        # zep/include do not fire in our translation units that include them.
        set_target_properties(Zep PROPERTIES SYSTEM TRUE)
    endif()
else()
    message(WARNING "Zep not found at external/zep")
endif()

# tree-sitter - editor-grade syntax parsing for dirty Python buffers.
set(LFS_TREE_SITTER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tree-sitter")
set(LFS_TREE_SITTER_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tree-sitter-python")

set(LFS_TREE_SITTER_SOURCES
    "${LFS_TREE_SITTER_ROOT}/lib/src/alloc.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/get_changed_ranges.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/language.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/lexer.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/node.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/parser.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/point.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/query.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/stack.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/subtree.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/tree.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/tree_cursor.c"
    "${LFS_TREE_SITTER_ROOT}/lib/src/wasm_store.c"
)

set(LFS_TREE_SITTER_REQUIRED_FILES
    "${LFS_TREE_SITTER_ROOT}/lib/include/tree_sitter/api.h"
    "${LFS_TREE_SITTER_ROOT}/lib/src/wasm_store.h"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/bindings/c/tree_sitter/tree-sitter-python.h"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/parser.c"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/scanner.c"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/tree_sitter/parser.h"
    ${LFS_TREE_SITTER_SOURCES}
)

foreach(required_file IN LISTS LFS_TREE_SITTER_REQUIRED_FILES)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required tree-sitter file is missing: ${required_file}")
    endif()
endforeach()

message(STATUS "Adding tree-sitter Python parser")

add_library(lfs_tree_sitter STATIC ${LFS_TREE_SITTER_SOURCES})
add_library(lfs::tree_sitter ALIAS lfs_tree_sitter)
target_include_directories(lfs_tree_sitter
    PUBLIC
        "${LFS_TREE_SITTER_ROOT}/lib/include"
    PRIVATE
        "${LFS_TREE_SITTER_ROOT}/lib/src"
)
target_compile_definitions(lfs_tree_sitter
    PRIVATE
        _POSIX_C_SOURCE=200112L
        _DEFAULT_SOURCE
        _BSD_SOURCE
        _DARWIN_C_SOURCE
)
set_target_properties(lfs_tree_sitter PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

add_library(lfs_tree_sitter_python STATIC
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/parser.c"
    "${LFS_TREE_SITTER_PYTHON_ROOT}/src/scanner.c"
)
add_library(lfs::tree_sitter_python ALIAS lfs_tree_sitter_python)
target_include_directories(lfs_tree_sitter_python
    PUBLIC
        "${LFS_TREE_SITTER_PYTHON_ROOT}/bindings/c"
    PRIVATE
        "${LFS_TREE_SITTER_PYTHON_ROOT}/src"
)
set_target_properties(lfs_tree_sitter_python PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# OpenMesh - Half-edge mesh data structure
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenMesh/CMakeLists.txt")
    message(STATUS "Adding OpenMesh")
    set(OPENMESH_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(OPENMESH_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(OpenMesh EXCLUDE_FROM_ALL)
    # Silence warnings from vendored OpenMesh TUs; treat headers as SYSTEM so
    # deprecated-enum-float and similar do not leak into our translation units.
    foreach(_lfs_openmesh_tgt OpenMeshCore OpenMeshTools OpenMeshCoreStatic OpenMeshToolsStatic)
        if(TARGET ${_lfs_openmesh_tgt})
            target_compile_options(${_lfs_openmesh_tgt} PRIVATE -w)
            set_target_properties(${_lfs_openmesh_tgt} PROPERTIES SYSTEM TRUE)
        endif()
    endforeach()
    unset(_lfs_openmesh_tgt)
else()
    message(WARNING "OpenMesh not found at external/OpenMesh")
endif()

# nvImageCodec - GPU-accelerated image codec library
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nvImageCodec/CMakeLists.txt")
    message(STATUS "Adding nvImageCodec")
    # Keep the codec's architectures consistent with the floor the app advertises at startup
    # (LFS_MIN_SM). A portable PTX-only build must not ship a codec built only for the
    # architecture of whatever machine produced it.
    if(DEFINED LFS_RUNTIME_MIN_SM)
        set(CUDA_TARGET_ARCHS "${LFS_RUNTIME_MIN_SM}" CACHE STRING "" FORCE)
    endif()
    set(BUILD_NVJPEG_EXT ON CACHE BOOL "" FORCE)
    set(BUILD_NVJPEG2K_EXT ON CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG OFF CACHE BOOL "" FORCE)
    set(WITH_DYNAMIC_NVJPEG2K ON CACHE BOOL "" FORCE)
    add_subdirectory(nvImageCodec EXCLUDE_FROM_ALL)
    # Suppress all warnings in vendored nvImageCodec object/static targets.
    foreach(_lfs_nvimg_tgt
            nvimgcodec_obj
            nvimgcodec
            nvimgcodec_static
            nvimgcodec_imgproc_static
            nvjpeg_ext_obj
            nvjpeg_ext
            nvjpeg_ext_static
            nvjpeg2k_ext_obj
            nvjpeg2k_ext
            nvjpeg2k_ext_static)
        if(TARGET ${_lfs_nvimg_tgt})
            target_compile_options(${_lfs_nvimg_tgt} PRIVATE -w)
        endif()
    endforeach()
    unset(_lfs_nvimg_tgt)
else()
    message(WARNING "nvImageCodec not found - GPU image loading unavailable")
endif()

# apriltag - AprilRobotics fiducial marker detection library
set(LFS_APRILTAG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/apriltag")

set(LFS_APRILTAG_SOURCES
    "${LFS_APRILTAG_ROOT}/apriltag.c"
    "${LFS_APRILTAG_ROOT}/apriltag_pose.c"
    "${LFS_APRILTAG_ROOT}/apriltag_quad_thresh.c"
    "${LFS_APRILTAG_ROOT}/tag16h5.c"
    "${LFS_APRILTAG_ROOT}/tag25h9.c"
    "${LFS_APRILTAG_ROOT}/tag36h10.c"
    "${LFS_APRILTAG_ROOT}/tag36h11.c"
    "${LFS_APRILTAG_ROOT}/tagCircle21h7.c"
    "${LFS_APRILTAG_ROOT}/tagStandard41h12.c"
    "${LFS_APRILTAG_ROOT}/common/g2d.c"
    "${LFS_APRILTAG_ROOT}/common/getopt.c"
    "${LFS_APRILTAG_ROOT}/common/homography.c"
    "${LFS_APRILTAG_ROOT}/common/image_u8.c"
    "${LFS_APRILTAG_ROOT}/common/image_u8_parallel.c"
    "${LFS_APRILTAG_ROOT}/common/image_u8x3.c"
    "${LFS_APRILTAG_ROOT}/common/image_u8x4.c"
    "${LFS_APRILTAG_ROOT}/common/matd.c"
    "${LFS_APRILTAG_ROOT}/common/pam.c"
    "${LFS_APRILTAG_ROOT}/common/pjpeg-idct.c"
    "${LFS_APRILTAG_ROOT}/common/pjpeg.c"
    "${LFS_APRILTAG_ROOT}/common/pnm.c"
    "${LFS_APRILTAG_ROOT}/common/pthreads_cross.c"
    "${LFS_APRILTAG_ROOT}/common/string_util.c"
    "${LFS_APRILTAG_ROOT}/common/svd22.c"
    "${LFS_APRILTAG_ROOT}/common/time_util.c"
    "${LFS_APRILTAG_ROOT}/common/unionfind.c"
    "${LFS_APRILTAG_ROOT}/common/workerpool.c"
    "${LFS_APRILTAG_ROOT}/common/zarray.c"
    "${LFS_APRILTAG_ROOT}/common/zhash.c"
    "${LFS_APRILTAG_ROOT}/common/zmaxheap.c"
)

foreach(required_file IN LISTS LFS_APRILTAG_SOURCES)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required apriltag file is missing: ${required_file}")
    endif()
endforeach()

message(STATUS "Adding AprilTag fiducial detector")

add_library(lfs_apriltag STATIC ${LFS_APRILTAG_SOURCES})
add_library(lfs::apriltag ALIAS lfs_apriltag)
target_include_directories(lfs_apriltag
    SYSTEM PUBLIC
        "${LFS_APRILTAG_ROOT}"
    PRIVATE
        "${LFS_APRILTAG_ROOT}/common"
)
if(MSVC)
    target_compile_definitions(lfs_apriltag
        PUBLIC
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
    )
    target_compile_options(lfs_apriltag PRIVATE /w)
else()
    target_compile_definitions(lfs_apriltag
        PRIVATE
            _POSIX_C_SOURCE=200809L
            _DEFAULT_SOURCE
            _BSD_SOURCE
            _DARWIN_C_SOURCE
    )
    target_compile_options(lfs_apriltag PRIVATE -w)
    target_link_libraries(lfs_apriltag PUBLIC Threads::Threads)
    if(UNIX)
        target_link_libraries(lfs_apriltag PUBLIC m)
    endif()
endif()
set_target_properties(lfs_apriltag PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    DEBUG_POSTFIX d
)

