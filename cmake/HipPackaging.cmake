# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
if(NOT LFS_HIP_RUNTIME)
    return()
endif()
set(_hip_modules "${LFS_HIP_RUNTIME_LIBRARY}")
set(_hip_dependencies lfs_hip_runtime_build)
if(LFS_HIP_TRAINING_RUNTIME)
    list(APPEND _hip_modules "${LFS_HIP_TRAINING_LIBRARY}")
    list(APPEND _hip_dependencies lfs_hip_training_build)
endif()
set(_hip_copy_commands COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/hip")
foreach(_module IN LISTS _hip_modules)
    list(APPEND _hip_copy_commands COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_module}" "${CMAKE_BINARY_DIR}/hip/")
    install(FILES "${_module}" DESTINATION "${CMAKE_INSTALL_BINDIR}/hip")
endforeach()
if(LFS_HIP_PROFILE_LIBRARY)
    list(APPEND _hip_dependencies lfs_hip_gfx1150_profile)
    list(APPEND _hip_copy_commands COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/hip/profiles/gfx1150"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${LFS_HIP_PROFILE_LIBRARY}" "${CMAKE_BINARY_DIR}/hip/profiles/gfx1150/")
    install(FILES "${LFS_HIP_PROFILE_LIBRARY}" DESTINATION "${CMAKE_INSTALL_BINDIR}/hip/profiles/gfx1150")
endif()
if(WIN32)
    include("${CMAKE_CURRENT_LIST_DIR}/CollectHipRuntime.cmake")
    lfs_hip_windows_runtime_files(_hip_dependencies_runtime "${LFS_HIP_ROOT}")
else()
    file(GLOB _hip_dependencies_runtime
        "${LFS_HIP_ROOT}/lib*/libamdhip64.so*" "${LFS_HIP_ROOT}/lib*/libamd_comgr.so*"
        "${LFS_HIP_ROOT}/lib*/libhiprtc*.so*" "${LFS_HIP_ROOT}/lib*/libhsa-runtime64.so*"
        "${LFS_HIP_ROOT}/lib*/librocprofiler-register.so*"
        "${LFS_HIP_ROOT}/lib*/librocm_kpack.so*")
endif()
set(_hip_runtime_manifest "${CMAKE_CURRENT_BINARY_DIR}/hip-runtime-files.txt")
file(WRITE "${_hip_runtime_manifest}" "${_hip_dependencies_runtime}")
list(APPEND _hip_copy_commands COMMAND "${CMAKE_COMMAND}"
    "-DLFS_HIP_RUNTIME_MANIFEST=${_hip_runtime_manifest}"
    "-DLFS_HIP_RUNTIME_DEST=${CMAKE_BINARY_DIR}"
    -P "${CMAKE_CURRENT_LIST_DIR}/StageHipRuntime.cmake")
install(FILES ${_hip_dependencies_runtime} DESTINATION "${CMAKE_INSTALL_BINDIR}")
# Retain the SDK's own notices without distributing its development payload.
file(GLOB_RECURSE _hip_sdk_notices LIST_DIRECTORIES false
    "${LFS_HIP_ROOT}/share/doc/*LICENSE*"
    "${LFS_HIP_ROOT}/share/doc/*COPYING*"
    "${LFS_HIP_ROOT}/share/doc/*NOTICE*")
foreach(_notice IN LISTS _hip_sdk_notices)
    file(RELATIVE_PATH _notice_relative "${LFS_HIP_ROOT}/share/doc" "${_notice}")
    cmake_path(GET _notice_relative PARENT_PATH _notice_directory)
    install(FILES "${_notice}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/licenses/rocm/${_notice_directory}")
endforeach()
add_custom_target(lfs_hip_package ALL ${_hip_copy_commands} DEPENDS ${_hip_dependencies} VERBATIM)
add_dependencies(${PROJECT_NAME} lfs_hip_package)
