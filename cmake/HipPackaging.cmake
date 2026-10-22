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
    file(GLOB _hip_dependencies_runtime
        "${LFS_HIP_ROOT}/bin/amdhip64*.dll" "${LFS_HIP_ROOT}/bin/amd_comgr.dll"
        "${LFS_HIP_ROOT}/bin/amdocl64.dll" "${LFS_HIP_ROOT}/bin/hiprtc*.dll"
        "${LFS_HIP_ROOT}/bin/rocm_kpack.dll")
else()
    file(GLOB _hip_dependencies_runtime
        "${LFS_HIP_ROOT}/lib*/libamdhip64.so*" "${LFS_HIP_ROOT}/lib*/libamd_comgr.so*"
        "${LFS_HIP_ROOT}/lib*/libhiprtc*.so*" "${LFS_HIP_ROOT}/lib*/libhsa-runtime64.so*"
        "${LFS_HIP_ROOT}/lib*/librocprofiler-register.so*")
endif()
foreach(_dependency IN LISTS _hip_dependencies_runtime)
    list(APPEND _hip_copy_commands COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_dependency}" "${CMAKE_BINARY_DIR}/")
endforeach()
install(FILES ${_hip_dependencies_runtime} DESTINATION "${CMAKE_INSTALL_BINDIR}")
add_custom_target(lfs_hip_package ALL ${_hip_copy_commands} DEPENDS ${_hip_dependencies} VERBATIM)
add_dependencies(${PROJECT_NAME} lfs_hip_package)
