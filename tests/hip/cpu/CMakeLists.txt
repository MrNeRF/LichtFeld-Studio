# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(HipCpuContracts LANGUAGES CXX)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
get_filename_component(LFS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
enable_testing()
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_path(GLM_INCLUDE_DIR glm/glm.hpp HINTS "${LFS_CPU_DEPENDENCY_INCLUDE}" REQUIRED)
add_executable(hip_external_access_cpu ../external_access_contract.cpp
    "${LFS_ROOT}/src/core/tensor/backend/hip/external_access_coordinator.cpp")
add_executable(hip_policy_cpu policy_contract.cpp)
foreach(target hip_external_access_cpu hip_policy_cpu)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_compile_definitions(${target} PRIVATE LFS_STATIC_BUILD LFS_HAS_CUDA=0 LFS_TENSOR_HIP)
    target_include_directories(${target} PRIVATE "${LFS_ROOT}/src" "${LFS_ROOT}/src/core/include"
        "${GLM_INCLUDE_DIR}")
    target_link_libraries(${target} PRIVATE Threads::Threads)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "hip;cpu" TIMEOUT 30)
endforeach()
foreach(policy hip_architectures hip_offload_compression windows_system_library_exclusions)
    add_test(NAME ${policy} COMMAND "${CMAKE_COMMAND}" "-DTEST_SDK=${TEST_SDK}"
        -P "${LFS_ROOT}/tests/cmake/${policy}.cmake")
    set_tests_properties(${policy} PROPERTIES LABELS "hip;cpu" TIMEOUT 30)
endforeach()
add_test(NAME hip_reject_invalid_architecture COMMAND "${CMAKE_COMMAND}"
    -P "${CMAKE_CURRENT_LIST_DIR}/reject_invalid_architecture.cmake")
set_tests_properties(hip_reject_invalid_architecture PROPERTIES LABELS "hip;cpu" TIMEOUT 30)
add_test(NAME hip_generators_cpu COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${LFS_ROOT}/tools" -p test_hip_generators.py)
set_tests_properties(hip_generators_cpu PROPERTIES LABELS "hip;cpu" TIMEOUT 120)
