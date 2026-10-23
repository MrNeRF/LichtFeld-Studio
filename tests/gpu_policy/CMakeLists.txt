# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(GpuPolicyCpuContracts LANGUAGES CXX)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
get_filename_component(LFS_POLICY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
enable_testing()
option(LFS_CPU_FETCH_GLM "Fetch the pinned header-only GLM dependency if absent" OFF)
option(LFS_CPU_PYTHON_UI_TESTS "Run synthetic Python preferences/viewer tests" OFF)
find_path(GLM_INCLUDE_DIR glm/glm.hpp HINTS "${LFS_CPU_DEPENDENCY_INCLUDE}")
if(NOT GLM_INCLUDE_DIR AND LFS_CPU_FETCH_GLM)
    include(FetchContent)
    set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
    set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(policy_glm
        URL https://github.com/g-truc/glm/archive/1.0.3.tar.gz
        URL_HASH SHA512=0a490f0c79cd4a8ba54f37358f8917cef961dab9e61417c84ae0959c61bc860e5b83f4fb7f27169fb3d08eef1d84131bddde23d60876922310205c901b1273aa
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    FetchContent_MakeAvailable(policy_glm)
    set(GLM_INCLUDE_DIR "${policy_glm_SOURCE_DIR}")
endif()
if(NOT EXISTS "${GLM_INCLUDE_DIR}/glm/glm.hpp")
    message(FATAL_ERROR "Provide LFS_CPU_DEPENDENCY_INCLUDE or explicitly enable LFS_CPU_FETCH_GLM")
endif()
foreach(flavor IN ITEMS base hip)
    set(target gpu_policy_cpu_${flavor})
    add_executable(${target} policy_contract.cpp)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_compile_definitions(${target} PRIVATE LFS_STATIC_BUILD LFS_HAS_CUDA=0)
    if(flavor STREQUAL "hip")
        target_compile_definitions(${target} PRIVATE LFS_TENSOR_HIP)
    endif()
    target_include_directories(${target} PRIVATE "${LFS_POLICY_ROOT}/src"
        "${LFS_POLICY_ROOT}/src/core/include" "${GLM_INCLUDE_DIR}")
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "cpu;configuration" TIMEOUT 30)
endforeach()
add_test(NAME gpu_policy_windows_dependencies COMMAND "${CMAKE_COMMAND}"
    -P "${LFS_POLICY_ROOT}/tests/cmake/windows_system_library_exclusions.cmake")
set_tests_properties(gpu_policy_windows_dependencies PROPERTIES LABELS "cpu;configuration" TIMEOUT 30)
if(LFS_CPU_PYTHON_UI_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME gpu_policy_python_ui COMMAND "${Python3_EXECUTABLE}" -m pytest -q
        "${LFS_POLICY_ROOT}/tests/python/test_gpu_preferences.py"
        "${LFS_POLICY_ROOT}/tests/python/test_preferences_panel.py"
        "${LFS_POLICY_ROOT}/tests/python/test_rendering_panel_simplify.py"
        "${LFS_POLICY_ROOT}/tests/python/test_rendering_panel_regressions.py"
        "${LFS_POLICY_ROOT}/tests/python/test_keymap_bindings.py")
    set_tests_properties(gpu_policy_python_ui PROPERTIES LABELS "cpu;configuration;python" TIMEOUT 120
        ENVIRONMENT "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1;LFS_TEST_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/no-native-runtime")
endif()
