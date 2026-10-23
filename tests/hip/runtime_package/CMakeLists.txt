# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(HipRuntimePackage LANGUAGES CXX)
if(NOT WIN32)
    message(FATAL_ERROR "This fixture validates Windows PE runtime dependencies")
endif()
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if(NOT EXISTS "${TEST_SDK}/lib/llvm/bin/llvm-objdump.exe")
    message(FATAL_ERROR "Pass TEST_SDK with llvm-objdump")
endif()
set(_sdk "${CMAKE_CURRENT_BINARY_DIR}/sdk")
file(MAKE_DIRECTORY "${_sdk}/bin" "${_sdk}/lib/llvm/bin")
file(CREATE_LINK "${TEST_SDK}/lib/llvm/bin/llvm-objdump.exe"
    "${_sdk}/lib/llvm/bin/llvm-objdump.exe" COPY_ON_ERROR)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dependency.cpp"
    "extern \"C\" __declspec(dllexport) int package_dependency() { return 42; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/runtime.cpp"
    "extern \"C\" __declspec(dllimport) int package_dependency();\nextern \"C\" __declspec(dllexport) int package_runtime() { return package_dependency(); }\n")
add_library(package_dependency SHARED "${CMAKE_CURRENT_BINARY_DIR}/dependency.cpp")
add_library(package_runtime SHARED "${CMAKE_CURRENT_BINARY_DIR}/runtime.cpp")
set_target_properties(package_dependency PROPERTIES OUTPUT_NAME amdocl64 RUNTIME_OUTPUT_DIRECTORY "${_sdk}/bin")
set_target_properties(package_runtime PROPERTIES OUTPUT_NAME amdhip64_fixture RUNTIME_OUTPUT_DIRECTORY "${_sdk}/bin")
target_link_libraries(package_runtime PRIVATE package_dependency)
foreach(component amd_comgr hiprtc_fixture)
    add_library(${component} SHARED "${CMAKE_CURRENT_BINARY_DIR}/dependency.cpp")
    set_target_properties(${component} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${_sdk}/bin")
endforeach()
file(WRITE "${_sdk}/bin/unrelated.dll" "Not a runtime dependency\n")
enable_testing()
add_test(NAME hip_runtime_package COMMAND "${CMAKE_COMMAND}" "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
    "-DTEST_SDK=${_sdk}" -P "${CMAKE_CURRENT_LIST_DIR}/verify.cmake")
