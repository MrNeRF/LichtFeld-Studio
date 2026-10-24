# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(tensor_operation_contracts
    ${CMAKE_CURRENT_LIST_DIR}/test_tensor_contract_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_tensor_operation_contracts.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_tensor_backend_semantics.cpp)
target_compile_features(tensor_operation_contracts PRIVATE cxx_std_23)
# Only the existing low-level semantic edge cases need private access.
set_property(SOURCE ${CMAKE_CURRENT_LIST_DIR}/test_tensor_backend_semantics.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS LFS_TENSOR_PRIVATE_ACCESS)
target_include_directories(tensor_operation_contracts PRIVATE
    ${LFS_CORE_ABI_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(tensor_operation_contracts PRIVATE lfs_core GTest::gtest)

set(_tensor_cpu_filter "Backends/TensorOperationContracts.*/CPU:Backends/TensorBackendSemantics.*/CPU")
add_test(NAME TensorOperationContracts.CPU
    COMMAND tensor_operation_contracts "--gtest_filter=${_tensor_cpu_filter}")
set_tests_properties(TensorOperationContracts.CPU PROPERTIES LABELS "cpu;tensor" TIMEOUT 120)
set(_tensor_contract_tests TensorOperationContracts.CPU)

set(_tensor_contract_backends)
if(LFS_HAS_CUDA)
    list(APPEND _tensor_contract_backends CUDA)
endif()
if(LFS_TENSOR_VULKAN)
    list(APPEND _tensor_contract_backends Vulkan)
endif()
if(LFS_TENSOR_METAL)
    list(APPEND _tensor_contract_backends Metal)
endif()
foreach(_backend IN LISTS _tensor_contract_backends)
    add_test(NAME TensorOperationContracts.${_backend}
        COMMAND tensor_operation_contracts "--require-tensor-backend=${_backend}"
        "--gtest_filter=Backends/TensorOperationContracts.*/${_backend}:Backends/TensorBackendSemantics.*/${_backend}")
    set_tests_properties(TensorOperationContracts.${_backend} PROPERTIES
        LABELS "gpu;tensor" TIMEOUT 180)
    list(APPEND _tensor_contract_tests TensorOperationContracts.${_backend})
    add_test(NAME TensorOperationContracts.Runner.${_backend}
        COMMAND ${CMAKE_COMMAND} "-DRUNNER=$<TARGET_FILE:tensor_operation_contracts>"
        "-DBACKEND=${_backend}" -DCASE=required-backend
        -P ${CMAKE_CURRENT_LIST_DIR}/cmake/test_tensor_contract_runner.cmake)
    set_tests_properties(TensorOperationContracts.Runner.${_backend} PROPERTIES
        LABELS "gpu;tensor;runner" TIMEOUT 30)
    list(APPEND _tensor_contract_tests TensorOperationContracts.Runner.${_backend})
endforeach()

set(_tensor_listing_backend "")
if(_tensor_contract_backends)
    list(GET _tensor_contract_backends 0 _tensor_listing_backend)
endif()
add_test(NAME TensorOperationContracts.Runner
    COMMAND ${CMAKE_COMMAND} "-DRUNNER=$<TARGET_FILE:tensor_operation_contracts>"
    "-DBACKEND=${_tensor_listing_backend}"
    -P ${CMAKE_CURRENT_LIST_DIR}/cmake/test_tensor_contract_runner.cmake)
set_tests_properties(TensorOperationContracts.Runner PROPERTIES LABELS "cpu;tensor;runner" TIMEOUT 120)
list(APPEND _tensor_contract_tests TensorOperationContracts.Runner)

if(WIN32)
    # Test EXEs need not share Core's output directory. Resolve both Release
    # and Debug dependency locations through the targets, not a hardcoded path.
    set_tests_properties(${_tensor_contract_tests} PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:GTest::gtest>;PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_core>")
endif()
