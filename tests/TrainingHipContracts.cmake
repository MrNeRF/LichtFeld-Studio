# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# Isolate registry dispatch and the HIP primitive from the full trainer. The
# non-HIP table providers in the test are inert identity fixtures, not kernels.
add_executable(training_hip_contracts
    ${CMAKE_CURRENT_LIST_DIR}/test_tensor_contract_main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test_training_hip_contracts.cpp
    ${CMAKE_SOURCE_DIR}/src/training/ops/registry.cpp
    ${CMAKE_SOURCE_DIR}/src/training/ops/table_hip.cpp)
target_compile_features(training_hip_contracts PRIVATE cxx_std_23)
target_include_directories(training_hip_contracts PRIVATE
    ${LFS_CORE_ABI_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/training/include)
target_link_libraries(training_hip_contracts PRIVATE lfs_core GTest::gtest)
add_test(NAME TrainingBackendContracts.CPU COMMAND training_hip_contracts
    --gtest_filter=TrainingBackendRegistry.*)
set_tests_properties(TrainingBackendContracts.CPU PROPERTIES LABELS "cpu;training" TIMEOUT 120)
set(_training_hip_contract_tests TrainingBackendContracts.CPU)
if(LFS_TENSOR_HIP)
    target_sources(training_hip_contracts PRIVATE ${CMAKE_SOURCE_DIR}/src/training/ops/extra_loss_hip.cpp)
    add_test(NAME TrainingPrimitiveContracts.HIP COMMAND training_hip_contracts
        --require-tensor-backend=HIP --gtest_filter=Backends/HipTrainingPrimitives.*/HIP)
    set_tests_properties(TrainingPrimitiveContracts.HIP PROPERTIES LABELS "gpu;training;hip" TIMEOUT 180)
    list(APPEND _training_hip_contract_tests TrainingPrimitiveContracts.HIP)
endif()
if(WIN32)
    set_tests_properties(${_training_hip_contract_tests} PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:GTest::gtest>;PATH=path_list_prepend:$<TARGET_FILE_DIR:lfs_core>")
endif()
