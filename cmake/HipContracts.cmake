# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
if(NOT LFS_TENSOR_HIP)
    return()
endif()
# Hosted CI compiles the host-side contracts, but never executes GPU tests.
# Local execution is a separate, explicit opt-in and keeps the normal test
# configuration of CUDA/Vulkan/Metal builds unchanged.
add_custom_target(lfs_hip_compile_contracts)
option(LFS_HIP_REGISTER_GPU_TESTS "Register HIP GPU contracts for local CTest execution" OFF)
if(LFS_HIP_REGISTER_GPU_TESTS)
    enable_testing()
endif()
set(_hip_training_module_tests
    lfs_hip_tensor_training_smoke
    lfs_hip_application_fast_contract lfs_hip_application_adam_contract
    lfs_hip_application_mcmc_contract lfs_hip_application_refine_contract
    lfs_hip_application_photo_contract lfs_hip_application_sh_contract
    lfs_hip_application_extra_contract lfs_hip_application_lpips_contract
    lfs_hip_application_image_contract lfs_hip_application_morton_contract
    lfs_hip_shared_viewer_contract
    lfs_hip_application_fast_densification_contract lfs_hip_fast_budget_contract
    lfs_hip_popspa_primitives_contract lfs_hip_popspa_controller_contract)
get_property(_hip_targets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY BUILDSYSTEM_TARGETS)
foreach(_target IN LISTS _hip_targets)
    if(NOT _target MATCHES "^lfs_hip_.*(contract|smoke)$" AND NOT _target STREQUAL "lfs_hip_training_admission")
        continue()
    endif()
    add_dependencies(lfs_hip_compile_contracts ${_target})
    if(NOT LFS_HIP_REGISTER_GPU_TESTS)
        continue()
    endif()
    # These contracts require an explicit model/reference fixture or a fresh
    # output directory. Keep their documented CLI protocols instead of
    # registering commands that fail due to absent data or mutate reused data.
    if(_target MATCHES "(core_moge_model|application_lpips_model|checkpoint_read)_contract$")
        continue()
    endif()
    set(_args)
    if(_target IN_LIST _hip_training_module_tests)
        list(APPEND _args "${LFS_HIP_TRAINING_LIBRARY}")
    elseif(_target STREQUAL "lfs_hip_os_metrics_contract" OR _target STREQUAL "lfs_hip_external_memory_contract")
        list(APPEND _args "${LFS_HIP_RUNTIME_LIBRARY}")
    endif()
    add_test(NAME hip.gpu.${_target} COMMAND ${_target} ${_args})
    set_tests_properties(hip.gpu.${_target} PROPERTIES LABELS "hip;gpu" TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" RUN_SERIAL TRUE)
endforeach()
