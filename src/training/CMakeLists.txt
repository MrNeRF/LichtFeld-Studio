# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

if(LFS_HAS_CUDA)
# Add fastlfs backend
add_subdirectory(rasterization/fastgs)

# Add LibTorch-free gsplat backend
add_subdirectory(rasterization/gsplat)

# LibTorch-free CUDA kernels
set(LFS_TRAINING_KERNEL_SOURCES
    kernels/loss_tensor_contract.cpp
    kernels/camera_loss_heatmap.cu
    kernels/ssim.cu
    kernels/l1_loss.cu
    kernels/ssim_reduction.cu
    kernels/regularization.cu
    kernels/bilateral_grid_forward.cu
    kernels/bilateral_grid_backward.cu
    kernels/bilateral_grid_tv.cu
    kernels/ppisp.cu
    kernels/pruning_kernels.cu
    kernels/mcmc_kernels.cu
    kernels/mrnf_kernels.cu
    kernels/densification_kernels.cu
    kernels/morton_reorder_kernels.cu
    kernels/grad_alpha.cu
    ops/geometry_cuda.cpp
    kernels/depth_loss.cu
    kernels/normal_loss.cu
    kernels/normal_consistency_loss.cu
    kernels/roi_weight_map.cu
    kernels/image_kernels.cu
    kernels/mask_preprocess.cu
    kernels/sparsity_optimizer_kernels.cu
)

# Create CUDA kernels library
add_library(lfs_training_kernels STATIC ${LFS_TRAINING_KERNEL_SOURCES})

target_include_directories(lfs_training_kernels
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(lfs_training_kernels
    PUBLIC
        lfs_core         # For lfs::core::Tensor (tensor merged into lfs_core)
        CUDA::cudart
        lfs_core_cuda    # q16 encode/decode kernels
)

set_target_properties(lfs_training_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

# CUDA compiler options for lfs_training_kernels
target_compile_options(lfs_training_kernels PRIVATE
    # CUDA device code + MSVC host compiler flags (Windows only)
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g --extended-lambda --expt-relaxed-constexpr -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>

    # CUDA device code for non-Windows
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g --extended-lambda --expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr>
)

if(MSVC)
    target_compile_definitions(lfs_training_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()
endif()

# Shared depth-anchor fit + sidecar cache (reused by the preprocess CLI)
add_subdirectory(depth_anchor)

# Optimizer sources
set(TRAINING_NEW_SOURCES
    optimizer/adam_optimizer.cpp
    sh_value_codec.cpp
    sh_value_storage.cpp
    morton_reorder.cpp
    live_model_mutation_guard.cpp
    optimizer/scheduler.cpp
    losses/regularization.cpp
    ops/registry.cpp
    ops/requirements.cpp
    ops/fast_services.cpp
    ops/gsplat_services.cpp
    losses/mask_loss.cpp
    strategies/strategy_utils.cpp
    strategies/mcmc.cpp
    strategies/mrnf.cpp
    strategies/strategy_factory.cpp
    strategies/improved_gs_plus.cpp
    components/config_serialization.cpp
    components/bilateral_grid.cpp
    components/ppisp.cpp
    components/ppisp_controller_pool.cpp
    components/ppisp_file.cpp
    components/sparsity_optimizer.cpp
    control/command_api.cpp
    metrics/eval_mask.cpp
    metrics/metrics.cpp
        checkpoint.cpp
        project_snapshot_chapters.cpp
        training_snapshot_service.cpp
        progress.cpp
        training_cropbox_mask.cpp
        trainer.cpp
        normal_auto_generate.cpp
        vram_ledger.cpp
        perf_bench.cpp
    training_setup.cpp
)
if(LFS_HAS_CUDA)
    list(APPEND TRAINING_NEW_SOURCES
        ops/table_cuda.cpp
        ops/session_cuda.cpp
        ops/photometric_cuda.cpp
        ops/gsplat_cuda.cpp
        ops/adam_cuda.cpp
        ops/lpips_cuda.cpp
        ops/bilateral_cuda.cpp
        ops/training_image_cuda.cpp
        ops/ppisp_cuda.cpp
        ops/mcmc_cuda.cpp
        ops/mrnf_cuda.cpp
        ops/morton_cuda.cpp
        ops/masks_cuda.cpp
        ops/extra_loss_cuda.cpp
        ops/sh_cuda.cpp
        ops/refine_cuda.cpp
        rasterization/fast_rasterizer_cuda.cpp
        rasterization/gsplat_rasterizer_cuda.cpp
        kernels/ppisp_controller.cu
    )
else()
    list(APPEND TRAINING_NEW_SOURCES
        ops/cuda_table_stub.cpp
    )
endif()

# Metal trainer kernels are embedded as MSL and compiled at runtime, so the
# build needs no Metal toolchain. common.metal comes first.
if(APPLE)
    set(LFS_TRAINING_METAL_KERNELS
        kernels/metal/common.metal
        kernels/metal/joint_adam.metal
        kernels/metal/sh_storage.metal
        kernels/metal/training_image.metal
        kernels/metal/adam.metal
        kernels/metal/sh.metal
        kernels/metal/morton.metal
        kernels/metal/fast_raster.metal
        kernels/metal/fast_backward.metal
        kernels/metal/gsplat.metal
        kernels/metal/photometric.metal
        kernels/metal/masks.metal
        kernels/metal/extra_loss.metal
        kernels/metal/geometry.metal
        kernels/metal/mrnf.metal
        kernels/metal/refine.metal
        kernels/metal/mcmc.metal
        kernels/metal/ppisp.metal
        kernels/metal/bilateral.metal
        kernels/metal/lpips.metal
    )
    set(_lfs_training_metal_cpp ${CMAKE_CURRENT_BINARY_DIR}/lfs_training_metal_kernels.cpp)
    list(TRANSFORM LFS_TRAINING_METAL_KERNELS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/
         OUTPUT_VARIABLE _lfs_training_metal_inputs)
    string(REPLACE ";" "$<SEMICOLON>" _lfs_training_metal_inputs "${_lfs_training_metal_inputs}")
    set(_lfs_embed_script ${CMAKE_SOURCE_DIR}/src/core/tensor/backend/metal/embed_kernels.cmake)
    add_custom_command(
        OUTPUT ${_lfs_training_metal_cpp}
        COMMAND ${CMAKE_COMMAND}
            "-DINPUTS=${_lfs_training_metal_inputs}"
            -DOUTPUT=${_lfs_training_metal_cpp}
            -DNAMESPACE=lfs::training::metal
            -DSYMBOL=kTrainingKernelSource
            -P ${_lfs_embed_script}
        DEPENDS ${LFS_TRAINING_METAL_KERNELS} ${_lfs_embed_script}
        COMMENT "Embedding Metal trainer kernels"
        VERBATIM)
    list(APPEND TRAINING_NEW_SOURCES
        ${_lfs_training_metal_cpp}
        ops/metal_kernels.cpp
        ops/table_metal.cpp
        ops/session_metal.cpp
        ops/training_image_metal.cpp
        ops/adam_metal.cpp
        ops/sh_metal.cpp
        ops/morton_metal.cpp
        ops/fast_metal.cpp
        ops/gsplat_metal.cpp
        ops/photometric_metal.cpp
        ops/masks_metal.cpp
        ops/extra_loss_metal.cpp
        ops/geometry_metal.cpp
        ops/mrnf_metal.cpp
        ops/refine_metal.cpp
        ops/mcmc_metal.cpp
        ops/bilateral_metal.cpp
        ops/ppisp_metal.cpp
        ops/lpips_metal.cpp
    )
else()
    list(APPEND TRAINING_NEW_SOURCES
        ops/metal_table_stub.cpp
    )
endif()

# lfs_training - Training module with optimizer
add_library(lfs_training STATIC ${TRAINING_NEW_SOURCES})

target_include_directories(lfs_training
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src                   # For old training components (TODO: port)
        ${CMAKE_SOURCE_DIR}/src/training          # For old training components (TODO: port)
        ${CMAKE_SOURCE_DIR}/src/training/fastgs  # For adam_api.h (old)
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/fastgs/rasterization/include  # For rasterization_api.h
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/fastgs/optimizer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/fastgs/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization  # For gsplat headers
)
# indicators.hpp is an implementation detail of progress.cpp.
target_include_directories(lfs_training SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>
)

# Link dependencies
target_link_libraries(lfs_training
    PUBLIC
        lfs_core                # Depends on new core module (SplatData, Tensor)
        lfs_diagnostics         # Live VRAM diagnostic accounting
        lfs_io                  # For CacheLoader, exporter, pipelined loader
        lfs_event_bridge        # Shared event bridge for cross-module singletons
        lfs_depth_anchor        # Shared depth-anchor fit + sidecar cache
    PRIVATE
        lfs_preprocessing       # MoGe-2 normal auto-generate
        lfs_python_utils
)
if(LFS_HAS_CUDA)
    target_link_libraries(lfs_training PUBLIC
        lfs_training_kernels
        fastlfs_backend
        gsplat_backend_lfs
    )
endif()

# Set C++ standard
set_target_properties(lfs_training PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "+------------------------------------------------------------------+")
message(STATUS "|  lfs_training - Training Module with Optimizer & Losses          |")
message(STATUS "+------------------------------------------------------------------+")
message(STATUS "  - Contains: AdamOptimizer, LR Schedulers, Losses (LibTorch-free API)")
message(STATUS "  - Namespace: lfs::training")
message(STATUS "  - Uses: CUDA kernels from gs_training_kernels, fastgs_backend")
