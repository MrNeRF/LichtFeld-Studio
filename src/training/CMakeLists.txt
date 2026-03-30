# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Add fastlfs backend
add_subdirectory(rasterization/fastgs)

# Add LibTorch-free gsplat backend
add_subdirectory(rasterization/gsplat)

# Add igs+ edge-compute rasterizer
add_subdirectory(rasterization/edge_compute)

# LibTorch-free CUDA kernels
set(LFS_TRAINING_KERNEL_SOURCES
    kernels/ssim.cu
    kernels/l1_loss.cu
    kernels/ssim_reduction.cu
    kernels/regularization.cu
    kernels/bilateral_grid_forward.cu
    kernels/bilateral_grid_backward.cu
    kernels/bilateral_grid_tv.cu
    kernels/ppisp.cu
    kernels/mcmc_kernels.cu
    kernels/mrnf_kernels.cu
    kernels/densification_kernels.cu
    kernels/grad_alpha.cu
    kernels/image_kernels.cu
    components/sparsity_optimizer_kernels.cu
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
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:-O0 -g ${LFS_CUDA_DEVICE_DEBUG_FLAGS} -lineinfo --extended-lambda --expt-relaxed-constexpr -Xcompiler=/Od -Xcompiler=/Z7 -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v -Xcompiler=/O2 -Xcompiler=/DNDEBUG -Xcompiler=/utf-8 -Xcompiler=/D_CRT_SECURE_NO_WARNINGS --diag-suppress=27>

    # CUDA device code for non-Windows
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Debug>>:-O0 -g ${LFS_CUDA_DEVICE_DEBUG_FLAGS} -lineinfo --extended-lambda --expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3 -use_fast_math --extended-lambda --expt-relaxed-constexpr --ptxas-options=-v>
)

if(MSVC)
    target_compile_definitions(lfs_training_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
endif()
# Optimizer sources
set(TRAINING_NEW_SOURCES
    optimizer/adam_optimizer.cpp
    optimizer/scheduler.cpp
    losses/regularization.cpp
    losses/photometric_loss.cpp
    strategies/strategy_utils.cpp
    strategies/mcmc.cpp
    strategies/mrnf.cpp
    strategies/strategy_factory.cpp
    strategies/improved_gs_plus.cpp
    rasterization/fast_rasterizer.cpp
    rasterization/gsplat_rasterizer.cpp
    rasterization/edge_rasterizer.cpp
    components/bilateral_grid.cpp
    components/ppisp.cpp
    components/ppisp_controller.cu
    components/ppisp_controller_pool.cu
    components/ppisp_file.cpp
    components/sparsity_optimizer.cpp
    control/command_api.cpp
    metrics/metrics.cpp
    checkpoint.cpp
    trainer.cpp
    training_setup.cpp
)

# lfs_training - Training module with optimizer
add_library(lfs_training STATIC ${TRAINING_NEW_SOURCES})

target_include_directories(lfs_training
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>  # For indicators.hpp (used in progress.hpp)
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src                   # For old training components (TODO: port)
        ${CMAKE_SOURCE_DIR}/src/training          # For old training components (TODO: port)
        ${CMAKE_SOURCE_DIR}/src/training/fastgs  # For adam_api.h (old)
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/fastgs/rasterization/include  # For rasterization_api.h
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/fastgs/optimizer/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/fastgs/utils
	${CMAKE_CURRENT_SOURCE_DIR}/rasterization/edge_compute/rasterization/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization/edge_compute/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterization  # For gsplat headers
)

# Link dependencies
target_link_libraries(lfs_training
    PUBLIC
        lfs_core                # Depends on new core module (SplatData, Tensor)
        lfs_io                  # For CacheLoader, exporter, pipelined loader
        lfs_event_bridge        # Shared event bridge for cross-module singletons
        lfs_training_kernels    # LibTorch-free SSIM kernels
        fastlfs_backend         # For adam_step_raw CUDA kernel (LibTorch-free)
        gsplat_backend_lfs      # LibTorch-free gsplat rasterization backend
	edge_compute_backend    # For igs+ edge score rasterization
    PRIVATE
        lfs_python_utils
)

# Set C++ standard
set_target_properties(lfs_training PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  lfs_training - Training Module with Optimizer & Losses       ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Contains: AdamOptimizer, LR Schedulers, Losses (LibTorch-free API)")
message(STATUS "  • Namespace: lfs::training")
message(STATUS "  • Uses: CUDA kernels from gs_training_kernels, fastgs_backend")
