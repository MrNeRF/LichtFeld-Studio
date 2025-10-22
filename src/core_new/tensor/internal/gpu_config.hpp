/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GPU Configuration - Query GPU properties for optimal kernel launch
 * Inspired by llm.c/llmc/cuda_common.h
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core {

    /**
     * @brief GPU configuration and properties
     *
     * Provides cached access to GPU properties for optimal kernel configuration.
     * Singleton pattern ensures properties are queried only once.
     */
    struct GPUConfig {
        int device_id = 0;
        int sm_count = 0;                    // Number of streaming multiprocessors
        int max_threads_per_sm = 0;          // Max threads per SM
        int max_threads_per_block = 0;       // Max threads per block
        int max_shared_memory_per_block = 0; // Shared memory per block (bytes)
        int warp_size = 32;                  // Warp size (always 32 on NVIDIA)
        int cuda_arch = 0;                   // Compute capability (e.g., 800 for SM80)

        /**
         * @brief Get singleton instance with cached GPU properties
         */
        static const GPUConfig& get() {
            static GPUConfig config = []() {
                GPUConfig cfg;
                cudaDeviceProp prop;

                // Query device 0 (can be extended for multi-GPU)
                cudaError_t err = cudaGetDeviceProperties(&prop, 0);
                if (err != cudaSuccess) {
                    // Fallback to safe defaults
                    cfg.sm_count = 108; // Assume A100/H100 for safety
                    cfg.max_threads_per_sm = 2048;
                    cfg.max_threads_per_block = 1024;
                    cfg.max_shared_memory_per_block = 49152;
                    cfg.warp_size = 32;
                    cfg.cuda_arch = 800;
                    return cfg;
                }

                cfg.device_id = 0;
                cfg.sm_count = prop.multiProcessorCount;
                cfg.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
                cfg.max_threads_per_block = prop.maxThreadsPerBlock;
                cfg.max_shared_memory_per_block = static_cast<int>(prop.sharedMemPerBlock);
                cfg.warp_size = prop.warpSize;
                cfg.cuda_arch = prop.major * 100 + prop.minor * 10; // e.g., 800 for SM_80

                return cfg;
            }();
            return config;
        }

        /**
         * @brief Calculate optimal grid size to fill the GPU
         *
         * This ensures we launch exactly enough blocks to saturate all SMs,
         * avoiding tail effects where the last block waits for stragglers.
         *
         * @param block_size Threads per block
         * @return Grid size (number of blocks)
         */
        int optimal_grid_size(int block_size) const {
            // Formula: (threads_per_sm * sm_count) / threads_per_block
            // This fills the GPU with exactly enough blocks
            return (max_threads_per_sm * sm_count) / block_size;
        }

        /**
         * @brief Calculate optimal grid size with manual cap
         *
         * Sometimes we want to cap the grid size to avoid excessive blocks.
         *
         * @param block_size Threads per block
         * @param max_blocks Maximum number of blocks
         * @return Grid size (capped)
         */
        int optimal_grid_size_capped(int block_size, int max_blocks) const {
            int optimal = optimal_grid_size(block_size);
            return (optimal < max_blocks) ? optimal : max_blocks;
        }

        /**
         * @brief Check if we're on A100 or H100 (SM80+)
         *
         * These architectures have 2× the register file and can run
         * 2 blocks of 1024 threads simultaneously.
         */
        bool is_ampere_or_newer() const {
            return cuda_arch >= 800;
        }

        /**
         * @brief Get max blocks per SM for 1024-thread blocks
         *
         * A100/H100 can run 2 blocks of 1024 threads, older GPUs only 1.
         */
        int max_1024_thread_blocks_per_sm() const {
            return is_ampere_or_newer() ? 2 : 1;
        }

        /**
         * @brief Get optimal shared memory size for a given block config
         *
         * @param smem_per_block Requested shared memory per block
         * @return True if the configuration is valid
         */
        bool check_shared_memory(int smem_per_block) const {
            return smem_per_block <= max_shared_memory_per_block;
        }
    };

    /**
     * @brief Global device properties (for use in kernels)
     *
     * This is exported so kernels can access GPU properties without
     * additional function calls.
     */
    inline const GPUConfig& gpu_config = GPUConfig::get();

// ============================================================================
// Compile-Time Constants (for __launch_bounds__)
// ============================================================================

// Maximum blocks per SM for 1024-thread blocks
// This is used in __launch_bounds__ to hint the compiler
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
    constexpr int MAX_1024_THREADS_BLOCKS = 2;
#else
    constexpr int MAX_1024_THREADS_BLOCKS = 1;
#endif

    // Warp size (always 32 for NVIDIA GPUs)
    constexpr int WARP_SIZE = 32;

// ============================================================================
// Helper Macros
// ============================================================================

/**
 * @brief Calculate grid size for a given number of elements
 *
 * This is the classic CEIL_DIV pattern: (N + M - 1) / M
 */
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/**
 * @brief Get optimal grid size for a kernel launch
 *
 * Example:
 *   int grid = OPTIMAL_GRID_SIZE(256);  // 256 threads per block
 *   my_kernel<<<grid, 256>>>(...)
 */
#define OPTIMAL_GRID_SIZE(block_size) \
    (lfs::core::gpu_config.optimal_grid_size(block_size))

/**
 * @brief Get optimal grid size with a cap
 *
 * Example:
 *   int grid = OPTIMAL_GRID_SIZE_CAPPED(256, 2048);  // Max 2048 blocks
 */
#define OPTIMAL_GRID_SIZE_CAPPED(block_size, max_blocks) \
    (lfs::core::gpu_config.optimal_grid_size_capped(block_size, max_blocks))

} // namespace lfs::core
