/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#ifndef LFS_HIP_LOGICAL_SUBGROUP32_HPP
#define LFS_HIP_LOGICAL_SUBGROUP32_HPP
#include <hip/hip_runtime.h>
#include <cstdint>

namespace lfs::hip::logical32 {
    // CUDA algorithms retain 32-lane tiles, including on a physical wave64.
    // The two halves have independent membership masks and ballot results.
    constexpr uint64_t physical_mask(uint32_t mask, unsigned lane) {
        return uint64_t(mask) << (lane & 32u);
    }
    constexpr uint32_t local_ballot(uint64_t ballot, unsigned lane) {
        return uint32_t(ballot >> (lane & 32u));
    }
    __device__ inline unsigned physical_lane() {
        const unsigned linear = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        return linear & (unsigned(warpSize) - 1u);
    }
    __device__ inline uint64_t members(uint64_t mask) {
        // warpSize is compiler-selected; recent SDKs need not define the
        // older __AMDGCN_WAVEFRONT_SIZE preprocessor macro.
        return warpSize == 64 ? (physical_mask(uint32_t(mask), physical_lane()) & __ballot(true)) : mask;
    }
    template<class T> __device__ inline T shuffle(uint64_t mask, T value, int source, int width = 32) {
        return __shfl_sync(members(mask), value, source, width);
    }
    template<class T> __device__ inline T shuffle_up(uint64_t mask, T value, unsigned delta, int width = 32) {
        return __shfl_up_sync(members(mask), value, delta, width);
    }
    template<class T> __device__ inline T shuffle_down(uint64_t mask, T value, unsigned delta, int width = 32) {
        return __shfl_down_sync(members(mask), value, delta, width);
    }
    template<class T> __device__ inline T shuffle_xor(uint64_t mask, T value, int delta, int width = 32) {
        return __shfl_xor_sync(members(mask), value, delta, width);
    }
    __device__ inline uint64_t ballot(uint64_t mask, int predicate) {
        const auto result = __ballot_sync(members(mask), predicate);
        return warpSize == 64 ? local_ballot(result, physical_lane()) : result;
    }
    __device__ inline void sync(uint64_t mask = 0xffffffffull) {
        __syncwarp(members(mask));
    }
}

#endif // LFS_HIP_LOGICAL_SUBGROUP32_HPP
