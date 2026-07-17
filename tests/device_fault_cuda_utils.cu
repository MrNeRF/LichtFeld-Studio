/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 6C-P3 positive nvcc TU (Ruling 1): includes the FOUR allowed headers
// and exercises DeviceFaultRecord layout + device_fault_try_record_first.
// Also provides a many-OOB-threads kernel for first-fault-wins unit tests.

#include "core/device_fault.hpp"
#include "core/cuda_error.hpp"
#include "core/error_codes.hpp"
#include "core/source_site.hpp"

#include <cstdint>
#include <cuda_runtime.h>

// Host-side ABI static_asserts already live in device_fault.hpp; repeating
// the size/offset contract here forces the nvcc TU to re-check them.
static_assert(sizeof(lfs::core::DeviceFaultRecord) == 32);
static_assert(alignof(lfs::core::DeviceFaultRecord) == 8);

namespace device_fault_test {
    namespace {

        // Every thread records the same OOB value/bound. First-fault-wins CAS
        // must leave exactly one record (code != 0) regardless of grid size.
        __global__ void many_oob_record_kernel(lfs::core::DeviceFaultRecord* fault,
                                               const std::uint32_t op_id,
                                               const std::int64_t value,
                                               const std::int64_t bound,
                                               const bool trap_after_record) {
            const std::uint64_t thread_id =
                (static_cast<std::uint64_t>(blockIdx.x) << 32) |
                static_cast<std::uint64_t>(threadIdx.x);
            lfs::core::device_fault_try_record_first(
                fault, op_id, value, bound, thread_id, trap_after_record);
        }

    } // namespace

    cudaError_t launch_many_oob_record(lfs::core::DeviceFaultRecord* fault,
                                       const std::uint32_t op_id,
                                       const std::int64_t value,
                                       const std::int64_t bound,
                                       const bool trap_after_record,
                                       const int blocks,
                                       const int threads_per_block,
                                       const cudaStream_t stream) {
        many_oob_record_kernel<<<blocks, threads_per_block, 0, stream>>>(
            fault, op_id, value, bound, trap_after_record);
        return cudaGetLastError();
    }

} // namespace device_fault_test
