/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/device_fault.hpp"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace device_fault_test {

    // Launch a kernel where every thread attempts first-fault record.
    // Used by first-fault-wins tests (deterministic: all threads OOB).
    [[nodiscard]] cudaError_t launch_many_oob_record(
        lfs::core::DeviceFaultRecord* fault,
        std::uint32_t op_id,
        std::int64_t value,
        std::int64_t bound,
        bool trap_after_record,
        int blocks,
        int threads_per_block,
        cudaStream_t stream);

} // namespace device_fault_test
