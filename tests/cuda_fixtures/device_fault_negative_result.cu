/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 6C-P3 NEGATIVE nvcc TU fixture: monadic Result is forbidden in .cu.
// Not built into any target; nvcc compilation of std::expected/core/error.hpp
// MUST fail under __CUDACC__. DeviceFaultNvccFixtures enforces presence,
// content, and absence-from-SOURCES.

#include "core/error.hpp"

#include <expected>

// Force a Result-like surface that nvcc TUs must not declare.
using ForbiddenResult = std::expected<void, lfs::Error>;
static_assert(sizeof(ForbiddenResult) > 0);

__global__ void device_fault_negative_result_should_not_compile() {}
