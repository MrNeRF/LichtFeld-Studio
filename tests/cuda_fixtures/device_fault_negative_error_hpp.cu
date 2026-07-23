/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 6C-P3 NEGATIVE nvcc TU fixture (Ruling 1 / §1.11).
//
// This translation unit deliberately includes a host-only Error header that is
// FORBIDDEN in CUDA-language TUs. Compiling it under nvcc MUST fail
// (core/error.hpp #errors under __CUDACC__). It must NOT be added to any
// production or test target source list; DeviceFaultNvccFixtures enforces
// presence, content, and absence-from-SOURCES.
//
// Allowed nvcc headers (FOUR, Ruling 1 amendment of §7.2.1 item 3):
//   1. core/source_site.hpp
//   2. core/error_codes.hpp
//   3. core/cuda_error.hpp  (CUDA-safe facade; DiagnosticMode lives here)
//   4. core/device_fault.hpp
// Forbidden examples: core/error.hpp, Result/Exception monadic APIs.

#include "core/error.hpp"

// Touch a host-only type so the include cannot be optimized away as unused.
static_assert(sizeof(lfs::Error) > 0, "negative fixture must reference Error");

__global__ void device_fault_negative_should_not_compile() {
    // Intentionally empty — the #include above is the failure under test.
}
