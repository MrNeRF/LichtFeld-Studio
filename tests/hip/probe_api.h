/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <stddef.h>

// C-only boundary for an isolated MSVC/CUDA + Clang/HIP coexistence test.
// No HIP/CUDA runtime types or C++ exceptions cross this boundary.
#ifdef __cplusplus
extern "C" {
#endif
typedef int (*LfsHipProbeAffine)(const float* input, float* output, size_t count,
                                 char* error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
