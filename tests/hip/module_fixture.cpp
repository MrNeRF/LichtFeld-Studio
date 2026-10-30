/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <cstdint>
#ifdef _WIN32
#define HIP_FIXTURE_EXPORT __declspec(dllexport)
#else
#define HIP_FIXTURE_EXPORT __attribute__((visibility("default")))
#endif
// Intentionally incomplete: ABI mismatch must precede other symbol lookups.
extern "C" HIP_FIXTURE_EXPORT uint32_t lfs_hip_runtime_abi() { return HIP_FIXTURE_ABI; }
