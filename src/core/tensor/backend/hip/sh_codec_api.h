/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
typedef struct LfsHipShCodec {
    uint32_t struct_bytes, source_format, destination_format, index_type, scatter, source_rest, destination_rest, match_cpu_rounding;
    size_t source_rows, destination_rows, count, source_offset, destination_offset;
} LfsHipShCodec;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_API int lfs_hip_sh_codec(LfsHipContext*, const LfsHipShCodec*, LfsHipView source, LfsHipView destination,
                                 LfsHipView indices, LfsHipView source_bounds, LfsHipView destination_bounds, LfsHipError*);
#ifdef __cplusplus
}
#endif
