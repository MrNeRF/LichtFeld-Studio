/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct LfsHipPredicate {
    uint32_t struct_bytes, op, dtype, scalar_kind;
    uint64_t count;
    int64_t integer_scalar;
    float float_scalar;
    uint32_t reserved;
} LfsHipPredicate;
/* Integer arithmetic and typed predicates use reserved=1 for a scalar on the left, 0 otherwise. */
LFS_HIP_API int lfs_hip_integer_pointwise(LfsHipContext*, LfsHipView lhs, LfsHipView rhs,
                                          LfsHipView output, const LfsHipPredicate*, LfsHipError*);
LFS_HIP_API int lfs_hip_predicate_typed(LfsHipContext*, LfsHipView lhs, LfsHipView rhs,
                                        LfsHipView output, const LfsHipPredicate*, LfsHipError*);
#ifdef __cplusplus
}
#endif
