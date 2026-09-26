/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"

void expression_native_atan2(const float* x, const float* y, float* output, size_t count) {
    lfs::core::tensor_ops::launch_binary_op_generic(x, y, output, count, lfs::core::ops::atan2_op{}, nullptr);
}
