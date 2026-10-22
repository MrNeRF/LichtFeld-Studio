/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../readback_buffer.hpp"
#include "hip_backend_ops.hpp"
#include "movement_ops.hpp"
#include <core/tensor.hpp>
#include <core/tensor_export.hpp>
#include <stdexcept>
namespace lfs::core::internal {
#ifndef LFS_HIP_EXPRESSION_JIT
    void HipBackendOps::compiled_expression(const ExpressionLaunch& launch, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: compiled_expression"); }
#endif
    void HipBackendOps::nn_attention(StorageRef q, StorageRef k, StorageRef v, std::optional<StorageRef> mask, StorageRef output, const AttentionProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: nn_attention"); }
    void HipBackendOps::nn_conv2d(StorageRef input, StorageRef weight, std::optional<StorageRef> bias, StorageRef output, const ConvProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: nn_conv2d"); }
} // namespace lfs::core::internal
