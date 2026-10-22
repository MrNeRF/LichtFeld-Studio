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
    void HipBackendOps::rasterize_points(const PointRasterProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: rasterize_points"); }
    std::tuple<Tensor, Tensor> HipBackendOps::kmeans_sh(const Tensor& shN, int n_points, int sh_coeffs, int k, int iterations, bool fast, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: kmeans_sh"); }
    void HipBackendOps::assign_sh3(const Tensor& shN, const Tensor& centroids, const Tensor& norms, Tensor& labels, bool fast, bool have_labels, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: assign_sh3"); }
    void HipBackendOps::decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest, std::vector<uint32_t>& idx, std::vector<float>& cost, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: decimate_candidates"); }
    DecimateMerge HipBackendOps::decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest, const std::vector<int>& member_group, const std::vector<uint32_t>& minimum, const std::vector<uint32_t>& members, const std::vector<uint32_t>& offsets, size_t removed, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: decimate_merge"); }
    void HipBackendOps::nn_linear(StorageRef a, StorageRef w, std::optional<StorageRef> bias, std::optional<StorageRef> scale, std::optional<StorageRef> residual, StorageRef output, const LinearProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: nn_linear"); }
    void HipBackendOps::nn_attention(StorageRef q, StorageRef k, StorageRef v, std::optional<StorageRef> mask, StorageRef output, const AttentionProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: nn_attention"); }
    void HipBackendOps::nn_norm(StorageRef input, StorageRef weight, std::optional<StorageRef> bias, StorageRef output, const NormProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: nn_norm"); }
    void HipBackendOps::nn_conv2d(StorageRef input, StorageRef weight, std::optional<StorageRef> bias, StorageRef output, const ConvProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: nn_conv2d"); }
    void HipBackendOps::multinomial(StorageRef weights, StorageRef output, const RandomProgram& program, ExecContext context) { throw std::runtime_error("HIP Tensor operation not implemented: multinomial"); }
} // namespace lfs::core::internal
