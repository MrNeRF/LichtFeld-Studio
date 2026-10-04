/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "mesh_mask.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include <stdexcept>
namespace lfs::training {
    namespace {
        auto mesh_ops() {
            const auto* ops = training_ops(core::default_gpu_backend()).masks;
            if (!ops || !ops->mesh_coverage)
                throw std::runtime_error("Evaluation mesh coverage is unavailable on the selected backend");
            return ops->mesh_coverage;
        }
    } // namespace
    core::Tensor rasterize_mesh_coverage(const core::Tensor& vertices, const core::Tensor& indices,
                                         const MeshMaskCamera& camera, float z_near, void*) {
        return mesh_ops()(vertices, indices, camera, {}, nullptr, z_near);
    }
    core::Tensor rasterize_mesh_coverage(const core::Tensor& vertices, const core::Tensor& indices,
                                         const MeshMaskCamera& camera, const core::Tensor& samples,
                                         const core::UndistortParams& distortion, float z_near, void*) {
        return mesh_ops()(vertices, indices, camera, samples, &distortion, z_near);
    }
} // namespace lfs::training
