/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "flip.hpp"
#include "core/assert.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_execution.hpp"
#include "lfs/training/ops/registry.hpp"
#include <cmath>

namespace lfs::training {
    core::Tensor flip_error_map(const core::Tensor& reference, const core::Tensor& test, float pixels_per_degree) {
        LFS_ASSERT_MSG(reference.device() == core::Device::GPU && reference.dtype() == core::DataType::Float32 && reference.ndim() == 3 && reference.shape()[0] == 3, "FLIP reference must be a GPU Float32 [3,H,W] tensor");
        LFS_ASSERT_MSG(test.shape() == reference.shape() && test.device() == core::Device::GPU && test.dtype() == core::DataType::Float32 && core::gpu_backend_of(test) == core::gpu_backend_of(reference), "FLIP test image must match the reference");
        LFS_ASSERT_MSG(std::isfinite(pixels_per_degree) && pixels_per_degree > 0, "FLIP pixels per degree must be positive and finite");
        const core::TensorExecutionTarget::Scope scope(reference.execution_target());
        const auto* ops = training_ops(*core::gpu_backend_of(reference)).training_image;
        LFS_ASSERT_MSG(ops && ops->flip_error_map, "FLIP is unavailable for this tensor backend");
        return ops->flip_error_map(reference, test, pixels_per_degree);
    }
    core::Tensor flip_error_image(const core::Tensor& error_map) {
        LFS_ASSERT_MSG(error_map.device() == core::Device::GPU && error_map.dtype() == core::DataType::Float32 && error_map.ndim() == 2, "FLIP error image needs a GPU Float32 [H,W] error map");
        const core::TensorExecutionTarget::Scope scope(error_map.execution_target());
        const auto* ops = training_ops(*core::gpu_backend_of(error_map)).training_image;
        LFS_ASSERT_MSG(ops && ops->flip_error_image, "FLIP is unavailable for this tensor backend");
        return ops->flip_error_image(error_map);
    }
} // namespace lfs::training
