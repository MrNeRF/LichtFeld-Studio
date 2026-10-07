/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/structure.hpp"
#include "core/assert.hpp"
#include "lfs/training/ops/registry.hpp"
#include <algorithm>
#include <stdexcept>
namespace lfs::training::kernels {
    float structure_base_denominator(const lfs::core::Tensor& base_weight, const int height, const int width,
                                     const bool valid_padding) {
        using namespace lfs::core;
        LFS_ASSERT(height > 0 && width > 0);
        float pixels;
        if (base_weight.is_valid()) {
            LFS_ASSERT(base_weight.ndim() == 2);
            LFS_ASSERT(base_weight.shape()[0] == static_cast<size_t>(height) &&
                       base_weight.shape()[1] == static_cast<size_t>(width));
            // Byte masks weight a pixel by mask != 0, as in the photometric weight.
            if (base_weight.dtype() == DataType::UInt8 || base_weight.dtype() == DataType::Bool)
                pixels = static_cast<float>(base_weight.count_nonzero());
            else {
                LFS_ASSERT(base_weight.dtype() == DataType::Float32);
                pixels = base_weight.sum().item<float>();
            }
        } else {
            pixels = static_cast<float>(valid_padding && height > 10 && width > 10
                                            ? (height - 10) * (width - 10)
                                            : height * width);
        }
        return 3.0f * pixels + 1e-8f;
    }

    size_t RidgeWorkspace::band_rows(const size_t height, const size_t width) const {
        LFS_ASSERT(height > 0 && width > 0);
        const size_t budget_rows = band_bytes / (3 * width * sizeof(float));
        const size_t rows = budget_rows > 2 * 10 ? budget_rows - 2 * 10 : 0;
        const size_t aligned = std::max<size_t>(8, rows / 8 * 8);
        const size_t padded_height = (height + 8 - 1) / 8 * 8;
        return std::min(aligned, padded_height);
    }

    void RidgeWorkspace::ensure_size(const size_t band_rows, const size_t width) {
        const lfs::core::TensorShape shape{3, band_rows + 2 * 10, width};
        if (!horizontal.is_valid() || horizontal.shape() != shape)
            horizontal = lfs::core::Tensor::empty(shape, lfs::core::Device::GPU);
        if (!reduction.is_valid())
            reduction = lfs::core::Tensor::empty({3072}, lfs::core::Device::GPU);
    }

    void GradientResidualWorkspace::ensure_allocated() {
        using namespace lfs::core;
        if (partial.is_valid())
            return;
        partial = Tensor::empty({2048}, Device::GPU);
        totals = Tensor::empty({2}, Device::GPU);
        loss = Tensor::empty({1}, Device::GPU);
    }

    namespace {
        const gpu_ops::StructureOps& operations(const core::Tensor& tensor) {
            const auto backend = core::gpu_backend_of(tensor).value_or(core::default_gpu_backend());
            const auto* result = training_ops(backend).structure;
            if (!result)
                throw std::runtime_error(unavailable_training_family(backend, Family::Structure)
                                             .value_or("Structure training ops are unavailable"));
            return *result;
        }
    } // namespace
    void ridge_structure_map(const core::Tensor& image, core::Tensor& output, RidgeWorkspace& workspace) {
        operations(image).ridge(image, output, workspace);
    }
    void structure_photometric_weight(const core::Tensor& structure, const core::Tensor& base,
                                      core::Tensor& output, float gain, bool valid_padding) {
        operations(structure).photometric_weight(structure, base, output, gain, valid_padding);
    }
    void structure_densification_weight(core::Tensor& error, const core::Tensor& structure, float gain) {
        operations(error).densification_weight(error, structure, gain);
    }
    core::Tensor gradient_residual_loss_gradient(const core::Tensor& image, const core::Tensor& target,
                                                 const core::Tensor& mask, core::Tensor& gradient, float weight,
                                                 GradientResidualWorkspace& workspace) {
        if (weight == 0.0f)
            return {};
        return operations(image).gradient_residual(image, target, mask, gradient, weight, workspace);
    }
} // namespace lfs::training::kernels
