/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/assert.hpp"
#include "lfs/training/ops/structure.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lfs::training::structure_detail {
    struct Parameters {
        uint64_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0;
        int32_t width = 0, height = 0, first_row = 0, rows = 0;
        uint32_t stride = 0, blocks = 0, image_byte = 0, mask_byte = 0;
        uint32_t radius = 0, first_scale = 0, valid_padding = 0, reserved = 0;
        float gain = 0, sigma_sq = 0;
        float filter_g[21]{}, filter_d1[21]{}, filter_d2[21]{};
    };
    static_assert(offsetof(Parameters, width) == 56 && offsetof(Parameters, gain) == 104);
    static_assert(sizeof(Parameters) == 368);

    template <class Backend>
    struct Implementation {
        using Tensor = core::Tensor;
        using Workspace = kernels::RidgeWorkspace;
        using GradientWorkspace = kernels::GradientResidualWorkspace;
        static void ridge(const Tensor& image, Tensor& output, Workspace& workspace) {
            LFS_ASSERT(image.device() == core::Device::GPU && image.ndim() == 3 && image.shape()[0] == 3);
            LFS_ASSERT(image.is_contiguous() && image.shape()[1] > 0 && image.shape()[2] > 0);
            LFS_ASSERT(image.dtype() == core::DataType::Float32 || image.dtype() == core::DataType::UInt8);
            const auto h = image.shape()[1], w = image.shape()[2];
            LFS_ASSERT((output.shape() == core::TensorShape{h, w}));
            LFS_ASSERT(output.dtype() == core::DataType::Float32 && output.is_contiguous());
            const auto band_rows = workspace.band_rows(h, w);
            workspace.ensure_size(band_rows, w);
            Parameters p;
            p.a = Backend::address(image);
            p.b = Backend::address(workspace.horizontal);
            p.c = Backend::address(output);
            p.d = Backend::address(workspace.reduction);
            p.width = static_cast<int>(w);
            p.height = static_cast<int>(h);
            p.stride = static_cast<uint32_t>(workspace.horizontal.numel() / 3);
            p.image_byte = image.dtype() == core::DataType::UInt8;
            p.first_scale = 1;
            for (double sigma : {0.8, 1.2, 1.8, 2.5}) {
                p.radius = static_cast<uint32_t>(4.0 * sigma + 0.5);
                p.sigma_sq = static_cast<float>(sigma * sigma);
                double sum = 0;
                for (int k = -int(p.radius); k <= int(p.radius); ++k)
                    sum += std::exp(-0.5 * k * k / (sigma * sigma));
                for (int k = -int(p.radius); k <= int(p.radius); ++k) {
                    const double g = std::exp(-0.5 * k * k / (sigma * sigma)) / sum;
                    const int j = k + int(p.radius);
                    p.filter_g[j] = static_cast<float>(g);
                    p.filter_d1[j] = static_cast<float>(k * g / (sigma * sigma));
                    p.filter_d2[j] = static_cast<float>((k * k / (sigma * sigma) - 1.0) * g / (sigma * sigma));
                }
                for (size_t row = 0; row < h; row += band_rows) {
                    p.first_row = static_cast<int>(row);
                    p.rows = static_cast<int>(std::min(band_rows, (h - row + 7) / 8 * 8));
                    Backend::launch(0, p, {&image}, {&workspace.horizontal}, (size_t(p.rows) + 2 * p.radius) * w);
                    Backend::launch(1, p, {&workspace.horizontal, &output}, {&output}, size_t(p.rows) * w);
                }
                p.first_scale = 0;
            }
            p.blocks = static_cast<uint32_t>(std::min((h * w + 255) / 256, size_t{1024}));
            Backend::launch(2, p, {&image, &output}, {&workspace.reduction}, size_t(p.blocks) * 256);
            Backend::launch(3, p, {&workspace.reduction, &output}, {&output}, h * w);
        }
        static void photometric_weight(const Tensor& structure, const Tensor& base, Tensor& output, float gain, bool valid_padding) {
            LFS_ASSERT(structure.device() == core::Device::GPU && structure.ndim() == 2 && structure.dtype() == core::DataType::Float32);
            LFS_ASSERT(structure.is_contiguous() && std::isfinite(gain) && gain >= 0 && gain <= 4);
            LFS_ASSERT(!base.is_valid() || (base.device() == core::Device::GPU && base.shape() == structure.shape() && base.is_contiguous()));
            LFS_ASSERT(!base.is_valid() || base.dtype() == core::DataType::Float32 || base.dtype() == core::DataType::UInt8 || base.dtype() == core::DataType::Bool);
            if (!output.is_valid() || output.shape() != structure.shape())
                output = Tensor::empty(structure.shape(), core::Device::GPU);
            Parameters p;
            p.a = Backend::address(structure);
            p.b = Backend::address(base);
            p.c = Backend::address(output);
            p.height = static_cast<int>(structure.shape()[0]);
            p.width = static_cast<int>(structure.shape()[1]);
            p.gain = gain;
            p.valid_padding = valid_padding;
            p.mask_byte = base.is_valid() && base.dtype() != core::DataType::Float32;
            Backend::launch(4, p, {&structure, &base}, {&output}, structure.numel());
        }
        static void densification_weight(Tensor& error, const Tensor& structure, float gain) {
            LFS_ASSERT(std::isfinite(gain) && gain >= 0);
            if (gain == 0)
                return;
            LFS_ASSERT(error.device() == core::Device::GPU && error.ndim() == 2 && error.dtype() == core::DataType::Float32 && error.is_contiguous());
            LFS_ASSERT(structure.device() == core::Device::GPU && structure.shape() == error.shape() && structure.dtype() == core::DataType::Float32 && structure.is_contiguous());
            Parameters p;
            p.a = Backend::address(error);
            p.b = Backend::address(structure);
            p.height = static_cast<int>(error.shape()[0]);
            p.width = static_cast<int>(error.shape()[1]);
            p.gain = gain;
            Backend::launch(5, p, {&error, &structure}, {&error}, error.numel());
        }
        static Tensor gradient(const Tensor& image, const Tensor& target, const Tensor& mask, Tensor& gradient, float weight, GradientWorkspace& workspace) {
            LFS_ASSERT(image.device() == core::Device::GPU && image.dtype() == core::DataType::Float32);
            LFS_ASSERT(image.ndim() == 3 && image.shape()[0] == 3 && image.is_contiguous());
            LFS_ASSERT(target.shape() == image.shape() && target.device() == core::Device::GPU && target.is_contiguous());
            LFS_ASSERT(target.dtype() == core::DataType::Float32 || target.dtype() == core::DataType::UInt8);
            LFS_ASSERT(gradient.shape() == image.shape() && gradient.device() == core::Device::GPU);
            LFS_ASSERT(gradient.dtype() == core::DataType::Float32 && gradient.is_contiguous());
            LFS_ASSERT(std::isfinite(weight) && weight > 0 && weight <= 8);
            const auto h = image.shape()[1], w = image.shape()[2];
            LFS_ASSERT(h > 0 && w > 0);
            if (mask.is_valid()) {
                LFS_ASSERT((mask.shape() == core::TensorShape{h, w}));
                LFS_ASSERT(mask.device() == core::Device::GPU && mask.is_contiguous());
                LFS_ASSERT(mask.dtype() == core::DataType::Float32 || mask.dtype() == core::DataType::UInt8 || mask.dtype() == core::DataType::Bool);
            }
            workspace.ensure_allocated();
            Parameters p;
            p.a = Backend::address(image);
            p.b = Backend::address(target);
            p.c = Backend::address(mask);
            p.d = Backend::address(workspace.partial);
            p.e = Backend::address(workspace.totals);
            p.f = Backend::address(workspace.loss);
            p.g = Backend::address(gradient);
            p.width = static_cast<int>(w);
            p.height = static_cast<int>(h);
            p.image_byte = target.dtype() == core::DataType::UInt8;
            p.mask_byte = mask.is_valid() && mask.dtype() != core::DataType::Float32;
            p.blocks = static_cast<uint32_t>(std::min((h * w + 255) / 256, size_t{1024}));
            p.gain = weight;
            Backend::launch(6, p, {&image, &target, &mask}, {&workspace.partial}, size_t(p.blocks) * 256);
            Backend::launch(7, p, {&workspace.partial}, {&workspace.totals, &workspace.loss}, 256);
            Backend::launch(8, p, {&image, &target, &mask, &workspace.totals, &gradient}, {&gradient}, ((w + 31) / 32) * ((h + 7) / 8) * 256);
            return workspace.loss;
        }
        static const gpu_ops::StructureOps& table() {
            static const gpu_ops::StructureOps result{ridge, photometric_weight, densification_weight, gradient};
            return result;
        }
    };
} // namespace lfs::training::structure_detail
