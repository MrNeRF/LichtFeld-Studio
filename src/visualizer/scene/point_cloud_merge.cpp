/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_merge.hpp"

#include <vector>

namespace lfs::vis {

    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Tensor on_gpu_float(const Tensor& tensor) {
            const Tensor gpu = tensor.device() == Device::GPU ? tensor : tensor.to(Device::GPU);
            return gpu.to(DataType::Float32).contiguous();
        }
    } // namespace

    Tensor transformPointsToWorld(const Tensor& means, const glm::mat4& world_transform) {
        const size_t count = means.is_valid() ? means.size(0) : 0;
        if (count == 0) {
            return Tensor::zeros({size_t{0}, size_t{3}}, Device::CPU, DataType::Float32);
        }
        // glm is column-major: world_transform[column][row]. Points are rows, so
        // the rotation goes in transposed and the translation broadcasts per row.
        std::vector<float> rotation_transposed;
        rotation_transposed.reserve(9);
        for (int column = 0; column < 3; ++column) {
            for (int row = 0; row < 3; ++row) {
                rotation_transposed.push_back(world_transform[column][row]);
            }
        }
        const std::vector<float> translation{world_transform[3][0], world_transform[3][1], world_transform[3][2]};
        const Tensor rotation = Tensor::from_vector(rotation_transposed, {size_t{3}, size_t{3}}, Device::CPU).to(Device::GPU);
        const Tensor offset = Tensor::from_vector(translation, {size_t{1}, size_t{3}}, Device::CPU).to(Device::GPU);
        return on_gpu_float(means).matmul(rotation).add(offset).cpu().contiguous();
    }

    Tensor pointColorsAsFloat(const Tensor& colors) {
        const size_t count = colors.is_valid() ? colors.size(0) : 0;
        if (count == 0) {
            return Tensor::zeros({size_t{0}, size_t{3}}, Device::CPU, DataType::Float32);
        }
        const Tensor scaled = colors.dtype() == DataType::UInt8 ? on_gpu_float(colors).mul(1.0f / 255.0f) : on_gpu_float(colors);
        return scaled.cpu().contiguous();
    }

} // namespace lfs::vis
