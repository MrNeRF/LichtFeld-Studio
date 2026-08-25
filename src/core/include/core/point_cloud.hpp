/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace lfs::core {
    // Unified point cloud structure using lfs::core::Tensor
    struct PointCloud {
        Tensor means;  // [N, 3] float32
        Tensor colors; // [N, 3] uint8 or float32

        // For Gaussian point clouds (optional, can be empty for basic point clouds)
        Tensor normals;  // [N, 3] float32
        Tensor sh0;      // [N, 3, 1] float32
        Tensor shN;      // [N, 3, (sh_degree+1)^2-1] float32
        Tensor opacity;  // [N, 1] float32
        Tensor scaling;  // [N, 3] float32
        Tensor rotation; // [N, 4] float32

        // Metadata
        std::vector<std::string> attribute_names;

        // Constructor for basic point cloud (means + colors only)
        PointCloud(Tensor pos, Tensor col)
            : means(std::move(pos)),
              colors(std::move(col)) {}

        // Default constructor
        PointCloud() = default;

        // Check if this is a Gaussian point cloud (has additional attributes)
        bool is_gaussian() const {
            return sh0.numel() > 0;
        }

        // Get number of points
        int64_t size() const {
            return means.numel() > 0 ? means.shape()[0] : 0;
        }

        // Move to device
        PointCloud to(Device device) const {
            PointCloud pc;
            pc.means = means.is_valid() ? means.to(device) : means;
            pc.colors = colors.is_valid() ? colors.to(device) : colors;
            pc.normals = normals.is_valid() ? normals.to(device) : normals;
            pc.sh0 = sh0.is_valid() ? sh0.to(device) : sh0;
            pc.shN = shN.is_valid() ? shN.to(device) : shN;
            pc.opacity = opacity.is_valid() ? opacity.to(device) : opacity;
            pc.scaling = scaling.is_valid() ? scaling.to(device) : scaling;
            pc.rotation = rotation.is_valid() ? rotation.to(device) : rotation;
            pc.attribute_names = attribute_names;
            return pc;
        }

        // Convert colors to float [0,1] if they're uint8
        void normalize_colors() {
            if (colors.numel() > 0 && colors.dtype() == DataType::UInt8) {
                colors = colors.to(DataType::Float32) / 255.0f;
            }
        }
    };
} // namespace lfs::core