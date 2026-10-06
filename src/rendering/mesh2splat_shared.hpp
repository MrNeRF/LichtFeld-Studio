/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025-2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Backend-independent halves of Mesh2Splat shared by the Vulkan converter and
// the tensor-program converter: input validation and triangle expansion before
// the GPU pass, SplatData assembly after it.

#include "core/error.hpp"
#include "core/mesh2splat.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace lfs::core {
    struct MeshData;
    class SplatData;
    struct TextureImage;
} // namespace lfs::core

namespace lfs::rendering::mesh2splat_detail {

    struct GaussianVertex {
        glm::vec4 position;
        glm::vec4 color;
        glm::vec4 scale;
        glm::vec4 normal;
        glm::vec4 rotation;
        glm::vec4 pbr;
    };
    static_assert(sizeof(GaussianVertex) == 6 * sizeof(glm::vec4));

    struct PerVertexData {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec4 tangent;
        glm::vec2 uv;
        glm::vec2 normalized_uv;
        glm::vec3 scale;
        glm::vec4 color;
    };
    static_assert(sizeof(PerVertexData) == 21 * sizeof(float));

    struct SubmeshGeometry {
        std::vector<PerVertexData> vertices;
        glm::vec3 bbox_min{std::numeric_limits<float>::max()};
        glm::vec3 bbox_max{std::numeric_limits<float>::lowest()};
        size_t material_index = 0;
    };

    struct PreparedConversion {
        std::vector<SubmeshGeometry> submeshes;
        glm::vec3 global_min{std::numeric_limits<float>::max()};
        glm::vec3 global_max{std::numeric_limits<float>::lowest()};
        float scene_scale = 0.0f;
        uint64_t triangle_count = 0;
        uint32_t output_capacity = 0; // GaussianVertex entries the GPU pass may write
    };

    [[nodiscard]] lfs::Error conversion_error(lfs::ErrorCode code, std::string detail,
                                              core::SourceSite detection = LFS_SOURCE_SITE_CURRENT());

    // Validates the options, reports "Preparing mesh data" and expands the mesh
    // into per-submesh triangle lists with the global bounding box.
    [[nodiscard]] lfs::Result<PreparedConversion> prepare_conversion(
        const core::MeshData& mesh,
        const core::Mesh2SplatOptions& options,
        const core::Mesh2SplatProgressCallback& progress);

    [[nodiscard]] lfs::Result<std::vector<uint8_t>> to_rgba8(const core::TextureImage& img);

    [[nodiscard]] std::unique_ptr<core::SplatData> build_splat_data(const std::vector<GaussianVertex>& data,
                                                                    float scale_multiplier,
                                                                    float scene_scale);

} // namespace lfs::rendering::mesh2splat_detail
