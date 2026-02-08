/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/material.hpp"
#include "core/tensor.hpp"
#include <cassert>
#include <utility>
#include <vector>

namespace lfs::core {

    struct LFS_CORE_API MeshData {
        Tensor vertices;  // [V, 3] Float32
        Tensor normals;   // [V, 3] Float32
        Tensor tangents;  // [V, 4] Float32 (xyz + handedness w)
        Tensor texcoords; // [V, 2] Float32
        Tensor colors;    // [V, 4] Float32
        Tensor indices;   // [F, 3] Int32

        std::vector<Material> materials;
        std::vector<std::pair<size_t, size_t>> submeshes; // (start_index, count) per material

        MeshData() = default;

        MeshData(Tensor verts, Tensor idx)
            : vertices(std::move(verts)),
              indices(std::move(idx)) {
            assert(vertices.ndim() == 2 && vertices.shape()[1] == 3);
            assert(indices.ndim() == 2 && indices.shape()[1] == 3);
        }

        int64_t vertex_count() const {
            return vertices.is_valid() ? vertices.shape()[0] : 0;
        }

        int64_t face_count() const {
            return indices.is_valid() ? indices.shape()[0] : 0;
        }

        bool has_normals() const { return normals.is_valid() && normals.numel() > 0; }
        bool has_tangents() const { return tangents.is_valid() && tangents.numel() > 0; }
        bool has_texcoords() const { return texcoords.is_valid() && texcoords.numel() > 0; }
        bool has_colors() const { return colors.is_valid() && colors.numel() > 0; }

        MeshData to(Device device) const {
            MeshData m;
            m.vertices = vertices.is_valid() ? vertices.to(device) : vertices;
            m.normals = normals.is_valid() ? normals.to(device) : normals;
            m.tangents = tangents.is_valid() ? tangents.to(device) : tangents;
            m.texcoords = texcoords.is_valid() ? texcoords.to(device) : texcoords;
            m.colors = colors.is_valid() ? colors.to(device) : colors;
            m.indices = indices.is_valid() ? indices.to(device) : indices;
            m.materials = materials;
            m.submeshes = submeshes;
            return m;
        }

        void compute_normals();
    };

} // namespace lfs::core
