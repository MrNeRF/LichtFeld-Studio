/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/material.hpp"
#include "core/tensor.hpp"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <vector>

namespace lfs::core {

    struct TextureImage {
        std::vector<uint8_t> pixels;
        int width = 0;
        int height = 0;
        int channels = 0;
    };

    struct Submesh {
        size_t start_index = 0;
        size_t index_count = 0;
        size_t material_index = 0;
    };

    struct LFS_CORE_API MeshData {
        Tensor vertices;  // [V, 3] Float32
        Tensor normals;   // [V, 3] Float32
        Tensor tangents;  // [V, 4] Float32 (xyz + handedness w)
        Tensor texcoords; // [V, 2] Float32
        Tensor colors;    // [V, 4] Float32
        Tensor indices;   // [F, 3] Int32

        std::vector<Material> materials;
        std::vector<Submesh> submeshes;
        std::vector<TextureImage> texture_images;
        std::atomic<uint32_t> generation_{0};

        MeshData() = default;

        MeshData(Tensor verts, Tensor idx)
            : vertices(std::move(verts)),
              indices(std::move(idx)) {
            assert(vertices.ndim() == 2 && vertices.shape()[1] == 3);
            assert(vertices.dtype() == DataType::Float32);
            assert(indices.ndim() == 2 && indices.shape()[1] == 3);
            assert(indices.dtype() == DataType::Int32);
        }

        MeshData(const MeshData&) = delete;
        MeshData& operator=(const MeshData&) = delete;

        MeshData(MeshData&& o) noexcept
            : vertices(std::move(o.vertices)),
              normals(std::move(o.normals)),
              tangents(std::move(o.tangents)),
              texcoords(std::move(o.texcoords)),
              colors(std::move(o.colors)),
              indices(std::move(o.indices)),
              materials(std::move(o.materials)),
              submeshes(std::move(o.submeshes)),
              texture_images(std::move(o.texture_images)),
              generation_(o.generation_.load(std::memory_order_relaxed)) {}

        MeshData& operator=(MeshData&& o) noexcept {
            if (this != &o) {
                vertices = std::move(o.vertices);
                normals = std::move(o.normals);
                tangents = std::move(o.tangents);
                texcoords = std::move(o.texcoords);
                colors = std::move(o.colors);
                indices = std::move(o.indices);
                materials = std::move(o.materials);
                submeshes = std::move(o.submeshes);
                texture_images = std::move(o.texture_images);
                generation_.store(o.generation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
            return *this;
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

        uint32_t generation() const { return generation_.load(std::memory_order_relaxed); }
        void mark_dirty() { generation_.fetch_add(1, std::memory_order_relaxed); }

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
            m.texture_images = texture_images;
            m.generation_.store(generation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return m;
        }

        void compute_normals();
    };

} // namespace lfs::core
