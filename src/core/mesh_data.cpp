/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <cassert>
#include <stdexcept>
#include <utility>

namespace lfs::core {

    using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;

    uint64_t MeshData::next_id() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    MeshData::MeshData(Tensor verts, Tensor idx)
        : vertices(std::move(verts)),
          indices(std::move(idx)) {
        validate_indices();
    }

    void MeshData::validate_indices() const {
        if (!vertices.is_valid() || vertices.ndim() != 2 || vertices.shape()[1] != 3 ||
            vertices.dtype() != DataType::Float32) {
            throw std::invalid_argument("MeshData vertices must be a float32 tensor with shape [V, 3]");
        }
        if (!indices.is_valid() || indices.ndim() != 2 || indices.shape()[1] != 3 ||
            indices.dtype() != DataType::Int32) {
            throw std::invalid_argument("MeshData indices must be an int32 tensor with shape [F, 3]");
        }

        const int64_t vertex_count = vertices.shape()[0];
        auto cpu_indices = indices.to(Device::CPU).contiguous();
        auto accessor = cpu_indices.accessor<int32_t, 2>();
        for (int64_t face = 0; face < cpu_indices.shape()[0]; ++face) {
            const int32_t i0 = accessor(face, 0);
            const int32_t i1 = accessor(face, 1);
            const int32_t i2 = accessor(face, 2);
            if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count)
                throw std::invalid_argument("MeshData face index is outside the vertex range");
        }
    }

    void MeshData::compute_normals() {
        validate_indices();

        auto cpu_verts = vertices.to(Device::CPU).contiguous();
        auto cpu_idx = indices.to(Device::CPU).contiguous();
        const int64_t nv = vertex_count();
        const int64_t nf = face_count();

        TriMesh mesh;
        mesh.request_vertex_normals();
        mesh.request_face_normals();

        auto vacc = cpu_verts.accessor<float, 2>();
        std::vector<TriMesh::VertexHandle> vhandles(nv);
        for (int64_t i = 0; i < nv; ++i) {
            vhandles[i] = mesh.add_vertex(TriMesh::Point(vacc(i, 0), vacc(i, 1), vacc(i, 2)));
        }

        auto iacc = cpu_idx.accessor<int32_t, 2>();
        for (int64_t i = 0; i < nf; ++i) {
            const int32_t i0 = iacc(i, 0), i1 = iacc(i, 1), i2 = iacc(i, 2);
            mesh.add_face(vhandles[i0], vhandles[i1], vhandles[i2]);
        }

        mesh.update_normals();

        const size_t n = static_cast<size_t>(nv);
        normals = Tensor::empty({n, 3}, Device::CPU, DataType::Float32);
        auto nacc = normals.accessor<float, 2>();
        for (int64_t i = 0; i < nv; ++i) {
            const auto n = mesh.normal(vhandles[i]);
            nacc(i, 0) = n[0];
            nacc(i, 1) = n[1];
            nacc(i, 2) = n[2];
        }

        if (vertices.device() == Device::CUDA) {
            normals = normals.to(Device::CUDA);
        }

        mark_dirty();
    }

} // namespace lfs::core
