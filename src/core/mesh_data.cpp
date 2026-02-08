/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <cassert>

namespace lfs::core {

    using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;

    void MeshData::compute_normals() {
        assert(vertices.is_valid() && vertices.ndim() == 2 && vertices.shape()[1] == 3);
        assert(indices.is_valid() && indices.ndim() == 2 && indices.shape()[1] == 3);

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
            mesh.add_face(vhandles[iacc(i, 0)], vhandles[iacc(i, 1)], vhandles[iacc(i, 2)]);
        }

        mesh.update_normals();

        normals = Tensor::empty({nv, 3}, Device::CPU, DataType::Float32);
        auto nacc = normals.accessor<float, 2>();
        for (int64_t i = 0; i < nv; ++i) {
            auto n = mesh.normal(vhandles[i]);
            nacc(i, 0) = n[0];
            nacc(i, 1) = n[1];
            nacc(i, 2) = n[2];
        }

        if (vertices.device() == Device::CUDA) {
            normals = normals.to(Device::CUDA);
        }
    }

} // namespace lfs::core
