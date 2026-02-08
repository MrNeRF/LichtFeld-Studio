/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "openmesh_bridge.hpp"
#include <algorithm>
#include <cassert>
#include <limits>

namespace lfs::io::mesh {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    lfs::core::MeshData from_openmesh(const TriMesh& mesh) {
        const int64_t nv = static_cast<int64_t>(mesh.n_vertices());
        const int64_t nf = static_cast<int64_t>(mesh.n_faces());
        assert(nv > 0 && nf > 0);
        assert(nv <= std::numeric_limits<int>::max());

        auto vertices = Tensor::empty({static_cast<size_t>(nv), size_t{3}}, Device::CPU, DataType::Float32);
        auto vacc = vertices.accessor<float, 2>();
        for (int64_t i = 0; i < nv; ++i) {
            const auto p = mesh.point(TriMesh::VertexHandle(static_cast<int>(i)));
            vacc(i, 0) = p[0];
            vacc(i, 1) = p[1];
            vacc(i, 2) = p[2];
        }

        auto indices = Tensor::empty({static_cast<size_t>(nf), size_t{3}}, Device::CPU, DataType::Int32);
        auto iacc = indices.accessor<int32_t, 2>();
        int64_t fi = 0;
        for (auto fh : mesh.faces()) {
            int vi = 0;
            for (auto fv : mesh.fv_range(fh)) {
                assert(vi < 3);
                iacc(fi, vi) = fv.idx();
                ++vi;
            }
            assert(vi == 3);
            ++fi;
        }
        assert(fi == nf);

        lfs::core::MeshData result(std::move(vertices), std::move(indices));

        if (mesh.has_vertex_normals()) {
            result.normals = Tensor::empty({static_cast<size_t>(nv), size_t{3}}, Device::CPU, DataType::Float32);
            auto nacc = result.normals.accessor<float, 2>();
            for (int64_t i = 0; i < nv; ++i) {
                const auto n = mesh.normal(TriMesh::VertexHandle(static_cast<int>(i)));
                nacc(i, 0) = n[0];
                nacc(i, 1) = n[1];
                nacc(i, 2) = n[2];
            }
        }

        if (mesh.has_vertex_texcoords2D()) {
            result.texcoords = Tensor::empty({static_cast<size_t>(nv), size_t{2}}, Device::CPU, DataType::Float32);
            auto tacc = result.texcoords.accessor<float, 2>();
            for (int64_t i = 0; i < nv; ++i) {
                const auto tc = mesh.texcoord2D(TriMesh::VertexHandle(static_cast<int>(i)));
                tacc(i, 0) = tc[0];
                tacc(i, 1) = tc[1];
            }
        }

        if (mesh.has_vertex_colors()) {
            result.colors = Tensor::empty({static_cast<size_t>(nv), size_t{4}}, Device::CPU, DataType::Float32);
            auto cacc = result.colors.accessor<float, 2>();
            for (int64_t i = 0; i < nv; ++i) {
                const auto c = mesh.color(TriMesh::VertexHandle(static_cast<int>(i)));
                cacc(i, 0) = static_cast<float>(c[0]) / 255.0f;
                cacc(i, 1) = static_cast<float>(c[1]) / 255.0f;
                cacc(i, 2) = static_cast<float>(c[2]) / 255.0f;
                cacc(i, 3) = 1.0f;
            }
        }

        return result;
    }

    TriMesh to_openmesh(const lfs::core::MeshData& data) {
        assert(data.vertices.is_valid() && data.indices.is_valid());

        auto cpu_verts = data.vertices.to(Device::CPU).contiguous();
        auto cpu_idx = data.indices.to(Device::CPU).contiguous();
        const int64_t nv = data.vertex_count();
        const int64_t nf = data.face_count();
        assert(nv <= std::numeric_limits<int>::max());

        TriMesh mesh;

        const bool has_normals = data.has_normals();
        const bool has_texcoords = data.has_texcoords();
        const bool has_colors = data.has_colors();

        if (has_normals)
            mesh.request_vertex_normals();
        if (has_texcoords)
            mesh.request_vertex_texcoords2D();
        if (has_colors)
            mesh.request_vertex_colors();

        auto vacc = cpu_verts.accessor<float, 2>();
        std::vector<TriMesh::VertexHandle> vhandles(nv);
        for (int64_t i = 0; i < nv; ++i) {
            vhandles[i] = mesh.add_vertex(TriMesh::Point(vacc(i, 0), vacc(i, 1), vacc(i, 2)));
        }

        if (has_normals) {
            auto cpu_normals = data.normals.to(Device::CPU).contiguous();
            auto nacc = cpu_normals.accessor<float, 2>();
            for (int64_t i = 0; i < nv; ++i) {
                mesh.set_normal(vhandles[i], TriMesh::Normal(nacc(i, 0), nacc(i, 1), nacc(i, 2)));
            }
        }

        if (has_texcoords) {
            auto cpu_tc = data.texcoords.to(Device::CPU).contiguous();
            auto tacc = cpu_tc.accessor<float, 2>();
            for (int64_t i = 0; i < nv; ++i) {
                mesh.set_texcoord2D(vhandles[i], TriMesh::TexCoord2D(tacc(i, 0), tacc(i, 1)));
            }
        }

        if (has_colors) {
            auto cpu_colors = data.colors.to(Device::CPU).contiguous();
            auto cacc = cpu_colors.accessor<float, 2>();
            for (int64_t i = 0; i < nv; ++i) {
                mesh.set_color(vhandles[i], TriMesh::Color(
                                                static_cast<unsigned char>(std::clamp(cacc(i, 0) * 255.0f, 0.0f, 255.0f)),
                                                static_cast<unsigned char>(std::clamp(cacc(i, 1) * 255.0f, 0.0f, 255.0f)),
                                                static_cast<unsigned char>(std::clamp(cacc(i, 2) * 255.0f, 0.0f, 255.0f))));
            }
        }

        auto iacc = cpu_idx.accessor<int32_t, 2>();
        for (int64_t i = 0; i < nf; ++i) {
            assert(iacc(i, 0) >= 0 && iacc(i, 0) < nv);
            assert(iacc(i, 1) >= 0 && iacc(i, 1) < nv);
            assert(iacc(i, 2) >= 0 && iacc(i, 2) < nv);
            mesh.add_face(vhandles[iacc(i, 0)], vhandles[iacc(i, 1)], vhandles[iacc(i, 2)]);
        }

        return mesh;
    }

} // namespace lfs::io::mesh
