/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_mesh.hpp"
#include "openmesh/circulator.hpp"
#include "openmesh/decimater.hpp"
#include "openmesh/handles.hpp"
#include "openmesh/io.hpp"
#include "openmesh/iterator.hpp"
#include "openmesh/mesh.hpp"

#include <cassert>
#include <format>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    using namespace openmesh_bindings;
    namespace OM = OpenMesh;

    PyMeshData PyMeshData::to_device(const std::string& device) const {
        const auto target = (device == "cuda" || device == "gpu") ? core::Device::CUDA : core::Device::CPU;
        return PyMeshData(std::make_shared<core::MeshData>(data_->to(target)));
    }

    std::string PyMeshData::repr() const {
        return std::format("MeshData(vertices={}, faces={}, normals={}, texcoords={})",
                           vertex_count(), face_count(), has_normals(), has_texcoords());
    }

    namespace {

        PyMeshData trimesh_to_mesh_data(const TriMesh& mesh) {
            using core::DataType;
            using core::Device;
            using core::Tensor;

            const auto nv = static_cast<int64_t>(mesh.n_vertices());
            const auto nf = static_cast<int64_t>(mesh.n_faces());

            auto verts = Tensor::empty({size_t(nv), size_t(3)}, Device::CPU, DataType::Float32);
            auto vacc = verts.accessor<float, 2>();
            for (auto vh : mesh.all_vertices()) {
                const auto p = mesh.point(vh);
                vacc(vh.idx(), 0) = static_cast<float>(p[0]);
                vacc(vh.idx(), 1) = static_cast<float>(p[1]);
                vacc(vh.idx(), 2) = static_cast<float>(p[2]);
            }

            auto idx = Tensor::empty({size_t(nf), size_t(3)}, Device::CPU, DataType::Int32);
            auto iacc = idx.accessor<int32_t, 2>();
            for (auto fh : mesh.all_faces()) {
                auto fv = mesh.cfv_iter(fh);
                iacc(fh.idx(), 0) = fv->idx();
                ++fv;
                iacc(fh.idx(), 1) = fv->idx();
                ++fv;
                iacc(fh.idx(), 2) = fv->idx();
            }

            auto data = std::make_shared<core::MeshData>(std::move(verts), std::move(idx));

            if (mesh.has_vertex_normals()) {
                data->normals =
                    Tensor::empty({size_t(nv), size_t(3)}, Device::CPU, DataType::Float32);
                auto nacc = data->normals.accessor<float, 2>();
                for (auto vh : mesh.all_vertices()) {
                    const auto n = mesh.normal(vh);
                    nacc(vh.idx(), 0) = static_cast<float>(n[0]);
                    nacc(vh.idx(), 1) = static_cast<float>(n[1]);
                    nacc(vh.idx(), 2) = static_cast<float>(n[2]);
                }
            }

            if (mesh.has_vertex_texcoords2D()) {
                data->texcoords =
                    Tensor::empty({size_t(nv), size_t(2)}, Device::CPU, DataType::Float32);
                auto tacc = data->texcoords.accessor<float, 2>();
                for (auto vh : mesh.all_vertices()) {
                    const auto t = mesh.texcoord2D(vh);
                    tacc(vh.idx(), 0) = static_cast<float>(t[0]);
                    tacc(vh.idx(), 1) = static_cast<float>(t[1]);
                }
            }

            if (mesh.has_vertex_colors()) {
                data->colors =
                    Tensor::empty({size_t(nv), size_t(4)}, Device::CPU, DataType::Float32);
                auto cacc = data->colors.accessor<float, 2>();
                for (auto vh : mesh.all_vertices()) {
                    const auto c = mesh.color(vh);
                    cacc(vh.idx(), 0) = c[0];
                    cacc(vh.idx(), 1) = c[1];
                    cacc(vh.idx(), 2) = c[2];
                    cacc(vh.idx(), 3) = c[3];
                }
            }

            return PyMeshData(std::move(data));
        }

        TriMesh mesh_data_to_trimesh(const PyMeshData& md) {
            TriMesh mesh;
            const auto& d = *md.data();
            assert(d.vertices.shape()[1] == 3);

            auto cpu_verts = d.vertices.to(core::Device::CPU).contiguous();
            auto vacc = cpu_verts.accessor<float, 2>();
            const int64_t nv = d.vertex_count();

            for (int64_t i = 0; i < nv; ++i) {
                mesh.add_vertex(TriMesh::Point(static_cast<double>(vacc(i, 0)),
                                               static_cast<double>(vacc(i, 1)),
                                               static_cast<double>(vacc(i, 2))));
            }

            if (d.face_count() > 0) {
                assert(d.indices.shape()[1] == 3);
                auto cpu_idx = d.indices.to(core::Device::CPU).contiguous();
                auto iacc = cpu_idx.accessor<int32_t, 2>();
                const int64_t nf = d.face_count();

                for (int64_t i = 0; i < nf; ++i) {
                    assert(iacc(i, 0) >= 0 && iacc(i, 0) < nv);
                    assert(iacc(i, 1) >= 0 && iacc(i, 1) < nv);
                    assert(iacc(i, 2) >= 0 && iacc(i, 2) < nv);
                    mesh.add_face(OM::VertexHandle(iacc(i, 0)), OM::VertexHandle(iacc(i, 1)),
                                  OM::VertexHandle(iacc(i, 2)));
                }
            }

            if (d.has_normals()) {
                mesh.request_vertex_normals();
                auto cpu_norms = d.normals.to(core::Device::CPU).contiguous();
                auto nacc = cpu_norms.accessor<float, 2>();
                for (auto vh : mesh.vertices()) {
                    mesh.set_normal(vh,
                                    TriMesh::Normal(static_cast<double>(nacc(vh.idx(), 0)),
                                                    static_cast<double>(nacc(vh.idx(), 1)),
                                                    static_cast<double>(nacc(vh.idx(), 2))));
                }
            }

            if (d.has_texcoords()) {
                mesh.request_vertex_texcoords2D();
                auto cpu_tc = d.texcoords.to(core::Device::CPU).contiguous();
                auto tacc = cpu_tc.accessor<float, 2>();
                for (auto vh : mesh.vertices()) {
                    mesh.set_texcoord2D(
                        vh, TriMesh::TexCoord2D(static_cast<double>(tacc(vh.idx(), 0)),
                                                static_cast<double>(tacc(vh.idx(), 1))));
                }
            }

            if (d.has_colors()) {
                mesh.request_vertex_colors();
                auto cpu_colors = d.colors.to(core::Device::CPU).contiguous();
                auto cacc = cpu_colors.accessor<float, 2>();
                for (auto vh : mesh.vertices()) {
                    mesh.set_color(vh, TriMesh::Color(cacc(vh.idx(), 0), cacc(vh.idx(), 1),
                                                      cacc(vh.idx(), 2), cacc(vh.idx(), 3)));
                }
            }

            return mesh;
        }

        PyMeshData read_mesh_data(const std::string& path) {
            TriMesh mesh;
            mesh.request_vertex_normals();
            mesh.request_vertex_texcoords2D();
            mesh.request_vertex_colors();

            OM::IO::Options opts;
            opts += OM::IO::Options::VertexNormal;
            opts += OM::IO::Options::VertexTexCoord;
            opts += OM::IO::Options::VertexColor;

            if (!OM::IO::read_mesh(mesh, path, opts)) {
                throw std::runtime_error("Failed to read mesh: " + path);
            }

            if (!opts.check(OM::IO::Options::VertexNormal)) {
                mesh.request_face_normals();
                mesh.update_normals();
                mesh.release_face_normals();
            }

            return trimesh_to_mesh_data(mesh);
        }

        void write_mesh_data(const PyMeshData& md, const std::string& path) {
            auto mesh = mesh_data_to_trimesh(md);
            if (!OM::IO::write_mesh(mesh, path)) {
                throw std::runtime_error("Failed to write mesh: " + path);
            }
        }

    } // namespace

    void register_mesh(nb::module_& m) {
        nb::class_<PyMeshData>(m, "MeshData")
            .def(
                "__init__",
                [](PyMeshData* self, const PyTensor& verts, const PyTensor& idx) {
                    new (self)
                        PyMeshData(std::make_shared<core::MeshData>(verts.tensor(), idx.tensor()));
                },
                nb::arg("vertices"), nb::arg("indices"))
            .def_prop_ro("vertices", &PyMeshData::vertices)
            .def_prop_ro("normals", &PyMeshData::normals)
            .def_prop_ro("tangents", &PyMeshData::tangents)
            .def_prop_ro("texcoords", &PyMeshData::texcoords)
            .def_prop_ro("colors", &PyMeshData::colors)
            .def_prop_ro("indices", &PyMeshData::indices)
            .def_prop_ro("vertex_count", &PyMeshData::vertex_count)
            .def_prop_ro("face_count", &PyMeshData::face_count)
            .def_prop_ro("has_normals", &PyMeshData::has_normals)
            .def_prop_ro("has_tangents", &PyMeshData::has_tangents)
            .def_prop_ro("has_texcoords", &PyMeshData::has_texcoords)
            .def_prop_ro("has_colors", &PyMeshData::has_colors)
            .def("set_vertices", &PyMeshData::set_vertices, nb::arg("tensor"))
            .def("set_normals", &PyMeshData::set_normals, nb::arg("tensor"))
            .def("set_indices", &PyMeshData::set_indices, nb::arg("tensor"))
            .def("set_texcoords", &PyMeshData::set_texcoords, nb::arg("tensor"))
            .def("set_colors", &PyMeshData::set_colors, nb::arg("tensor"))
            .def("compute_normals", &PyMeshData::compute_normals)
            .def("to", &PyMeshData::to_device, nb::arg("device"))
            .def(
                "to_trimesh", [](const PyMeshData& md) { return mesh_data_to_trimesh(md); },
                "Convert to OpenMesh TriMesh for topology operations")
            .def("__repr__", &PyMeshData::repr);

        expose_handles(m);

        expose_mesh<TriMesh>(m, "TriMesh");
        expose_mesh<PolyMesh>(m, "PolyMesh");

        expose_iterator<OM::PolyConnectivity::VertexIter, &OM::ArrayKernel::n_vertices>(
            m, "VertexIter");
        expose_iterator<OM::PolyConnectivity::HalfedgeIter, &OM::ArrayKernel::n_halfedges>(
            m, "HalfedgeIter");
        expose_iterator<OM::PolyConnectivity::EdgeIter, &OM::ArrayKernel::n_edges>(m, "EdgeIter");
        expose_iterator<OM::PolyConnectivity::FaceIter, &OM::ArrayKernel::n_faces>(m, "FaceIter");

        expose_circulator<OM::PolyConnectivity::VertexVertexIter, OM::VertexHandle>(
            m, "VertexVertexIter");
        expose_circulator<OM::PolyConnectivity::VertexIHalfedgeIter, OM::VertexHandle>(
            m, "VertexIHalfedgeIter");
        expose_circulator<OM::PolyConnectivity::VertexOHalfedgeIter, OM::VertexHandle>(
            m, "VertexOHalfedgeIter");
        expose_circulator<OM::PolyConnectivity::VertexEdgeIter, OM::VertexHandle>(
            m, "VertexEdgeIter");
        expose_circulator<OM::PolyConnectivity::VertexFaceIter, OM::VertexHandle>(
            m, "VertexFaceIter");
        expose_circulator<OM::PolyConnectivity::FaceVertexIter, OM::FaceHandle>(
            m, "FaceVertexIter");
        expose_circulator<OM::PolyConnectivity::FaceHalfedgeIter, OM::FaceHandle>(
            m, "FaceHalfedgeIter");
        expose_circulator<OM::PolyConnectivity::FaceEdgeIter, OM::FaceHandle>(m, "FaceEdgeIter");
        expose_circulator<OM::PolyConnectivity::FaceFaceIter, OM::FaceHandle>(m, "FaceFaceIter");
        expose_circulator<OM::PolyConnectivity::HalfedgeLoopIter, OM::HalfedgeHandle>(
            m, "HalfedgeLoopIter");

        expose_io(m);

        expose_decimater<TriMesh>(m, "TriMesh");
        expose_decimater<PolyMesh>(m, "PolyMesh");

        m.def("to_mesh_data", &trimesh_to_mesh_data, nb::arg("trimesh"),
              "Convert TriMesh to tensor-backed MeshData");
        m.def("from_mesh_data", &mesh_data_to_trimesh, nb::arg("mesh_data"),
              "Convert MeshData to TriMesh");

        m.def("read_mesh", &read_mesh_data, nb::arg("path"),
              "Read a mesh file into MeshData (auto-computes normals if missing)");
        m.def("write_mesh", &write_mesh_data, nb::arg("mesh_data"), nb::arg("path"),
              "Write MeshData to a mesh file");
    }

} // namespace lfs::python
