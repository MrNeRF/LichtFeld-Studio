/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#include "io.hpp"
#include "mesh_types.hpp"

#include <nanobind/stl/string.h>

namespace OM = OpenMesh;

namespace lfs::python::openmesh_bindings {

    template <class Mesh>
    void def_read_mesh(nb::module_& m, const char* _name) {
        m.def(
            _name,
            [](const std::string& _filename, bool _binary, bool _msb, bool _lsb, bool _swap,
               bool _vertex_normal, bool _vertex_color, bool _vertex_tex_coord,
               bool _halfedge_tex_coord, bool _edge_color, bool _face_normal, bool _face_color,
               bool _face_texture_index, bool _color_alpha, bool _color_float,
               bool _vertex_status, bool _halfedge_status, bool _edge_status, bool _face_status) {
                Mesh mesh;
                OM::IO::Options options;

                if (_binary)
                    options += OM::IO::Options::Binary;
                if (_msb)
                    options += OM::IO::Options::MSB;
                if (_lsb)
                    options += OM::IO::Options::LSB;
                if (_swap)
                    options += OM::IO::Options::Swap;

                if (_vertex_normal) {
                    options += OM::IO::Options::VertexNormal;
                    mesh.request_vertex_normals();
                }
                if (_vertex_color) {
                    options += OM::IO::Options::VertexColor;
                    mesh.request_vertex_colors();
                }
                if (_vertex_tex_coord) {
                    options += OM::IO::Options::VertexTexCoord;
                    mesh.request_vertex_texcoords1D();
                    mesh.request_vertex_texcoords2D();
                    mesh.request_vertex_texcoords3D();
                }
                if (_halfedge_tex_coord) {
                    options += OM::IO::Options::FaceTexCoord;
                    mesh.request_halfedge_texcoords1D();
                    mesh.request_halfedge_texcoords2D();
                    mesh.request_halfedge_texcoords3D();
                }
                if (_edge_color) {
                    options += OM::IO::Options::EdgeColor;
                    mesh.request_edge_colors();
                }
                if (_face_normal) {
                    options += OM::IO::Options::FaceNormal;
                    mesh.request_face_normals();
                }
                if (_face_color) {
                    options += OM::IO::Options::FaceColor;
                    mesh.request_face_colors();
                }
                if (_face_texture_index) {
                    mesh.request_face_texture_index();
                }
                if (_vertex_status) {
                    mesh.request_vertex_status();
                }
                if (_halfedge_status) {
                    mesh.request_halfedge_status();
                }
                if (_edge_status) {
                    mesh.request_edge_status();
                }
                if (_face_status) {
                    mesh.request_face_status();
                }
                if (_face_status || _halfedge_status || _edge_status || _vertex_status) {
                    options += OM::IO::Options::Status;
                }

                if (_color_alpha)
                    options += OM::IO::Options::ColorAlpha;
                if (_color_float)
                    options += OM::IO::Options::ColorFloat;

                if (!OM::IO::read_mesh(mesh, _filename, options)) {
                    throw std::runtime_error("File could not be read: " + _filename);
                }
                if (_vertex_normal && !options.vertex_has_normal()) {
                    throw std::runtime_error("Vertex normals could not be read.");
                }
                if (_vertex_color && !options.vertex_has_color()) {
                    throw std::runtime_error("Vertex colors could not be read.");
                }
                if (_vertex_tex_coord && !options.vertex_has_texcoord()) {
                    throw std::runtime_error("Vertex texcoords could not be read.");
                }
                if (_edge_color && !options.edge_has_color()) {
                    throw std::runtime_error("Edge colors could not be read.");
                }
                if (_face_normal && !options.face_has_normal()) {
                    throw std::runtime_error("Face normals could not be read.");
                }
                if (_face_color && !options.face_has_color()) {
                    throw std::runtime_error("Face colors could not be read.");
                }
                if (_halfedge_tex_coord && !options.face_has_texcoord()) {
                    throw std::runtime_error("Halfedge texcoords could not be read.");
                }

                return mesh;
            },
            nb::arg("filename"), nb::arg("binary") = false, nb::arg("msb") = false,
            nb::arg("lsb") = false, nb::arg("swap") = false,
            nb::arg("vertex_normal") = false, nb::arg("vertex_color") = false,
            nb::arg("vertex_tex_coord") = false, nb::arg("halfedge_tex_coord") = false,
            nb::arg("edge_color") = false, nb::arg("face_normal") = false,
            nb::arg("face_color") = false, nb::arg("face_texture_index") = false,
            nb::arg("color_alpha") = false, nb::arg("color_float") = false,
            nb::arg("vertex_status") = false, nb::arg("halfedge_status") = false,
            nb::arg("edge_status") = false, nb::arg("face_status") = false);
    }

    template <class Mesh>
    void def_write_mesh(nb::module_& m) {
        m.def(
            "write_mesh",
            [](const std::string& _filename, const Mesh& _mesh, bool _binary, bool _msb,
               bool _lsb, bool _swap, bool _vertex_normal, bool _vertex_color,
               bool _vertex_tex_coord, bool _halfedge_tex_coord, bool _edge_color,
               bool _face_normal, bool _face_color, bool _color_alpha, bool _color_float,
               bool _status) {
                OM::IO::Options options;

                if (_binary)
                    options += OM::IO::Options::Binary;
                if (_msb)
                    options += OM::IO::Options::MSB;
                if (_lsb)
                    options += OM::IO::Options::LSB;
                if (_swap)
                    options += OM::IO::Options::Swap;

                if (_vertex_normal)
                    options += OM::IO::Options::VertexNormal;
                if (_vertex_color)
                    options += OM::IO::Options::VertexColor;
                if (_vertex_tex_coord)
                    options += OM::IO::Options::VertexTexCoord;
                if (_halfedge_tex_coord)
                    options += OM::IO::Options::FaceTexCoord;
                if (_edge_color)
                    options += OM::IO::Options::EdgeColor;
                if (_face_normal)
                    options += OM::IO::Options::FaceNormal;
                if (_face_color)
                    options += OM::IO::Options::FaceColor;

                if (_color_alpha)
                    options += OM::IO::Options::ColorAlpha;
                if (_color_float)
                    options += OM::IO::Options::ColorFloat;

                if (_status)
                    options += OM::IO::Options::Status;

                if (!OM::IO::write_mesh(_mesh, _filename, options)) {
                    throw std::runtime_error("File could not be written: " + _filename);
                }
            },
            nb::arg("filename"), nb::arg("mesh"), nb::arg("binary") = false,
            nb::arg("msb") = false, nb::arg("lsb") = false, nb::arg("swap") = false,
            nb::arg("vertex_normal") = false, nb::arg("vertex_color") = false,
            nb::arg("vertex_tex_coord") = false, nb::arg("halfedge_tex_coord") = false,
            nb::arg("edge_color") = false, nb::arg("face_normal") = false,
            nb::arg("face_color") = false, nb::arg("color_alpha") = false,
            nb::arg("color_float") = false, nb::arg("status") = false);
    }

    void expose_io(nb::module_& m) {
        def_read_mesh<TriMesh>(m, "read_trimesh");
        def_read_mesh<PolyMesh>(m, "read_polymesh");
        def_write_mesh<TriMesh>(m);
        def_write_mesh<PolyMesh>(m);
    }

} // namespace lfs::python::openmesh_bindings
