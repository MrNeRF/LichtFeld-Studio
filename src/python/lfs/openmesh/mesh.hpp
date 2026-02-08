/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#pragma once

#include "circulator.hpp"
#include "iterator.hpp"
#include "mesh_types.hpp"
#include "utilities.hpp"

#include <OpenMesh/Core/Geometry/MathDefs.hh>

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace OM = OpenMesh;

namespace lfs::python::openmesh_bindings {

    template <class T>
    using np_array = nb::ndarray<nb::numpy, T, nb::c_contig>;

    template <class Mesh, class OtherMesh>
    void assign_connectivity(Mesh& _self, const OtherMesh& _other) {
        _self.assign_connectivity(_other);
    }

    template <class Mesh, class Iterator, size_t (OM::ArrayKernel::*n_items)() const>
    IteratorWrapperT<Iterator, n_items> get_iterator(Mesh& _self) {
        return IteratorWrapperT<Iterator, n_items>(_self, typename Iterator::value_type(0));
    }

    template <class Mesh, class Iterator, size_t (OM::ArrayKernel::*n_items)() const>
    IteratorWrapperT<Iterator, n_items> get_skipping_iterator(Mesh& _self) {
        return IteratorWrapperT<Iterator, n_items>(_self, typename Iterator::value_type(0), true);
    }

    template <class Mesh, class Circulator, class CenterEntityHandle>
    CirculatorWrapperT<Circulator, CenterEntityHandle> get_circulator(Mesh& _self, CenterEntityHandle _handle) {
        return CirculatorWrapperT<Circulator, CenterEntityHandle>(_self, _handle);
    }

    template <class Mesh>
    void garbage_collection(Mesh& _self, nb::list& _vh_to_update, nb::list& _hh_to_update,
                            nb::list& _fh_to_update, bool _v = true, bool _e = true, bool _f = true) {
        std::vector<OM::VertexHandle*> vh_vector;
        for (auto item : _vh_to_update) {
            if (nb::isinstance<OM::VertexHandle>(item))
                vh_vector.push_back(nb::cast<OM::VertexHandle*>(item));
        }
        std::vector<OM::HalfedgeHandle*> hh_vector;
        for (auto item : _hh_to_update) {
            if (nb::isinstance<OM::HalfedgeHandle>(item))
                hh_vector.push_back(nb::cast<OM::HalfedgeHandle*>(item));
        }
        std::vector<OM::FaceHandle*> fh_vector;
        for (auto item : _fh_to_update) {
            if (nb::isinstance<OM::FaceHandle>(item))
                fh_vector.push_back(nb::cast<OM::FaceHandle*>(item));
        }
        _self.garbage_collection(vh_vector, hh_vector, fh_vector, _v, _e, _f);
    }

    template <class Vector>
    nb::ndarray<nb::numpy, typename Vector::value_type> vec2numpy(const Vector& _vec) {
        using dtype = typename Vector::value_type;
        dtype* data = new dtype[_vec.size()];
        std::copy_n(_vec.data(), _vec.size(), data);
        return make_owned_array(data, {_vec.size()});
    }

    template <class Mesh, class Vector>
    nb::ndarray<nb::numpy, typename Vector::value_type> vec2numpy(Mesh& _mesh, Vector& _vec, size_t _n = 1) {
        using dtype = typename Vector::value_type;
        auto* ptr = const_cast<dtype*>(_vec.data());
        if (_n == 1) {
            return make_array(ptr, {_vec.size()}, nb::cast(_mesh));
        }
        return make_array(ptr, {_n, _vec.size()}, nb::cast(_mesh));
    }

    template <class Mesh>
    nb::ndarray<nb::numpy, float> flt2numpy(Mesh& _mesh, const float& _flt, size_t _n = 1) {
        return make_array(const_cast<float*>(&_flt), {_n}, nb::cast(_mesh));
    }

    template <class Mesh>
    nb::ndarray<nb::numpy, double> flt2numpy(Mesh& _mesh, const double& _flt, size_t _n = 1) {
        return make_array(const_cast<double*>(&_flt), {_n}, nb::cast(_mesh));
    }

    inline nb::ndarray<nb::numpy, int> face_vertex_indices_trimesh(TriMesh& _self) {
        if (_self.n_faces() == 0) {
            int* data = new int[0];
            return make_owned_array(data, {size_t(0), size_t(3)});
        }

        const bool has_status = _self.has_face_status();
        int* indices = new int[_self.n_faces() * 3];

        for (auto fh : _self.all_faces()) {
            if (has_status && _self.status(fh).deleted()) {
                delete[] indices;
                throw std::runtime_error("Mesh has deleted items. Please call garbage_collection() first.");
            }
            auto fv_it = _self.fv_iter(fh);
            indices[fh.idx() * 3 + 0] = fv_it->idx();
            ++fv_it;
            indices[fh.idx() * 3 + 1] = fv_it->idx();
            ++fv_it;
            indices[fh.idx() * 3 + 2] = fv_it->idx();
        }
        return make_owned_array(indices, {_self.n_faces(), size_t(3)});
    }

    struct FuncEdgeVertex {
        static void call(const OM::ArrayKernel& _mesh, OM::EdgeHandle _eh, int* _ptr) {
            const auto heh = _mesh.halfedge_handle(_eh, 0);
            _ptr[0] = _mesh.from_vertex_handle(heh).idx();
            _ptr[1] = _mesh.to_vertex_handle(heh).idx();
        }
    };

    struct FuncEdgeFace {
        static void call(const OM::ArrayKernel& _mesh, OM::EdgeHandle _eh, int* _ptr) {
            const auto heh1 = _mesh.halfedge_handle(_eh, 0);
            const auto heh2 = _mesh.halfedge_handle(_eh, 1);
            _ptr[0] = _mesh.face_handle(heh1).idx();
            _ptr[1] = _mesh.face_handle(heh2).idx();
        }
    };

    struct FuncEdgeHalfedge {
        static void call(const OM::ArrayKernel& _mesh, OM::EdgeHandle _eh, int* _ptr) {
            _ptr[0] = _mesh.halfedge_handle(_eh, 0).idx();
            _ptr[1] = _mesh.halfedge_handle(_eh, 1).idx();
        }
    };

    struct FuncHalfedgeToVertex {
        static void call(const OM::ArrayKernel& _mesh, OM::HalfedgeHandle _heh, int* _ptr) {
            *_ptr = _mesh.to_vertex_handle(_heh).idx();
        }
        static size_t dim() { return 1; }
    };

    struct FuncHalfedgeFromVertex {
        static void call(const OM::ArrayKernel& _mesh, OM::HalfedgeHandle _heh, int* _ptr) {
            *_ptr = _mesh.from_vertex_handle(_heh).idx();
        }
        static size_t dim() { return 1; }
    };

    struct FuncHalfedgeFace {
        static void call(const OM::ArrayKernel& _mesh, OM::HalfedgeHandle _heh, int* _ptr) {
            *_ptr = _mesh.face_handle(_heh).idx();
        }
        static size_t dim() { return 1; }
    };

    struct FuncHalfedgeEdge {
        static void call(const OM::ArrayKernel& _mesh, OM::HalfedgeHandle _heh, int* _ptr) {
            *_ptr = _mesh.edge_handle(_heh).idx();
        }
        static size_t dim() { return 1; }
    };

    struct FuncHalfedgeVertex {
        static void call(const OM::ArrayKernel& _mesh, OM::HalfedgeHandle _heh, int* _ptr) {
            _ptr[0] = _mesh.from_vertex_handle(_heh).idx();
            _ptr[1] = _mesh.to_vertex_handle(_heh).idx();
        }
        static size_t dim() { return 2; }
    };

    template <class Mesh, class CopyFunc>
    nb::ndarray<nb::numpy, int> edge_other_indices(Mesh& _self) {
        if (_self.n_edges() == 0) {
            int* data = new int[0];
            return make_owned_array(data, {size_t(0), size_t(2)});
        }

        const bool has_status = _self.has_edge_status();
        int* indices = new int[_self.n_edges() * 2];

        for (auto eh : _self.all_edges()) {
            if (has_status && _self.status(eh).deleted()) {
                delete[] indices;
                throw std::runtime_error("Mesh has deleted items. Please call garbage_collection() first.");
            }
            CopyFunc::call(_self, eh, &indices[eh.idx() * 2]);
        }
        return make_owned_array(indices, {_self.n_edges(), size_t(2)});
    }

    template <class Mesh, class CopyFunc>
    nb::ndarray<nb::numpy, int> halfedge_other_indices(Mesh& _self) {
        if (_self.n_halfedges() == 0) {
            int* data = new int[0];
            const size_t dim = CopyFunc::dim();
            if (dim == 1)
                return make_owned_array(data, {size_t(0)});
            return make_owned_array(data, {size_t(0), dim});
        }

        const bool has_status = _self.has_halfedge_status();
        const size_t dim = CopyFunc::dim();
        int* indices = new int[_self.n_halfedges() * dim];

        for (auto heh : _self.all_halfedges()) {
            if (has_status && _self.status(heh).deleted()) {
                delete[] indices;
                throw std::runtime_error("Mesh has deleted items. Please call garbage_collection() first.");
            }
            CopyFunc::call(_self, heh, &indices[heh.idx() * dim]);
        }

        if (dim == 1)
            return make_owned_array(indices, {_self.n_halfedges()});
        return make_owned_array(indices, {_self.n_halfedges(), dim});
    }

    template <class Mesh, class Handle, class Circulator>
    nb::ndarray<nb::numpy, int> indices(Mesh& _self) {
        const size_t n = _self.py_n_items(Handle());
        if (n == 0) {
            int* data = new int[0];
            return make_owned_array(data, {size_t(0)});
        }

        const bool has_status = _self.py_has_status(Handle());

        int max_valence = 0;
        for (size_t i = 0; i < n; ++i) {
            Handle hnd(static_cast<int>(i));
            if (has_status && _self.status(hnd).deleted()) {
                throw std::runtime_error("Mesh has deleted items. Please call garbage_collection() first.");
            }
            int valence = 0;
            for (auto it = Circulator(_self, hnd); it.is_valid(); ++it)
                valence++;
            max_valence = std::max(max_valence, valence);
        }

        int* idx_data = new int[n * max_valence];

        for (size_t i = 0; i < n; ++i) {
            int valence = 0;
            for (auto it = Circulator(_self, Handle(static_cast<int>(i))); it.is_valid(); ++it) {
                idx_data[i * max_valence + valence] = it->idx();
                valence++;
            }
            for (int j = valence; j < max_valence; ++j)
                idx_data[i * max_valence + j] = -1;
        }

        return make_owned_array(idx_data, {n, static_cast<size_t>(max_valence)});
    }

    template <class Mesh>
    void add_vertices(Mesh& _self, np_array<typename Mesh::Point::value_type> _points) {
        if (_points.size() == 0)
            return;
        if (_points.ndim() != 2 || _points.shape(1) != 3)
            throw std::runtime_error("Array 'points' must have shape (n, 3)");

        for (size_t i = 0; i < _points.shape(0); ++i) {
            _self.add_vertex(typename Mesh::Point(
                _points.data()[i * 3 + 0],
                _points.data()[i * 3 + 1],
                _points.data()[i * 3 + 2]));
        }
    }

    template <class Mesh>
    void add_faces(Mesh& _self, np_array<int> _faces) {
        if (_self.n_vertices() < 3 || _faces.size() == 0)
            return;
        if (_faces.ndim() != 2 || _faces.shape(1) < 3)
            throw std::runtime_error("Array 'face_vertex_indices' must have shape (n, m) with m > 2");

        const size_t cols = _faces.shape(1);
        for (size_t i = 0; i < _faces.shape(0); ++i) {
            std::vector<OM::VertexHandle> vhandles;
            for (size_t j = 0; j < cols; ++j) {
                int idx = _faces.data()[i * cols + j];
                if (idx >= 0 && idx < static_cast<int>(_self.n_vertices()))
                    vhandles.push_back(OM::VertexHandle(idx));
            }
            if (vhandles.size() >= 3)
                _self.add_face(vhandles);
        }
    }

    template <class Class>
    void expose_type_specific_functions(Class& _class) {
        // Default: no extra methods
    }

    template <>
    inline void expose_type_specific_functions(nb::class_<PolyMesh>& _class) {
        using Scalar = PolyMesh::Scalar;
        using Point = PolyMesh::Point;

        _class
            .def("add_face",
                 [](PolyMesh& _self, OM::VertexHandle _vh0, OM::VertexHandle _vh1,
                    OM::VertexHandle _vh2, OM::VertexHandle _vh3) {
                     return static_cast<OM::FaceHandle>(_self.add_face(_vh0, _vh1, _vh2, _vh3));
                 })

            .def("split",
                 [](PolyMesh& _self, OM::EdgeHandle _eh, np_array<typename Point::value_type> _arr) {
                     _self.split(_eh, Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            .def("split",
                 [](PolyMesh& _self, OM::FaceHandle _fh, np_array<typename Point::value_type> _arr) {
                     _self.split(_fh, Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            .def("insert_edge", &PolyMesh::insert_edge)

            .def("face_vertex_indices", &indices<PolyMesh, OM::FaceHandle, PolyMesh::FaceVertexIter>)
            .def("fv_indices", &indices<PolyMesh, OM::FaceHandle, PolyMesh::FaceVertexIter>)

            .def("calc_face_normal",
                 [](PolyMesh& _self, np_array<typename Point::value_type> _p0,
                    np_array<typename Point::value_type> _p1,
                    np_array<typename Point::value_type> _p2) {
                     const Point p0(_p0.data()[0], _p0.data()[1], _p0.data()[2]);
                     const Point p1(_p1.data()[0], _p1.data()[1], _p1.data()[2]);
                     const Point p2(_p2.data()[0], _p2.data()[1], _p2.data()[2]);
                     return vec2numpy(_self.calc_face_normal(p0, p1, p2));
                 });
    }

    template <>
    inline void expose_type_specific_functions(nb::class_<TriMesh>& _class) {
        using Scalar = TriMesh::Scalar;
        using Point = TriMesh::Point;

        void (TriMesh::*split_copy_eh_vh)(OM::EdgeHandle, OM::VertexHandle) = &TriMesh::split_copy;
        OM::HalfedgeHandle (TriMesh::*vertex_split_vh)(OM::VertexHandle, OM::VertexHandle,
                                                       OM::VertexHandle, OM::VertexHandle) = &TriMesh::vertex_split;

        _class
            .def("split",
                 [](TriMesh& _self, OM::EdgeHandle _eh, np_array<typename Point::value_type> _arr) {
                     return _self.split(_eh, Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            .def("split",
                 [](TriMesh& _self, OM::FaceHandle _fh, np_array<typename Point::value_type> _arr) {
                     return _self.split(_fh, Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            .def("split_copy", split_copy_eh_vh)

            .def("split_copy",
                 [](TriMesh& _self, OM::EdgeHandle _eh, np_array<typename Point::value_type> _arr) {
                     return _self.split_copy(_eh, Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            .def("split_copy",
                 [](TriMesh& _self, OM::FaceHandle _fh, np_array<typename Point::value_type> _arr) {
                     return _self.split_copy(_fh, Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            .def("opposite_vh", &TriMesh::opposite_vh)
            .def("opposite_he_opposite_vh", &TriMesh::opposite_he_opposite_vh)

            .def("vertex_split", vertex_split_vh)

            .def("vertex_split",
                 [](TriMesh& _self, np_array<typename Point::value_type> _arr,
                    OM::VertexHandle _v1, OM::VertexHandle _vl, OM::VertexHandle _vr) {
                     return _self.vertex_split(Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]),
                                               _v1, _vl, _vr);
                 })

            .def("is_flip_ok", &TriMesh::is_flip_ok)
            .def("flip", &TriMesh::flip)

            .def("face_vertex_indices", &face_vertex_indices_trimesh)
            .def("fv_indices", &face_vertex_indices_trimesh);
    }

    template <class Mesh>
    void expose_mesh(nb::module_& m, const char* _name) {
        using Scalar = typename Mesh::Scalar;
        using Point = typename Mesh::Point;
        using Normal = typename Mesh::Normal;
        using Color = typename Mesh::Color;
        using TexCoord1D = typename Mesh::TexCoord1D;
        using TexCoord2D = typename Mesh::TexCoord2D;
        using TexCoord3D = typename Mesh::TexCoord3D;
        using TextureIndex = typename Mesh::TextureIndex;

        // KernelT function pointers
        OM::VertexHandle (Mesh::*vertex_handle_uint)(unsigned int) const = &Mesh::vertex_handle;
        OM::HalfedgeHandle (Mesh::*halfedge_handle_uint)(unsigned int) const = &Mesh::halfedge_handle;
        OM::EdgeHandle (Mesh::*edge_handle_uint)(unsigned int) const = &Mesh::edge_handle;
        OM::FaceHandle (Mesh::*face_handle_uint)(unsigned int) const = &Mesh::face_handle;

        void (Mesh::*garbage_collection_bools)(bool, bool, bool) = &Mesh::garbage_collection;
        void (*garbage_collection_lists_bools)(Mesh&, nb::list&, nb::list&, nb::list&, bool, bool, bool) =
            &garbage_collection;

        OM::HalfedgeHandle (Mesh::*halfedge_handle_vh)(OM::VertexHandle) const = &Mesh::halfedge_handle;
        OM::HalfedgeHandle (Mesh::*halfedge_handle_fh)(OM::FaceHandle) const = &Mesh::halfedge_handle;
        OM::FaceHandle (Mesh::*face_handle_hh)(OM::HalfedgeHandle) const = &Mesh::face_handle;
        OM::HalfedgeHandle (Mesh::*prev_halfedge_handle_hh)(OM::HalfedgeHandle) const = &Mesh::prev_halfedge_handle;
        OM::EdgeHandle (Mesh::*edge_handle_hh)(OM::HalfedgeHandle) const = &Mesh::edge_handle;
        OM::HalfedgeHandle (Mesh::*halfedge_handle_eh_uint)(OM::EdgeHandle, unsigned int) const =
            &Mesh::halfedge_handle;

        void (Mesh::*set_halfedge_handle_vh_hh)(OM::VertexHandle, OM::HalfedgeHandle) =
            &Mesh::set_halfedge_handle;
        void (Mesh::*set_halfedge_handle_fh_hh)(OM::FaceHandle, OM::HalfedgeHandle) =
            &Mesh::set_halfedge_handle;

        OM::FaceHandle (Mesh::*new_face_void)(void) = &Mesh::new_face;
        OM::FaceHandle (Mesh::*new_face_face)(const typename Mesh::Face&) = &Mesh::new_face;

        // Kernel iterators
        IteratorWrapperT<typename Mesh::VertexIter, &Mesh::n_vertices> (*vertices)(Mesh&) = &get_iterator;
        IteratorWrapperT<typename Mesh::HalfedgeIter, &Mesh::n_halfedges> (*halfedges)(Mesh&) = &get_iterator;
        IteratorWrapperT<typename Mesh::EdgeIter, &Mesh::n_edges> (*edges)(Mesh&) = &get_iterator;
        IteratorWrapperT<typename Mesh::FaceIter, &Mesh::n_faces> (*faces)(Mesh&) = &get_iterator;

        IteratorWrapperT<typename Mesh::VertexIter, &Mesh::n_vertices> (*svertices)(Mesh&) =
            &get_skipping_iterator;
        IteratorWrapperT<typename Mesh::HalfedgeIter, &Mesh::n_halfedges> (*shalfedges)(Mesh&) =
            &get_skipping_iterator;
        IteratorWrapperT<typename Mesh::EdgeIter, &Mesh::n_edges> (*sedges)(Mesh&) = &get_skipping_iterator;
        IteratorWrapperT<typename Mesh::FaceIter, &Mesh::n_faces> (*sfaces)(Mesh&) = &get_skipping_iterator;

        // BaseKernel function pointers
        void (Mesh::*copy_all_properties_vh)(OM::VertexHandle, OM::VertexHandle, bool) =
            &Mesh::copy_all_properties;
        void (Mesh::*copy_all_properties_hh)(OM::HalfedgeHandle, OM::HalfedgeHandle, bool) =
            &Mesh::copy_all_properties;
        void (Mesh::*copy_all_properties_eh)(OM::EdgeHandle, OM::EdgeHandle, bool) =
            &Mesh::copy_all_properties;
        void (Mesh::*copy_all_properties_fh)(OM::FaceHandle, OM::FaceHandle, bool) =
            &Mesh::copy_all_properties;

        // PolyConnectivity function pointers
        void (*assign_connectivity_poly)(Mesh&, const PolyMesh&) = &assign_connectivity;
        void (*assign_connectivity_tri)(Mesh&, const TriMesh&) = &assign_connectivity;

        unsigned int (Mesh::*valence_vh)(OM::VertexHandle) const = &Mesh::valence;
        unsigned int (Mesh::*valence_fh)(OM::FaceHandle) const = &Mesh::valence;

        void (Mesh::*triangulate_fh)(OM::FaceHandle) = &Mesh::triangulate;
        void (Mesh::*triangulate_void)() = &Mesh::triangulate;

        // Circulators
        CirculatorWrapperT<typename Mesh::VertexVertexIter, OM::VertexHandle> (*vv)(Mesh&, OM::VertexHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::VertexIHalfedgeIter, OM::VertexHandle> (*vih)(Mesh&, OM::VertexHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::VertexOHalfedgeIter, OM::VertexHandle> (*voh)(Mesh&, OM::VertexHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::VertexEdgeIter, OM::VertexHandle> (*ve)(Mesh&, OM::VertexHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::VertexFaceIter, OM::VertexHandle> (*vf)(Mesh&, OM::VertexHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::FaceVertexIter, OM::FaceHandle> (*fv)(Mesh&, OM::FaceHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::FaceHalfedgeIter, OM::FaceHandle> (*fh)(Mesh&, OM::FaceHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::FaceEdgeIter, OM::FaceHandle> (*fe)(Mesh&, OM::FaceHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::FaceFaceIter, OM::FaceHandle> (*ff)(Mesh&, OM::FaceHandle) =
            &get_circulator;
        CirculatorWrapperT<typename Mesh::HalfedgeLoopIter, OM::HalfedgeHandle> (*hl)(Mesh&, OM::HalfedgeHandle) =
            &get_circulator;

        // Boundary tests
        bool (Mesh::*is_boundary_hh)(OM::HalfedgeHandle) const = &Mesh::is_boundary;
        bool (Mesh::*is_boundary_eh)(OM::EdgeHandle) const = &Mesh::is_boundary;
        bool (Mesh::*is_boundary_vh)(OM::VertexHandle) const = &Mesh::is_boundary;
        bool (Mesh::*is_boundary_fh)(OM::FaceHandle, bool) const = &Mesh::is_boundary;

        // PolyMeshT function pointers
        Scalar (Mesh::*calc_edge_length_eh)(OM::EdgeHandle) const = &Mesh::calc_edge_length;
        Scalar (Mesh::*calc_edge_length_hh)(OM::HalfedgeHandle) const = &Mesh::calc_edge_length;
        Scalar (Mesh::*calc_edge_sqr_length_eh)(OM::EdgeHandle) const = &Mesh::calc_edge_sqr_length;
        Scalar (Mesh::*calc_edge_sqr_length_hh)(OM::HalfedgeHandle) const = &Mesh::calc_edge_sqr_length;
        Scalar (Mesh::*calc_dihedral_angle_fast_hh)(OM::HalfedgeHandle) const = &Mesh::calc_dihedral_angle_fast;
        Scalar (Mesh::*calc_dihedral_angle_fast_eh)(OM::EdgeHandle) const = &Mesh::calc_dihedral_angle_fast;
        Scalar (Mesh::*calc_dihedral_angle_hh)(OM::HalfedgeHandle) const = &Mesh::calc_dihedral_angle;
        Scalar (Mesh::*calc_dihedral_angle_eh)(OM::EdgeHandle) const = &Mesh::calc_dihedral_angle;

        unsigned int (Mesh::*find_feature_edges)(Scalar) = &Mesh::find_feature_edges;

        void (Mesh::*split_fh_vh)(OM::FaceHandle, OM::VertexHandle) = &Mesh::split;
        void (Mesh::*split_eh_vh)(OM::EdgeHandle, OM::VertexHandle) = &Mesh::split;
        void (Mesh::*split_copy_fh_vh)(OM::FaceHandle, OM::VertexHandle) = &Mesh::split_copy;

        // ---- Class definition ----
        nb::class_<Mesh> class_mesh(m, _name);

        class_mesh
            .def(nb::init<>())

            .def(
                "__init__",
                [](Mesh* self, np_array<typename Point::value_type> _points, np_array<int> _faces) {
                    new (self) Mesh();
                    add_vertices(*self, _points);
                    add_faces(*self, _faces);
                },
                nb::arg("points"), nb::arg("face_vertex_indices"))

            .def(
                "__init__",
                [](Mesh* self, np_array<typename Point::value_type> _points) {
                    new (self) Mesh();
                    add_vertices(*self, _points);
                },
                nb::arg("points"))

            // Copy interface
            .def("__copy__", &Mesh::py_copy)
            .def("__deepcopy__", &Mesh::py_deepcopy)

            // KernelT
            .def("reserve", &Mesh::reserve)

            .def("vertex_handle", vertex_handle_uint)
            .def("halfedge_handle", halfedge_handle_uint)
            .def("edge_handle", edge_handle_uint)
            .def("face_handle", face_handle_uint)

            .def("clear", &Mesh::clear)
            .def("clean", &Mesh::clean)
            .def("garbage_collection", garbage_collection_bools,
                 nb::arg("v") = true, nb::arg("e") = true, nb::arg("f") = true)
            .def("garbage_collection", garbage_collection_lists_bools,
                 nb::arg("vh_to_update"), nb::arg("hh_to_update"), nb::arg("fh_to_update"),
                 nb::arg("v") = true, nb::arg("e") = true, nb::arg("f") = true)

            .def("n_vertices", &Mesh::n_vertices)
            .def("n_halfedges", &Mesh::n_halfedges)
            .def("n_edges", &Mesh::n_edges)
            .def("n_faces", &Mesh::n_faces)
            .def("vertices_empty", &Mesh::vertices_empty)
            .def("halfedges_empty", &Mesh::halfedges_empty)
            .def("edges_empty", &Mesh::edges_empty)
            .def("faces_empty", &Mesh::faces_empty)

            .def("halfedge_handle", halfedge_handle_vh)
            .def("set_halfedge_handle", set_halfedge_handle_vh_hh)

            .def("to_vertex_handle", &Mesh::to_vertex_handle)
            .def("from_vertex_handle", &Mesh::from_vertex_handle)
            .def("set_vertex_handle", &Mesh::set_vertex_handle)
            .def("face_handle", face_handle_hh)
            .def("set_face_handle", &Mesh::set_face_handle)
            .def("next_halfedge_handle",
                 [](Mesh& _self, OM::HalfedgeHandle heh) {
                     return static_cast<OM::HalfedgeHandle>(_self.next_halfedge_handle(heh));
                 })
            .def("set_next_halfedge_handle",
                 [](Mesh& _self, OM::HalfedgeHandle heh0, OM::HalfedgeHandle heh1) {
                     _self.set_next_halfedge_handle(heh0, heh1);
                 })
            .def("prev_halfedge_handle", prev_halfedge_handle_hh)
            .def("opposite_halfedge_handle",
                 [](Mesh& _self, OM::HalfedgeHandle heh) {
                     return static_cast<OM::HalfedgeHandle>(_self.opposite_halfedge_handle(heh));
                 })
            .def("ccw_rotated_halfedge_handle",
                 [](Mesh& _self, OM::HalfedgeHandle heh) {
                     return static_cast<OM::HalfedgeHandle>(_self.ccw_rotated_halfedge_handle(heh));
                 })
            .def("cw_rotated_halfedge_handle",
                 [](Mesh& _self, OM::HalfedgeHandle heh) {
                     return static_cast<OM::HalfedgeHandle>(_self.cw_rotated_halfedge_handle(heh));
                 })
            .def("edge_handle", edge_handle_hh)

            .def("halfedge_handle", halfedge_handle_eh_uint)
            .def("halfedge_handle", halfedge_handle_fh)
            .def("set_halfedge_handle", set_halfedge_handle_fh_hh)

            // Status: is_deleted / set_deleted
            .def("is_deleted",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_status())
                         return false;
                     return _self.status(_h).deleted();
                 })
            .def("set_deleted",
                 [](Mesh& _self, OM::VertexHandle _h, bool _val) {
                     if (!_self.has_vertex_status())
                         _self.request_vertex_status();
                     _self.status(_h).set_deleted(_val);
                 })
            .def("is_locked",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_status())
                         return false;
                     return _self.status(_h).locked();
                 })
            .def("set_locked",
                 [](Mesh& _self, OM::VertexHandle _h, bool _val) {
                     if (!_self.has_vertex_status())
                         _self.request_vertex_status();
                     _self.status(_h).set_locked(_val);
                 })
            .def("is_deleted",
                 [](Mesh& _self, OM::HalfedgeHandle _h) {
                     if (!_self.has_halfedge_status())
                         return false;
                     return _self.status(_h).deleted();
                 })
            .def("set_deleted",
                 [](Mesh& _self, OM::HalfedgeHandle _h, bool _val) {
                     if (!_self.has_halfedge_status())
                         _self.request_halfedge_status();
                     _self.status(_h).set_deleted(_val);
                 })
            .def("is_deleted",
                 [](Mesh& _self, OM::EdgeHandle _h) {
                     if (!_self.has_edge_status())
                         return false;
                     return _self.status(_h).deleted();
                 })
            .def("set_deleted",
                 [](Mesh& _self, OM::EdgeHandle _h, bool _val) {
                     if (!_self.has_edge_status())
                         _self.request_edge_status();
                     _self.status(_h).set_deleted(_val);
                 })
            .def("is_deleted",
                 [](Mesh& _self, OM::FaceHandle _h) {
                     if (!_self.has_face_status())
                         return false;
                     return _self.status(_h).deleted();
                 })
            .def("set_deleted",
                 [](Mesh& _self, OM::FaceHandle _h, bool _val) {
                     if (!_self.has_face_status())
                         _self.request_face_status();
                     _self.status(_h).set_deleted(_val);
                 })

            // Request/release/has properties
            .def("request_vertex_normals", &Mesh::request_vertex_normals)
            .def("request_vertex_colors", &Mesh::request_vertex_colors)
            .def("request_vertex_texcoords1D", &Mesh::request_vertex_texcoords1D)
            .def("request_vertex_texcoords2D", &Mesh::request_vertex_texcoords2D)
            .def("request_vertex_texcoords3D", &Mesh::request_vertex_texcoords3D)
            .def("request_halfedge_normals", &Mesh::request_halfedge_normals)
            .def("request_halfedge_colors", &Mesh::request_halfedge_colors)
            .def("request_halfedge_texcoords1D", &Mesh::request_halfedge_texcoords1D)
            .def("request_halfedge_texcoords2D", &Mesh::request_halfedge_texcoords2D)
            .def("request_halfedge_texcoords3D", &Mesh::request_halfedge_texcoords3D)
            .def("request_edge_colors", &Mesh::request_edge_colors)
            .def("request_face_normals", &Mesh::request_face_normals)
            .def("request_face_colors", &Mesh::request_face_colors)
            .def("request_face_texture_index", &Mesh::request_face_texture_index)

            .def("release_vertex_normals", &Mesh::release_vertex_normals)
            .def("release_vertex_colors", &Mesh::release_vertex_colors)
            .def("release_vertex_texcoords1D", &Mesh::release_vertex_texcoords1D)
            .def("release_vertex_texcoords2D", &Mesh::release_vertex_texcoords2D)
            .def("release_vertex_texcoords3D", &Mesh::release_vertex_texcoords3D)
            .def("release_halfedge_normals", &Mesh::release_halfedge_normals)
            .def("release_halfedge_colors", &Mesh::release_halfedge_colors)
            .def("release_halfedge_texcoords1D", &Mesh::release_halfedge_texcoords1D)
            .def("release_halfedge_texcoords2D", &Mesh::release_halfedge_texcoords2D)
            .def("release_halfedge_texcoords3D", &Mesh::release_halfedge_texcoords3D)
            .def("release_edge_colors", &Mesh::release_edge_colors)
            .def("release_face_normals", &Mesh::release_face_normals)
            .def("release_face_colors", &Mesh::release_face_colors)
            .def("release_face_texture_index", &Mesh::release_face_texture_index)

            .def("has_vertex_normals", &Mesh::has_vertex_normals)
            .def("has_vertex_colors", &Mesh::has_vertex_colors)
            .def("has_vertex_texcoords1D", &Mesh::has_vertex_texcoords1D)
            .def("has_vertex_texcoords2D", &Mesh::has_vertex_texcoords2D)
            .def("has_vertex_texcoords3D", &Mesh::has_vertex_texcoords3D)
            .def("has_halfedge_normals", &Mesh::has_halfedge_normals)
            .def("has_halfedge_colors", &Mesh::has_halfedge_colors)
            .def("has_halfedge_texcoords1D", &Mesh::has_halfedge_texcoords1D)
            .def("has_halfedge_texcoords2D", &Mesh::has_halfedge_texcoords2D)
            .def("has_halfedge_texcoords3D", &Mesh::has_halfedge_texcoords3D)
            .def("has_edge_colors", &Mesh::has_edge_colors)
            .def("has_face_normals", &Mesh::has_face_normals)
            .def("has_face_colors", &Mesh::has_face_colors)
            .def("has_face_texture_index", &Mesh::has_face_texture_index)

            .def("new_vertex",
                 [](Mesh& _self) { return static_cast<OM::VertexHandle>(_self.new_vertex()); })
            .def("new_vertex",
                 [](Mesh& _self, np_array<typename Point::value_type> _arr) {
                     return static_cast<OM::VertexHandle>(
                         _self.new_vertex(Point(_arr.data()[0], _arr.data()[1], _arr.data()[2])));
                 })

            .def("new_edge", &Mesh::new_edge)
            .def("new_face", new_face_void)
            .def("new_face", new_face_face)

            .def("vertices", vertices)
            .def("halfedges", halfedges)
            .def("edges", edges)
            .def("faces", faces)

            .def("svertices", svertices)
            .def("shalfedges", shalfedges)
            .def("sedges", sedges)
            .def("sfaces", sfaces)

            .def("texture_index",
                 [](Mesh& _self, OM::FaceHandle _h) {
                     if (!_self.has_face_texture_index())
                         _self.request_face_texture_index();
                     return _self.texture_index(_h);
                 })
            .def("set_texture_index",
                 [](Mesh& _self, OM::FaceHandle _h, TextureIndex _idx) {
                     if (!_self.has_face_texture_index())
                         _self.request_face_texture_index();
                     _self.set_texture_index(_h, _idx);
                 })

            .def("texture_name",
                 [](Mesh& _self, TextureIndex _idx) {
                     OM::MPropHandleT<std::map<TextureIndex, std::string>> prop;
                     if (_self.get_property_handle(prop, "TextureMapping")) {
                         const auto map = _self.property(prop);
                         if (map.count(_idx) == 0) {
                             throw nb::index_error();
                         }
                         return map.at(_idx);
                     }
                     throw std::runtime_error("Mesh has no textures.");
                 })

            .def("set_feature",
                 [](Mesh& _self, OM::EdgeHandle _h, bool val) {
                     if (!_self.has_edge_status())
                         _self.request_edge_status();
                     return _self.status(_h).set_feature(val);
                 })
            .def("feature",
                 [](Mesh& _self, OM::EdgeHandle _h) {
                     if (!_self.has_edge_status())
                         _self.request_edge_status();
                     return _self.status(_h).feature();
                 })
            .def("set_feature",
                 [](Mesh& _self, OM::VertexHandle _h, bool val) {
                     if (!_self.has_vertex_status())
                         _self.request_vertex_status();
                     return _self.status(_h).set_feature(val);
                 })
            .def("feature",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_status())
                         _self.request_vertex_status();
                     return _self.status(_h).feature();
                 })

            // BaseKernel
            .def("copy_all_properties", copy_all_properties_vh,
                 nb::arg("vh_from"), nb::arg("vh_to"), nb::arg("copy_build_in") = false)
            .def("copy_all_properties", copy_all_properties_hh,
                 nb::arg("hh_from"), nb::arg("hh_to"), nb::arg("copy_build_in") = false)
            .def("copy_all_properties", copy_all_properties_eh,
                 nb::arg("eh_from"), nb::arg("eh_to"), nb::arg("copy_build_in") = false)
            .def("copy_all_properties", copy_all_properties_fh,
                 nb::arg("fh_from"), nb::arg("fh_to"), nb::arg("copy_build_in") = false)

            // ArrayKernel
            .def("is_valid_handle", (bool(Mesh::*)(OM::VertexHandle) const) & Mesh::is_valid_handle)
            .def("is_valid_handle", (bool(Mesh::*)(OM::HalfedgeHandle) const) & Mesh::is_valid_handle)
            .def("is_valid_handle", (bool(Mesh::*)(OM::EdgeHandle) const) & Mesh::is_valid_handle)
            .def("is_valid_handle", (bool(Mesh::*)(OM::FaceHandle) const) & Mesh::is_valid_handle)

            .def("delete_isolated_vertices",
                 [](Mesh& _self) {
                     if (!_self.has_vertex_status())
                         _self.request_vertex_status();
                     _self.delete_isolated_vertices();
                 })

            // PolyConnectivity
            .def("assign_connectivity", assign_connectivity_poly)
            .def("assign_connectivity", assign_connectivity_tri)

            .def("add_face",
                 [](Mesh& _self, OM::VertexHandle _vh0, OM::VertexHandle _vh1, OM::VertexHandle _vh2) {
                     return static_cast<OM::FaceHandle>(_self.add_face(_vh0, _vh1, _vh2));
                 })
            .def("add_face",
                 [](Mesh& _self, const std::vector<OM::VertexHandle>& _vhs) {
                     return static_cast<OM::FaceHandle>(_self.add_face(_vhs));
                 })

            .def("opposite_face_handle", &Mesh::opposite_face_handle)
            .def("adjust_outgoing_halfedge", &Mesh::adjust_outgoing_halfedge)
            .def("find_halfedge",
                 [](Mesh& _self, OM::VertexHandle _vh0, OM::VertexHandle _vh1) {
                     return static_cast<OM::HalfedgeHandle>(_self.find_halfedge(_vh0, _vh1));
                 })
            .def("valence", valence_vh)
            .def("valence", valence_fh)
            .def("is_simple_link", &Mesh::is_simple_link)
            .def("is_simply_connected", &Mesh::is_simply_connected)
            .def("triangulate", triangulate_fh)
            .def("triangulate", triangulate_void)
            .def("split_edge", &Mesh::split_edge)
            .def("split_edge_copy", &Mesh::split_edge_copy)

            .def("remove_edge",
                 [](Mesh& _self, OM::EdgeHandle _eh) {
                     if (!_self.has_edge_status())
                         _self.request_edge_status();
                     if (!_self.has_face_status())
                         _self.request_face_status();
                     return _self.remove_edge(_eh);
                 })
            .def("reinsert_edge",
                 [](Mesh& _self, OM::EdgeHandle _eh) {
                     if (!_self.has_edge_status())
                         _self.request_edge_status();
                     if (!_self.has_face_status())
                         _self.request_face_status();
                     _self.reinsert_edge(_eh);
                 })
            .def("is_collapse_ok",
                 [](Mesh& _self, OM::HalfedgeHandle _heh) {
                     if (!_self.has_vertex_status())
                         _self.request_vertex_status();
                     if (!_self.has_halfedge_status())
                         _self.request_halfedge_status();
                     if (!_self.has_edge_status())
                         _self.request_edge_status();
                     if (!_self.has_face_status())
                         _self.request_face_status();
                     return _self.is_collapse_ok(_heh);
                 })
            .def("collapse",
                 [](Mesh& _self, OM::HalfedgeHandle _heh) {
                     if (!_self.has_vertex_status())
                         _self.request_vertex_status();
                     if (!_self.has_halfedge_status())
                         _self.request_halfedge_status();
                     if (!_self.has_edge_status())
                         _self.request_edge_status();
                     if (!_self.has_face_status())
                         _self.request_face_status();
                     _self.collapse(_heh);
                 })

            .def("add_vertex",
                 [](Mesh& _self, np_array<typename Point::value_type> _arr) {
                     return static_cast<OM::VertexHandle>(
                         _self.add_vertex(Point(_arr.data()[0], _arr.data()[1], _arr.data()[2])));
                 })

            .def(
                "delete_vertex",
                [](Mesh& _self, OM::VertexHandle _vh, bool _delete_isolated) {
                    if (!_self.has_vertex_status())
                        _self.request_vertex_status();
                    if (!_self.has_halfedge_status())
                        _self.request_halfedge_status();
                    if (!_self.has_edge_status())
                        _self.request_edge_status();
                    if (!_self.has_face_status())
                        _self.request_face_status();
                    _self.delete_vertex(_vh, _delete_isolated);
                },
                nb::arg("vh"), nb::arg("delete_isolated_vertices") = true)

            .def(
                "delete_edge",
                [](Mesh& _self, OM::EdgeHandle _eh, bool _delete_isolated) {
                    if (!_self.has_vertex_status() && _delete_isolated)
                        _self.request_vertex_status();
                    if (!_self.has_halfedge_status())
                        _self.request_halfedge_status();
                    if (!_self.has_edge_status())
                        _self.request_edge_status();
                    if (!_self.has_face_status())
                        _self.request_face_status();
                    _self.delete_edge(_eh, _delete_isolated);
                },
                nb::arg("eh"), nb::arg("delete_isolated_vertices") = true)

            .def(
                "delete_face",
                [](Mesh& _self, OM::FaceHandle _fh, bool _delete_isolated) {
                    if (!_self.has_vertex_status() && _delete_isolated)
                        _self.request_vertex_status();
                    if (!_self.has_halfedge_status())
                        _self.request_halfedge_status();
                    if (!_self.has_edge_status())
                        _self.request_edge_status();
                    if (!_self.has_face_status())
                        _self.request_face_status();
                    _self.delete_face(_fh, _delete_isolated);
                },
                nb::arg("fh"), nb::arg("delete_isolated_vertices") = true)

            .def("vv", vv)
            .def("vih", vih)
            .def("voh", voh)
            .def("ve", ve)
            .def("vf", vf)
            .def("fv", fv)
            .def("fh", fh)
            .def("fe", fe)
            .def("ff", ff)
            .def("hl", hl)

            .def("is_boundary", is_boundary_hh)
            .def("is_boundary", is_boundary_eh)
            .def("is_boundary", is_boundary_vh)
            .def("is_boundary", is_boundary_fh, nb::arg("fh"), nb::arg("check_vertex") = false)
            .def("is_manifold", &Mesh::is_manifold)

            .def_static("is_triangles", &Mesh::is_triangles)

            .def_ro_static("InvalidVertexHandle", &Mesh::InvalidVertexHandle)
            .def_ro_static("InvalidHalfedgeHandle", &Mesh::InvalidHalfedgeHandle)
            .def_ro_static("InvalidEdgeHandle", &Mesh::InvalidEdgeHandle)
            .def_ro_static("InvalidFaceHandle", &Mesh::InvalidFaceHandle)

            // PolyMeshT
            .def("calc_edge_length", calc_edge_length_eh)
            .def("calc_edge_length", calc_edge_length_hh)
            .def("calc_edge_sqr_length", calc_edge_sqr_length_eh)
            .def("calc_edge_sqr_length", calc_edge_sqr_length_hh)

            .def("calc_sector_angle", &Mesh::calc_sector_angle)
            .def("calc_sector_area", &Mesh::calc_sector_area)

            .def("calc_dihedral_angle_fast", calc_dihedral_angle_fast_hh)
            .def("calc_dihedral_angle_fast", calc_dihedral_angle_fast_eh)
            .def("calc_dihedral_angle", calc_dihedral_angle_hh)
            .def("calc_dihedral_angle", calc_dihedral_angle_eh)

            .def("find_feature_edges", find_feature_edges,
                 nb::arg("angle_tresh") = OM::deg_to_rad(44.0))

            .def("split", split_fh_vh)
            .def("split", split_eh_vh)
            .def("split_copy", split_copy_fh_vh)

            .def("update_normals",
                 [](Mesh& _self) {
                     if (!_self.has_face_normals())
                         _self.request_face_normals();
                     if (!_self.has_halfedge_normals())
                         _self.request_halfedge_normals();
                     if (!_self.has_vertex_normals())
                         _self.request_vertex_normals();
                     _self.update_normals();
                 })
            .def("update_normal",
                 [](Mesh& _self, OM::FaceHandle _fh) {
                     if (!_self.has_face_normals())
                         _self.request_face_normals();
                     _self.update_normal(_fh);
                 })
            .def("update_face_normals",
                 [](Mesh& _self) {
                     if (!_self.has_face_normals())
                         _self.request_face_normals();
                     _self.update_face_normals();
                 })
            .def(
                "update_normal",
                [](Mesh& _self, OM::HalfedgeHandle _hh, double _feature_angle) {
                    if (!_self.has_face_normals()) {
                        _self.request_face_normals();
                        _self.update_face_normals();
                    }
                    if (!_self.has_halfedge_normals())
                        _self.request_halfedge_normals();
                    _self.update_normal(_hh, _feature_angle);
                },
                nb::arg("heh"), nb::arg("feature_angle") = 0.8)
            .def(
                "update_halfedge_normals",
                [](Mesh& _self, double _feature_angle) {
                    if (!_self.has_face_normals()) {
                        _self.request_face_normals();
                        _self.update_face_normals();
                    }
                    if (!_self.has_halfedge_normals())
                        _self.request_halfedge_normals();
                    _self.update_halfedge_normals(_feature_angle);
                },
                nb::arg("feature_angle") = 0.8)
            .def("update_normal",
                 [](Mesh& _self, OM::VertexHandle _vh) {
                     if (!_self.has_face_normals()) {
                         _self.request_face_normals();
                         _self.update_face_normals();
                     }
                     if (!_self.has_vertex_normals())
                         _self.request_vertex_normals();
                     _self.update_normal(_vh);
                 })
            .def("update_vertex_normals",
                 [](Mesh& _self) {
                     if (!_self.has_face_normals()) {
                         _self.request_face_normals();
                         _self.update_face_normals();
                     }
                     if (!_self.has_vertex_normals())
                         _self.request_vertex_normals();
                     _self.update_vertex_normals();
                 })

            .def("is_estimated_feature_edge", &Mesh::is_estimated_feature_edge)
            .def_static("is_polymesh", &Mesh::is_polymesh)
            .def("is_trimesh", &Mesh::is_trimesh)

            // numpy calc_*
            .def("calc_face_normal",
                 [](Mesh& _self, OM::FaceHandle _fh) { return vec2numpy(_self.calc_face_normal(_fh)); })
            .def(
                "calc_halfedge_normal",
                [](Mesh& _self, OM::HalfedgeHandle _heh, double _feature_angle) {
                    if (!_self.has_face_normals()) {
                        _self.request_face_normals();
                        _self.update_face_normals();
                    }
                    return vec2numpy(_self.calc_halfedge_normal(_heh, _feature_angle));
                },
                nb::arg("heh"), nb::arg("feature_angle") = 0.8)
            .def("calc_vertex_normal",
                 [](Mesh& _self, OM::VertexHandle _vh) {
                     if (!_self.has_face_normals()) {
                         _self.request_face_normals();
                         _self.update_face_normals();
                     }
                     return vec2numpy(_self.calc_vertex_normal(_vh));
                 })
            .def("calc_vertex_normal_fast",
                 [](Mesh& _self, OM::VertexHandle _vh) {
                     if (!_self.has_face_normals()) {
                         _self.request_face_normals();
                         _self.update_face_normals();
                     }
                     typename Mesh::Normal n;
                     _self.calc_vertex_normal_fast(_vh, n);
                     return vec2numpy(n);
                 })
            .def("calc_vertex_normal_correct",
                 [](Mesh& _self, OM::VertexHandle _vh) {
                     typename Mesh::Normal n;
                     _self.calc_vertex_normal_correct(_vh, n);
                     return vec2numpy(n);
                 })
            .def("calc_vertex_normal_loop",
                 [](Mesh& _self, OM::VertexHandle _vh) {
                     typename Mesh::Normal n;
                     _self.calc_vertex_normal_loop(_vh, n);
                     return vec2numpy(n);
                 })
            .def("calc_face_centroid",
                 [](Mesh& _self, OM::FaceHandle _fh) { return vec2numpy(_self.calc_face_centroid(_fh)); })
            .def("calc_edge_vector",
                 [](Mesh& _self, OM::EdgeHandle _eh) { return vec2numpy(_self.calc_edge_vector(_eh)); })
            .def("calc_edge_vector",
                 [](Mesh& _self, OM::HalfedgeHandle _heh) { return vec2numpy(_self.calc_edge_vector(_heh)); })
            .def("calc_sector_vectors",
                 [](Mesh& _self, OM::HalfedgeHandle _heh) {
                     typename Mesh::Normal vec0, vec1;
                     _self.calc_sector_vectors(_heh, vec0, vec1);
                     return std::make_tuple(vec2numpy(vec0), vec2numpy(vec1));
                 })
            .def("calc_sector_normal",
                 [](Mesh& _self, OM::HalfedgeHandle _heh) {
                     typename Mesh::Normal n;
                     _self.calc_sector_normal(_heh, n);
                     return vec2numpy(n);
                 })

            // numpy vector getter
            .def("point",
                 [](Mesh& _self, OM::VertexHandle _h) { return vec2numpy(_self, _self.point(_h)); })
            .def("normal",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_normals())
                         _self.request_vertex_normals();
                     return vec2numpy(_self, _self.normal(_h));
                 })
            .def("normal",
                 [](Mesh& _self, OM::HalfedgeHandle _h) {
                     if (!_self.has_halfedge_normals())
                         _self.request_halfedge_normals();
                     return vec2numpy(_self, _self.normal(_h));
                 })
            .def("normal",
                 [](Mesh& _self, OM::FaceHandle _h) {
                     if (!_self.has_face_normals())
                         _self.request_face_normals();
                     return vec2numpy(_self, _self.normal(_h));
                 })

            .def("color",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_colors())
                         _self.request_vertex_colors();
                     return vec2numpy(_self, _self.color(_h));
                 })
            .def("color",
                 [](Mesh& _self, OM::HalfedgeHandle _h) {
                     if (!_self.has_halfedge_colors())
                         _self.request_halfedge_colors();
                     return vec2numpy(_self, _self.color(_h));
                 })
            .def("color",
                 [](Mesh& _self, OM::EdgeHandle _h) {
                     if (!_self.has_edge_colors())
                         _self.request_edge_colors();
                     return vec2numpy(_self, _self.color(_h));
                 })
            .def("color",
                 [](Mesh& _self, OM::FaceHandle _h) {
                     if (!_self.has_face_colors())
                         _self.request_face_colors();
                     return vec2numpy(_self, _self.color(_h));
                 })

            .def("texcoord1D",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_texcoords1D())
                         _self.request_vertex_texcoords1D();
                     return flt2numpy(_self, _self.texcoord1D(_h));
                 })
            .def("texcoord1D",
                 [](Mesh& _self, OM::HalfedgeHandle _h) {
                     if (!_self.has_halfedge_texcoords1D())
                         _self.request_halfedge_texcoords1D();
                     return flt2numpy(_self, _self.texcoord1D(_h));
                 })
            .def("texcoord2D",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_texcoords2D())
                         _self.request_vertex_texcoords2D();
                     return vec2numpy(_self, _self.texcoord2D(_h));
                 })
            .def("texcoord2D",
                 [](Mesh& _self, OM::HalfedgeHandle _h) {
                     if (!_self.has_halfedge_texcoords2D())
                         _self.request_halfedge_texcoords2D();
                     return vec2numpy(_self, _self.texcoord2D(_h));
                 })
            .def("texcoord3D",
                 [](Mesh& _self, OM::VertexHandle _h) {
                     if (!_self.has_vertex_texcoords3D())
                         _self.request_vertex_texcoords3D();
                     return vec2numpy(_self, _self.texcoord3D(_h));
                 })
            .def("texcoord3D",
                 [](Mesh& _self, OM::HalfedgeHandle _h) {
                     if (!_self.has_halfedge_texcoords3D())
                         _self.request_halfedge_texcoords3D();
                     return vec2numpy(_self, _self.texcoord3D(_h));
                 })

            // numpy vector setter
            .def("set_point",
                 [](Mesh& _self, OM::VertexHandle _h, np_array<typename Point::value_type> _arr) {
                     _self.point(_h) = Point(_arr.data()[0], _arr.data()[1], _arr.data()[2]);
                 })

            .def("set_normal",
                 [](Mesh& _self, OM::VertexHandle _h, np_array<typename Normal::value_type> _arr) {
                     if (!_self.has_vertex_normals())
                         _self.request_vertex_normals();
                     _self.set_normal(_h, Normal(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })
            .def("set_normal",
                 [](Mesh& _self, OM::HalfedgeHandle _h, np_array<typename Normal::value_type> _arr) {
                     if (!_self.has_halfedge_normals())
                         _self.request_halfedge_normals();
                     _self.set_normal(_h, Normal(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })
            .def("set_normal",
                 [](Mesh& _self, OM::FaceHandle _h, np_array<typename Normal::value_type> _arr) {
                     if (!_self.has_face_normals())
                         _self.request_face_normals();
                     _self.set_normal(_h, Normal(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            .def("set_color",
                 [](Mesh& _self, OM::VertexHandle _h, np_array<typename Color::value_type> _arr) {
                     if (!_self.has_vertex_colors())
                         _self.request_vertex_colors();
                     _self.set_color(_h, Color(_arr.data()[0], _arr.data()[1], _arr.data()[2], _arr.data()[3]));
                 })
            .def("set_color",
                 [](Mesh& _self, OM::HalfedgeHandle _h, np_array<typename Color::value_type> _arr) {
                     if (!_self.has_halfedge_colors())
                         _self.request_halfedge_colors();
                     _self.set_color(_h, Color(_arr.data()[0], _arr.data()[1], _arr.data()[2], _arr.data()[3]));
                 })
            .def("set_color",
                 [](Mesh& _self, OM::EdgeHandle _h, np_array<typename Color::value_type> _arr) {
                     if (!_self.has_edge_colors())
                         _self.request_edge_colors();
                     _self.set_color(_h, Color(_arr.data()[0], _arr.data()[1], _arr.data()[2], _arr.data()[3]));
                 })
            .def("set_color",
                 [](Mesh& _self, OM::FaceHandle _h, np_array<typename Color::value_type> _arr) {
                     if (!_self.has_face_colors())
                         _self.request_face_colors();
                     _self.set_color(_h, Color(_arr.data()[0], _arr.data()[1], _arr.data()[2], _arr.data()[3]));
                 })

            .def("set_texcoord1D",
                 [](Mesh& _self, OM::VertexHandle _h, np_array<TexCoord1D> _arr) {
                     if (!_self.has_vertex_texcoords1D())
                         _self.request_vertex_texcoords1D();
                     _self.set_texcoord1D(_h, _arr.data()[0]);
                 })
            .def("set_texcoord1D",
                 [](Mesh& _self, OM::HalfedgeHandle _h, np_array<TexCoord1D> _arr) {
                     if (!_self.has_halfedge_texcoords1D())
                         _self.request_halfedge_texcoords1D();
                     _self.set_texcoord1D(_h, _arr.data()[0]);
                 })

            .def("set_texcoord2D",
                 [](Mesh& _self, OM::VertexHandle _h, np_array<typename TexCoord2D::value_type> _arr) {
                     if (!_self.has_vertex_texcoords2D())
                         _self.request_vertex_texcoords2D();
                     _self.set_texcoord2D(_h, TexCoord2D(_arr.data()[0], _arr.data()[1]));
                 })
            .def("set_texcoord2D",
                 [](Mesh& _self, OM::HalfedgeHandle _h, np_array<typename TexCoord2D::value_type> _arr) {
                     if (!_self.has_halfedge_texcoords2D())
                         _self.request_halfedge_texcoords2D();
                     _self.set_texcoord2D(_h, TexCoord2D(_arr.data()[0], _arr.data()[1]));
                 })

            .def("set_texcoord3D",
                 [](Mesh& _self, OM::VertexHandle _h, np_array<typename TexCoord3D::value_type> _arr) {
                     if (!_self.has_vertex_texcoords3D())
                         _self.request_vertex_texcoords3D();
                     _self.set_texcoord3D(_h, TexCoord3D(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })
            .def("set_texcoord3D",
                 [](Mesh& _self, OM::HalfedgeHandle _h, np_array<typename TexCoord3D::value_type> _arr) {
                     if (!_self.has_halfedge_texcoords3D())
                         _self.request_halfedge_texcoords3D();
                     _self.set_texcoord3D(_h, TexCoord3D(_arr.data()[0], _arr.data()[1], _arr.data()[2]));
                 })

            // numpy matrix getter
            .def("points",
                 [](Mesh& _self) {
                     return vec2numpy(_self, _self.point(OM::VertexHandle(0)), _self.n_vertices());
                 })
            .def("vertex_normals",
                 [](Mesh& _self) {
                     if (!_self.has_vertex_normals())
                         _self.request_vertex_normals();
                     return vec2numpy(_self, _self.normal(OM::VertexHandle(0)), _self.n_vertices());
                 })
            .def("vertex_colors",
                 [](Mesh& _self) {
                     if (!_self.has_vertex_colors())
                         _self.request_vertex_colors();
                     return vec2numpy(_self, _self.color(OM::VertexHandle(0)), _self.n_vertices());
                 })
            .def("vertex_texcoords1D",
                 [](Mesh& _self) {
                     if (!_self.has_vertex_texcoords1D())
                         _self.request_vertex_texcoords1D();
                     return flt2numpy(_self, _self.texcoord1D(OM::VertexHandle(0)), _self.n_vertices());
                 })
            .def("vertex_texcoords2D",
                 [](Mesh& _self) {
                     if (!_self.has_vertex_texcoords2D())
                         _self.request_vertex_texcoords2D();
                     return vec2numpy(_self, _self.texcoord2D(OM::VertexHandle(0)), _self.n_vertices());
                 })
            .def("vertex_texcoords3D",
                 [](Mesh& _self) {
                     if (!_self.has_vertex_texcoords3D())
                         _self.request_vertex_texcoords3D();
                     return vec2numpy(_self, _self.texcoord3D(OM::VertexHandle(0)), _self.n_vertices());
                 })

            .def("halfedge_normals",
                 [](Mesh& _self) {
                     if (!_self.has_halfedge_normals())
                         _self.request_halfedge_normals();
                     return vec2numpy(_self, _self.normal(OM::HalfedgeHandle(0)), _self.n_halfedges());
                 })
            .def("halfedge_colors",
                 [](Mesh& _self) {
                     if (!_self.has_halfedge_colors())
                         _self.request_halfedge_colors();
                     return vec2numpy(_self, _self.color(OM::HalfedgeHandle(0)), _self.n_halfedges());
                 })
            .def("halfedge_texcoords1D",
                 [](Mesh& _self) {
                     if (!_self.has_halfedge_texcoords1D())
                         _self.request_halfedge_texcoords1D();
                     return flt2numpy(_self, _self.texcoord1D(OM::HalfedgeHandle(0)), _self.n_halfedges());
                 })
            .def("halfedge_texcoords2D",
                 [](Mesh& _self) {
                     if (!_self.has_halfedge_texcoords2D())
                         _self.request_halfedge_texcoords2D();
                     return vec2numpy(_self, _self.texcoord2D(OM::HalfedgeHandle(0)), _self.n_halfedges());
                 })
            .def("halfedge_texcoords3D",
                 [](Mesh& _self) {
                     if (!_self.has_halfedge_texcoords3D())
                         _self.request_halfedge_texcoords3D();
                     return vec2numpy(_self, _self.texcoord3D(OM::HalfedgeHandle(0)), _self.n_halfedges());
                 })

            .def("edge_colors",
                 [](Mesh& _self) {
                     if (!_self.has_edge_colors())
                         _self.request_edge_colors();
                     return vec2numpy(_self, _self.color(OM::EdgeHandle(0)), _self.n_edges());
                 })

            .def("face_normals",
                 [](Mesh& _self) {
                     if (!_self.has_face_normals())
                         _self.request_face_normals();
                     return vec2numpy(_self, _self.normal(OM::FaceHandle(0)), _self.n_faces());
                 })
            .def("face_colors",
                 [](Mesh& _self) {
                     if (!_self.has_face_colors())
                         _self.request_face_colors();
                     return vec2numpy(_self, _self.color(OM::FaceHandle(0)), _self.n_faces());
                 })

            // numpy indices
            .def("vertex_vertex_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexVertexIter>)
            .def("vv_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexVertexIter>)
            .def("vertex_face_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexFaceIter>)
            .def("vf_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexFaceIter>)
            .def("vertex_edge_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexEdgeIter>)
            .def("ve_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexEdgeIter>)
            .def("vertex_outgoing_halfedge_indices",
                 &indices<Mesh, OM::VertexHandle, typename Mesh::VertexOHalfedgeIter>)
            .def("voh_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexOHalfedgeIter>)
            .def("vertex_incoming_halfedge_indices",
                 &indices<Mesh, OM::VertexHandle, typename Mesh::VertexIHalfedgeIter>)
            .def("vih_indices", &indices<Mesh, OM::VertexHandle, typename Mesh::VertexIHalfedgeIter>)

            .def("face_face_indices", &indices<Mesh, OM::FaceHandle, typename Mesh::FaceFaceIter>)
            .def("ff_indices", &indices<Mesh, OM::FaceHandle, typename Mesh::FaceFaceIter>)
            .def("face_edge_indices", &indices<Mesh, OM::FaceHandle, typename Mesh::FaceEdgeIter>)
            .def("fe_indices", &indices<Mesh, OM::FaceHandle, typename Mesh::FaceEdgeIter>)
            .def("face_halfedge_indices", &indices<Mesh, OM::FaceHandle, typename Mesh::FaceHalfedgeIter>)
            .def("fh_indices", &indices<Mesh, OM::FaceHandle, typename Mesh::FaceHalfedgeIter>)

            .def("edge_vertex_indices", &edge_other_indices<Mesh, FuncEdgeVertex>)
            .def("ev_indices", &edge_other_indices<Mesh, FuncEdgeVertex>)
            .def("edge_face_indices", &edge_other_indices<Mesh, FuncEdgeFace>)
            .def("ef_indices", &edge_other_indices<Mesh, FuncEdgeFace>)
            .def("edge_halfedge_indices", &edge_other_indices<Mesh, FuncEdgeHalfedge>)
            .def("eh_indices", &edge_other_indices<Mesh, FuncEdgeHalfedge>)

            .def("halfedge_vertex_indices", &halfedge_other_indices<Mesh, FuncHalfedgeVertex>)
            .def("hv_indices", &halfedge_other_indices<Mesh, FuncHalfedgeVertex>)
            .def("halfedge_to_vertex_indices", &halfedge_other_indices<Mesh, FuncHalfedgeToVertex>)
            .def("htv_indices", &halfedge_other_indices<Mesh, FuncHalfedgeToVertex>)
            .def("halfedge_from_vertex_indices", &halfedge_other_indices<Mesh, FuncHalfedgeFromVertex>)
            .def("hfv_indices", &halfedge_other_indices<Mesh, FuncHalfedgeFromVertex>)
            .def("halfedge_face_indices", &halfedge_other_indices<Mesh, FuncHalfedgeFace>)
            .def("hf_indices", &halfedge_other_indices<Mesh, FuncHalfedgeFace>)
            .def("halfedge_edge_indices", &halfedge_other_indices<Mesh, FuncHalfedgeEdge>)
            .def("he_indices", &halfedge_other_indices<Mesh, FuncHalfedgeEdge>)

            // numpy add vertices & faces
            .def("add_vertices", &add_vertices<Mesh>, nb::arg("points"))
            .def("add_faces", &add_faces<Mesh>, nb::arg("face_vertex_indices"))

            .def("resize_points",
                 [](Mesh& _self, size_t _n_vertices) {
                     _self.resize(_n_vertices, _self.n_edges(), _self.n_faces());
                 })

            // Property interface: single item
            .def("vertex_property",
                 &Mesh::template py_property<OM::VertexHandle, typename Mesh::VPropHandle>)
            .def("halfedge_property",
                 &Mesh::template py_property<OM::HalfedgeHandle, typename Mesh::HPropHandle>)
            .def("edge_property",
                 &Mesh::template py_property<OM::EdgeHandle, typename Mesh::EPropHandle>)
            .def("face_property",
                 &Mesh::template py_property<OM::FaceHandle, typename Mesh::FPropHandle>)

            .def("set_vertex_property",
                 &Mesh::template py_set_property<OM::VertexHandle, typename Mesh::VPropHandle>)
            .def("set_halfedge_property",
                 &Mesh::template py_set_property<OM::HalfedgeHandle, typename Mesh::HPropHandle>)
            .def("set_edge_property",
                 &Mesh::template py_set_property<OM::EdgeHandle, typename Mesh::EPropHandle>)
            .def("set_face_property",
                 &Mesh::template py_set_property<OM::FaceHandle, typename Mesh::FPropHandle>)

            .def("has_vertex_property", &Mesh::template py_has_property<OM::VertexHandle>)
            .def("has_halfedge_property", &Mesh::template py_has_property<OM::HalfedgeHandle>)
            .def("has_edge_property", &Mesh::template py_has_property<OM::EdgeHandle>)
            .def("has_face_property", &Mesh::template py_has_property<OM::FaceHandle>)

            .def("remove_vertex_property", &Mesh::template py_remove_property<OM::VertexHandle>)
            .def("remove_halfedge_property", &Mesh::template py_remove_property<OM::HalfedgeHandle>)
            .def("remove_edge_property", &Mesh::template py_remove_property<OM::EdgeHandle>)
            .def("remove_face_property", &Mesh::template py_remove_property<OM::FaceHandle>)

            // Property interface: generic (list)
            .def("vertex_property",
                 &Mesh::template py_property_generic<OM::VertexHandle, typename Mesh::VPropHandle>)
            .def("halfedge_property",
                 &Mesh::template py_property_generic<OM::HalfedgeHandle, typename Mesh::HPropHandle>)
            .def("edge_property",
                 &Mesh::template py_property_generic<OM::EdgeHandle, typename Mesh::EPropHandle>)
            .def("face_property",
                 &Mesh::template py_property_generic<OM::FaceHandle, typename Mesh::FPropHandle>)

            .def("set_vertex_property",
                 &Mesh::template py_set_property_generic<OM::VertexHandle, typename Mesh::VPropHandle>)
            .def("set_halfedge_property",
                 &Mesh::template py_set_property_generic<OM::HalfedgeHandle, typename Mesh::HPropHandle>)
            .def("set_edge_property",
                 &Mesh::template py_set_property_generic<OM::EdgeHandle, typename Mesh::EPropHandle>)
            .def("set_face_property",
                 &Mesh::template py_set_property_generic<OM::FaceHandle, typename Mesh::FPropHandle>)

            // Property interface: array
            .def("vertex_property_array",
                 &Mesh::template py_property_array<OM::VertexHandle, typename Mesh::VPropHandle>)
            .def("halfedge_property_array",
                 &Mesh::template py_property_array<OM::HalfedgeHandle, typename Mesh::HPropHandle>)
            .def("edge_property_array",
                 &Mesh::template py_property_array<OM::EdgeHandle, typename Mesh::EPropHandle>)
            .def("face_property_array",
                 &Mesh::template py_property_array<OM::FaceHandle, typename Mesh::FPropHandle>)

            .def("set_vertex_property_array",
                 &Mesh::template py_set_property_array<OM::VertexHandle, typename Mesh::VPropHandle>)
            .def("set_halfedge_property_array",
                 &Mesh::template py_set_property_array<OM::HalfedgeHandle, typename Mesh::HPropHandle>)
            .def("set_edge_property_array",
                 &Mesh::template py_set_property_array<OM::EdgeHandle, typename Mesh::EPropHandle>)
            .def("set_face_property_array",
                 &Mesh::template py_set_property_array<OM::FaceHandle, typename Mesh::FPropHandle>)

            // Property interface: copy
            .def("copy_property",
                 &Mesh::template py_copy_property<OM::VertexHandle, typename Mesh::VPropHandle>)
            .def("copy_property",
                 &Mesh::template py_copy_property<OM::HalfedgeHandle, typename Mesh::HPropHandle>)
            .def("copy_property",
                 &Mesh::template py_copy_property<OM::EdgeHandle, typename Mesh::EPropHandle>)
            .def("copy_property",
                 &Mesh::template py_copy_property<OM::FaceHandle, typename Mesh::FPropHandle>);

        expose_type_specific_functions(class_mesh);
    }

} // namespace lfs::python::openmesh_bindings
