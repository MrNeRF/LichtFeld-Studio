/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#pragma once

#define OM_STATIC_BUILD

#include "utilities.hpp"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <map>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <string>

namespace nb = nanobind;

namespace lfs::python::openmesh_bindings {

    struct MeshTraits : public OpenMesh::DefaultTraits {
        typedef OpenMesh::Vec3d Point;
        typedef OpenMesh::Vec3d Normal;
        typedef OpenMesh::Vec4f Color;
        typedef double TexCoord1D;
        typedef OpenMesh::Vec2d TexCoord2D;
        typedef OpenMesh::Vec3d TexCoord3D;
    };

    template <class Mesh>
    class MeshWrapperT : public Mesh {
    public:
        typedef OpenMesh::VPropHandleT<nb::object> VPropHandle;
        typedef OpenMesh::HPropHandleT<nb::object> HPropHandle;
        typedef OpenMesh::EPropHandleT<nb::object> EPropHandle;
        typedef OpenMesh::FPropHandleT<nb::object> FPropHandle;

        template <class Handle, class PropHandle>
        nb::object py_property(const std::string& _name, Handle _h) {
            const auto prop = py_prop_on_demand<Handle, PropHandle>(_name);
            return Mesh::property(prop, _h);
        }

        template <class Handle, class PropHandle>
        void py_set_property(const std::string& _name, Handle _h, nb::object _val) {
            const auto prop = py_prop_on_demand<Handle, PropHandle>(_name);
            Mesh::property(prop, _h) = _val;
        }

        template <class Handle>
        bool py_has_property(const std::string& _name) {
            auto& prop_map = py_prop_map(Handle());
            return prop_map.count(_name);
        }

        template <class Handle>
        void py_remove_property(const std::string& _name) {
            auto& prop_map = py_prop_map(Handle());
            if (prop_map.count(_name) != 0) {
                Mesh::remove_property(prop_map.at(_name));
                prop_map.erase(_name);
            }
        }

        template <class Handle, class PropHandle>
        nb::list py_property_generic(const std::string& _name) {
            const size_t n = py_n_items(Handle());
            const auto prop = py_prop_on_demand<Handle, PropHandle>(_name);
            nb::list res;
            for (size_t i = 0; i < n; ++i) {
                res.append(Mesh::property(prop, Handle(static_cast<int>(i))));
            }
            return res;
        }

        template <class Handle, class PropHandle>
        void py_set_property_generic(const std::string& _name, nb::list _list) {
            const size_t n = py_n_items(Handle());
            const auto prop = py_prop_on_demand<Handle, PropHandle>(_name);
            if (nb::len(_list) != n) {
                throw std::runtime_error("List must have length n.");
            }
            for (size_t i = 0; i < n; ++i) {
                Mesh::property(prop, Handle(static_cast<int>(i))) = nb::borrow(_list[i]);
            }
        }

        template <class Handle, class PropHandle>
        nb::ndarray<nb::numpy, double> py_property_array(const std::string& _name) {
            const size_t n = py_n_items(Handle());
            const auto prop = py_prop_on_demand<Handle, PropHandle>(_name);

            const nb::object tmp_obj = Mesh::property(prop, Handle(0));
            auto first_arr = nb::cast<nb::ndarray<double, nb::numpy, nb::c_contig>>(tmp_obj);
            const size_t size = first_arr.size();

            if (size == 0) {
                throw std::runtime_error("One of the arrays has size 0.");
            }

            std::vector<size_t> shape;
            shape.push_back(n);
            for (size_t d = 0; d < first_arr.ndim(); ++d) {
                shape.push_back(first_arr.shape(d));
            }

            double* data = new double[size * n];

            for (size_t i = 0; i < n; ++i) {
                const nb::object obj = Mesh::property(prop, Handle(static_cast<int>(i)));
                auto arr = nb::cast<nb::ndarray<double, nb::numpy, nb::c_contig>>(obj);
                if (arr.size() != size) {
                    delete[] data;
                    throw std::runtime_error("Array sizes do not match.");
                }
                if (arr.ndim() != first_arr.ndim()) {
                    delete[] data;
                    throw std::runtime_error("Array dimensions do not match.");
                }
                std::copy(arr.data(), arr.data() + size, &data[size * i]);
            }

            return make_owned_array(data, shape);
        }

        template <class Handle, class PropHandle>
        void py_set_property_array(const std::string& _name, nb::ndarray<double, nb::numpy, nb::c_contig> _arr) {
            const size_t n = py_n_items(Handle());
            const auto prop = py_prop_on_demand<Handle, PropHandle>(_name);

            if (_arr.size() == 0 || _arr.ndim() < 1 || _arr.shape(0) != n) {
                throw std::runtime_error("Array must have shape (n, ...).");
            }

            if (_arr.ndim() == 1) {
                for (size_t i = 0; i < n; ++i) {
                    Mesh::property(prop, Handle(static_cast<int>(i))) = nb::float_(_arr.data()[i]);
                }
            } else {
                const size_t row_size = _arr.size() / n;
                for (size_t i = 0; i < n; ++i) {
                    double* sub_data = new double[row_size];
                    std::copy(_arr.data() + i * row_size, _arr.data() + (i + 1) * row_size, sub_data);
                    std::vector<size_t> sub_shape;
                    for (size_t d = 1; d < _arr.ndim(); ++d) {
                        sub_shape.push_back(_arr.shape(d));
                    }
                    auto tmp = make_owned_array(sub_data, sub_shape);
                    Mesh::property(prop, Handle(static_cast<int>(i))) = nb::cast(tmp);
                }
            }
        }

        template <class Handle, class PropHandle>
        void py_copy_property(const std::string& _name, Handle _from, Handle _to) {
            auto prop = py_prop_on_demand<Handle, PropHandle>(_name);
            Mesh::copy_property(prop, _from, _to);
        }

        nb::object py_copy() {
            return nb::cast(MeshWrapperT(*this));
        }

        nb::object py_deepcopy(nb::dict _memo) {
            nb::object id_func = nb::module_::import_("builtins").attr("id");
            nb::object deepcopy = nb::module_::import_("copy").attr("deepcopy");

            MeshWrapperT* copy = new MeshWrapperT(*this);
            nb::object copy_pyobj = nb::cast(copy, nb::rv_policy::take_ownership);
            _memo[id_func(nb::cast(this))] = copy_pyobj;

            py_deepcopy_prop<OpenMesh::VertexHandle>(copy, deepcopy, _memo);
            py_deepcopy_prop<OpenMesh::HalfedgeHandle>(copy, deepcopy, _memo);
            py_deepcopy_prop<OpenMesh::EdgeHandle>(copy, deepcopy, _memo);
            py_deepcopy_prop<OpenMesh::FaceHandle>(copy, deepcopy, _memo);

            return copy_pyobj;
        }

        size_t py_n_items(OpenMesh::VertexHandle) const { return Mesh::n_vertices(); }
        size_t py_n_items(OpenMesh::HalfedgeHandle) const { return Mesh::n_halfedges(); }
        size_t py_n_items(OpenMesh::EdgeHandle) const { return Mesh::n_edges(); }
        size_t py_n_items(OpenMesh::FaceHandle) const { return Mesh::n_faces(); }

        size_t py_has_status(OpenMesh::VertexHandle) const { return Mesh::has_vertex_status(); }
        size_t py_has_status(OpenMesh::HalfedgeHandle) const { return Mesh::has_halfedge_status(); }
        size_t py_has_status(OpenMesh::EdgeHandle) const { return Mesh::has_edge_status(); }
        size_t py_has_status(OpenMesh::FaceHandle) const { return Mesh::has_face_status(); }

    private:
        template <class Handle>
        void py_deepcopy_prop(MeshWrapperT* _copy, nb::object _copyfunc, nb::dict _memo) {
            for (const auto& item : py_prop_map(Handle())) {
                const auto prop = item.second;
                for (size_t i = 0; i < py_n_items(Handle()); ++i) {
                    const Handle h(static_cast<int>(i));
                    _copy->property(prop, h) = _copyfunc(this->property(prop, h), _memo);
                }
            }
        }

        template <class Handle, class PropHandle>
        PropHandle py_prop_on_demand(const std::string& _name) {
            auto& prop_map = py_prop_map(Handle());
            if (prop_map.count(_name) == 0) {
                PropHandle prop;
                Mesh::add_property(prop, _name);
                prop_map[_name] = prop;
                const size_t n = py_n_items(Handle());
                for (size_t i = 0; i < n; ++i) {
                    Mesh::property(prop, Handle(static_cast<int>(i))) = nb::none();
                }
            }
            return prop_map.at(_name);
        }

        std::map<std::string, VPropHandle>& py_prop_map(OpenMesh::VertexHandle) { return vprop_map_; }
        std::map<std::string, HPropHandle>& py_prop_map(OpenMesh::HalfedgeHandle) { return hprop_map_; }
        std::map<std::string, EPropHandle>& py_prop_map(OpenMesh::EdgeHandle) { return eprop_map_; }
        std::map<std::string, FPropHandle>& py_prop_map(OpenMesh::FaceHandle) { return fprop_map_; }

        std::map<std::string, VPropHandle> vprop_map_;
        std::map<std::string, HPropHandle> hprop_map_;
        std::map<std::string, EPropHandle> eprop_map_;
        std::map<std::string, FPropHandle> fprop_map_;
    };

    using TriMesh = MeshWrapperT<OpenMesh::TriMesh_ArrayKernelT<MeshTraits>>;
    using PolyMesh = MeshWrapperT<OpenMesh::PolyMesh_ArrayKernelT<MeshTraits>>;

} // namespace lfs::python::openmesh_bindings
