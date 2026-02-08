/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#pragma once

#include "mesh_types.hpp"

#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModAspectRatioT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Tools/Decimater/ModEdgeLengthT.hh>
#include <OpenMesh/Tools/Decimater/ModHausdorffT.hh>
#include <OpenMesh/Tools/Decimater/ModIndependentSetsT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalDeviationT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalFlippingT.hh>
#include <OpenMesh/Tools/Decimater/ModProgMeshT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/ModRoundnessT.hh>

#include <cstdio>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
namespace OM = OpenMesh;

namespace lfs::python::openmesh_bindings {

    template <class Handle>
    void expose_module_handle(nb::module_& m, const char* _name) {
        nb::class_<Handle>(m, _name)
            .def(nb::init<>())
            .def("is_valid", &Handle::is_valid);
    }

    template <class Module>
    nb::list infolist(Module& _self) {
        const typename Module::InfoList& infos = _self.infolist();
        nb::list res;
        for (size_t i = 0; i < infos.size(); ++i) {
            res.append(infos[i]);
        }
        return res;
    }

    template <class Mesh>
    void copy_locked_vertex_prop(Mesh& _mesh, const std::string& _name) {
        if (!_mesh.template py_has_property<OM::VertexHandle>(_name)) {
            throw std::runtime_error("Vertex property \"" + _name + "\" does not exist.");
        }
        for (auto vh : _mesh.vertices()) {
            const auto val =
                _mesh.template py_property<OM::VertexHandle, typename Mesh::VPropHandle>(_name, vh);
            _mesh.status(vh).set_locked(nb::cast<bool>(val));
        }
    }

    template <class Mesh>
    void copy_feature_vertex_prop(Mesh& _mesh, const std::string& _name) {
        if (!_mesh.template py_has_property<OM::VertexHandle>(_name)) {
            throw std::runtime_error("Vertex property \"" + _name + "\" does not exist.");
        }
        for (auto vh : _mesh.vertices()) {
            const auto val =
                _mesh.template py_property<OM::VertexHandle, typename Mesh::VPropHandle>(_name, vh);
            _mesh.status(vh).set_feature(nb::cast<bool>(val));
        }
    }

    template <class Mesh>
    void copy_feature_edge_prop(Mesh& _mesh, const std::string& _name) {
        if (!_mesh.template py_has_property<OM::EdgeHandle>(_name)) {
            throw std::runtime_error("Edge property \"" + _name + "\" does not exist.");
        }
        for (auto eh : _mesh.edges()) {
            const auto val =
                _mesh.template py_property<OM::EdgeHandle, typename Mesh::EPropHandle>(_name, eh);
            _mesh.status(eh).set_feature(nb::cast<bool>(val));
        }
    }

    template <class Mesh>
    void copy_status_props(Mesh& _mesh, nb::object _locked_vertex_propname,
                           nb::object _feature_vertex_propname, nb::object _feature_edge_propname) {
        if (!_locked_vertex_propname.is_none())
            copy_locked_vertex_prop(_mesh, nb::cast<std::string>(_locked_vertex_propname));
        if (!_feature_vertex_propname.is_none())
            copy_feature_vertex_prop(_mesh, nb::cast<std::string>(_feature_vertex_propname));
        if (!_feature_edge_propname.is_none())
            copy_feature_edge_prop(_mesh, nb::cast<std::string>(_feature_edge_propname));
    }

    template <class Mesh>
    void expose_decimater(nb::module_& m, const char* _name) {
        using ModBase = OM::Decimater::ModBaseT<Mesh>;
        using ModAspectRatio = OM::Decimater::ModAspectRatioT<Mesh>;
        using ModEdgeLength = OM::Decimater::ModEdgeLengthT<Mesh>;
        using ModHausdorff = OM::Decimater::ModHausdorffT<Mesh>;
        using ModIndependentSets = OM::Decimater::ModIndependentSetsT<Mesh>;
        using ModNormalDeviation = OM::Decimater::ModNormalDeviationT<Mesh>;
        using ModNormalFlipping = OM::Decimater::ModNormalFlippingT<Mesh>;
        using ModProgMesh = OM::Decimater::ModProgMeshT<Mesh>;
        using ModQuadric = OM::Decimater::ModQuadricT<Mesh>;
        using ModRoundness = OM::Decimater::ModRoundnessT<Mesh>;

        using ModAspectRatioHandle = OM::Decimater::ModHandleT<ModAspectRatio>;
        using ModEdgeLengthHandle = OM::Decimater::ModHandleT<ModEdgeLength>;
        using ModHausdorffHandle = OM::Decimater::ModHandleT<ModHausdorff>;
        using ModIndependentSetsHandle = OM::Decimater::ModHandleT<ModIndependentSets>;
        using ModNormalDeviationHandle = OM::Decimater::ModHandleT<ModNormalDeviation>;
        using ModNormalFlippingHandle = OM::Decimater::ModHandleT<ModNormalFlipping>;
        using ModProgMeshHandle = OM::Decimater::ModHandleT<ModProgMesh>;
        using ModQuadricHandle = OM::Decimater::ModHandleT<ModQuadric>;
        using ModRoundnessHandle = OM::Decimater::ModHandleT<ModRoundness>;

        using Decimater = OM::Decimater::DecimaterT<Mesh>;
        using Info = typename ModProgMesh::Info;

        char buffer[64];

        // Decimater
        snprintf(buffer, sizeof buffer, "%sDecimater", _name);

        nb::class_<Decimater>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())

            .def(
                "decimate",
                [](Decimater& _self, size_t _n_collapses, nb::object _locked_vertex_propname,
                   nb::object _feature_vertex_propname, nb::object _feature_edge_propname) {
                    copy_status_props(_self.mesh(), _locked_vertex_propname,
                                      _feature_vertex_propname, _feature_edge_propname);
                    _self.decimate(_n_collapses);
                },
                nb::arg("n_collapses") = 0, nb::arg("locked_vertex_propname") = nb::none(),
                nb::arg("feature_vertex_propname") = nb::none(),
                nb::arg("feature_edge_propname") = nb::none())

            .def(
                "decimate_to",
                [](Decimater& _self, size_t _n_vertices, nb::object _locked_vertex_propname,
                   nb::object _feature_vertex_propname, nb::object _feature_edge_propname) {
                    copy_status_props(_self.mesh(), _locked_vertex_propname,
                                      _feature_vertex_propname, _feature_edge_propname);
                    _self.decimate_to(_n_vertices);
                },
                nb::arg("n_vertices"), nb::arg("locked_vertex_propname") = nb::none(),
                nb::arg("feature_vertex_propname") = nb::none(),
                nb::arg("feature_edge_propname") = nb::none())

            .def(
                "decimate_to_faces",
                [](Decimater& _self, size_t _n_vertices, size_t _n_faces,
                   nb::object _locked_vertex_propname, nb::object _feature_vertex_propname,
                   nb::object _feature_edge_propname) {
                    copy_status_props(_self.mesh(), _locked_vertex_propname,
                                      _feature_vertex_propname, _feature_edge_propname);
                    _self.decimate_to_faces(_n_vertices, _n_faces);
                },
                nb::arg("n_vertices") = 0, nb::arg("n_faces") = 0,
                nb::arg("locked_vertex_propname") = nb::none(),
                nb::arg("feature_vertex_propname") = nb::none(),
                nb::arg("feature_edge_propname") = nb::none())

            .def("initialize", [](Decimater& _self) { return _self.initialize(); })
            .def("is_initialized", [](Decimater& _self) { return _self.is_initialized(); })

            .def("add", [](Decimater& _self, ModAspectRatioHandle& _mod) { return _self.add(_mod); })
            .def("add", [](Decimater& _self, ModEdgeLengthHandle& _mod) { return _self.add(_mod); })
            .def("add", [](Decimater& _self, ModHausdorffHandle& _mod) { return _self.add(_mod); })
            .def("add",
                 [](Decimater& _self, ModIndependentSetsHandle& _mod) { return _self.add(_mod); })
            .def("add",
                 [](Decimater& _self, ModNormalDeviationHandle& _mod) { return _self.add(_mod); })
            .def("add",
                 [](Decimater& _self, ModNormalFlippingHandle& _mod) { return _self.add(_mod); })
            .def("add", [](Decimater& _self, ModProgMeshHandle& _mod) { return _self.add(_mod); })
            .def("add", [](Decimater& _self, ModQuadricHandle& _mod) { return _self.add(_mod); })
            .def("add", [](Decimater& _self, ModRoundnessHandle& _mod) { return _self.add(_mod); })

            .def("remove",
                 [](Decimater& _self, ModAspectRatioHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModEdgeLengthHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModHausdorffHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModIndependentSetsHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModNormalDeviationHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModNormalFlippingHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModProgMeshHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModQuadricHandle& _mod) { return _self.remove(_mod); })
            .def("remove",
                 [](Decimater& _self, ModRoundnessHandle& _mod) { return _self.remove(_mod); })

            .def(
                "module",
                [](Decimater& _self, ModAspectRatioHandle& _mod) -> ModAspectRatio& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModEdgeLengthHandle& _mod) -> ModEdgeLength& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModHausdorffHandle& _mod) -> ModHausdorff& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModIndependentSetsHandle& _mod) -> ModIndependentSets& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModNormalDeviationHandle& _mod) -> ModNormalDeviation& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModNormalFlippingHandle& _mod) -> ModNormalFlipping& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModProgMeshHandle& _mod) -> ModProgMesh& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModQuadricHandle& _mod) -> ModQuadric& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference)
            .def(
                "module",
                [](Decimater& _self, ModRoundnessHandle& _mod) -> ModRoundness& {
                    return _self.module(_mod);
                },
                nb::rv_policy::reference);

        // ModBase
        snprintf(buffer, sizeof buffer, "%sModBase", _name);

        nb::class_<ModBase>(m, buffer)
            .def("name", &ModBase::name, nb::rv_policy::copy)
            .def("is_binary", &ModBase::is_binary)
            .def("set_binary", &ModBase::set_binary)
            .def("initialize", &ModBase::initialize)
            .def("collapse_priority", &ModBase::collapse_priority)
            .def("preprocess_collapse", &ModBase::preprocess_collapse)
            .def("postprocess_collapse", &ModBase::postprocess_collapse)
            .def("set_error_tolerance_factor", &ModBase::set_error_tolerance_factor);

        // ModAspectRatio
        snprintf(buffer, sizeof buffer, "%sModAspectRatio", _name);

        nb::class_<ModAspectRatio, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("aspect_ratio", &ModAspectRatio::aspect_ratio)
            .def("set_aspect_ratio", &ModAspectRatio::set_aspect_ratio);

        snprintf(buffer, sizeof buffer, "%sModAspectRatioHandle", _name);
        expose_module_handle<ModAspectRatioHandle>(m, buffer);

        // ModEdgeLength
        snprintf(buffer, sizeof buffer, "%sModEdgeLength", _name);

        nb::class_<ModEdgeLength, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("edge_length", &ModEdgeLength::edge_length)
            .def("set_edge_length", &ModEdgeLength::set_edge_length);

        snprintf(buffer, sizeof buffer, "%sModEdgeLengthHandle", _name);
        expose_module_handle<ModEdgeLengthHandle>(m, buffer);

        // ModHausdorff
        snprintf(buffer, sizeof buffer, "%sModHausdorff", _name);

        nb::class_<ModHausdorff, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("tolerance", &ModHausdorff::tolerance)
            .def("set_tolerance", &ModHausdorff::set_tolerance);

        snprintf(buffer, sizeof buffer, "%sModHausdorffHandle", _name);
        expose_module_handle<ModHausdorffHandle>(m, buffer);

        // ModIndependentSets
        snprintf(buffer, sizeof buffer, "%sModIndependentSets", _name);

        nb::class_<ModIndependentSets, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>());

        snprintf(buffer, sizeof buffer, "%sModIndependentSetsHandle", _name);
        expose_module_handle<ModIndependentSetsHandle>(m, buffer);

        // ModNormalDeviation
        snprintf(buffer, sizeof buffer, "%sModNormalDeviation", _name);

        nb::class_<ModNormalDeviation, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("normal_deviation", &ModNormalDeviation::normal_deviation)
            .def("set_normal_deviation", &ModNormalDeviation::set_normal_deviation);

        snprintf(buffer, sizeof buffer, "%sModNormalDeviationHandle", _name);
        expose_module_handle<ModNormalDeviationHandle>(m, buffer);

        // ModNormalFlipping
        snprintf(buffer, sizeof buffer, "%sModNormalFlipping", _name);

        nb::class_<ModNormalFlipping, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("max_normal_deviation", &ModNormalFlipping::max_normal_deviation)
            .def("set_max_normal_deviation", &ModNormalFlipping::set_max_normal_deviation);

        snprintf(buffer, sizeof buffer, "%sModNormalFlippingHandle", _name);
        expose_module_handle<ModNormalFlippingHandle>(m, buffer);

        // ModProgMesh
        snprintf(buffer, sizeof buffer, "%sModProgMeshInfo", _name);

        nb::class_<Info>(m, buffer)
            .def_rw("v0", &Info::v0)
            .def_rw("v1", &Info::v1)
            .def_rw("vl", &Info::vl)
            .def_rw("vr", &Info::vr);

        snprintf(buffer, sizeof buffer, "%sModProgMesh", _name);

        nb::class_<ModProgMesh, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("pmi", &infolist<ModProgMesh>)
            .def("infolist", &infolist<ModProgMesh>)
            .def("write", &ModProgMesh::write);

        snprintf(buffer, sizeof buffer, "%sModProgMeshHandle", _name);
        expose_module_handle<ModProgMeshHandle>(m, buffer);

        // ModQuadric
        snprintf(buffer, sizeof buffer, "%sModQuadric", _name);

        nb::class_<ModQuadric, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("set_max_err", &ModQuadric::set_max_err,
                 nb::arg("err"), nb::arg("binary") = true)
            .def("unset_max_err", &ModQuadric::unset_max_err)
            .def("max_err", &ModQuadric::max_err);

        snprintf(buffer, sizeof buffer, "%sModQuadricHandle", _name);
        expose_module_handle<ModQuadricHandle>(m, buffer);

        // ModRoundness
        snprintf(buffer, sizeof buffer, "%sModRoundness", _name);

        nb::class_<ModRoundness, ModBase>(m, buffer)
            .def(nb::init<Mesh&>(), nb::keep_alive<1, 2>())
            .def("set_min_angle", &ModRoundness::set_min_angle)
            .def("set_min_roundness", &ModRoundness::set_min_roundness,
                 nb::arg("min_roundness"), nb::arg("binary") = true)
            .def("unset_min_roundness", &ModRoundness::unset_min_roundness)
            .def("roundness", &ModRoundness::roundness);

        snprintf(buffer, sizeof buffer, "%sModRoundnessHandle", _name);
        expose_module_handle<ModRoundnessHandle>(m, buffer);
    }

} // namespace lfs::python::openmesh_bindings
