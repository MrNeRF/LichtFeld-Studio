/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#pragma once

#include "mesh_types.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python::openmesh_bindings {

    template <class Circulator, class CenterEntityHandle>
    class CirculatorWrapperT {
    public:
        CirculatorWrapperT(PolyMesh& _mesh, CenterEntityHandle _center)
            : circulator_(_mesh, _center) {}

        CirculatorWrapperT(TriMesh& _mesh, CenterEntityHandle _center)
            : circulator_(_mesh, _center) {}

        CirculatorWrapperT iter() const { return *this; }

        typename Circulator::ValueHandle next() {
            if (circulator_.is_valid()) {
                typename Circulator::ValueHandle res((*circulator_).idx());
                ++circulator_;
                return res;
            }
            throw nb::stop_iteration();
        }

    private:
        Circulator circulator_;
    };

    template <class Circulator, class CenterEntityHandle>
    void expose_circulator(nb::module_& m, const char* _name) {
        using Wrapper = CirculatorWrapperT<Circulator, CenterEntityHandle>;
        nb::class_<Wrapper>(m, _name)
            .def("__init__",
                 [](Wrapper* self, TriMesh& mesh, CenterEntityHandle center) {
                     new (self) Wrapper(mesh, center);
                 })
            .def("__init__",
                 [](Wrapper* self, PolyMesh& mesh, CenterEntityHandle center) {
                     new (self) Wrapper(mesh, center);
                 })
            .def("__iter__", &Wrapper::iter)
            .def("__next__", &Wrapper::next);
    }

} // namespace lfs::python::openmesh_bindings
