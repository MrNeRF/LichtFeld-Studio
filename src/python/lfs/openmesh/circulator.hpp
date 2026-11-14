/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#pragma once

#include "mesh_types.hpp"

#include <nanobind/nanobind.h>

#include <type_traits>

namespace nb = nanobind;

namespace lfs::python::openmesh_bindings {

    // Plain handle type of a (deprecated) circulator: ValueHandle in current
    // OpenMesh (value_type is the smart handle there), value_type in OpenMesh 11.0.
    template <class Circulator, class = void>
    struct CirculatorValueHandle {
        using type = typename Circulator::value_type;
    };

    template <class Circulator>
    struct CirculatorValueHandle<Circulator, std::void_t<typename Circulator::ValueHandle>> {
        using type = typename Circulator::ValueHandle;
    };

    template <class Circulator, class CenterEntityHandle>
    class CirculatorWrapperT {
        using ValueHandle = typename CirculatorValueHandle<Circulator>::type;

    public:
        CirculatorWrapperT(PolyMesh& _mesh, CenterEntityHandle _center)
            : circulator_(_mesh, _center) {}

        CirculatorWrapperT(TriMesh& _mesh, CenterEntityHandle _center)
            : circulator_(_mesh, _center) {}

        CirculatorWrapperT iter() const { return *this; }

        ValueHandle next() {
            if (circulator_.is_valid()) {
                ValueHandle res((*circulator_).idx());
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
