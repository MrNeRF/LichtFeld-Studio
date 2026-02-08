/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#include "handles.hpp"

#include <OpenMesh/Core/Mesh/Handles.hh>
#include <nanobind/nanobind.h>

namespace OM = OpenMesh;

namespace lfs::python::openmesh_bindings {

    void expose_handles(nb::module_& m) {
        nb::class_<OM::BaseHandle>(m, "BaseHandle")
            .def(nb::init<>())
            .def(nb::init<int>())
            .def("idx", &OM::BaseHandle::idx)
            .def("is_valid", &OM::BaseHandle::is_valid)
            .def("reset", &OM::BaseHandle::reset)
            .def("invalidate", &OM::BaseHandle::invalidate)
            .def("__eq__", &OM::BaseHandle::operator==)
            .def("__ne__", &OM::BaseHandle::operator!=)
            .def("__lt__", &OM::BaseHandle::operator<);

        nb::class_<OM::VertexHandle, OM::BaseHandle>(m, "VertexHandle")
            .def(nb::init<>())
            .def(nb::init<int>());
        nb::class_<OM::HalfedgeHandle, OM::BaseHandle>(m, "HalfedgeHandle")
            .def(nb::init<>())
            .def(nb::init<int>());
        nb::class_<OM::EdgeHandle, OM::BaseHandle>(m, "EdgeHandle")
            .def(nb::init<>())
            .def(nb::init<int>());
        nb::class_<OM::FaceHandle, OM::BaseHandle>(m, "FaceHandle")
            .def(nb::init<>())
            .def(nb::init<int>());
    }

} // namespace lfs::python::openmesh_bindings
