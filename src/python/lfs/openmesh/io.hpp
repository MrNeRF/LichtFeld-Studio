/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python::openmesh_bindings {

    void expose_io(nb::module_& m);

} // namespace lfs::python::openmesh_bindings
