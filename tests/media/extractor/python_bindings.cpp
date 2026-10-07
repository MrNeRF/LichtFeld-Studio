// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/error.hpp"
#include "python/lfs/py_media.hpp"
#include <nanobind/nanobind.h>
NB_MODULE(media_test_bindings, m) {
    nanobind::exception<lfs::Exception>(m, "MediaError", PyExc_RuntimeError);
    lfs::python::register_media(m);
}
