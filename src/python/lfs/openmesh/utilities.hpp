/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Ported from openmesh-python (BSD-3-Clause) to nanobind. */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;

namespace lfs::python::openmesh_bindings {

    template <class T>
    nb::capsule free_when_done(T* data) {
        return nb::capsule(data, [](void* p) noexcept {
            delete[] static_cast<T*>(p);
        });
    }

    template <class T>
    nb::ndarray<nb::numpy, T> make_array(T* data, std::vector<size_t> shape, nb::handle owner) {
        return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
    }

    template <class T>
    nb::ndarray<nb::numpy, T> make_owned_array(T* data, std::vector<size_t> shape) {
        nb::capsule owner = free_when_done(data);
        return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), std::move(owner));
    }

} // namespace lfs::python::openmesh_bindings
