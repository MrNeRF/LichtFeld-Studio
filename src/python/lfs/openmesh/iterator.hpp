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

    template <class Iterator, size_t (OpenMesh::ArrayKernel::*n_items)() const>
    class IteratorWrapperT {
    public:
        IteratorWrapperT(const PolyMesh& _mesh, typename Iterator::value_type _hnd, bool _skip = false)
            : mesh_(_mesh),
              n_items_(n_items),
              iterator_(_mesh, _hnd, _skip),
              iterator_end_(_mesh, typename Iterator::value_type(static_cast<int>((_mesh.*n_items)()))) {}

        IteratorWrapperT(const TriMesh& _mesh, typename Iterator::value_type _hnd, bool _skip = false)
            : mesh_(_mesh),
              n_items_(n_items),
              iterator_(_mesh, _hnd, _skip),
              iterator_end_(_mesh, typename Iterator::value_type(static_cast<int>((_mesh.*n_items)()))) {}

        IteratorWrapperT iter() const { return *this; }

        typename Iterator::value_handle next() {
            if (iterator_ != iterator_end_) {
                typename Iterator::value_handle res((*iterator_).idx());
                ++iterator_;
                return res;
            }
            throw nb::stop_iteration();
        }

        unsigned int len() const { return static_cast<unsigned int>((mesh_.*n_items_)()); }

    private:
        const OpenMesh::PolyConnectivity& mesh_;
        size_t (OpenMesh::ArrayKernel::*n_items_)() const;
        Iterator iterator_;
        Iterator iterator_end_;
    };

    template <class Iterator, size_t (OpenMesh::ArrayKernel::*n_items)() const>
    void expose_iterator(nb::module_& m, const char* _name) {
        using Wrapper = IteratorWrapperT<Iterator, n_items>;
        nb::class_<Wrapper>(m, _name)
            .def("__init__",
                 [](Wrapper* self, PolyMesh& mesh, typename Iterator::value_type hnd) {
                     new (self) Wrapper(mesh, hnd);
                 })
            .def("__init__",
                 [](Wrapper* self, PolyMesh& mesh, typename Iterator::value_type hnd, bool skip) {
                     new (self) Wrapper(mesh, hnd, skip);
                 })
            .def("__init__",
                 [](Wrapper* self, TriMesh& mesh, typename Iterator::value_type hnd) {
                     new (self) Wrapper(mesh, hnd);
                 })
            .def("__init__",
                 [](Wrapper* self, TriMesh& mesh, typename Iterator::value_type hnd, bool skip) {
                     new (self) Wrapper(mesh, hnd, skip);
                 })
            .def("__iter__", &Wrapper::iter)
            .def("__next__", &Wrapper::next)
            .def("__len__", &Wrapper::len);
    }

} // namespace lfs::python::openmesh_bindings
