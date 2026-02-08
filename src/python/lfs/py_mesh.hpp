/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/mesh_data.hpp"
#include "py_tensor.hpp"
#include <cassert>
#include <memory>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python {

    class PyMeshData {
    public:
        explicit PyMeshData(std::shared_ptr<core::MeshData> data)
            : data_(std::move(data)) {
            assert(data_);
        }

        PyTensor vertices() const { return PyTensor(data_->vertices, false); }
        PyTensor normals() const { return PyTensor(data_->normals, false); }
        PyTensor tangents() const { return PyTensor(data_->tangents, false); }
        PyTensor texcoords() const { return PyTensor(data_->texcoords, false); }
        PyTensor colors() const { return PyTensor(data_->colors, false); }
        PyTensor indices() const { return PyTensor(data_->indices, false); }

        int64_t vertex_count() const { return data_->vertex_count(); }
        int64_t face_count() const { return data_->face_count(); }
        bool has_normals() const { return data_->has_normals(); }
        bool has_tangents() const { return data_->has_tangents(); }
        bool has_texcoords() const { return data_->has_texcoords(); }
        bool has_colors() const { return data_->has_colors(); }

        void compute_normals() { data_->compute_normals(); }

        PyMeshData to_device(const std::string& device) const;

        void set_vertices(const PyTensor& t) { data_->vertices = t.tensor(); }
        void set_normals(const PyTensor& t) { data_->normals = t.tensor(); }
        void set_indices(const PyTensor& t) { data_->indices = t.tensor(); }
        void set_texcoords(const PyTensor& t) { data_->texcoords = t.tensor(); }
        void set_colors(const PyTensor& t) { data_->colors = t.tensor(); }

        std::shared_ptr<core::MeshData> data() const { return data_; }

        std::string repr() const;

    private:
        std::shared_ptr<core::MeshData> data_;
    };

    void register_mesh(nb::module_& m);

} // namespace lfs::python
