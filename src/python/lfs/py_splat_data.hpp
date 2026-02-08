/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "py_tensor.hpp"
#include <nanobind/nanobind.h>
#include <optional>

namespace nb = nanobind;

namespace lfs::python {

    class PySplatData {
    public:
        explicit PySplatData(core::SplatData* data) : data_(data) {
            assert(data_ != nullptr);
        }

        // Raw tensor access (views - no copy)
        PyTensor means_raw() const;
        PyTensor sh0_raw() const;
        PyTensor shN_raw() const;
        PyTensor scaling_raw() const;
        PyTensor rotation_raw() const;
        PyTensor opacity_raw() const;

        // Computed getters (may involve GPU computation)
        PyTensor get_means() const;
        PyTensor get_opacity() const;
        PyTensor get_rotation() const;
        PyTensor get_scaling() const;
        PyTensor get_shs() const;

        // RGB color accessors (handle SH0 encoding internally)
        PyTensor get_colors_rgb() const;
        void set_colors_rgb(const PyTensor& colors);

        // Metadata
        int active_sh_degree() const { return data_->get_active_sh_degree(); }
        int max_sh_degree() const { return data_->get_max_sh_degree(); }
        float scene_scale() const { return data_->get_scene_scale(); }
        size_t num_points() const { return data_->size(); }

        // Soft deletion
        PyTensor deleted() const;
        bool has_deleted_mask() const { return data_->has_deleted_mask(); }
        size_t visible_count() const { return data_->visible_count(); }
        PyTensor soft_delete(const PyTensor& mask);
        void undelete(const PyTensor& mask);
        void clear_deleted() { data_->clear_deleted(); }
        size_t apply_deleted() { return data_->apply_deleted(); }

        // SH degree management
        void increment_sh_degree() { data_->increment_sh_degree(); }
        void set_active_sh_degree(int degree) { data_->set_active_sh_degree(degree); }
        void set_max_sh_degree(int degree) { data_->set_max_sh_degree(degree); }

        // Capacity management
        void reserve_capacity(size_t capacity) { data_->reserve_capacity(capacity); }

        // Access underlying data (for internal use)
        core::SplatData* data() { return data_; }
        const core::SplatData* data() const { return data_; }

    private:
        core::SplatData* data_;
    };

    // Register PySplatData with nanobind module
    void register_splat_data(nb::module_& m);

} // namespace lfs::python
