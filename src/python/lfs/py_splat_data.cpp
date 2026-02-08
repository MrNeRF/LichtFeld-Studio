/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_splat_data.hpp"
#include <nanobind/stl/optional.h>

namespace {
    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float SH_DC_OFFSET = 0.5f;
} // namespace

namespace lfs::python {

    // Raw tensor access - return views (no copy)
    PyTensor PySplatData::means_raw() const {
        return PyTensor(data_->means_raw(), false);
    }

    PyTensor PySplatData::sh0_raw() const {
        return PyTensor(data_->sh0_raw(), false);
    }

    PyTensor PySplatData::shN_raw() const {
        return PyTensor(data_->shN_raw(), false);
    }

    PyTensor PySplatData::scaling_raw() const {
        return PyTensor(data_->scaling_raw(), false);
    }

    PyTensor PySplatData::rotation_raw() const {
        return PyTensor(data_->rotation_raw(), false);
    }

    PyTensor PySplatData::opacity_raw() const {
        return PyTensor(data_->opacity_raw(), false);
    }

    // Computed getters
    PyTensor PySplatData::get_means() const {
        return PyTensor(data_->get_means(), true);
    }

    PyTensor PySplatData::get_opacity() const {
        return PyTensor(data_->get_opacity(), true);
    }

    PyTensor PySplatData::get_rotation() const {
        return PyTensor(data_->get_rotation(), true);
    }

    PyTensor PySplatData::get_scaling() const {
        return PyTensor(data_->get_scaling(), true);
    }

    PyTensor PySplatData::get_shs() const {
        return PyTensor(data_->get_shs(), true);
    }

    PyTensor PySplatData::get_colors_rgb() const {
        auto sh0 = data_->sh0_raw();
        assert(sh0.shape().rank() == 3 && sh0.shape()[1] == 1 && sh0.shape()[2] == 3);
        auto flat = sh0.reshape({static_cast<int>(sh0.shape()[0]), 3});
        return PyTensor(flat * SH_C0 + SH_DC_OFFSET, true);
    }

    void PySplatData::set_colors_rgb(const PyTensor& colors) {
        const auto& rgb = colors.tensor();
        assert(rgb.shape().rank() == 2 && rgb.shape()[1] == 3);
        assert(rgb.shape()[0] == static_cast<int64_t>(data_->size()));
        auto sh0_values = (rgb - SH_DC_OFFSET) / SH_C0;
        auto sh0 = data_->sh0_raw();
        sh0.copy_(sh0_values.reshape({static_cast<int>(rgb.shape()[0]), 1, 3}));
    }

    // Soft deletion
    PyTensor PySplatData::deleted() const {
        if (!data_->has_deleted_mask()) {
            return PyTensor(core::Tensor(), false);
        }
        return PyTensor(data_->deleted(), false);
    }

    PyTensor PySplatData::soft_delete(const PyTensor& mask) {
        core::Tensor prev_state = data_->soft_delete(mask.tensor());
        return PyTensor(std::move(prev_state), true);
    }

    void PySplatData::undelete(const PyTensor& mask) {
        data_->undelete(mask.tensor());
    }

    void register_splat_data(nb::module_& m) {
        nb::class_<PySplatData>(m, "SplatData")
            // Raw tensor access (views)
            .def_prop_ro("means_raw", &PySplatData::means_raw,
                         "Raw means tensor [N, 3] (view)")
            .def_prop_ro("sh0_raw", &PySplatData::sh0_raw,
                         "Raw SH0 tensor [N, 1, 3] (view)")
            .def_prop_ro("shN_raw", &PySplatData::shN_raw,
                         "Raw SHN tensor [N, (degree+1)^2-1, 3] (view)")
            .def_prop_ro("scaling_raw", &PySplatData::scaling_raw,
                         "Raw scaling tensor [N, 3] (log-space, view)")
            .def_prop_ro("rotation_raw", &PySplatData::rotation_raw,
                         "Raw rotation tensor [N, 4] (quaternions, view)")
            .def_prop_ro("opacity_raw", &PySplatData::opacity_raw,
                         "Raw opacity tensor [N, 1] (logit-space, view)")

            // Computed getters
            .def("get_means", &PySplatData::get_means,
                 "Get means (same as means_raw for now)")
            .def("get_opacity", &PySplatData::get_opacity,
                 "Get opacity with sigmoid applied")
            .def("get_rotation", &PySplatData::get_rotation,
                 "Get normalized rotation quaternions")
            .def("get_scaling", &PySplatData::get_scaling,
                 "Get scaling with exp applied")
            .def("get_shs", &PySplatData::get_shs,
                 "Get concatenated SH coefficients (sh0 + shN)")

            // RGB color accessors
            .def("get_colors_rgb", &PySplatData::get_colors_rgb,
                 "Get RGB colors [N, 3] in [0,1] range (handles SH0 encoding)")
            .def("set_colors_rgb", &PySplatData::set_colors_rgb, nb::arg("colors"),
                 "Set RGB colors from [N, 3] tensor in [0,1] range (handles SH0 encoding)")

            // Metadata
            .def_prop_ro("active_sh_degree", &PySplatData::active_sh_degree,
                         "Current active SH degree")
            .def_prop_ro("max_sh_degree", &PySplatData::max_sh_degree,
                         "Maximum SH degree")
            .def_prop_ro("scene_scale", &PySplatData::scene_scale,
                         "Scene scale factor")
            .def_prop_ro("num_points", &PySplatData::num_points,
                         "Number of Gaussians")

            // Soft deletion
            .def_prop_ro("deleted", &PySplatData::deleted,
                         "Deletion mask tensor [N] (bool)")
            .def("has_deleted_mask", &PySplatData::has_deleted_mask,
                 "Check if deletion mask exists")
            .def("visible_count", &PySplatData::visible_count,
                 "Number of visible (non-deleted) Gaussians")
            .def("soft_delete", &PySplatData::soft_delete, nb::arg("mask"),
                 "Mark Gaussians as deleted by mask, returns previous state for undo")
            .def("undelete", &PySplatData::undelete, nb::arg("mask"),
                 "Restore deleted Gaussians by mask")
            .def("clear_deleted", &PySplatData::clear_deleted,
                 "Clear all soft-deleted flags")
            .def("apply_deleted", &PySplatData::apply_deleted,
                 "Permanently remove deleted Gaussians, returns count removed")

            // SH degree management
            .def("increment_sh_degree", &PySplatData::increment_sh_degree,
                 "Increment active SH degree by 1")
            .def("set_active_sh_degree", &PySplatData::set_active_sh_degree, nb::arg("degree"),
                 "Set active SH degree")
            .def("set_max_sh_degree", &PySplatData::set_max_sh_degree, nb::arg("degree"),
                 "Set maximum SH degree")

            // Capacity
            .def("reserve_capacity", &PySplatData::reserve_capacity, nb::arg("capacity"),
                 "Reserve capacity for Gaussians (for densification)");
    }

} // namespace lfs::python
