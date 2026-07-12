/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/property_registry.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python {

    class PyOptimizationParams {
    public:
        PyOptimizationParams() = default;

        [[nodiscard]] nb::object get(const std::string& prop_id) const;
        void set(const std::string& prop_id, nb::object value);
        [[nodiscard]] nb::dict prop_info(const std::string& prop_id) const;
        void reset(const std::string& prop_id);
        [[nodiscard]] nb::list properties() const;
        [[nodiscard]] nb::dict get_all_properties() const;

        // Returns reference to ParamManager's active params (or throws if unavailable)
        core::param::OptimizationParameters& params();
        [[nodiscard]] const core::param::OptimizationParameters& params() const;

        // Check if params are available
        [[nodiscard]] bool has_params() const;
    };

    class PyDatasetConfig {
    public:
        PyDatasetConfig() = default;

        [[nodiscard]] nb::object get(const std::string& prop_id) const;
        void set(const std::string& prop_id, nb::object value);
        [[nodiscard]] nb::dict prop_info(const std::string& prop_id) const;
        [[nodiscard]] nb::list properties() const;
        [[nodiscard]] nb::dict get_all_properties() const;

        core::param::DatasetConfig& params();
        [[nodiscard]] core::param::DatasetConfig params() const;

        [[nodiscard]] bool has_params() const;
        [[nodiscard]] bool can_edit() const;
    };

    void register_optimization_properties();
    void register_dataset_properties();
    void register_params(nb::module_& m);

} // namespace lfs::python
