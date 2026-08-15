/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include "python/gil.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include <string_view>

namespace nb = nanobind;

namespace lfs::vis::gui {

    inline std::string capture_python_panel_chrome(const nb::object& instance) {
        if (!instance.is_valid() || !lfs::python::can_acquire_gil())
            return {};
        const lfs::python::GilAcquire gil;
        if (!nb::hasattr(instance, "capture_chrome"))
            return {};
        try {
            nb::object result = instance.attr("capture_chrome")();
            if (!result.is_valid() || result.is_none())
                return {};
            nb::object dumps = nb::module_::import_("json").attr("dumps");
            return nb::cast<std::string>(dumps(result));
        } catch (const std::exception& error) {
            LOG_WARN("Panel capture_chrome failed: {}", error.what());
            return {};
        }
    }

    inline void apply_python_panel_chrome(const nb::object& instance,
                                          const std::string_view json) {
        if (!instance.is_valid() || !lfs::python::can_acquire_gil())
            return;
        const lfs::python::GilAcquire gil;
        if (!nb::hasattr(instance, "apply_chrome"))
            return;
        try {
            const std::string text = json.empty() ? std::string("{}") : std::string(json);
            nb::object loads = nb::module_::import_("json").attr("loads");
            nb::object payload = loads(text);
            instance.attr("apply_chrome")(payload);
        } catch (const std::exception& error) {
            LOG_WARN("Panel apply_chrome failed: {}", error.what());
        }
    }

} // namespace lfs::vis::gui
