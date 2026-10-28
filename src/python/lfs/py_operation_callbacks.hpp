/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include "io/project_operations.hpp"

#include <nanobind/nanobind.h>

#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace nb = nanobind;

namespace lfs::python {

    // Progress and cancel callbacks for native project operations, which
    // copy them while the GIL is released. Share the Python handle so those
    // copies only change C++ reference counts.
    struct PyProgressCallback {
        std::shared_ptr<nb::object> callback;
        bool present;

        explicit PyProgressCallback(nb::object value)
            : callback(std::make_shared<nb::object>(std::move(value))),
              present(*callback && !callback->is_none()) {}

        // The native callback, or none when Python passed None.
        [[nodiscard]] lfs::io::project::ProjectOperationProgress operation() const {
            return present ? lfs::io::project::ProjectOperationProgress(*this)
                           : lfs::io::project::ProjectOperationProgress{};
        }

        void operator()(float progress, const std::string& message) const {
            nb::gil_scoped_acquire gil;
            if (!present)
                return;
            try {
                (*callback)(progress, message);
            } catch (const std::exception& e) {
                LOG_ERROR("Python progress callback error: {}", e.what());
            }
        }
    };

    struct PyCancelCallback {
        std::shared_ptr<nb::object> callback;
        bool present;

        explicit PyCancelCallback(nb::object value)
            : callback(std::make_shared<nb::object>(std::move(value))),
              present(*callback && !callback->is_none()) {}

        // The native callback, or none when Python passed None.
        [[nodiscard]] lfs::io::project::ProjectOperationCancel operation() const {
            return present ? lfs::io::project::ProjectOperationCancel(*this)
                           : lfs::io::project::ProjectOperationCancel{};
        }

        bool operator()() const {
            nb::gil_scoped_acquire gil;
            if (!present)
                return false;
            try {
                return nb::cast<bool>((*callback)());
            } catch (const std::exception& e) {
                LOG_ERROR("Python cancellation callback error: {}", e.what());
                return true;
            }
        }
    };

} // namespace lfs::python
