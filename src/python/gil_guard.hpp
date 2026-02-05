/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <Python.h>

#include "python_runtime.hpp"

namespace lfs::python {

    // Unconditional GIL acquire. Use when Python is guaranteed initialized.
    class GilAcquire {
        const PyGILState_STATE state_;

    public:
        GilAcquire() noexcept : state_(PyGILState_Ensure()) {}
        ~GilAcquire() { PyGILState_Release(state_); }

        GilAcquire(const GilAcquire&) = delete;
        GilAcquire& operator=(const GilAcquire&) = delete;
        GilAcquire(GilAcquire&&) = delete;
        GilAcquire& operator=(GilAcquire&&) = delete;
    };

    // Conditional GIL acquire. Checks Py_IsInitialized() && is_gil_state_ready() first.
    class GilGuard {
        PyGILState_STATE state_{};
        bool acquired_;

    public:
        GilGuard() noexcept : acquired_(Py_IsInitialized() && is_gil_state_ready()) {
            if (acquired_)
                state_ = PyGILState_Ensure();
        }
        ~GilGuard() {
            if (acquired_)
                PyGILState_Release(state_);
        }

        explicit operator bool() const noexcept { return acquired_; }

        GilGuard(const GilGuard&) = delete;
        GilGuard& operator=(const GilGuard&) = delete;
        GilGuard(GilGuard&&) = delete;
        GilGuard& operator=(GilGuard&&) = delete;
    };

} // namespace lfs::python
