/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <Python.h>

#include "python_runtime.hpp"

namespace lfs::python {

    inline bool can_acquire_gil() noexcept {
        return Py_IsInitialized() && is_gil_state_ready();
    }

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

} // namespace lfs::python
