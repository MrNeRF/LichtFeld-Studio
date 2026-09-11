/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/source_site.hpp"

#include <nanobind/nanobind.h>

#include <cstdint>
#include <string_view>
#include <utility>

namespace nb = nanobind;

namespace lfs::python {

    // Phase 9 Section 2.2: per-site policy for a contained Python callback failure.
    enum class PyCallbackPolicy : std::uint8_t {
        DisableAndReport, // caller unregisters the callback; publish a Toast
        FailOwner,        // caller converts to the owning operation's failure; no publish
        WarnAndContinue,  // keep the callback registered; publish a StatusOnly notification
    };

    // Publish a contained Python-callback Error to the ErrorBus per policy:
    // DisableAndReport -> Toast, WarnAndContinue -> StatusOnly, FailOwner -> no
    // publish (the owner surfaces the returned Error). GIL-free (value Error).
    void report_python_callback_error(const lfs::Error& error, PyCallbackPolicy policy) noexcept;

    // Contain the Python exception currently held by `e` (Section 2.2). Precondition:
    // GIL held. Restores the pending error, extracts it via error_from_python (type +
    // traceback preserved), reports it per policy, and returns the typed Error (used by
    // FailOwner owners). Never throws; never leaves a Python error pending.
    lfs::Error contain_python_callback(nb::python_error& e, PyCallbackPolicy policy,
                                       lfs::core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) noexcept;

    // Contain a non-Python C++ callback exception (Section 2.2 fallback): normalize to
    // an Error{domain=Python, code=Internal, user_message=what}, report per policy, and
    // return it. GIL-free.
    lfs::Error contain_cxx_callback(std::string_view what, PyCallbackPolicy policy,
                                    lfs::core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) noexcept;

    // Creates the lichtfeld.Error Python exception hierarchy (Phase 9 Section 1.1)
    // and installs the single LIFO exception translator (Section 1.3). Must be
    // called once inside NB_MODULE, BEFORE any binding group registers, so a throw
    // during any m.def / method unwind maps to the typed subclass. Registration
    // order inside the module body is therefore load-bearing.
    void register_errors(nb::module_& m);

    // Set the pending Python error to the typed lichtfeld exception for `error`
    // (subclass by code, attributes populated). Precondition: GIL held. Never
    // throws; leaves a Python error pending. Reused by the _testing bindings.
    void raise_typed_python_error(const lfs::Error& error) noexcept;

    // Registers lichtfeld._testing (unstable, test-only): raise_error,
    // probe_python_error, tick_frame. See Phase 9 Section 5.1.
    void register_testing(nb::module_& m);

    // Unwrap a Result at a binding boundary (Section 1.4). On failure throws
    // lfs::Exception carrying the Error; the registered translator converts it to
    // the typed Python exception during unwind. Zero cost on success (move-out).
    template <class T>
    T unwrap(lfs::Result<T>&& r) {
        if (!r) {
            throw lfs::Exception(std::move(r).error());
        }
        return std::move(r).value();
    }

    inline void unwrap(lfs::Status&& s) {
        if (!s) {
            throw lfs::Exception(std::move(s).error());
        }
    }

} // namespace lfs::python
