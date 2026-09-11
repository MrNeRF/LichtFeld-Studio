/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_error.hpp"

#include "core/error_bus.hpp"
#include "core/error_codes.hpp"
#include "core/memory_pressure.hpp"
#include "core/source_site.hpp"
#include "io/loader.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace lfs::python {

    namespace {

        // Immortal for the interpreter's lifetime (standard practice for static
        // exception types). Created before any binding group registers.
        nb::object g_error;
        nb::object g_invalid_argument;
        nb::object g_not_found;
        nb::object g_cancelled;
        nb::object g_resource;

        PyObject* subclass_for(const lfs::ErrorCode code) noexcept {
            switch (code) {
            case lfs::ErrorCode::InvalidArgument:
            case lfs::ErrorCode::BoundsViolation:
                return g_invalid_argument.ptr();
            case lfs::ErrorCode::NotFound:
                return g_not_found.ptr();
            case lfs::ErrorCode::Cancelled:
                return g_cancelled.ptr();
            case lfs::ErrorCode::ResourceExhausted:
                return g_resource.ptr();
            default:
                return g_error.ptr();
            }
        }

        std::string display_message(const lfs::Error& error) {
            std::string message(error.user_message());
            if (message.empty())
                message.assign(error.detail());
            if (message.empty()) {
                const std::string developer = lfs::format_for_developer(error);
                const auto newline = developer.find('\n');
                message = newline == std::string::npos ? developer : developer.substr(0, newline);
            }
            return message;
        }

        void set_attr_steal(PyObject* inst, const char* name, PyObject* value) {
            if (!value) {
                PyErr_Clear();
                return;
            }
            if (PyObject_SetAttrString(inst, name, value) != 0)
                PyErr_Clear();
            Py_DECREF(value);
        }

        void set_context_attr(PyObject* inst, const lfs::Error& error) {
            const auto frames = error.frames();
            PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(frames.size()));
            if (!tuple) {
                PyErr_Clear();
                return;
            }
            for (std::size_t i = 0; i < frames.size(); ++i) {
                const auto& frame = frames[i];
                const char* file = frame.source.file_name() ? frame.source.file_name() : "";
                const std::string line = std::format("{} @ {}:{}", frame.operation, file,
                                                     static_cast<unsigned long>(frame.source.line()));
                PyObject* py = PyUnicode_FromString(line.c_str());
                if (!py) {
                    PyErr_Clear();
                    Py_INCREF(Py_None);
                    py = Py_None;
                }
                PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(i), py);
            }
            set_attr_steal(inst, "context", tuple);
        }

        void set_bytes_attr(PyObject* inst, const char* key, const lfs::Error& error) {
            std::optional<unsigned long long> found;
            for (const auto& frame : error.frames()) {
                for (const auto& entry : frame.fields.entries()) {
                    if (entry.key != key)
                        continue;
                    if (const auto* u = std::get_if<std::uint64_t>(&entry.value)) {
                        found = *u;
                    } else if (const auto* s = std::get_if<std::int64_t>(&entry.value)) {
                        found = static_cast<unsigned long long>(*s);
                    }
                }
            }
            PyObject* value = nullptr;
            if (found) {
                value = PyLong_FromUnsignedLongLong(*found);
            } else {
                Py_INCREF(Py_None);
                value = Py_None;
            }
            set_attr_steal(inst, key, value);
        }

    } // namespace

    void raise_typed_python_error(const lfs::Error& error) noexcept {
        PyObject* type = subclass_for(error.code());
        try {
            const std::string message = display_message(error);
            PyObject* inst = PyObject_CallFunction(type, "s", message.c_str());
            if (!inst) {
                PyErr_Clear();
                PyErr_SetString(type, message.c_str());
                return;
            }

            set_attr_steal(inst, "code", PyUnicode_FromString(lfs::to_string(error.code())));
            set_attr_steal(inst, "domain", PyUnicode_FromString(lfs::to_string(error.domain())));
            set_attr_steal(inst, "user_message", PyUnicode_FromString(message.c_str()));
            set_attr_steal(inst, "operation_id",
                           PyLong_FromUnsignedLongLong(error.operation_id().value()));
            set_attr_steal(inst, "retryable",
                           PyBool_FromLong(error.retryability() != lfs::Retryability::NotRetryable));
            set_attr_steal(inst, "details",
                           PyUnicode_FromString(lfs::format_for_developer(error).c_str()));
            set_context_attr(inst, error);
            if (error.code() == lfs::ErrorCode::ResourceExhausted) {
                set_bytes_attr(inst, "requested_bytes", error);
                set_bytes_attr(inst, "available_bytes", error);
            }

            PyErr_SetObject(type, inst);
            Py_DECREF(inst);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): noexcept boundary; on any host-side
            // failure fall back to a bare typed error so unwind still delivers one.
            PyErr_SetString(type, std::string(error.user_message()).c_str());
        }
    }

    void report_python_callback_error(const lfs::Error& error, const PyCallbackPolicy policy) noexcept {
        lfs::ErrorSurface surface = lfs::ErrorSurface::Toast;
        switch (policy) {
        case PyCallbackPolicy::DisableAndReport:
            surface = lfs::ErrorSurface::Toast;
            break;
        case PyCallbackPolicy::WarnAndContinue:
            surface = lfs::ErrorSurface::StatusOnly;
            break;
        case PyCallbackPolicy::FailOwner:
            return; // the owning operation surfaces the returned Error itself
        }
        lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
            .error = error,
            .surface = surface,
            .operation_id = error.operation_id(),
        });
    }

    lfs::Error contain_python_callback(nb::python_error& e, const PyCallbackPolicy policy,
                                       const lfs::core::SourceSite site) noexcept {
        // python_error fetched+cleared the pending error on construction; restore it
        // so error_from_python can format+consume it under the (held) GIL.
        e.restore();
        lfs::Error error = error_from_python(site);
        report_python_callback_error(error, policy);
        return error;
    }

    lfs::Error contain_cxx_callback(const std::string_view what, const PyCallbackPolicy policy,
                                    const lfs::core::SourceSite site) noexcept {
        lfs::Error error = lfs::make_error({
            .code = lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::Python,
            .severity = lfs::Severity::Error,
            .user_message = std::string(what),
            .detection = site,
        });
        report_python_callback_error(error, policy);
        return error;
    }

    namespace {

        void translate_lfs_exception(const std::exception_ptr& p, void* /*payload*/) {
            // LIFO: this translator runs before nanobind's default. A type this
            // clause does not catch propagates out of the lambda so the dispatch
            // loop forwards it to the next (default) translator.
            try {
                std::rethrow_exception(p);
            } catch (const lfs::Exception& e) {
                raise_typed_python_error(e.error());
            } catch (const lfs::core::MemoryAllocationError& e) {
                raise_typed_python_error(lfs::core::to_error(e.failure(), LFS_SOURCE_SITE_CURRENT()));
            } catch (const lfs::io::LoadCancelledError& e) {
                raise_typed_python_error(lfs::make_error({
                    .code = lfs::ErrorCode::Cancelled,
                    .domain = lfs::ErrorDomain::IO,
                    .severity = lfs::Severity::Error,
                    .user_message = e.what(),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
        }

        nb::object new_exception(const char* qualified_name, const char* doc, PyObject* base) {
            PyObject* type = PyErr_NewExceptionWithDoc(qualified_name, doc, base, nullptr);
            return nb::steal(type);
        }

    } // namespace

    void register_errors(nb::module_& m) {
        g_error = new_exception(
            "lichtfeld.Error",
            "Base class for every typed LichtFeld error crossing the Python boundary. "
            "Derives RuntimeError for backward compatibility. Carries .code, .domain, "
            ".user_message, .operation_id, .retryable, .details, and .context.",
            PyExc_RuntimeError);

        {
            PyObject* bases = PyTuple_Pack(2, g_error.ptr(), PyExc_ValueError);
            g_invalid_argument = new_exception(
                "lichtfeld.InvalidArgumentError",
                "Raised for InvalidArgument / BoundsViolation errors.", bases);
            Py_XDECREF(bases);
        }
        {
            PyObject* bases = PyTuple_Pack(2, g_error.ptr(), PyExc_FileNotFoundError);
            g_not_found = new_exception(
                "lichtfeld.NotFoundError",
                "Raised for NotFound errors; also a FileNotFoundError.", bases);
            Py_XDECREF(bases);
        }
        g_cancelled = new_exception(
            "lichtfeld.CancelledError",
            "Raised when an operation was cancelled.", g_error.ptr());
        g_resource = new_exception(
            "lichtfeld.ResourceError",
            "Raised for ResourceExhausted errors (memory/disk/queue). Carries "
            ".requested_bytes / .available_bytes (int or None).",
            g_error.ptr());

        m.attr("Error") = g_error;
        m.attr("InvalidArgumentError") = g_invalid_argument;
        m.attr("NotFoundError") = g_not_found;
        m.attr("CancelledError") = g_cancelled;
        m.attr("ResourceError") = g_resource;

        // One LIFO translator covering every m.def / method in all binding files.
        nb::register_exception_translator(translate_lfs_exception, nullptr);
    }

    void register_testing(nb::module_& m) {
        nb::module_ testing = m.def_submodule(
            "_testing", "Test-only hooks for the error boundary (unstable API).");

        testing.def(
            "raise_error",
            [](const std::string& code, const std::string& domain, const std::string& user_message,
               std::optional<std::int64_t> requested_bytes) {
                lfs::SmallFields fields;
                if (requested_bytes)
                    fields.add("requested_bytes", static_cast<std::uint64_t>(*requested_bytes));
                throw lfs::Exception(lfs::make_error({
                    .code = error_code_from_string(code),
                    .domain = error_domain_from_string(domain),
                    .severity = lfs::Severity::Error,
                    .user_message = user_message,
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                    .fields = std::move(fields),
                }));
            },
            nb::arg("code"), nb::arg("domain"), nb::arg("user_message"), nb::kw_only(),
            nb::arg("requested_bytes") = nb::none(),
            "Throw lfs::Exception(make_error(...)) so the LIFO translator maps it to a typed "
            "lichtfeld exception.");

        testing.def(
            "probe_python_error",
            [](nb::callable fn) -> nb::dict {
                lfs::Error error = lfs::make_error({
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Python,
                    .severity = lfs::Severity::Error,
                    .user_message = "callable did not raise",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
                try {
                    fn();
                } catch (nb::python_error& e) {
                    // python_error already fetched the pending error on construction;
                    // restore it so error_from_python can format+consume it under the GIL.
                    e.restore();
                    error = error_from_python(LFS_SOURCE_SITE_CURRENT());
                }

                std::string py_type;
                for (const auto& frame : error.frames()) {
                    for (const auto& entry : frame.fields.entries()) {
                        if (entry.key == "py_type") {
                            if (const auto* s = std::get_if<std::string>(&entry.value))
                                py_type = *s;
                        }
                    }
                }

                nb::dict result;
                result["code"] = std::string(lfs::to_string(error.code()));
                result["domain"] = std::string(lfs::to_string(error.domain()));
                result["user_message"] = std::string(error.user_message());
                result["detail"] = std::string(error.detail());
                result["py_type"] = py_type;
                return result;
            },
            nb::arg("callable"),
            "Invoke callable under the error_from_python guard; return {code, domain, "
            "user_message, detail, py_type}.");

        testing.def(
            "raise_memory_allocation_error",
            [](std::int64_t requested_bytes) {
                lfs::core::AllocationFailure failure;
                failure.domain = lfs::core::MemoryDomain::CudaDevice;
                failure.requested_bytes = static_cast<std::size_t>(requested_bytes);
                failure.operation = "test.raise_memory_allocation_error";
                throw lfs::core::MemoryAllocationError(failure);
            },
            nb::arg("requested_bytes"),
            "Throw core::MemoryAllocationError so the LIFO translator maps it to "
            "lichtfeld.ResourceError (translator rule 2).");

        testing.def(
            "tick_frame", [](float dt) { lfs::python::tick_frame_callback(dt); }, nb::arg("dt"),
            "Invoke the registered on_frame callback once (see lichtfeld.on_frame).");
    }

} // namespace lfs::python
