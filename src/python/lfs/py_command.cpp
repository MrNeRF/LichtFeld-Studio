/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_command.hpp"
#include "core/logger.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>

namespace lfs::python {

    namespace {
        vis::DirtyMask normalize_dirty_flags(const uint32_t flags) {
            return flags == 0 ? vis::DirtyFlag::ALL : static_cast<vis::DirtyMask>(flags);
        }

        void validate_undo_callback(const nb::object& callback, const char* name) {
            if (callback.is_none()) {
                return;
            }
            if (!nb::isinstance<nb::callable>(callback)) {
                throw nb::type_error((std::string(name) + " must be callable or None").c_str());
            }
        }
    } // namespace

    PyUndoEntry::PyUndoEntry(std::string name,
                             nb::object undo_fn,
                             nb::object redo_fn,
                             std::string id,
                             std::string source,
                             std::string scope,
                             const size_t estimated_bytes,
                             const vis::DirtyMask dirty_flags,
                             const std::chrono::milliseconds merge_window)
        : name_(std::move(name)),
          id_(std::move(id)),
          source_(std::move(source)),
          scope_(std::move(scope)),
          undo_fn_(std::move(undo_fn)),
          redo_fn_(std::move(redo_fn)),
          estimated_bytes_(estimated_bytes),
          dirty_flags_(dirty_flags == 0 ? vis::DirtyFlag::ALL : dirty_flags),
          updated_at_(std::chrono::steady_clock::now()),
          merge_window_(merge_window) {}

    PyUndoEntry::~PyUndoEntry() {
        nb::gil_scoped_acquire gil;
        undo_fn_ = nb::object();
        redo_fn_ = nb::object();
    }

    void PyUndoEntry::undo() {
        nb::gil_scoped_acquire gil;
        try {
            if (undo_fn_.is_valid() && !undo_fn_.is_none()) {
                undo_fn_();
            }
        } catch (const std::exception& e) {
            LOG_ERROR("PyUndoEntry undo: {}", e.what());
            throw;
        }
    }

    void PyUndoEntry::redo() {
        nb::gil_scoped_acquire gil;
        try {
            if (redo_fn_.is_valid() && !redo_fn_.is_none()) {
                redo_fn_();
            }
        } catch (const std::exception& e) {
            LOG_ERROR("PyUndoEntry redo: {}", e.what());
            throw;
        }
    }

    vis::op::UndoMetadata PyUndoEntry::metadata() const {
        return vis::op::UndoMetadata{
            .id = id_.empty() ? std::string("python.custom") : id_,
            .label = name_,
            .source = source_.empty() ? std::string("python") : source_,
            .scope = scope_.empty() ? std::string("custom") : scope_,
        };
    }

    size_t PyUndoEntry::estimatedBytes() const {
        if (estimated_bytes_ > 0) {
            return estimated_bytes_;
        }
        return sizeof(*this) + name_.size() + id_.size() + source_.size() + scope_.size() + 128;
    }

    vis::DirtyMask PyUndoEntry::dirtyFlags() const {
        return dirty_flags_;
    }

    bool PyUndoEntry::tryMerge(const vis::op::UndoEntry& incoming) {
        if (merge_window_.count() <= 0 || id_.empty()) {
            return false;
        }

        const auto* other = dynamic_cast<const PyUndoEntry*>(&incoming);
        if (!other || other->id_ != id_ || other->merge_window_.count() <= 0) {
            return false;
        }

        const auto elapsed = other->updated_at_ - updated_at_;
        if (elapsed < std::chrono::milliseconds{0} || elapsed > merge_window_) {
            return false;
        }

        nb::gil_scoped_acquire gil;
        name_ = other->name_;
        redo_fn_ = other->redo_fn_;
        estimated_bytes_ = std::max(estimated_bytes_, other->estimated_bytes_);
        dirty_flags_ |= other->dirty_flags_;
        updated_at_ = other->updated_at_;
        return true;
    }

    PyTransaction::PyTransaction(std::string name)
        : name_(std::move(name)) {}

    PyTransaction::~PyTransaction() {
        if (!active_) {
            return;
        }
        try {
            exit(false);
        } catch (const std::exception& e) {
            LOG_ERROR("PyTransaction destructor rollback failed: {}", e.what());
        } catch (...) {
            LOG_ERROR("PyTransaction destructor rollback failed: unknown exception");
        }
    }

    void PyTransaction::enter() {
        vis::op::undoHistory().beginTransaction(name_);
        active_ = true;
    }

    void PyTransaction::exit(const bool commit) {
        if (!active_)
            return;
        active_ = false;

        if (!commit) {
            vis::op::undoHistory().rollbackTransaction();
            return;
        }

        vis::op::undoHistory().commitTransaction();
    }

    void PyTransaction::add(nb::object undo_fn, nb::object redo_fn) {
        nb::gil_scoped_acquire gil;
        validate_undo_callback(undo_fn, "undo");
        validate_undo_callback(redo_fn, "redo");
        try {
            if (redo_fn.is_valid() && !redo_fn.is_none())
                redo_fn();
        } catch (const std::exception& e) {
            LOG_ERROR("Transaction add: {}", e.what());
            throw;
        }

        auto entry = std::make_unique<PyUndoEntry>(
            name_,
            std::move(undo_fn),
            std::move(redo_fn),
            "python.transaction",
            "python",
            "grouped");
        vis::op::undoHistory().push(std::move(entry));
    }

    void register_commands(nb::module_& m) {
        const auto stack_item_to_dict = [](const vis::op::UndoStackItem& item) {
            nb::dict result;
            result["id"] = item.metadata.id;
            result["label"] = item.metadata.label;
            result["source"] = item.metadata.source;
            result["scope"] = item.metadata.scope;
            result["estimated_bytes"] = item.estimated_bytes;
            result["cpu_bytes"] = item.cpu_bytes;
            result["gpu_bytes"] = item.gpu_bytes;
            return result;
        };
        const auto history_result_to_dict = [](const vis::op::HistoryResult& result) {
            nb::dict payload;
            payload["success"] = result.success;
            payload["changed"] = result.changed;
            payload["steps_performed"] = result.steps_performed;
            payload["error"] = result.error;
            return payload;
        };

        // lf.undo submodule - main undo API
        auto undo = m.def_submodule("undo", "Undo/redo system");

        undo.def(
            "push",
            [](const std::string& name,
               nb::object undo_fn,
               nb::object redo_fn,
               bool validate,
               const std::string& id,
               const std::string& source,
               const std::string& scope,
               const size_t estimated_bytes,
               const uint32_t dirty_flags,
               const uint64_t merge_window_ms) {
                validate_undo_callback(undo_fn, "undo");
                validate_undo_callback(redo_fn, "redo");
                if (validate) {
                    size_t dot_count = std::count(name.begin(), name.end(), '.');
                    bool has_space = name.find(' ') != std::string::npos;
                    if (dot_count != 1 || has_space) {
                        LOG_WARN("lf.undo.push(): Operation name '{}' should be 'category.action' format", name);
                    }
                }
                auto entry = std::make_unique<PyUndoEntry>(
                    name,
                    std::move(undo_fn),
                    std::move(redo_fn),
                    id,
                    source,
                    scope,
                    estimated_bytes,
                    normalize_dirty_flags(dirty_flags),
                    std::chrono::milliseconds(merge_window_ms));
                vis::op::undoHistory().push(std::move(entry));
            },
            nb::arg("name"),
            nb::arg("undo"),
            nb::arg("redo"),
            nb::arg("validate") = false,
            nb::arg("id") = "",
            nb::arg("source") = "python",
            nb::arg("scope") = "custom",
            nb::arg("estimated_bytes") = 0,
            nb::arg("dirty_flags") = 0,
            nb::arg("merge_window_ms") = 0,
            "Push an undo step with undo/redo functions");

        undo.def(
            "undo", []() { return vis::op::undoHistory().undo().success; }, "Undo last operation");
        undo.def(
            "redo", []() { return vis::op::undoHistory().redo().success; }, "Redo last undone operation");
        undo.def(
            "jump",
            [history_result_to_dict](const std::string& stack, size_t count) {
                if (stack == "undo") {
                    return history_result_to_dict(vis::op::undoHistory().undoMultiple(count));
                }
                if (stack == "redo") {
                    return history_result_to_dict(vis::op::undoHistory().redoMultiple(count));
                }
                throw std::runtime_error("stack must be 'undo' or 'redo'");
            },
            nb::arg("stack"),
            nb::arg("count"),
            "Apply multiple undo/redo steps for history navigation");
        undo.def(
            "can_undo", []() { return vis::op::undoHistory().canUndo(); }, "Check if undo is available");
        undo.def(
            "can_redo", []() { return vis::op::undoHistory().canRedo(); }, "Check if redo is available");
        undo.def(
            "clear", []() { vis::op::undoHistory().clear(); }, "Clear undo history");

        undo.def(
            "get_undo_name",
            []() -> std::string {
                if (!vis::op::undoHistory().canUndo())
                    return "";
                return vis::op::undoHistory().undoName();
            },
            "Get name of next undo operation");

        undo.def(
            "get_redo_name",
            []() -> std::string {
                if (!vis::op::undoHistory().canRedo())
                    return "";
                return vis::op::undoHistory().redoName();
            },
            "Get name of next redo operation");

        undo.def(
            "undo_names",
            []() { return vis::op::undoHistory().undoNames(); },
            "Get the undo stack names, newest first");

        undo.def(
            "redo_names",
            []() { return vis::op::undoHistory().redoNames(); },
            "Get the redo stack names, newest first");

        undo.def(
            "undo_bytes",
            []() { return vis::op::undoHistory().undoBytes(); },
            "Get estimated bytes retained by undo history");

        undo.def(
            "redo_bytes",
            []() { return vis::op::undoHistory().redoBytes(); },
            "Get estimated bytes retained by redo history");

        undo.def(
            "transaction_bytes",
            []() { return vis::op::undoHistory().transactionBytes(); },
            "Get estimated bytes retained by active grouped history transactions");
        undo.def(
            "max_bytes",
            []() { return vis::op::undoHistory().maxBytes(); },
            "Get the configured total retained history byte budget");
        undo.def(
            "set_max_bytes",
            [](size_t max_bytes) { vis::op::undoHistory().setMaxBytes(max_bytes); },
            nb::arg("max_bytes"),
            "Set the retained history byte budget");

        undo.def(
            "total_bytes",
            []() { return vis::op::undoHistory().totalBytes(); },
            "Get estimated bytes retained by undo and redo history");
        undo.def(
            "total_cpu_bytes",
            []() { return vis::op::undoHistory().totalMemory().cpu_bytes; },
            "Get estimated CPU-resident bytes retained by history");
        undo.def(
            "total_gpu_bytes",
            []() { return vis::op::undoHistory().totalMemory().gpu_bytes; },
            "Get estimated GPU-resident bytes retained by history");

        undo.def(
            "has_active_transaction",
            []() { return vis::op::undoHistory().hasActiveTransaction(); },
            "Check if a grouped history transaction is active");

        undo.def(
            "transaction_depth",
            []() { return vis::op::undoHistory().transactionDepth(); },
            "Get the current grouped history transaction nesting depth");
        undo.def(
            "transaction_age_ms",
            []() { return vis::op::undoHistory().transactionAgeMs(); },
            "Get the age of the active grouped history transaction in milliseconds");

        undo.def(
            "active_transaction_name",
            []() { return vis::op::undoHistory().activeTransactionName(); },
            "Get the current grouped history transaction label");
        undo.def(
            "generation",
            []() { return vis::op::undoHistory().generation(); },
            "Get the shared history change generation");
        undo.def(
            "subscribe",
            [](nb::callable callback) {
                nb::object cb = nb::borrow<nb::object>(callback);
                return vis::op::undoHistory().subscribe([cb]() {
                    nb::gil_scoped_acquire gil;
                    try {
                        cb();
                    } catch (const std::exception& e) {
                        LOG_ERROR("lf.undo.subscribe callback failed: {}", e.what());
                    } catch (...) {
                        LOG_ERROR("lf.undo.subscribe callback failed: unknown exception");
                    }
                });
            },
            nb::arg("callback"),
            "Subscribe to shared history changes and return a subscription id");
        undo.def(
            "unsubscribe",
            [](uint64_t subscription_id) { vis::op::undoHistory().unsubscribe(subscription_id); },
            nb::arg("subscription_id"),
            "Unsubscribe a shared history observer");
        undo.def(
            "shrink_to_fit",
            [](size_t target_gpu_bytes) { vis::op::undoHistory().shrinkToFit(target_gpu_bytes); },
            nb::arg("target_gpu_bytes"),
            "Offload history to CPU and evict cold entries until GPU usage fits the requested budget");

        undo.def(
            "stack",
            [stack_item_to_dict]() {
                nb::dict payload;
                nb::list undo_items;
                for (const auto& item : vis::op::undoHistory().undoItems()) {
                    undo_items.append(stack_item_to_dict(item));
                }
                nb::list redo_items;
                for (const auto& item : vis::op::undoHistory().redoItems()) {
                    redo_items.append(stack_item_to_dict(item));
                }
                payload["undo"] = undo_items;
                payload["redo"] = redo_items;
                payload["undo_bytes"] = vis::op::undoHistory().undoBytes();
                payload["redo_bytes"] = vis::op::undoHistory().redoBytes();
                payload["transaction_bytes"] = vis::op::undoHistory().transactionBytes();
                payload["total_bytes"] = vis::op::undoHistory().totalBytes();
                payload["max_bytes"] = vis::op::undoHistory().maxBytes();
                const auto undo_memory = vis::op::undoHistory().undoMemory();
                const auto redo_memory = vis::op::undoHistory().redoMemory();
                const auto transaction_memory = vis::op::undoHistory().transactionMemory();
                const auto total_memory = vis::op::undoHistory().totalMemory();
                payload["undo_cpu_bytes"] = undo_memory.cpu_bytes;
                payload["undo_gpu_bytes"] = undo_memory.gpu_bytes;
                payload["redo_cpu_bytes"] = redo_memory.cpu_bytes;
                payload["redo_gpu_bytes"] = redo_memory.gpu_bytes;
                payload["transaction_cpu_bytes"] = transaction_memory.cpu_bytes;
                payload["transaction_gpu_bytes"] = transaction_memory.gpu_bytes;
                payload["total_cpu_bytes"] = total_memory.cpu_bytes;
                payload["total_gpu_bytes"] = total_memory.gpu_bytes;
                payload["transaction_active"] = vis::op::undoHistory().hasActiveTransaction();
                payload["transaction_depth"] = vis::op::undoHistory().transactionDepth();
                payload["transaction_name"] = vis::op::undoHistory().activeTransactionName();
                payload["transaction_age_ms"] = vis::op::undoHistory().transactionAgeMs();
                payload["generation"] = vis::op::undoHistory().generation();
                return payload;
            },
            "Get the structured undo/redo stack state");

        nb::class_<PyTransaction>(undo, "Transaction")
            .def(nb::init<const std::string&>(), nb::arg("name") = "Grouped Changes")
            .def(
                "__enter__", [](PyTransaction& self) { self.enter(); return &self; }, "Begin transaction context")
            .def(
                "__exit__", [](PyTransaction& self, nb::args args) {
                    const bool commit = args.size() > 0 ? args[0].is_none() : true;
                    self.exit(commit);
                    return false;
                },
                "Commit transaction on context exit")
            .def("add", &PyTransaction::add, nb::arg("undo"), nb::arg("redo"), "Add an undo/redo pair to the transaction");

        undo.def(
            "transaction", [](const std::string& name) {
                return PyTransaction(name);
            },
            nb::arg("name") = "Grouped Changes", "Create a transaction for grouping undo steps");
    }

} // namespace lfs::python
