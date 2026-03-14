/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_command.hpp"
#include "core/logger.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>

namespace lfs::python {

    PyUndoEntry::PyUndoEntry(std::string name, nb::object undo_fn, nb::object redo_fn)
        : name_(std::move(name)),
          undo_fn_(std::move(undo_fn)),
          redo_fn_(std::move(redo_fn)) {}

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
            .id = "python.custom",
            .label = name_,
            .source = "python",
            .scope = "custom",
        };
    }

    PyTransaction::PyTransaction(std::string name)
        : name_(std::move(name)) {}

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
        try {
            if (redo_fn.is_valid() && !redo_fn.is_none())
                redo_fn();
        } catch (const std::exception& e) {
            LOG_ERROR("Transaction add: {}", e.what());
            throw;
        }

        auto entry = std::make_unique<PyUndoEntry>(name_, std::move(undo_fn), std::move(redo_fn));
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
            [](const std::string& name, nb::object undo_fn, nb::object redo_fn, bool validate) {
                if (validate) {
                    size_t dot_count = std::count(name.begin(), name.end(), '.');
                    bool has_space = name.find(' ') != std::string::npos;
                    if (dot_count != 1 || has_space) {
                        LOG_WARN("lf.undo.push(): Operation name '{}' should be 'category.action' format", name);
                    }
                }
                auto entry = std::make_unique<PyUndoEntry>(name, std::move(undo_fn), std::move(redo_fn));
                vis::op::undoHistory().push(std::move(entry));
            },
            nb::arg("name"), nb::arg("undo"), nb::arg("redo"), nb::arg("validate") = false,
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
            "total_bytes",
            []() { return vis::op::undoHistory().totalBytes(); },
            "Get estimated bytes retained by undo and redo history");

        undo.def(
            "has_active_transaction",
            []() { return vis::op::undoHistory().hasActiveTransaction(); },
            "Check if a grouped history transaction is active");

        undo.def(
            "transaction_depth",
            []() { return vis::op::undoHistory().transactionDepth(); },
            "Get the current grouped history transaction nesting depth");

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
                payload["total_bytes"] = vis::op::undoHistory().totalBytes();
                payload["transaction_active"] = vis::op::undoHistory().hasActiveTransaction();
                payload["transaction_depth"] = vis::op::undoHistory().transactionDepth();
                payload["transaction_name"] = vis::op::undoHistory().activeTransactionName();
                payload["generation"] = vis::op::undoHistory().generation();
                return payload;
            },
            "Get the structured undo/redo stack state");

        nb::class_<PyTransaction>(undo, "Transaction")
            .def(nb::init<const std::string&>(), nb::arg("name") = "Grouped Changes")
            .def(
                "__enter__", [](PyTransaction& self) { self.enter(); return &self; }, "Begin transaction context")
            .def(
                "__exit__", [](PyTransaction& self, nb::object exc_type, nb::object, nb::object) {
                    self.exit(exc_type.is_none());
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
