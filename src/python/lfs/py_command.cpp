/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_command.hpp"
#include "core/logger.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <nanobind/stl/string.h>

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
        }
    }

    PyTransaction::PyTransaction(std::string name)
        : name_(std::move(name)) {}

    void PyTransaction::enter() {
        active_ = true;
        entries_.clear();
    }

    void PyTransaction::exit(const bool commit) {
        if (!active_)
            return;
        active_ = false;

        if (!commit || entries_.empty())
            return;

        class CompoundEntry final : public vis::op::UndoEntry {
        public:
            CompoundEntry(std::string name, std::vector<std::pair<nb::object, nb::object>> entries)
                : name_(std::move(name)),
                  entries_(std::move(entries)) {}

            void undo() override {
                nb::gil_scoped_acquire gil;
                for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
                    try {
                        if (it->first.is_valid() && !it->first.is_none())
                            it->first();
                    } catch (const std::exception& e) {
                        LOG_ERROR("CompoundEntry undo: {}", e.what());
                    }
                }
            }

            void redo() override {
                nb::gil_scoped_acquire gil;
                for (auto& entry : entries_) {
                    try {
                        if (entry.second.is_valid() && !entry.second.is_none())
                            entry.second();
                    } catch (const std::exception& e) {
                        LOG_ERROR("CompoundEntry redo: {}", e.what());
                    }
                }
            }

            [[nodiscard]] std::string name() const override { return name_; }

        private:
            std::string name_;
            std::vector<std::pair<nb::object, nb::object>> entries_;
        };

        auto compound = std::make_unique<CompoundEntry>(name_, std::move(entries_));
        vis::op::undoHistory().push(std::move(compound));
    }

    void PyTransaction::add(nb::object undo_fn, nb::object redo_fn) {
        if (!active_) {
            auto entry = std::make_unique<PyUndoEntry>(name_, std::move(undo_fn), std::move(redo_fn));
            entry->redo();
            vis::op::undoHistory().push(std::move(entry));
            return;
        }

        nb::gil_scoped_acquire gil;
        try {
            if (redo_fn.is_valid() && !redo_fn.is_none())
                redo_fn();
        } catch (const std::exception& e) {
            LOG_ERROR("Transaction add: {}", e.what());
        }
        entries_.emplace_back(std::move(undo_fn), std::move(redo_fn));
    }

    void register_commands(nb::module_& m) {
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
            "undo", []() { vis::op::undoHistory().undo(); }, "Undo last operation");
        undo.def(
            "redo", []() { vis::op::undoHistory().redo(); }, "Redo last undone operation");
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

        nb::class_<PyTransaction>(undo, "Transaction")
            .def(nb::init<const std::string&>(), nb::arg("name") = "Grouped Changes")
            .def(
                "__enter__", [](PyTransaction& self) { self.enter(); return &self; }, "Begin transaction context")
            .def(
                "__exit__", [](PyTransaction& self, nb::args) {
                    self.exit(true);
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
