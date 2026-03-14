/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "visualizer/operation/undo_entry.hpp"

#include <nanobind/nanobind.h>

#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace lfs::python {

    class PyUndoEntry final : public vis::op::UndoEntry {
    public:
        PyUndoEntry(std::string name, nb::object undo_fn, nb::object redo_fn);

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string name() const override { return name_; }
        [[nodiscard]] vis::op::UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override { return sizeof(*this) + name_.size() + 128; }

    private:
        std::string name_;
        nb::object undo_fn_;
        nb::object redo_fn_;
    };

    class PyTransaction {
    public:
        explicit PyTransaction(std::string name);

        void enter();
        void exit(bool commit = true);
        void add(nb::object undo_fn, nb::object redo_fn);

    private:
        std::string name_;
        bool active_ = false;
    };

    void register_commands(nb::module_& m);

} // namespace lfs::python
