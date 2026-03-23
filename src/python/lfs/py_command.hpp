/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "visualizer/operation/undo_entry.hpp"
#include "visualizer/rendering/dirty_flags.hpp"

#include <nanobind/nanobind.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace lfs::python {

    class PyUndoEntry final : public vis::op::UndoEntry {
    public:
        PyUndoEntry(std::string name,
                    nb::object undo_fn,
                    nb::object redo_fn,
                    std::string id = {},
                    std::string source = "python",
                    std::string scope = "custom",
                    size_t estimated_bytes = 0,
                    vis::DirtyMask dirty_flags = vis::DirtyFlag::ALL,
                    std::chrono::milliseconds merge_window = std::chrono::milliseconds{0});
        ~PyUndoEntry() override;

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string name() const override { return name_; }
        [[nodiscard]] vis::op::UndoMetadata metadata() const override;
        [[nodiscard]] size_t estimatedBytes() const override;
        [[nodiscard]] vis::DirtyMask dirtyFlags() const override;
        bool tryMerge(const vis::op::UndoEntry& incoming) override;

    private:
        std::string name_;
        std::string id_;
        std::string source_;
        std::string scope_;
        nb::object undo_fn_;
        nb::object redo_fn_;
        size_t estimated_bytes_ = 0;
        vis::DirtyMask dirty_flags_ = vis::DirtyFlag::ALL;
        std::chrono::steady_clock::time_point updated_at_;
        std::chrono::milliseconds merge_window_{0};
    };

    class PyTransaction {
    public:
        explicit PyTransaction(std::string name);
        ~PyTransaction();

        void enter();
        void exit(bool commit = true);
        void add(nb::object undo_fn, nb::object redo_fn);

    private:
        std::string name_;
        bool active_ = false;
    };

    void register_commands(nb::module_& m);

} // namespace lfs::python
