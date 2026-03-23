/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "operator/operator.hpp"

namespace lfs::vis::op {

    class LFS_VIS_API UndoOperator : public Operator {
    public:
        static const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
    };

    class LFS_VIS_API RedoOperator : public Operator {
    public:
        static const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
    };

    class LFS_VIS_API DeleteOperator : public Operator {
    public:
        static const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
    };

    LFS_VIS_API void registerEditOperators();
    LFS_VIS_API void unregisterEditOperators();

} // namespace lfs::vis::op
