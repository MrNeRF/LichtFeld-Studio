/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "operator/operator.hpp"
#include "selection/selection_service.hpp"
#include "core/export.hpp"

namespace lfs::vis::op {

    class LFS_VIS_API SelectionStrokeOperator : public Operator {
    public:
        static const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    private:
        lfs::vis::SelectionShape shape_ = lfs::vis::SelectionShape::Brush;
        lfs::vis::SelectionMode mode_ = lfs::vis::SelectionMode::Replace;
        float brush_radius_ = 20.0f;
        SelectionFilterState filters_{};
    };

    void registerSelectionOperators();
    void unregisterSelectionOperators();

} // namespace lfs::vis::op
