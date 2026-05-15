/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "operator/operator.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace lfs::vis::op {

    class AlignPickPointOperator : public Operator {
    public:
        static const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    private:
        std::vector<glm::vec3> picked_points_;
        int pick_button_ = 0;

        glm::vec3 unprojectScreenPoint(const OperatorContext& ctx, double x, double y) const;
        void applyAlignment(OperatorContext& ctx);
    };

    void registerAlignOperators();
    void unregisterAlignOperators();

} // namespace lfs::vis::op
