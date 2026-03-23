/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "operator/operator.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace lfs::vis::op {

    enum class BrushMode : uint8_t {
        Select,    // 0 - selection mode
        Saturation // 1 - saturation adjustment mode
    };

    enum class BrushAction : uint8_t {
        Add,   // 0 - add to selection / increase saturation
        Remove // 1 - remove from selection / decrease saturation
    };

    class BrushStrokeOperator : public Operator {
    public:
        static const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    private:
        BrushMode mode_ = BrushMode::Select;
        BrushAction action_ = BrushAction::Add;
        float brush_radius_ = 20.0f;
        float saturation_amount_ = 0.5f;
        glm::vec2 last_stroke_pos_{0.0f};

        // Selection mode state
        lfs::core::Tensor cumulative_selection_;
        std::shared_ptr<lfs::core::Tensor> selection_before_;

        // Saturation mode state
        std::shared_ptr<lfs::core::Tensor> sh0_before_;
        std::string saturation_node_name_;

        void beginSelectionStroke(OperatorContext& ctx);
        void beginSaturationStroke(OperatorContext& ctx);
        void updateSelectionAtPoint(double x, double y, OperatorContext& ctx);
        void updateSaturationAtPoint(double x, double y, OperatorContext& ctx);
        void finalizeSelectionStroke(OperatorContext& ctx);
        void finalizeSaturationStroke(OperatorContext& ctx);
        void clearBrushState();
    };

    void registerBrushOperators();
    void unregisterBrushOperators();

} // namespace lfs::vis::op
