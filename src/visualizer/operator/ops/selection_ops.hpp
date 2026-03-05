/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "operator/operator.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace lfs::vis::op {

    enum class SelectionMode : uint8_t {
        Brush,     // 0 - matches SelectionSubMode::Centers
        Rectangle, // 1 - matches SelectionSubMode::Rectangle
        Polygon,   // 2 - matches SelectionSubMode::Polygon
        Lasso,     // 3 - matches SelectionSubMode::Lasso
        Rings      // 4 - matches SelectionSubMode::Rings
    };

    enum class SelectionOp : uint8_t {
        Replace,
        Add,
        Remove
    };

    class SelectionStrokeOperator : public Operator {
    public:
        static const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    private:
        SelectionMode mode_ = SelectionMode::Brush;
        SelectionOp op_ = SelectionOp::Replace;
        lfs::core::Tensor stroke_selection_;
        std::shared_ptr<lfs::core::Tensor> selection_before_;
        glm::vec2 last_stroke_pos_{0.0f};
        float brush_radius_ = 20.0f;
        bool use_depth_filter_ = false;

        // Rectangle mode state
        glm::vec2 rect_start_{0.0f};
        glm::vec2 rect_end_{0.0f};

        // Lasso mode state
        std::vector<glm::vec2> lasso_points_;

        // Polygon mode state (world-space anchored points)
        std::vector<glm::vec3> polygon_world_points_;
        bool polygon_closed_ = false;
        static constexpr float POLYGON_CLOSE_THRESHOLD = 12.0f;

        void beginStroke(OperatorContext& ctx);
        void updateBrushSelection(double x, double y, OperatorContext& ctx);
        void updateRectPreview(OperatorContext& ctx);
        void updateLassoPreview(OperatorContext& ctx);
        void updatePolygonPreview(OperatorContext& ctx);
        void computeRectSelection(OperatorContext& ctx);
        void computeLassoSelection(OperatorContext& ctx);
        void computePolygonSelection(OperatorContext& ctx);
        void finalizeSelection(OperatorContext& ctx);
        void clearPreview(OperatorContext& ctx);
    };

    void registerSelectionOperators();
    void unregisterSelectionOperators();

} // namespace lfs::vis::op
