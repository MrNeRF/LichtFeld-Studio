/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "builtin_common.hpp"
#include "core/nodes/curve.hpp"
#include "core/tensor_procedural.hpp"
namespace lfs::nodes::builtin {
    inline std::vector<CurvePoint> curve_points(const NodeContext& context, std::string_view property) {
        return curve_control_points(context.properties().value(std::string(property), nlohmann::json::array()));
    }
    inline std::vector<core::CurveKnot> curve_knots(const std::vector<CurvePoint>& points) {
        std::vector<core::CurveKnot> result;
        result.reserve(points.size());
        for (const auto& point : points)
            result.push_back({point.x, point.y, point.tangent});
        return result;
    }
    inline Tensor evaluate_curve(const Tensor& x, const std::vector<CurvePoint>& points, bool clamp) {
        return core::interpolate_curve(x, curve_knots(points), clamp);
    }

} // namespace lfs::nodes::builtin
