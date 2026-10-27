/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "point_pick_operator.hpp"
#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace lfs::vis::op {

    class LFS_VIS_API ScaleDistancePickPointOperator : public PointPickOperator {
    public:
        ScaleDistancePickPointOperator();

        static LFS_LOCAL_SYMBOL const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;

    private:
        friend class ScaleDistanceOperatorTest;

        void setPickedPointsForTesting(const std::vector<glm::vec3>& pts) {
            picked_points_ = pts;
            syncPickedPointsToServices();
        }
        bool applyPickedPoints(OperatorContext& ctx, float target_distance) override;
    };

    void registerScaleDistanceOperators();
    void unregisterScaleDistanceOperators();

    // Compute scale matrix around the midpoint of p0 and p1 that rescales distance ||p1 - p0|| to target_distance.
    // Returns std::nullopt if points are too close (distance < 1e-6) or target_distance <= 1e-6.
    [[nodiscard]] LFS_VIS_API std::optional<glm::mat4> computeScaleDistanceTransform(
        const glm::vec3& p0, const glm::vec3& p1, float target_distance);

} // namespace lfs::vis::op
