/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operator/ops/align_ops.hpp"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace {

    [[nodiscard]] glm::vec3 applyRot(const glm::mat4& m, const glm::vec3& v) {
        return glm::mat3(m) * v;
    }

    TEST(AlignEdgeToWorldX, IdentityUpMapsPositiveZEdgeToPlusX) {
        const glm::mat4 up(1.0f);
        const glm::vec3 p0(0.0f, 0.0f, 0.0f);
        const glm::vec3 p1(0.0f, 0.0f, 2.0f); // +Z edge

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up, p0, p1);
        const glm::mat4 total = yaw * up;
        const glm::vec3 rotated = applyRot(total, p1 - p0);

        EXPECT_NEAR(rotated.x, 2.0f, 1e-5f);
        EXPECT_NEAR(rotated.y, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated.z, 0.0f, 1e-5f);
    }

    TEST(AlignEdgeToWorldX, IdentityUpMapsNegativeXEdgeToPlusX) {
        const glm::mat4 up(1.0f);
        const glm::vec3 p0(1.0f, 0.0f, 0.0f);
        const glm::vec3 p1(-1.0f, 0.0f, 0.0f); // -X edge length 2

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up, p0, p1);
        const glm::vec3 rotated = applyRot(yaw * up, p1 - p0);

        EXPECT_NEAR(rotated.x, 2.0f, 1e-5f);
        EXPECT_NEAR(rotated.y, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated.z, 0.0f, 1e-5f);
    }

    TEST(AlignEdgeToWorldX, DegenerateVerticalEdgeYieldsIdentity) {
        const glm::mat4 up(1.0f);
        const glm::vec3 p0(0.0f, 0.0f, 0.0f);
        const glm::vec3 p1(0.0f, 3.0f, 0.0f); // pure +Y — zero ground projection

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up, p0, p1);
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                const float expected = (row == col) ? 1.0f : 0.0f;
                EXPECT_NEAR(yaw[col][row], expected, 1e-6f);
            }
        }
    }

    TEST(AlignEdgeToWorldX, ComposesWithUpRotation) {
        // Plane originally facing +Z; rotate normal to +Y then yaw edge to +X.
        const glm::vec3 p0(0.0f, 0.0f, 0.0f);
        const glm::vec3 p1(1.0f, 0.0f, 0.0f);
        const glm::vec3 p2(0.0f, 1.0f, 0.0f);
        const glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0)); // +Z
        const glm::vec3 target_up(0.0f, 1.0f, 0.0f);
        const glm::vec3 axis = glm::normalize(glm::cross(normal, target_up));
        const float angle = std::acos(glm::clamp(glm::dot(normal, target_up), -1.0f, 1.0f));
        const glm::mat4 up_rot = glm::rotate(glm::mat4(1.0f), angle, axis);

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up_rot, p0, p1);
        const glm::mat4 total = yaw * up_rot;
        const glm::vec3 rotated_edge = applyRot(total, p1 - p0);
        const glm::vec3 rotated_normal = applyRot(total, normal);

        EXPECT_NEAR(rotated_normal.x, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated_normal.y, 1.0f, 1e-5f);
        EXPECT_NEAR(rotated_normal.z, 0.0f, 1e-5f);
        EXPECT_GT(rotated_edge.x, 0.5f);
        EXPECT_NEAR(rotated_edge.z, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated_edge.y, 0.0f, 1e-4f);
    }

    TEST(AlignTransform, NormalMapsToPlusY) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 1.0f, 0.0f);
        in.p1 = glm::vec3(2.0f, 1.4f, 0.3f);
        in.p2 = glm::vec3(0.4f, 0.2f, 1.7f);
        in.camera_pos = glm::vec3(0.0f, 8.0f, 4.0f);

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        glm::vec3 normal = glm::normalize(glm::cross(in.p1 - in.p0, in.p2 - in.p0));
        lfs::vis::op::faceNormalTowardCamera(normal, (in.p0 + in.p1 + in.p2) / 3.0f, in.camera_pos);
        const glm::vec3 transformed = glm::mat3(*xform) * normal;
        EXPECT_NEAR(transformed.x, 0.0f, 1e-5f);
        EXPECT_NEAR(transformed.y, 1.0f, 1e-5f);
        EXPECT_NEAR(transformed.z, 0.0f, 1e-5f);
    }

    TEST(AlignTransform, CentroidLandsOnGround) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(1.0f, 2.0f, 3.0f);
        in.p1 = glm::vec3(2.0f, 2.5f, 3.4f);
        in.p2 = glm::vec3(1.4f, 1.2f, 4.1f);
        in.camera_pos = glm::vec3(1.5f, 9.0f, 3.5f);

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        const glm::vec3 center = (in.p0 + in.p1 + in.p2) / 3.0f;
        const glm::vec3 transformed = glm::vec3((*xform) * glm::vec4(center, 1.0f));
        EXPECT_NEAR(transformed.y, 0.0f, 1e-5f);
        EXPECT_NEAR(transformed.x, center.x, 1e-5f);
        EXPECT_NEAR(transformed.z, center.z, 1e-5f);
    }

    TEST(AlignTransform, AntiParallelNormalRotates180) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 1.0f, 0.0f);
        in.p1 = glm::vec3(1.0f, 1.0f, 0.0f);
        in.p2 = glm::vec3(0.0f, 1.0f, 1.0f);
        in.camera_pos = glm::vec3(0.0f, -5.0f, 0.0f);

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        glm::vec3 normal = glm::normalize(glm::cross(in.p1 - in.p0, in.p2 - in.p0));
        lfs::vis::op::faceNormalTowardCamera(normal, (in.p0 + in.p1 + in.p2) / 3.0f, in.camera_pos);
        EXPECT_LT(normal.y, 0.0f);

        const glm::vec3 transformed = glm::normalize(glm::mat3(*xform) * normal);
        EXPECT_NEAR(transformed.x, 0.0f, 1e-5f);
        EXPECT_NEAR(transformed.y, 1.0f, 1e-5f);
        EXPECT_NEAR(transformed.z, 0.0f, 1e-5f);
    }

    TEST(AlignTransform, AxisSnapWithinTolerance) {
        glm::vec3 normal = glm::normalize(glm::vec3(1.0f, std::tan(glm::radians(2.0f)), 0.0f));
        EXPECT_TRUE(lfs::vis::op::snapAlignNormalToNodeAxes(normal, glm::mat4(1.0f)));
        EXPECT_NEAR(normal.x, 1.0f, 1e-5f);
        EXPECT_NEAR(normal.y, 0.0f, 1e-5f);
        EXPECT_NEAR(normal.z, 0.0f, 1e-5f);
    }

    TEST(AlignTransform, AxisSnapOutsideTolerance) {
        glm::vec3 normal = glm::normalize(glm::vec3(1.0f, std::tan(glm::radians(5.0f)), 0.0f));
        const glm::vec3 original = normal;
        EXPECT_FALSE(lfs::vis::op::snapAlignNormalToNodeAxes(normal, glm::mat4(1.0f)));
        EXPECT_NEAR(normal.x, original.x, 1e-6f);
        EXPECT_NEAR(normal.y, original.y, 1e-6f);
        EXPECT_NEAR(normal.z, original.z, 1e-6f);
    }

    TEST(AlignTransform, EdgeToWorldXComposes) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 0.0f, 0.0f);
        in.p1 = glm::vec3(0.0f, 0.0f, 2.0f);
        in.p2 = glm::vec3(1.0f, 0.0f, 0.0f);
        in.camera_pos = glm::vec3(0.0f, 5.0f, 0.0f);
        in.edge_to_world_x = true;

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        const glm::vec3 edge = glm::mat3(*xform) * (in.p1 - in.p0);
        EXPECT_NEAR(edge.z, 0.0f, 1e-5f);
        EXPECT_GT(edge.x, 0.0f);
    }

    TEST(AlignTransform, DegenerateReturnsNullopt) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 0.0f, 0.0f);
        in.p1 = glm::vec3(1.0f, 0.0f, 0.0f);
        in.p2 = glm::vec3(2.0f, 0.0f, 0.0f);
        in.camera_pos = glm::vec3(0.0f, 5.0f, 0.0f);
        EXPECT_FALSE(lfs::vis::op::computeAlignTransform(in).has_value());
    }

} // namespace
