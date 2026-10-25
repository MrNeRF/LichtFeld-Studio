/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_fixture.hpp"
#include "core/collision_walk.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
namespace {
    using namespace lfs::core;
    using collision_test::box;
    TEST(CollisionWalk, AnalyticalClearanceAndTransformedDiagonalSegment) {
        const auto volume = box({0, 0, 0}, {1, 1, 1});
        const glm::mat4 identity(1);
        EXPECT_FALSE(collisionCapsuleIntersects(*volume, identity, {-0.25f, 0, .5f}, {0, 1, 0}, 2, .25f));
        EXPECT_TRUE(collisionCapsuleIntersects(*volume, identity, {-0.249f, 0, .5f}, {0, 1, 0}, 2, .25f));
        EXPECT_FALSE(collisionCapsuleIntersects(*volume, identity, {1.2f, 1.2f, .5f}, {0, 0, 1}, .4f, .2f));
        const auto rotation = glm::rotate(identity, .5f, glm::vec3(0, 0, 1));
        EXPECT_TRUE(collisionCapsuleIntersects(*volume, rotation, {.3f, .3f, .5f}, {0, 1, 0}, 1, .1f));
    }

    TEST(CollisionWalk, SettlesAndSweepsThinWallWithoutFallingForever) {
        const auto floor = box({-2, -1, -2}, {4, 1, 4});
        const auto wall = box({.5f, 0, -2}, {1, 400, 400}, .01f);
        const CapsuleQuery query = [&](auto feet, auto up, float height, float radius) {
            return collisionCapsuleIntersects(*floor, glm::mat4(1), feet, up, height, radius) ||
                   collisionCapsuleIntersects(*wall, glm::mat4(1), feet, up, height, radius);
        };
        WalkSettings settings;
        WalkState state{.eye = {0, 2, -1.99f}, .presentation = {0, 2, -1.99f}};
        settleWalk(state, settings, {0, 1, 0}, query);
        EXPECT_NEAR(state.eye.y, settings.height, .001f);
        for (int i = 0; i < 500; ++i)
            advanceWalk(state, settings, {0, 1, 0}, {10, 0, 0}, .1f, query);
        EXPECT_LE(state.eye.x, .251f);
        WalkState unsupported{.eye = {100, 100, 100}};
        EXPECT_THROW(settleWalk(unsupported, settings, {0, 1, 0}, query), std::runtime_error);
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {0, 0, 100}, .1f, query), std::runtime_error);
    }

    TEST(CollisionWalk, MinimumStepFitsBelowCeilingAndLowerSlabBlocksLift) {
        const auto floor = box({-2, -1, -2}, {4, 1, 4});
        const auto step = box({.5f, 0, -2}, {100, 10, 400}, .01f);
        auto run = [&](float ceiling_height) {
            const auto ceiling = box({-2, ceiling_height, -2}, {400, 10, 400}, .01f);
            const CapsuleQuery query = [&](auto feet, auto up, float height, float radius) {
                for (const auto& volume : {floor, step, ceiling})
                    if (collisionCapsuleIntersects(*volume, glm::mat4(1), feet, up, height, radius))
                        return true;
                return false;
            };
            WalkSettings settings;
            WalkState state{.eye = {0, 1.7f, 0}, .presentation = {0, 1.7f, 0}};
            settleWalk(state, settings, {0, 1, 0}, query);
            for (int i = 0; i < 50; ++i) {
                SCOPED_TRACE(::testing::Message() << ceiling_height << " step " << i << " eye " << state.eye.x << "," << state.eye.y);
                EXPECT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, {.5f, 0, 0}, .05f, query));
                if (::testing::Test::HasFailure())
                    break;
            }
            for (int i = 0; i < 500; ++i) {
                EXPECT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query));
                EXPECT_FALSE(query(state.feet, {0, 1, 0}, settings.height, settings.radius));
            }
            return state;
        };
        EXPECT_GT(run(1.85f).eye.x, 1);
        EXPECT_NEAR(run(1.85f).eye.y, 1.8f, .001f);
        EXPECT_LT(run(1.75f).eye.x, .5f);
        EXPECT_LE(run(1.75f).presentation.y, 1.75f);
    }

    TEST(CollisionWalk, RepeatedGroundedFramesAcrossTransforms) {
        const auto floor = box({-20, -1, -20}, {40, 1, 40});
        for (float translation : {0.f, 100.f, 1000.f, -10000.f})
            for (float scale : {.01f, 1.f, 100.f})
                for (float angle : {0.f, .37f, 1.1f}) {
                    SCOPED_TRACE(::testing::Message() << translation << " scale " << scale << " angle " << angle);
                    auto transform = glm::translate(glm::mat4(1), glm::vec3(translation, translation, translation));
                    transform = glm::scale(glm::rotate(transform, angle, glm::normalize(glm::vec3(1, 2, 3))), glm::vec3(scale));
                    const auto up = glm::normalize(glm::vec3(transform * glm::vec4(0, 1, 0, 0)));
                    WalkSettings settings{1.7f * scale, .25f * scale, .3f * scale, 2 * scale};
                    const CapsuleQuery query = [&](auto feet, auto axis, float height, float radius) {
                        return collisionCapsuleIntersects(*floor, transform, feet, axis, height, radius);
                    };
                    WalkState state{.eye = glm::vec3(transform * glm::vec4(0, 2, 0, 1))};
                    ASSERT_NO_THROW(settleWalk(state, settings, up, query));
                    for (int frame = 0; frame < 200; ++frame) {
                        SCOPED_TRACE(frame);
                        const auto velocity = frame < 100 ? glm::vec3(0) : glm::vec3(transform * glm::vec4(.2f, 0, 0, 0));
                        ASSERT_NO_THROW(advanceWalk(state, settings, up, velocity, .016f, query));
                        ASSERT_FALSE(query(state.feet, up, settings.height, settings.radius));
                    }
                }
    }

    TEST(CollisionWalk, GroundContactNearCoordinatePrecisionBoundaries) {
        for (float altitude : {.1234567f, 1.99999f, 15.9999f, 127.999f, 1023.9999f, -1024.0001f, 8191.9995f})
            for (float height : {.7f, 1.7f, 3.14159f, 17.f})
                for (float radius : {.025f, .25f}) {
                    SCOPED_TRACE(::testing::Message() << altitude << " height " << height << " radius " << radius);
                    const auto floor = box({-20, altitude - 1, -20}, {40, 1, 40});
                    const CapsuleQuery query = [&](auto feet, auto up, float h, float r) {
                        return collisionCapsuleIntersects(*floor, glm::mat4(1), feet, up, h, r);
                    };
                    WalkSettings settings{height, radius, .3f, 2};
                    WalkState state{.eye = {0, altitude + height + radius, 0}};
                    ASSERT_NO_THROW(settleWalk(state, settings, {0, 1, 0}, query));
                    const auto grounded = state.feet;
                    for (int frame = 0; frame < 500; ++frame) {
                        ASSERT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query));
                        ASSERT_EQ(state.feet, grounded);
                        ASSERT_FALSE(query(state.feet, {0, 1, 0}, height, radius));
                    }
                }
    }

    TEST(CollisionWalk, RealGeometryAndCapsuleChangesStillRejectOverlap) {
        const auto floor = box({-2, -1, -2}, {4, 1, 4});
        const auto ceiling = box({-2, 2, -2}, {4, 1, 4});
        auto floor_transform = glm::mat4(1);
        const CapsuleQuery query = [&](auto feet, auto up, float height, float radius) {
            return collisionCapsuleIntersects(*floor, floor_transform, feet, up, height, radius) ||
                   collisionCapsuleIntersects(*ceiling, glm::mat4(1), feet, up, height, radius);
        };
        WalkSettings settings;
        WalkState state{.eye = {0, 1.8f, 0}};
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query), std::invalid_argument);
        settleWalk(state, settings, {0, 1, 0}, query);
        const auto feet = state.feet;
        floor_transform[3].y = .1f;
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query), std::runtime_error);
        EXPECT_EQ(state.feet, feet);
        floor_transform[3].y = 0;
        settings.height = 2.1f;
        EXPECT_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query), std::runtime_error);
        EXPECT_EQ(state.feet, feet);
        settings.height = 1.7f;
        EXPECT_NO_THROW(advanceWalk(state, settings, {0, 1, 0}, {}, .016f, query));
    }
} // namespace
