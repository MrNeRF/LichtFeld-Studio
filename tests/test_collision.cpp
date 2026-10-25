/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/cuda/collision_voxelization.hpp"
#include <gtest/gtest.h>
using namespace lfs::core;

TEST(Collision, LogicalBoundsAndPackedBricks) {
    CollisionVolume v;
    v.edge = 4;
    v.dimensions = {1, 3, 3};
    v.words = {CollisionVolume::brick_tag | 1, 0x00110011, 0};
    v.occupied_count = 4;
    ASSERT_NO_THROW(v.validate());
    EXPECT_TRUE(v.occupied({0, 1, 1}));
    EXPECT_FALSE(v.occupied({1, 1, 1}));
    EXPECT_FALSE(v.occupied({0, 3, 1}));
    EXPECT_FALSE(v.occupied({-1, 0, 0}));
    v.words[1] |= 2;
    EXPECT_THROW(v.validate(), std::invalid_argument);
    v.words = {1};
    v.occupied_count = 9;
    EXPECT_NO_THROW(v.validate());
}

TEST(Collision, MalformedTopologyAndSettings) {
    CollisionVolume v;
    v.edge = 8;
    v.dimensions = {5, 5, 5};
    v.words.assign(9, 0);
    v.words[0] = 3;
    EXPECT_NO_THROW(v.validate());
    for (const auto pointer : {2u, 4u, 0x80000000u, 0xffffffffu}) {
        v.words[1] = pointer;
        EXPECT_THROW(v.validate(), std::invalid_argument);
    }
    v.words = {1};
    v.occupied_count = 125;
    EXPECT_NO_THROW(v.validate());
    v.occupied_count++;
    EXPECT_THROW(v.validate(), std::invalid_argument);
    for (const float invalid : {0.f, -1.f, INFINITY, NAN}) {
        v.settings.voxel_size = invalid;
        EXPECT_THROW(v.settings.validate(), std::invalid_argument);
    }
    std::stop_source stop;
    stop.request_stop();
    EXPECT_THROW(generateCollision({}, {}, stop.get_token()), std::runtime_error);
    const CollisionSource invalid;
    EXPECT_THROW(generateCollision({&invalid, 1}, {}), std::invalid_argument);
}

TEST(Collision, OutputLayoutAcrossDispatchesAndAddressOverflow) {
    size_t records = 0;
    for (size_t batch = 0; batch < 61; ++batch) {
        const auto bytes = cuda::collisionOutputBytes(records, 1u << 20);
        records = bytes / sizeof(cuda::CollisionBrick);
        EXPECT_EQ(bytes, (batch + 1) * (1u << 20) * sizeof(cuda::CollisionBrick));
    }
    EXPECT_EQ(records, 63'963'136u);
    EXPECT_THROW(cuda::collisionOutputBytes(records, 1u << 20), std::length_error);
    EXPECT_EQ(cuda::collisionOutputBytes(records, 36'864), size_t(64'000'000) * sizeof(cuda::CollisionBrick));
    EXPECT_THROW(cuda::collisionOutputBytes(records, 36'865), std::length_error);
    EXPECT_EQ(cuda::collisionOutputBytes(records, 17), cuda::collisionOutputBytes(records + 17));
    EXPECT_THROW(cuda::collisionOutputBytes(SIZE_MAX, 1), std::length_error);
    EXPECT_THROW(cuda::collisionOutputBytes(std::vector<cuda::CollisionBrick>().max_size(), 1), std::length_error);
}
