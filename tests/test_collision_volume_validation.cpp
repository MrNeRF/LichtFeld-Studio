/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_volume.hpp"
#include "project/collision_serialization.hpp"
#include <bit>
#include <gtest/gtest.h>
#include <limits>

namespace {
    using lfs::core::CollisionVolume;

    CollisionVolume brick(glm::ivec3 dimensions, uint64_t mask) {
        CollisionVolume volume;
        volume.dimensions = dimensions;
        volume.edge = 4;
        volume.words = {CollisionVolume::brick_tag | 1u, uint32_t(mask), uint32_t(mask >> 32)};
        volume.occupied_count = std::popcount(mask);
        return volume;
    }

    TEST(CollisionVolumeValidation, EveryBoundaryCellRetainsTheSamePaddingRules) {
        for (int z = 1; z <= 4; ++z)
            for (int y = 1; y <= 4; ++y)
                for (int x = 1; x <= 4; ++x)
                    for (int bit = 0; bit < 64; ++bit) {
                        auto volume = brick({x, y, z}, uint64_t{1} << bit);
                        const glm::ivec3 cell{bit & 3, (bit >> 2) & 3, bit >> 4};
                        if (glm::all(glm::lessThan(cell, volume.dimensions))) {
                            EXPECT_NO_THROW(volume.validate());
                            EXPECT_TRUE(volume.occupied(cell));
                        } else {
                            EXPECT_THROW(volume.validate(), std::exception);
                        }
                    }
    }

    TEST(CollisionVolumeValidation, FullInteriorMaskCountMatchesAnExhaustiveCellScan) {
        uint64_t random = 0x123456789abcdef0ull;
        for (int i = 0; i < 512; ++i) {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            auto volume = brick({4, 4, 4}, random);
            uint64_t count = 0;
            for (int z = 0; z < 4; ++z)
                for (int y = 0; y < 4; ++y)
                    for (int x = 0; x < 4; ++x)
                        count += volume.occupied({x, y, z});
            EXPECT_EQ(count, volume.occupied_count);
            EXPECT_NO_THROW(volume.validate());
            ++volume.occupied_count;
            EXPECT_THROW(volume.validate(), std::exception);
        }
    }

    TEST(CollisionVolumeValidation, RejectsSharedPointersUnreachableWordsAndInvalidBounds) {
        auto volume = brick({4, 4, 4}, ~uint64_t{0});
        volume.words.push_back(0);
        EXPECT_THROW(volume.validate(), std::exception);
        volume = brick({4, 4, 4}, 1);
        volume.words[0] = CollisionVolume::brick_tag; // mask overlaps its node
        EXPECT_THROW(volume.validate(), std::exception);
        volume = brick({4, 4, 4}, 1);
        volume.words[0] = CollisionVolume::brick_tag | 100u;
        EXPECT_THROW(volume.validate(), std::exception);
        volume = brick({4, 4, 4}, 1);
        volume.origin.x = std::numeric_limits<float>::quiet_NaN();
        EXPECT_THROW(volume.validate(), std::exception);
        volume = brick({4, 4, 4}, 1);
        volume.edge = 3;
        EXPECT_THROW(volume.validate(), std::exception);
        volume = brick({4, 4, 4}, 1);
        volume.edge = 8;
        volume.words = {3, 0x80000009u, 0x80000009u, 0, 0, 0, 0, 0, 0, 1, 0};
        volume.occupied_count = 2;
        EXPECT_THROW(volume.validate(), std::exception); // shared mask words
    }

    TEST(CollisionVolumeValidation, UniformNodesUseClippedBoundsAndRejectUnreachableData) {
        CollisionVolume volume;
        volume.edge = 4;
        volume.dimensions = {3, 2, 1};
        volume.words = {1};
        volume.occupied_count = 6;
        EXPECT_NO_THROW(volume.validate());
        volume.words.push_back(0);
        EXPECT_THROW(volume.validate(), std::exception);
        volume.words = {0};
        volume.occupied_count = 0;
        EXPECT_NO_THROW(volume.validate());
    }

    TEST(CollisionVolumeValidation, EncodedWordsRemainCanonicalLittleEndianAndRoundTripExactly) {
        const auto original = brick({4, 4, 4}, 0x0123456789abcdefull);
        const auto bytes = lfs::io::project::encodeCollision(original);
        ASSERT_EQ(bytes.size(), 68u + original.words.size() * 4);
        for (size_t i = 0; i < original.words.size(); ++i)
            for (size_t byte = 0; byte < 4; ++byte)
                EXPECT_EQ(std::to_integer<uint8_t>(bytes[68 + 4 * i + byte]),
                          uint8_t(original.words[i] >> (8 * byte)));
        auto decoded = lfs::io::project::decodeCollision(bytes);
        EXPECT_EQ(decoded->words, original.words);
        EXPECT_EQ(decoded->occupied_count, original.occupied_count);
        EXPECT_EQ(decoded->dimensions, original.dimensions);
        auto invalid = bytes;
        invalid.pop_back();
        EXPECT_THROW(lfs::io::project::decodeCollision(invalid), std::exception);
        lfs::io::project::chapter_binary::write_u32(invalid = bytes, 64, 100);
        EXPECT_THROW(lfs::io::project::decodeCollision(invalid), std::exception);
    }
} // namespace
