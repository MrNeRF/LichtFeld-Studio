/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/uuid.hpp"

#include <gtest/gtest.h>

#include <thread>
#include <unordered_set>
#include <vector>

TEST(UuidTest, NilAndCanonicalTextRoundTrip) {
    const lfs::core::Uuid nil;
    EXPECT_TRUE(nil.is_nil());
    EXPECT_EQ(nil.to_string(), "00000000-0000-0000-0000-000000000000");

    const auto parsed = lfs::core::Uuid::from_string("00112233-4455-6677-8899-aabbccddeeff");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_FALSE(parsed->is_nil());
    EXPECT_EQ(parsed->to_string(), "00112233-4455-6677-8899-aabbccddeeff");
    EXPECT_EQ(lfs::core::Uuid::from_string("00112233445566778899aabbccddeeff"), std::nullopt);
    EXPECT_EQ(lfs::core::Uuid::from_string("00112233-4455-6677-8899-aabbccddeezz"), std::nullopt);
}

TEST(UuidTest, GeneratedValuesUseVersionFourAndRfc4122Variant) {
    for (int i = 0; i < 128; ++i) {
        const auto uuid = lfs::core::generate_uuid_v4();
        EXPECT_FALSE(uuid.is_nil());
        EXPECT_EQ(uuid.bytes[6] & 0xf0, 0x40);
        EXPECT_EQ(uuid.bytes[8] & 0xc0, 0x80);

        const auto reparsed = lfs::core::Uuid::from_string(uuid.to_string());
        ASSERT_TRUE(reparsed.has_value());
        EXPECT_EQ(*reparsed, uuid);
    }
}

TEST(UuidTest, ConcurrentGenerationProducesUniqueValues) {
    constexpr std::size_t THREAD_COUNT = 8;
    constexpr std::size_t UUIDS_PER_THREAD = 128;
    std::vector<lfs::core::Uuid> generated(THREAD_COUNT * UUIDS_PER_THREAD);
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);

    for (std::size_t thread_index = 0; thread_index < THREAD_COUNT; ++thread_index) {
        threads.emplace_back([&, thread_index] {
            const std::size_t begin = thread_index * UUIDS_PER_THREAD;
            for (std::size_t i = 0; i < UUIDS_PER_THREAD; ++i) {
                generated[begin + i] = lfs::core::generate_uuid_v4();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    std::unordered_set<lfs::core::Uuid> unique(generated.begin(), generated.end());
    EXPECT_EQ(unique.size(), generated.size());
}
