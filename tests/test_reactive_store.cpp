/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/reactive/observable.hpp"
#include "core/reactive/store.hpp"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

using lfs::core::reactive::BatchUpdate;
using lfs::core::reactive::Observable;
using lfs::core::reactive::Store;

TEST(ReactiveStoreTest, DedupsFieldsWithinFrame) {
    Store store;
    Observable<int> value(store, 1, "value", 0);

    int callback_count = 0;
    int observed = 0;
    auto token = value.subscribe([&](const int& v) {
        ++callback_count;
        observed = v;
    });

    value.set(1);
    value.set(2);

    EXPECT_TRUE(store.drain_dirty_into_frame());
    EXPECT_EQ(callback_count, 1);
    EXPECT_EQ(observed, 2);
    EXPECT_FALSE(store.has_dirty());
}

TEST(ReactiveStoreTest, BatchDefersPublicationUntilEnd) {
    Store store;
    Observable<int> a(store, 1, "a", 0);
    Observable<int> b(store, 2, "b", 0);

    int callback_count = 0;
    auto token_a = a.subscribe([&](const int&) { ++callback_count; });
    auto token_b = b.subscribe([&](const int&) { ++callback_count; });

    {
        BatchUpdate batch(store);
        a.set(1);
        b.set(2);
        EXPECT_FALSE(store.has_dirty());
    }

    EXPECT_TRUE(store.has_dirty());
    EXPECT_TRUE(store.drain_dirty_into_frame());
    EXPECT_EQ(callback_count, 2);
}

TEST(ReactiveStoreTest, ReentrantSetsDrainOnNextFrame) {
    Store store;
    Observable<int> a(store, 1, "a", 0);
    Observable<int> b(store, 2, "b", 0);

    int a_count = 0;
    int b_count = 0;
    auto token_a = a.subscribe([&](const int&) {
        ++a_count;
        b.set(7);
    });
    auto token_b = b.subscribe([&](const int&) { ++b_count; });

    a.set(1);
    EXPECT_TRUE(store.drain_dirty_into_frame());
    EXPECT_EQ(a_count, 1);
    EXPECT_EQ(b_count, 0);
    EXPECT_TRUE(store.has_dirty());

    EXPECT_TRUE(store.drain_dirty_into_frame());
    EXPECT_EQ(a_count, 1);
    EXPECT_EQ(b_count, 1);
}

TEST(ReactiveStoreTest, MultiThreadProducersSingleGuiDrain) {
    constexpr int kFields = 100;
    constexpr int kThreads = 64;
    constexpr int kWritesPerThread = 1000;

    Store store;
    std::vector<std::unique_ptr<Observable<int>>> fields;
    fields.reserve(kFields);
    for (int i = 0; i < kFields; ++i)
        fields.push_back(std::make_unique<Observable<int>>(store, i + 1, "field", 0));

    std::atomic<int> callback_count{0};
    std::vector<lfs::core::reactive::SubscriptionToken> tokens;
    tokens.reserve(kFields);
    for (auto& field : fields) {
        tokens.push_back(field->subscribe([&](const int&) {
            callback_count.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    std::vector<std::thread> producers;
    producers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        producers.emplace_back([&, t] {
            for (int i = 0; i < kWritesPerThread; ++i) {
                auto& field = *fields[(t + i) % kFields];
                field.set(t * kWritesPerThread + i + 1);
            }
        });
    }

    for (auto& producer : producers)
        producer.join();

    EXPECT_TRUE(store.drain_dirty_into_frame());
    EXPECT_EQ(callback_count.load(), kFields);
}

TEST(ReactiveStoreTest, SubscriberExceptionDoesNotStopRemainingSubscribers) {
    Store store;
    Observable<int> value(store, 1, "value", 0);

    int callback_count = 0;
    auto throwing_token = value.subscribe([](const int&) {
        throw std::runtime_error("subscriber failure");
    });
    auto succeeding_token = value.subscribe([&](const int&) {
        ++callback_count;
    });

    value.set(1);

    EXPECT_TRUE(store.drain_dirty_into_frame());
    EXPECT_EQ(callback_count, 1);
    EXPECT_FALSE(store.has_dirty());
}

TEST(ReactiveStoreTest, EnqueueWakesMainThread) {
    Store store;
    Observable<int> value(store, 1, "value", 0);
    std::atomic<int> wake_count{0};

    Store::set_wake_callback(+[] {});
    Store::set_wake_callback(nullptr);

    static std::atomic<int>* wake_ptr = nullptr;
    wake_ptr = &wake_count;
    Store::set_wake_callback(+[] {
        wake_ptr->fetch_add(1, std::memory_order_relaxed);
    });

    value.set(1);
    Store::set_wake_callback(nullptr);
    wake_ptr = nullptr;

    EXPECT_EQ(wake_count.load(), 1);
}
