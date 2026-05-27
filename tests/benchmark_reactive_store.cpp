/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/reactive/observable.hpp"
#include "core/reactive/store.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

TEST(ReactiveStoreBenchmark, MultiProducerDrainCost) {
    constexpr int kFields = 100;
    constexpr int kThreads = 64;
    constexpr int kWritesPerThread = 1000;

    lfs::core::reactive::Store store;
    std::vector<std::unique_ptr<lfs::core::reactive::Observable<int>>> fields;
    fields.reserve(kFields);
    for (int i = 0; i < kFields; ++i)
        fields.push_back(std::make_unique<lfs::core::reactive::Observable<int>>(store, i + 1, "field", 0));

    std::vector<lfs::core::reactive::SubscriptionToken> tokens;
    tokens.reserve(kFields);
    for (auto& field : fields)
        tokens.push_back(field->subscribe([](const int&) {}));

    std::vector<std::thread> producers;
    producers.reserve(kThreads);
    const auto write_start = std::chrono::steady_clock::now();
    for (int t = 0; t < kThreads; ++t) {
        producers.emplace_back([&, t] {
            for (int i = 0; i < kWritesPerThread; ++i)
                fields[(t + i) % kFields]->set(t * kWritesPerThread + i + 1);
        });
    }
    for (auto& producer : producers)
        producer.join();
    const auto write_end = std::chrono::steady_clock::now();

    const auto drain_start = std::chrono::steady_clock::now();
    EXPECT_TRUE(store.drain_dirty_into_frame());
    const auto drain_end = std::chrono::steady_clock::now();

    const auto write_us = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count();
    const auto drain_us = std::chrono::duration_cast<std::chrono::microseconds>(drain_end - drain_start).count();
    std::cout << "ReactiveStore writes_us=" << write_us
              << " drain_us=" << drain_us
              << " fields=" << kFields
              << " threads=" << kThreads
              << " writes=" << (kThreads * kWritesPerThread)
              << std::endl;
}
