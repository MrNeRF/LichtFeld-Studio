// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "core/scene.hpp"
#include "python/python_runtime.hpp"

namespace lfs::python {

    class SceneValidityTest : public ::testing::Test {
    protected:
        void SetUp() override {
            set_application_scene(nullptr);
        }

        void TearDown() override {
            set_application_scene(nullptr);
        }

        core::Scene dummy_scene_;
    };

    TEST_F(SceneValidityTest, GenerationNonNegative) {
        auto gen = get_scene_generation();
        EXPECT_GE(gen, 0u);
    }

    TEST_F(SceneValidityTest, GenerationIncrementsOnSet) {
        auto gen1 = get_scene_generation();
        set_application_scene(&dummy_scene_);
        auto gen2 = get_scene_generation();
        EXPECT_GT(gen2, gen1);
    }

    TEST_F(SceneValidityTest, GenerationIncrementsOnClear) {
        set_application_scene(&dummy_scene_);
        auto gen1 = get_scene_generation();
        set_application_scene(nullptr);
        auto gen2 = get_scene_generation();
        EXPECT_GT(gen2, gen1);
    }

    TEST_F(SceneValidityTest, GetApplicationSceneReturnsCorrectPointer) {
        EXPECT_EQ(get_application_scene(), nullptr);
        set_application_scene(&dummy_scene_);
        EXPECT_EQ(get_application_scene(), &dummy_scene_);
        set_application_scene(nullptr);
        EXPECT_EQ(get_application_scene(), nullptr);
    }

    TEST_F(SceneValidityTest, ConcurrentReadsAreSafe) {
        set_application_scene(&dummy_scene_);
        std::atomic<int> success_count{0};
        std::vector<std::thread> threads;

        for (int i = 0; i < 10; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < 1000; ++j) {
                    auto gen = get_scene_generation();
                    auto* scene = get_application_scene();
                    EXPECT_GE(gen, 0u);
                    EXPECT_EQ(scene, &dummy_scene_);
                }
                success_count++;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        EXPECT_EQ(success_count.load(), 10);
    }

    TEST_F(SceneValidityTest, GenerationIsMonotonic) {
        std::vector<uint64_t> generations;
        generations.push_back(get_scene_generation());

        for (int i = 0; i < 10; ++i) {
            set_application_scene(&dummy_scene_);
            generations.push_back(get_scene_generation());
            set_application_scene(nullptr);
            generations.push_back(get_scene_generation());
        }

        for (size_t i = 1; i < generations.size(); ++i) {
            EXPECT_GT(generations[i], generations[i - 1]);
        }
    }

} // namespace lfs::python
