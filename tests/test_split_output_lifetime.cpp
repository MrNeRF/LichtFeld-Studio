/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/rendering_manager.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace lfs::vis {
    struct SplitOutputLifetimeTestAccess {
        static void checkRetainedPair(const bool retire_left) {
            RenderingManager manager;
            manager.vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
            auto& renderer = *manager.vksplat_viewport_renderer_;
            auto& pool = renderer.output_pool_;
            const OutputImagePool::Key key{
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .extent = {64, 64},
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
                .external = true};
            const auto create = [&](const std::uintptr_t id, const std::size_t logical) {
                VulkanContext::ExternalImage image{};
                image.image = reinterpret_cast<VkImage>(id);
                image.view = reinterpret_cast<VkImageView>(id + 100);
                const auto acquired = pool.registerCreated(key, std::move(image));
                auto& slot = renderer.ring_.slotAt(logical, 0);
                slot.image = acquired.image;
                slot.color_pool_serial = acquired.acquisition_serial;
                slot.generation = renderer.ring_.bumpGeneration(logical);
                renderer.ring_.markLatest(logical, 0);
                return acquired;
            };
            const auto left = create(1, 1);
            const auto right = create(2, 2);
            RenderingManager::VulkanMeshFrame frame;
            frame.split_view.enabled = true;
            frame.split_view.left.external_image_view = left.image.view;
            frame.split_view.left.external_image_generation = 1;
            frame.split_view.right.external_image_view = right.image.view;
            frame.split_view.right.external_image_generation = 1;
            manager.setVulkanMeshFrame(frame);

            // Measure the publication boundary with unchanged bindings. This
            // matches the no-change path used on successive viewport updates.
            std::vector<double> timings;
            for (int repetition = 0; repetition < 9; ++repetition) {
                const auto start = std::chrono::steady_clock::now();
                for (int update = 0; update < 20000; ++update) {
                    manager.setVulkanMeshFrame(frame);
                }
                timings.push_back(std::chrono::duration<double, std::nano>(
                                      std::chrono::steady_clock::now() - start)
                                      .count() /
                                  20000.0);
            }
            std::sort(timings.begin(), timings.end());
            std::cout << "split publication median_ns=" << timings[timings.size() / 2] << '\n';

            // A successful panel resize releases its previous ring image. If
            // the other panel defers, the manager retains the published pair.
            // Prior GPU work is complete; this publication is a future reader.
            const auto retired = retire_left ? left : right;
            pool.release(retired.acquisition_serial, 10, 20);
            renderer.ring_.slotAt(retire_left ? 1 : 2, 0) = {};
            std::vector<VkImageView> destroyed;
            const auto destroy = [&](VulkanContext::ExternalImage& image) {
                destroyed.push_back(image.view);
                image = {};
            };
            const auto producer_done = [](const VulkanContext::ExternalImage&, std::uint64_t) { return true; };
            const auto consumer_done = [](std::uint64_t) { return true; };
            pool.drain(false, producer_done, consumer_done, destroy);
            EXPECT_EQ(pool.freeCount(), 0u) << "A published image must not be reused";
            for (std::uint64_t attempt = 0; attempt <= OutputImagePool::kIdleTrimTicks + 1; ++attempt) {
                pool.drain(false, producer_done, consumer_done, destroy);
                pool.trimAged(destroy);
            }
            const auto cached = manager.getVulkanMeshFrame();
            EXPECT_EQ(cached.split_view.left.external_image_view, left.image.view);
            EXPECT_EQ(cached.split_view.right.external_image_view, right.image.view);
            EXPECT_EQ(std::count(destroyed.begin(), destroyed.end(), retired.image.view), 0)
                << "A still-published split panel was destroyed after a partial pair render";
            EXPECT_EQ(pool.freeCount(), 0u) << "A published image must not be reused either";

            manager.clearVulkanMeshFrame();
            pool.drain(false, producer_done, consumer_done, destroy);
            pool.trimIdle(destroy);
            EXPECT_EQ(std::count(destroyed.begin(), destroyed.end(), retired.image.view), 1);
            // Scripted handles never reach Vulkan teardown.
            renderer.ring_.reset();
        }
    };
} // namespace lfs::vis

TEST(SplitOutputLifetimeTest, RetainsLeftImageWhenRightPanelDefers) {
    lfs::vis::SplitOutputLifetimeTestAccess::checkRetainedPair(true);
}

TEST(SplitOutputLifetimeTest, RetainsRightImageWhenLeftPanelDefers) {
    lfs::vis::SplitOutputLifetimeTestAccess::checkRetainedPair(false);
}
