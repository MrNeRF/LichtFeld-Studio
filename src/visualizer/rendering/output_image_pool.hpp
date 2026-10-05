/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gpu_resource_pool.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Bucket allocation extent for viewport output images (64-px).
    [[nodiscard]] constexpr std::uint32_t ceil64(const std::uint32_t v) noexcept {
        return ((v + 63u) / 64u) * 64u;
    }

    // Map full-image UV [0,1] onto the valid top-left sub-rect of a padded output.
    // Per axis: scale = valid/alloc; 1.0 when alloc<=0 or valid>=alloc.
    [[nodiscard]] inline glm::vec2 outputUvScale(const glm::ivec2 valid,
                                                 const glm::ivec2 alloc) noexcept {
        glm::vec2 scale{1.0f, 1.0f};
        if (alloc.x > 0 && valid.x > 0 && valid.x < alloc.x) {
            scale.x = static_cast<float>(valid.x) / static_cast<float>(alloc.x);
        }
        if (alloc.y > 0 && valid.y > 0 && valid.y < alloc.y) {
            scale.y = static_cast<float>(valid.y) / static_cast<float>(alloc.y);
        }
        return scale;
    }

    // Half-texel max UV to stop linear-filter bleed from padding.
    // Per axis: (valid - 0.5) / alloc; 1.0 when alloc<=0 or valid<=0.
    [[nodiscard]] inline glm::vec2 outputUvClampMax(const glm::ivec2 valid,
                                                    const glm::ivec2 alloc) noexcept {
        glm::vec2 clamp_max{1.0f, 1.0f};
        if (alloc.x > 0 && valid.x > 0) {
            clamp_max.x = (static_cast<float>(valid.x) - 0.5f) / static_cast<float>(alloc.x);
        }
        if (alloc.y > 0 && valid.y > 0) {
            clamp_max.y = (static_cast<float>(valid.y) - 0.5f) / static_cast<float>(alloc.y);
        }
        return clamp_max;
    }

    // Shared key for GPU resource free-lists (format/extent/usage/external).
    struct GpuResourcePoolKey {
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{0, 0};
        VkImageUsageFlags usage = 0;
        bool external = false;

        [[nodiscard]] bool operator==(const GpuResourcePoolKey& other) const noexcept {
            return format == other.format && extent.width == other.extent.width &&
                   extent.height == other.extent.height && usage == other.usage &&
                   external == other.external;
        }
    };

    template <typename Payload>
    using GpuResourcePool = BasicGpuResourcePool<Payload, GpuResourcePoolKey>;

    class LFS_VIS_API OutputImagePool {
    public:
        static constexpr std::uint64_t kIdleTrimTicks = GpuResourcePool<VulkanContext::ExternalImage>::kIdleTrimTicks;

        using Key = GpuResourcePoolKey;
        using DestroyFn = GpuResourcePool<VulkanContext::ExternalImage>::DestroyFn;
        // Payload-aware producer predicate (mechanical upgrade from value-only).
        using TimelinePred = GpuResourcePool<VulkanContext::ExternalImage>::ProducerPred;
        using FramePred = GpuResourcePool<VulkanContext::ExternalImage>::FramePred;

        struct Acquired {
            VulkanContext::ExternalImage image{};
            std::uint64_t acquisition_serial = 0;
        };

        OutputImagePool();

        [[nodiscard]] std::optional<Acquired> acquire(const Key& key);

        [[nodiscard]] Acquired registerCreated(const Key& key, VulkanContext::ExternalImage&& image);

        void release(std::uint64_t acquisition_serial,
                     std::uint64_t producer_value,
                     std::uint64_t consumer_serial,
                     bool evict = false);

        void drain(bool force,
                   const TimelinePred& producer_done,
                   const FramePred& consumer_done,
                   const DestroyFn& destroy);

        void trimIdle(const DestroyFn& destroy);
        void trimAged(const DestroyFn& destroy);

        [[nodiscard]] std::size_t idleBytes() const;
        [[nodiscard]] std::size_t liveCount() const;
        [[nodiscard]] std::size_t retiredCount() const;
        [[nodiscard]] std::size_t freeCount() const;
        [[nodiscard]] bool misuseFlagged() const;

    private:
        GpuResourcePool<VulkanContext::ExternalImage> pool_;
    };

} // namespace lfs::vis
