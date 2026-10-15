/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "rendering/frame_contract.hpp"
#include <array>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    namespace vr {
        struct EyeImage {
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkSemaphore ready = VK_NULL_HANDLE;
            std::uint64_t ready_value = 0;
            std::uint64_t generation = 0;
            glm::ivec2 size{0};
            glm::ivec2 alloc_size{0};
            bool flip_y = false;
        };

        // The public boundary deliberately contains no OpenXR headers. Desktop-only
        // builds use the same interface without linking an XR loader.
        class LFS_VIS_API OpenXrService {
        public:
            using RenderEyes = std::function<std::expected<std::array<EyeImage, 2>, std::string>(
                const std::array<lfs::rendering::FrameView, 2>&)>;

            OpenXrService();
            ~OpenXrService();
            OpenXrService(const OpenXrService&) = delete;
            OpenXrService& operator=(const OpenXrService&) = delete;

            static bool compiled();
            bool prepare();
            bool prepared() const;
            VkResult createVulkanInstance(const VkInstanceCreateInfo& info, VkInstance& instance);
            VkPhysicalDevice graphicsDevice(VkInstance instance);
            VkResult createVulkanDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo& info, VkDevice& device);

            std::expected<void, std::string> start(VulkanContext& context,
                                                   const glm::mat3& rotation,
                                                   const glm::vec3& position);
            // Only VulkanContext's post-device-drain boundary may pass true.
            void stop(bool graphics_retired = false);
            void reportUnavailable(std::string reason);
            bool active() const;
            bool running() const;
            bool requested() const;
            void request(bool enabled);
            void recenter();
            void setKeyboardNavigation(glm::vec3 movement, float turn, bool recenter);
            const std::string& status() const;
            std::expected<void, std::string> frame(const RenderEyes& render);
            EyeImage mirror() const;

        private:
            struct Impl;
            std::unique_ptr<Impl> impl_;
        };
    } // namespace vr
} // namespace lfs::vis
