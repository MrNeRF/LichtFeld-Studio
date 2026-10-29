/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include <cstring>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace lfs::test {
    struct ImageDifference {
        float max = 0.0f;
        std::size_t count = 0;
    };

    inline ImageDifference compareImages(const std::vector<float>& a, const std::vector<float>& b) {
        ImageDifference d;
        if (a.size() != b.size())
            return {INFINITY, std::max(a.size(), b.size())};
        for (std::size_t i = 0; i < a.size(); ++i) {
            const float delta = std::abs(a[i] - b[i]);
            d.count += !(delta == 0.0f);
            d.max = std::max(d.max, delta);
        }
        return d;
    }

    inline std::vector<float> rgbChannels(const float* data, const std::size_t pixels, const std::size_t channels) {
        std::vector<float> rgb;
        for (std::size_t i = 0; i < pixels; ++i)
            rgb.insert(rgb.end(), {data[channels * i], data[channels * i + 1], data[channels * i + 2]});
        return rgb;
    }
    // Mean red minus green (or blue) over the pixels of `mask`, or over every
    // lit pixel without one.
    inline float excess(const std::vector<float>& rgb, const std::vector<bool>* mask = nullptr, const int other = 1) {
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < rgb.size() / 3; ++i) {
            const float* p = &rgb[3 * i];
            if (mask ? !(*mask)[i] : p[0] + p[1] + p[2] < 0.15f)
                continue;
            sum += p[0] - p[other];
            ++count;
        }
        return count ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
    }

    inline std::vector<float> readBuffer(VulkanGSRenderer& renderer, const _VulkanBuffer& buffer,
                                         VmaAllocator allocator, std::size_t count) {
        const VkDeviceSize bytes = count * sizeof(float);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer readback = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo mapped{};
        if (vmaCreateBuffer(allocator, &info, &allocation_info, &readback, &allocation, &mapped) != VK_SUCCESS) {
            ADD_FAILURE() << "readback allocation failed";
            return {};
        }
        renderer.beginCommandBatch();
        const lfs::rendering::vulkan::DeclaredAccess access{.buffer = &buffer,
                                                            .use = lfs::rendering::vulkan::BufferUse::TransferRead};
        renderer.planTransfer(std::span{&access, 1});
        const VkBufferCopy copy{buffer.offset, 0, bytes};
        vkCmdCopyBuffer(renderer.activeCommandBuffer(), buffer.buffer, readback, 1, &copy);
        renderer.endCommandBatch();
        vmaInvalidateAllocation(allocator, allocation, 0, bytes);
        std::vector<float> values(count);
        std::memcpy(values.data(), mapped.pMappedData, bytes);
        vmaDestroyBuffer(allocator, readback, allocation);
        return values;
    }
} // namespace lfs::test
