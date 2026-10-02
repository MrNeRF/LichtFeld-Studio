/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Exercise the production HiGS compose shader with projected inputs. No Metal,
// tensor GPU backend, viewer preferences, Python runtime or downloaded asset.
#include "core/headless_vulkan_device.hpp"
#include "gs_renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void check(VkResult result, const char* operation) {
        require(result == VK_SUCCESS, std::string(operation) + ": " + std::to_string(result));
    }

    struct Allocator {
        VmaAllocator value = VK_NULL_HANDLE;
        explicit Allocator(const lfs::core::VulkanDeviceHandles& h) {
            VmaAllocatorCreateInfo info{};
            info.instance = static_cast<VkInstance>(h.instance);
            info.physicalDevice = static_cast<VkPhysicalDevice>(h.physical_device);
            info.device = static_cast<VkDevice>(h.device);
            info.vulkanApiVersion = VK_API_VERSION_1_2;
            check(vmaCreateAllocator(&info, &value), "vmaCreateAllocator");
        }
        ~Allocator() { vmaDestroyAllocator(value); }
    };

    struct HostBuffer {
        VmaAllocator allocator;
        _VulkanBuffer view;
        void* data = nullptr;
        HostBuffer(VmaAllocator a, size_t bytes) : allocator(a) {
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = bytes;
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            VmaAllocationCreateInfo memory{};
            memory.usage = VMA_MEMORY_USAGE_AUTO;
            memory.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo allocation{};
            check(vmaCreateBuffer(a, &info, &memory, &view.buffer, &view.allocation, &allocation), "vmaCreateBuffer");
            view.allocSize = view.capacity = view.size = bytes;
            data = allocation.pMappedData;
            std::memset(data, 0, bytes);
        }
        ~HostBuffer() { vmaDestroyBuffer(allocator, view.buffer, view.allocation); }
        template <class T> T* as() { return static_cast<T*>(data); }
        void flush() { check(vmaFlushAllocation(allocator, view.allocation, 0, VK_WHOLE_SIZE), "vmaFlushAllocation"); }
        void invalidate() { check(vmaInvalidateAllocation(allocator, view.allocation, 0, VK_WHOLE_SIZE), "vmaInvalidateAllocation"); }
    };

    class Compose final : public VulkanGSPipeline {
        _ComputePipeline plain_{12}, overlays_{18};

    public:
        Compose(const lfs::core::VulkanDeviceHandles& h, VmaAllocator allocator, const std::filesystem::path& shaders) {
            initializeExternal(static_cast<VkInstance>(h.instance), static_cast<VkPhysicalDevice>(h.physical_device),
                               static_cast<VkDevice>(h.device), static_cast<VkQueue>(h.queue), h.queue_family, allocator);
            createComputePipeline(plain_, (shaders / "macro_compose.spv").string());
            createComputePipeline(overlays_, (shaders / "macro_compose_overlays.spv").string());
            createPendingComputePipelines();
        }
        // Pipeline descriptors are members: destroy their handles before members
        // go out of scope and the base destructor visits its pipeline registry.
        ~Compose() { cleanup(); }

        void run(const VulkanGSRendererUniforms& u, const std::vector<std::unique_ptr<HostBuffer>>& buffers, bool overlays) {
            std::vector<_VulkanBuffer> bindings;
            for (size_t i = 0; i < (overlays ? 18u : 12u); ++i) {
                buffers[i]->flush();
                bindings.push_back(buffers[i]->view);
            }
            beginCommandBatch();
            VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(activeCommandBuffer(), VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &before, 0, nullptr, 0, nullptr);
            executeCompute({{u.image_width, HIGS_TILE_WIDTH}, {u.image_height, HIGS_TILE_HEIGHT}},
                           &u, sizeof(u), overlays ? overlays_ : plain_, bindings);
            VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(activeCommandBuffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &after, 0, nullptr, 0, nullptr);
            endCommandBatch();
            waitForPendingBatch();
            for (const auto& buffer : buffers)
                buffer->invalidate();
        }
    };

    // IEEE binary16 constants, supplied directly so the test does not reproduce
    // the shader's FP16 accumulation algorithm. Both 0.50005 and 0.49995 round
    // to 0x3800 (0.5); 0x3bec is the nearest half to 0.99.
    constexpr uint16_t kHalfHalf = 0x3800, kHalfPoint99 = 0x3bec;
    constexpr float kFarDepth = 1e10f, kGuard = -12345.f;

    void test_case(Compose& pipeline, VmaAllocator allocator, bool overlays, uint32_t profile,
                   float first_alpha, float expected, uint32_t count, uint32_t width, bool split_wave) {
        constexpr size_t partial_pixels = 2 * HIGS_MACRO_TILE_SIZE_TILES * HIGS_TILE_SIZE;
        const size_t pixels = size_t(width) * HIGS_TILE_HEIGHT;
        const std::array<size_t, 18> sizes = {
            count * 4, 2 * 4, 4, count * 2 * 4, count * 4 * 4, count * 3 * 4, count * 4,
            partial_pixels * 4 * 2, 2 * 4, (pixels + 4) * 4 * 4, (pixels + 4) * 4, (pixels + 4) * 4,
            count + 4, count + 4, 16 * 4 * 4, count + 4, 32 * 4 * 4, count * 4};
        std::vector<std::unique_ptr<HostBuffer>> buffers;
        for (size_t bytes : sizes)
            buffers.push_back(std::make_unique<HostBuffer>(allocator, bytes));
        auto* ids = buffers[0]->as<int32_t>();
        std::iota(ids, ids + count, 0);
        std::copy_n(ids, count, buffers[17]->as<int32_t>());
        buffers[1]->as<int32_t>()[1] = int32_t(count);
        buffers[2]->as<int32_t>()[0] = split_wave ? 1 : 2;
        auto* xy = buffers[3]->as<float>();
        auto* conic = buffers[4]->as<float>();
        auto* depths = buffers[6]->as<float>();
        for (uint32_t i = 0; i < count; ++i) {
            xy[i * 2] = (i == 0 || i + 1 == count) ? 0.f : 1000.f;
            conic[i * 4] = conic[i * 4 + 2] = 100.f;
            conic[i * 4 + 3] = i == 0 ? first_alpha : .01f;
            depths[i] = i == 0 ? 4.f : 6.f;
        }
        auto* partials = buffers[7]->as<uint16_t>();
        for (size_t i = 0; i < partial_pixels; ++i) {
            partials[i * 4] = 0x3400; // 0.25, an exactly represented color partial.
            partials[i * 4 + 3] = i < HIGS_MACRO_TILE_SIZE_TILES * HIGS_TILE_SIZE ? kHalfHalf : kHalfPoint99;
        }
        buffers[8]->as<uint32_t>()[0] = buffers[8]->as<uint32_t>()[1] = 1;
        for (size_t i : {9u, 10u, 11u})
            std::fill_n(buffers[i]->as<float>(), sizes[i] / 4, kGuard);

        VulkanGSRendererUniforms u{};
        u.image_width = width;
        u.image_height = HIGS_TILE_HEIGHT;
        u.grid_width = u.grid_height = 1;
        u.mip_filter = 2; // Production exact median request.
        u.splat_render_profile = profile;
        pipeline.run(u, buffers, overlays);
        if (split_wave) {
            require(buffers[10]->as<float>()[0] == kFarDepth, "First wave should not cross the median");
            require(buffers[11]->as<float>()[0] > .5f, "First wave lost FP32 depth transmittance");
            buffers[2]->as<int32_t>()[0] = 2;
            // The production partial pool is reused for each raster wave.
            std::fill_n(partials, partial_pixels * 4, uint16_t(0));
            for (size_t i = 0; i < partial_pixels; ++i)
                partials[i * 4 + 3] = kHalfPoint99;
            u.wave_base = 1;
            pipeline.run(u, buffers, overlays);
        }
        const float actual = buffers[10]->as<float>()[0];
        require(actual == expected, "Incorrect median: got " + std::to_string(actual) + " expected " + std::to_string(expected) +
                                       " (profile=" + std::to_string(profile) + ", overlays=" + std::to_string(overlays) +
                                       ", alpha=" + std::to_string(first_alpha) + ")");
        require(buffers[10]->as<float>()[pixels] == kGuard && buffers[11]->as<float>()[pixels] == kGuard &&
                    buffers[9]->as<float>()[pixels * 4] == kGuard, "Partial edge dispatch wrote outside the output extent");
        // Source zero is centered only on pixel (0,0); other pixels cannot
        // cross 50%. This also exercises inactive lanes as source broadcasters.
        for (size_t p = 1; p < pixels; ++p)
            require(buffers[10]->as<float>()[p] == kFarDepth, "Spurious depth outside Gaussian coverage");

        std::array<float, 4> exact_color;
        std::copy_n(buffers[9]->as<float>(), 4, exact_color.begin());
        if (!split_wave) {
            u.mip_filter = 0;
            pipeline.run(u, buffers, overlays);
            require(std::equal(exact_color.begin(), exact_color.end(), buffers[9]->as<float>()),
                    "Exact depth changed ordinary color composition");
        }
        // Reuse the same outputs for an empty frame: old medians must disappear.
        u.mip_filter = 2;
        u.wave_base = 0;
        buffers[2]->as<int32_t>()[0] = 0;
        pipeline.run(u, buffers, overlays);
        require(buffers[10]->as<float>()[0] == kFarDepth && buffers[11]->as<float>()[0] == 1.f,
                "Empty frame retained depth from a previous scene");
    }
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc <= 2, "Usage: vulkan_depth_contracts [generated-shader-directory]");
        const auto shaders = argc == 2 ? std::filesystem::path(argv[1])
                                       : std::filesystem::path(LFS_VULKAN_RASTERIZER_DEV_SPV_DIR) / "generated";
        auto owner = lfs::core::HeadlessAdoptedDevice::try_create(false, true);
        require(owner.has_value(), "A Vulkan device supporting the viewer shaders is required");
        Allocator allocator(owner->handles());
        Compose pipeline(owner->handles(), allocator.value, shaders);
        size_t cases = 0;
        for (bool overlays : {false, true}) {
            for (uint32_t profile : {0u, 1u}) {
                for (const auto& [alpha, expected] : std::array<std::pair<float, float>, 4>{{
                         {.49995f, 6.f}, {.50005f, 4.f}, {.5f, 4.f}, {.1f, kFarDepth}}}) {
                    test_case(pipeline, allocator.value, overlays, profile, alpha, expected, 1025, 8, false);
                    ++cases;
                }
                test_case(pipeline, allocator.value, overlays, profile, .49995f, 6.f, 1056, 1, false);
                test_case(pipeline, allocator.value, overlays, profile, .49995f, 6.f, 1025, 8, true);
                cases += 2;
            }
        }
        std::printf("Vulkan HiGS depth contracts passed: %zu cases (FP16 boundary, profiles, overlays, edge lanes, wave continuation, empty reuse).\n", cases);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Vulkan depth contracts failed: %s\n", e.what());
        return 1;
    }
}
