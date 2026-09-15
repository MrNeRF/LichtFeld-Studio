/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "shared_viewport_gpu_assets.hpp"

#include "core/mesh_data.hpp"
#include "window/vulkan_context.hpp"

#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    namespace shared_viewport_gpu_detail {
        struct OneShotResult {
            bool completed = false;
            // True means the caller may release resources referenced by the
            // submitted command. A failed bounded wait is only safe to clean
            // up after the failure-only device-idle containment succeeds.
            bool safe_to_release = true;
        };
    } // namespace shared_viewport_gpu_detail

    struct SharedMeshVertex {
        float position[3];
        float normal[3];
        float tangent[4];
        float texcoord[2];
        float color[4];
    };
    static_assert(sizeof(SharedMeshVertex) == 64, "SharedMeshVertex layout — 16-byte aligned");

    struct SharedMaterialUbo {
        float base_color[4];
        float emissive_metallic[4];
        float roughness_flags[4];
        float vertex_color_flags[4];
    };
    static_assert(sizeof(SharedMaterialUbo) == 64, "SharedMaterialUbo layout");

    struct SharedViewportGpuAssets::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;

        std::uint64_t epoch = 0;
        std::optional<std::size_t> epoch_frame_slot;

        VkSampler mesh_sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout material_layout = VK_NULL_HANDLE;
        std::vector<VkDescriptorPool> material_descriptor_pools;
        VkCommandPool mesh_transfer_pool = VK_NULL_HANDLE;

        struct GpuTexture {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            std::string vram_label;
        };
        GpuTexture white_pixel{};

        struct GpuMaterial {
            VkBuffer ubo = VK_NULL_HANDLE;
            VmaAllocation ubo_alloc = VK_NULL_HANDLE;
            VkDescriptorSet descriptor = VK_NULL_HANDLE;
            VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
            GpuTexture albedo{};
            GpuTexture normal{};
            GpuTexture metallic_roughness{};
        };

        struct GpuSubmesh {
            std::uint32_t start_index = 0;
            std::uint32_t index_count = 0;
            std::size_t material_index = 0;
        };

        struct GpuMesh {
            VkBuffer vertex_buffer = VK_NULL_HANDLE;
            VmaAllocation vertex_alloc = VK_NULL_HANDLE;
            VkBuffer index_buffer = VK_NULL_HANDLE;
            VmaAllocation index_alloc = VK_NULL_HANDLE;
            std::uint32_t total_index_count = 0;
            std::uint32_t generation = 0;
            std::uint64_t last_used_epoch = 0;
            std::vector<GpuMaterial> materials;
            std::vector<GpuSubmesh> submeshes;
            glm::vec3 aabb_min{0.0f};
            glm::vec3 aabb_max{0.0f};
        };

        std::unordered_map<std::uint64_t, GpuMesh> mesh_cache;

        VkSampler environment_sampler = VK_NULL_HANDLE;
        VkCommandPool environment_transfer_pool = VK_NULL_HANDLE;

        struct EnvironmentImage {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            std::string vram_label;
            std::filesystem::path path;
            std::uint64_t last_used_epoch = 0;
            std::uint64_t retire_after_epoch = 0;
        };
        std::vector<EnvironmentImage> environment_images;
        std::size_t live_environment_index = std::numeric_limits<std::size_t>::max();
        std::filesystem::path environment_failed_path;
        bool environment_load_failed = false;
        std::uint64_t environment_last_enabled_epoch = 0;

        ~Impl() { shutdown(); }

        [[nodiscard]] bool matches(const VulkanContext& ctx) const {
            return context == &ctx && device == ctx.device() && allocator == ctx.allocator() &&
                   device != VK_NULL_HANDLE && allocator != VK_NULL_HANDLE;
        }

        [[nodiscard]] bool ensureContext(VulkanContext& ctx);
        void beginFrame(std::size_t frame_slot);
        void shutdown();

        [[nodiscard]] bool initMeshInfrastructure();
        void shutdownMeshes();
        void prepareMeshes(const std::vector<VulkanMeshDrawItem>& items);
        [[nodiscard]] bool findMesh(std::uint64_t mesh_id, SharedMeshDrawAsset& out) const;
        void evictUnusedMeshes();

        [[nodiscard]] bool initEnvironmentInfrastructure();
        void shutdownEnvironment();
        void prepareEnvironment(const VulkanEnvironmentParams& params);
        void retireEnvironmentImages(bool force_wait);
        [[nodiscard]] SharedEnvironmentTexture environmentTexture() const;
        [[nodiscard]] bool hasLiveEnvironmentTexture() const;

        [[nodiscard]] VkCommandBuffer beginMeshCommands() const;
        [[nodiscard]] shared_viewport_gpu_detail::OneShotResult
        endMeshCommands(VkCommandBuffer cb) const;
        [[nodiscard]] VkCommandBuffer beginEnvironmentCommands() const;
        [[nodiscard]] shared_viewport_gpu_detail::OneShotResult
        endEnvironmentCommands(VkCommandBuffer cb) const;

        [[nodiscard]] bool createMeshSampler();
        [[nodiscard]] bool createMaterialLayout();
        [[nodiscard]] VkDescriptorPool createMaterialDescriptorPool();
        [[nodiscard]] bool createInitialMaterialDescriptorPool();
        [[nodiscard]] bool allocateMaterialDescriptor(GpuMaterial& material);
        [[nodiscard]] bool writeBuffer(VmaAllocation alloc, const void* src, std::size_t bytes) const;
        [[nodiscard]] bool createTexture(const std::uint8_t* rgba,
                                         int w,
                                         int h,
                                         GpuTexture& out,
                                         std::string_view label);
        void destroyTexture(GpuTexture& texture) const;
        // Detach resources whose submission cannot be proven retired. This
        // intentionally leaks the Vulkan objects until device teardown rather
        // than freeing them while the GPU may still reference them.
        void abandonTexture(GpuTexture& texture) const;
        [[nodiscard]] bool createWhitePixel();
        [[nodiscard]] bool uploadTextureFromMesh(const lfs::core::MeshData& mesh,
                                                 int tex_index,
                                                 GpuTexture& out,
                                                 std::string_view label);
        [[nodiscard]] bool uploadMaterial(const lfs::core::MeshData& mesh,
                                          std::size_t material_index,
                                          GpuMaterial& out);
        [[nodiscard]] bool createDeviceLocalBuffer(VkDeviceSize size,
                                                   VkBufferUsageFlags usage,
                                                   VkBuffer& buffer,
                                                   VmaAllocation& alloc) const;
        [[nodiscard]] bool createStagingBuffer(VkDeviceSize size,
                                               const void* data,
                                               VkBuffer& buffer,
                                               VmaAllocation& alloc) const;
        [[nodiscard]] bool uploadMesh(const lfs::core::MeshData& mesh, GpuMesh& destination);
        void destroyMaterial(GpuMaterial& material) const;
        void destroyMesh(GpuMesh& mesh) const;

        [[nodiscard]] bool loadEnvironmentFromPath(const std::filesystem::path& path);
        void destroyEnvironmentImage(EnvironmentImage& image) const;
        void abandonEnvironmentImage(EnvironmentImage& image) const;
        [[nodiscard]] bool createEnvironmentSampler();
    };

} // namespace lfs::vis
