/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;
    struct VulkanMeshDrawItem;
    struct VulkanEnvironmentParams;

    // Shared mesh/material and environment resources. Per-view descriptors,
    // light buffers, shadow maps, and pipelines remain on the individual passes.
    struct SharedMeshSubmesh {
        std::uint32_t start_index = 0;
        std::uint32_t index_count = 0;
        std::size_t material_index = 0;
    };

    struct SharedMeshDrawAsset {
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        VkBuffer index_buffer = VK_NULL_HANDLE;
        std::uint32_t total_index_count = 0;
        std::uint32_t generation = 0;
        glm::vec3 aabb_min{0.0f};
        glm::vec3 aabb_max{0.0f};
        std::vector<SharedMeshSubmesh> submeshes;
        std::vector<VkDescriptorSet> material_descriptors;
    };

    struct SharedEnvironmentTexture {
        VkImageView image_view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
    };

    class LFS_VIS_API SharedViewportGpuAssets {
    public:
        SharedViewportGpuAssets();
        ~SharedViewportGpuAssets();
        SharedViewportGpuAssets(const SharedViewportGpuAssets&) = delete;
        SharedViewportGpuAssets& operator=(const SharedViewportGpuAssets&) = delete;
        SharedViewportGpuAssets(SharedViewportGpuAssets&&) noexcept;
        SharedViewportGpuAssets& operator=(SharedViewportGpuAssets&&) noexcept;

        // Bind to a live VulkanContext. Safe to call repeatedly. A different
        // context, device, or allocator identity retires every GPU object
        // before the new device is used — old-device buffers are never reused.
        [[nodiscard]] bool ensureContext(VulkanContext& context);

        // Upload or retain GPU geometry/materials for `items`. Idempotent by
        // (MeshData::id, generation). Eviction waits only when stale entries
        // exist — never a per-frame global wait.
        void prepareMeshes(const std::vector<VulkanMeshDrawItem>& items, std::size_t frame_slot);

        // Copies handles for record-time binding. Valid until the next
        // prepareMeshes eviction of this id (not evicted while used this epoch).
        [[nodiscard]] bool findMesh(std::uint64_t mesh_id, SharedMeshDrawAsset& out) const;

        // Keep the previous environment image until submitted frames retire.
        // A disabled view must not destroy an image used by another view.
        void prepareEnvironment(const VulkanEnvironmentParams& params, std::size_t frame_slot);

        [[nodiscard]] SharedEnvironmentTexture environmentTexture() const;

        [[nodiscard]] VkDescriptorSetLayout meshMaterialLayout() const;

        [[nodiscard]] VkDevice device() const;

    private:
        friend class VulkanMeshPass;
        friend class VulkanEnvironmentPass;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
