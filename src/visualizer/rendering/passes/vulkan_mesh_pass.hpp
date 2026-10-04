/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/viewport_draw_types.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core {
    struct MeshData;
}

namespace lfs::vis {

    class VulkanContext;
    class SharedViewportGpuAssets;

    // GPU-rendered mesh draw item. Full PBR (Cook-Torrance: GGX + Smith + Schlick) with
    // albedo / normal / metallic-roughness textures, vertex colors, Reinhard tonemap +
    // sRGB gamma — feature-matched to master's `mesh_pbr.frag`. Shadow maps are not yet
    // wired (master defaults shadow_enabled=false; will be a follow-up depth-only pass).
    using VulkanMeshDrawItem = ViewportMeshDrawItem;

    using VulkanMeshPassParams = ViewportMeshPassDesc;

    using VulkanMeshViewportPanel = ViewportMeshPanel;

    class LFS_VIS_API VulkanMeshPass {
    public:
        VulkanMeshPass();
        ~VulkanMeshPass();

        VulkanMeshPass(const VulkanMeshPass&) = delete;
        VulkanMeshPass& operator=(const VulkanMeshPass&) = delete;
        VulkanMeshPass(VulkanMeshPass&&) noexcept;
        VulkanMeshPass& operator=(VulkanMeshPass&&) noexcept;

        // Attachment formats must match the render pass. A null shared_assets
        // creates a private geometry/material cache for single-pass callers.
        [[nodiscard]] bool init(VulkanContext& context,
                                VkFormat color_format,
                                VkFormat depth_stencil_format,
                                std::shared_ptr<SharedViewportGpuAssets> shared_assets = {});

        // Upload any new / changed mesh GPU buffers into the shared (or private)
        // asset cache. Idempotent — caches by (MeshData::id, generation). Per-view
        // light UBOs and shadow maps are prepared locally.
        void prepare(VulkanContext& context, const VulkanMeshPassParams& params);

        // Record draw commands into an already-active dynamic-rendering pass.
        // Caller must have transitioned the color attachment to
        // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL and bound a depth attachment.
        void record(VkCommandBuffer command_buffer,
                    VkRect2D viewport_rect,
                    const VulkanMeshPassParams& params);

        void discardImport(uint64_t mesh_id);
        void shutdown();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
