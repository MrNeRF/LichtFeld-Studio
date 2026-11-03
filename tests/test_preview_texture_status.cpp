/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_vk_backend.hpp"
#include <gtest/gtest.h>

namespace {
    // Exercise the public renderer ABI, including Windows' DLL boundary.
#ifdef _MSC_VER
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    void
    releaseTexture(Rml::RenderInterface& renderer, Rml::TextureHandle texture) {
        renderer.ReleaseTexture(texture);
    }
} // namespace

class RenderInterfaceVKTestAccess {
public:
    static void checkPreviewStatusLifetime() {
        RenderInterface_VK backend;
        RenderInterface_VK::texture_data_t texture;
        texture.m_preview_source = "preview://path=fixture&rev=1";
        texture.m_preview_status = "pending";
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), "unknown");
        backend.m_preview_textures_by_source[texture.m_preview_source] = &texture;
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), "pending");
        texture.m_preview_status = "ready";
        auto retired = texture;
        releaseTexture(backend, reinterpret_cast<Rml::TextureHandle>(&retired));
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), "ready");
        texture.m_preview_status = "error";
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), "error");
        releaseTexture(backend, reinterpret_cast<Rml::TextureHandle>(&texture));
        EXPECT_EQ(backend.previewTextureStatus(texture.m_preview_source), "unknown");
        // These two inert fixtures own no GPU resources and live on this stack.
        for (auto& pending : backend.m_pending_for_deletion_textures_by_frames)
            pending.clear();
    }
};

TEST(PreviewTextureStatus, FollowsLiveSourceAndRetirement) {
    RenderInterfaceVKTestAccess::checkPreviewStatusLifetime();
}
