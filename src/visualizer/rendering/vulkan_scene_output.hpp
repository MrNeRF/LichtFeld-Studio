/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "scene_output.hpp"
#include <vulkan/vulkan.h>

namespace lfs::vis {
    // The desktop compositor uses Vulkan on every platform. Keep its native
    // transport conversion here, outside the scene renderer request/results.
    inline SceneImageHandle sceneImageHandle(VkImage value) { return SceneImageHandle::fromNative(value); }
    inline SceneImageViewHandle sceneImageViewHandle(VkImageView value) { return SceneImageViewHandle::fromNative(value); }
    inline SceneImageLayout sceneImageLayout(VkImageLayout value) { return SceneImageLayout::fromNative(value); }
    inline SceneTimelineHandle sceneTimelineHandle(VkSemaphore value) { return SceneTimelineHandle::fromNative(value); }
    inline VkImage vulkanSceneImage(SceneImageHandle value) { return value.native<VkImage>(); }
    inline VkImageView vulkanSceneImageView(SceneImageViewHandle value) { return value.native<VkImageView>(); }
    inline VkImageLayout vulkanSceneImageLayout(SceneImageLayout value) { return value.native<VkImageLayout>(); }
    inline VkSemaphore vulkanSceneTimeline(SceneTimelineHandle value) { return value.native<VkSemaphore>(); }
} // namespace lfs::vis
