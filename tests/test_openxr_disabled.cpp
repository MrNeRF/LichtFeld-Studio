/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vr/openxr_service.hpp"
#include <gtest/gtest.h>

// Compile this suite together with openxr_service.cpp without LFS_ENABLE_OPENXR.
// It intentionally needs no VulkanContext, GPU, or OpenXR loader.
#ifndef LFS_ENABLE_OPENXR
namespace {
    using lfs::vis::vr::EyeImage;
    using lfs::vis::vr::OpenXrService;

    void expectEmptyMirror(const EyeImage& image) {
        EXPECT_EQ(image.image, VK_NULL_HANDLE);
        EXPECT_EQ(image.view, VK_NULL_HANDLE);
        EXPECT_EQ(image.ready, VK_NULL_HANDLE);
        EXPECT_EQ(image.layout, VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT_EQ(image.ready_value, 0u);
        EXPECT_EQ(image.generation, 0u);
        EXPECT_EQ(image.size, glm::ivec2(0));
        EXPECT_EQ(image.alloc_size, glm::ivec2(0));
        EXPECT_FALSE(image.flip_y);
    }

    TEST(OpenXrDisabled, ReportsUnavailableBuildWithoutCreatingSession) {
        OpenXrService service;
        EXPECT_FALSE(OpenXrService::compiled());
        EXPECT_FALSE(service.prepared());
        EXPECT_FALSE(service.active());
        EXPECT_FALSE(service.running());
        EXPECT_FALSE(service.requested());
        EXPECT_FALSE(service.status().empty());
        EXPECT_FALSE(service.prepare());
        EXPECT_NE(service.status().find("OpenXR"), std::string::npos);
        EXPECT_NE(service.status().find("LFS_ENABLE_OPENXR=ON"), std::string::npos);
        EXPECT_FALSE(service.prepared());
        EXPECT_FALSE(service.active());
        EXPECT_FALSE(service.running());
        expectEmptyMirror(service.mirror());
    }

    TEST(OpenXrDisabled, StopClearsRequestAndMirrorAndIsIdempotent) {
        OpenXrService service;
        service.request(true);
        EXPECT_TRUE(service.requested());
        EXPECT_FALSE(service.active());
        EXPECT_FALSE(service.running());
        service.stop();
        EXPECT_FALSE(service.requested());
        expectEmptyMirror(service.mirror());
        service.stop();
        service.stop(true);
        EXPECT_FALSE(service.requested());
        EXPECT_FALSE(service.active());
        EXPECT_FALSE(service.running());
        expectEmptyMirror(service.mirror());
    }

    TEST(OpenXrDisabled, FrameFailsWithoutInvokingRendererEvenWhenRequested) {
        OpenXrService service;
        ASSERT_FALSE(service.prepare());
        int render_calls = 0;
        const OpenXrService::RenderEyes render = [&](const auto&) {
            ++render_calls;
            return std::expected<std::array<EyeImage, 2>, std::string>(std::array<EyeImage, 2>{});
        };
        const auto frame = service.frame(render);
        ASSERT_FALSE(frame.has_value());
        EXPECT_EQ(frame.error(), service.status());
        service.request(true);
        const auto requested_frame = service.frame(render);
        ASSERT_FALSE(requested_frame.has_value());
        EXPECT_FALSE(requested_frame.error().empty());
        EXPECT_EQ(render_calls, 0);
        EXPECT_FALSE(service.active());
        EXPECT_FALSE(service.running());
        expectEmptyMirror(service.mirror());
        // An absent renderer is also safe: the disabled path must never call it.
        EXPECT_FALSE(service.frame({}).has_value());
    }

    TEST(OpenXrDisabled, VulkanCreationFailsAndPreservesOutputHandles) {
        OpenXrService service;
        // Opaque sentinel handles are never dereferenced or passed to Vulkan.
        auto instance = reinterpret_cast<VkInstance>(std::uintptr_t{0x1234});
        const auto original_instance = instance;
        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        EXPECT_EQ(service.createVulkanInstance(instance_info, instance), VK_ERROR_FEATURE_NOT_PRESENT);
        EXPECT_EQ(instance, original_instance);
        EXPECT_EQ(service.graphicsDevice(instance), VK_NULL_HANDLE);
        EXPECT_EQ(service.graphicsDevice(VK_NULL_HANDLE), VK_NULL_HANDLE);

        auto device = reinterpret_cast<VkDevice>(std::uintptr_t{0x5678});
        const auto original_device = device;
        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        EXPECT_EQ(service.createVulkanDevice(VK_NULL_HANDLE, device_info, device), VK_ERROR_FEATURE_NOT_PRESENT);
        EXPECT_EQ(device, original_device);

        instance = VK_NULL_HANDLE;
        device = VK_NULL_HANDLE;
        EXPECT_EQ(service.createVulkanInstance(instance_info, instance), VK_ERROR_FEATURE_NOT_PRESENT);
        EXPECT_EQ(service.createVulkanDevice(VK_NULL_HANDLE, device_info, device), VK_ERROR_FEATURE_NOT_PRESENT);
        EXPECT_EQ(instance, VK_NULL_HANDLE);
        EXPECT_EQ(device, VK_NULL_HANDLE);
    }

    TEST(OpenXrDisabled, RecenterAndKeyboardNavigationAreSafeWithoutSession) {
        OpenXrService service;
        service.recenter();
        service.setKeyboardNavigation({1.0f, -1.0f, 0.5f}, 1.0f, true);
        service.request(true);
        service.setKeyboardNavigation({}, -1.0f, false);
        service.recenter();
        EXPECT_TRUE(service.requested());
        EXPECT_FALSE(service.active());
        EXPECT_FALSE(service.running());
        service.stop();
        service.setKeyboardNavigation({}, 0.0f, false);
        service.recenter();
        EXPECT_FALSE(service.requested());
        expectEmptyMirror(service.mirror());
    }

    TEST(OpenXrDisabled, UnavailableStatusCanBeUpdatedAndPrepareRestoresBuildGuidance) {
        OpenXrService service;
        service.reportUnavailable("VR unavailable: test reason");
        EXPECT_EQ(service.status(), "VR unavailable: test reason");
        ASSERT_FALSE(service.frame({}).has_value());
        EXPECT_EQ(service.frame({}).error(), service.status());
        EXPECT_FALSE(service.prepare());
        EXPECT_NE(service.status().find("LFS_ENABLE_OPENXR=ON"), std::string::npos);
    }
} // namespace
#endif
