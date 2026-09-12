/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_upscaler_plugin_api.h"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

namespace lfs::vis {

    TEST(SceneUpscalerPluginApi, VersionOneBoundaryUsesPlainStandardLayoutRecords) {
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerExtensionSink>);
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerBootstrapConfigV1>);
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerRuntimeConfigV1>);
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerOptimalSettingsV1>);
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerFeatureConfigV1>);
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerImageV1>);
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerEvaluateV1>);
        static_assert(std::is_standard_layout_v<LfsSceneUpscalerPluginApiV1>);
        static_assert(std::is_trivially_copyable_v<LfsSceneUpscalerImageV1>);
        static_assert(std::is_trivially_copyable_v<LfsSceneUpscalerEvaluateV1>);

        EXPECT_EQ(LFS_SCENE_UPSCALER_PLUGIN_ABI_V1, 1u);
        EXPECT_STREQ(LFS_SCENE_UPSCALER_PLUGIN_ENTRY_V1,
                     "lfs_scene_upscaler_plugin_get_api_v1");
        EXPECT_EQ(LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT, 3);

        constexpr std::size_t LEGACY_EVALUATION_PREFIX =
            offsetof(LfsSceneUpscalerEvaluateV1, reset_flags) +
            sizeof(std::uint32_t);
        EXPECT_EQ(offsetof(LfsSceneUpscalerEvaluateV1, camera_near),
                  LEGACY_EVALUATION_PREFIX);
        EXPECT_EQ(offsetof(LfsSceneUpscalerEvaluateV1, camera_far),
                  offsetof(LfsSceneUpscalerEvaluateV1, camera_near) + sizeof(float));
        EXPECT_EQ(offsetof(LfsSceneUpscalerEvaluateV1,
                           camera_vertical_fov_radians),
                  offsetof(LfsSceneUpscalerEvaluateV1, camera_far) + sizeof(float));
        EXPECT_EQ(offsetof(LfsSceneUpscalerEvaluateV1, view_space_to_meters),
                  offsetof(LfsSceneUpscalerEvaluateV1,
                           camera_vertical_fov_radians) +
                      sizeof(float));
    }

    TEST(SceneUpscalerPluginApi, RejectsTruncatedWrongVersionAndIncompleteTables) {
        LfsSceneUpscalerPluginApiV1 api{
            .struct_size = sizeof(LfsSceneUpscalerPluginApiV1),
            .abi_version = LFS_SCENE_UPSCALER_PLUGIN_ABI_V1,
            .plugin_id = "test-plugin",
            .display_name = "Test plugin",
            .create = [](const LfsSceneUpscalerBootstrapConfigV1*) -> void* {
                return nullptr;
            },
            .destroy = [](void*) {},
            .required_instance_extensions =
                [](void*, const LfsSceneUpscalerExtensionSink*) {
                    return LFS_SCENE_UPSCALER_PLUGIN_OK;
                },
            .required_device_extensions =
                [](void*, VkInstance, VkPhysicalDevice, const LfsSceneUpscalerExtensionSink*) {
                    return LFS_SCENE_UPSCALER_PLUGIN_OK;
                },
            .initialize_runtime = [](void*, const LfsSceneUpscalerRuntimeConfigV1*) { return LFS_SCENE_UPSCALER_PLUGIN_OK; },
            .optimal_settings =
                [](void*, std::uint32_t, std::uint32_t, std::uint32_t,
                   LfsSceneUpscalerOptimalSettingsV1*) {
                    return LFS_SCENE_UPSCALER_PLUGIN_OK;
                },
            .create_feature =
                [](void*, VkCommandBuffer, const LfsSceneUpscalerFeatureConfigV1*) {
                    return LFS_SCENE_UPSCALER_PLUGIN_OK;
                },
            .evaluate = [](void*, const LfsSceneUpscalerEvaluateV1*) { return LFS_SCENE_UPSCALER_PLUGIN_OK; },
            .release_feature = [](void*, std::uint32_t) {},
            .shutdown_runtime = [](void*) {},
            .last_error = [](void*, char*, std::size_t) -> std::size_t {
                return 0;
            },
        };
        EXPECT_TRUE(lfs_scene_upscaler_plugin_api_v1_complete(&api));

        auto incompatible = api;
        incompatible.abi_version = LFS_SCENE_UPSCALER_PLUGIN_ABI_V1 + 1;
        EXPECT_FALSE(lfs_scene_upscaler_plugin_api_v1_complete(&incompatible));

        auto truncated = api;
        truncated.struct_size = offsetof(LfsSceneUpscalerPluginApiV1, last_error);
        EXPECT_FALSE(lfs_scene_upscaler_plugin_api_v1_complete(&truncated));

        auto incomplete = api;
        incomplete.evaluate = nullptr;
        EXPECT_FALSE(lfs_scene_upscaler_plugin_api_v1_complete(&incomplete));
    }

} // namespace lfs::vis
