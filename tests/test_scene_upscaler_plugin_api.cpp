/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/nvidia_dlss_plugin.hpp"
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
        EXPECT_EQ(LFS_SCENE_UPSCALER_PLUGIN_VIEW_INVALID, UINT32_MAX);
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

        auto legacy = api;
        legacy.struct_size = offsetof(LfsSceneUpscalerPluginApiV1, capabilities);
        EXPECT_TRUE(lfs_scene_upscaler_plugin_api_v1_complete(&legacy));
        EXPECT_FALSE(lfs_scene_upscaler_plugin_api_v1_supports_dynamic_view_ids(&legacy));

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

    TEST(SceneUpscalerPluginApi, DynamicViewCapabilityIsOptionalAndDetectable) {
        LfsSceneUpscalerPluginApiV1 legacy{};
        legacy.struct_size = offsetof(LfsSceneUpscalerPluginApiV1, capabilities);
        EXPECT_FALSE(lfs_scene_upscaler_plugin_api_v1_supports_dynamic_view_ids(&legacy));

        LfsSceneUpscalerPluginApiV1 dynamic{};
        dynamic.struct_size = sizeof(dynamic);
        dynamic.capabilities = LFS_SCENE_UPSCALER_PLUGIN_CAPABILITY_DYNAMIC_VIEW_IDS;
        EXPECT_TRUE(lfs_scene_upscaler_plugin_api_v1_supports_dynamic_view_ids(&dynamic));
        dynamic.capabilities = LFS_SCENE_UPSCALER_PLUGIN_CAPABILITY_NONE;
        EXPECT_FALSE(lfs_scene_upscaler_plugin_api_v1_supports_dynamic_view_ids(&dynamic));
    }

    TEST(SceneUpscalerPluginApi, ViewIdentityAllocatorAvoidsAliasesUntilRetirement) {
        NvidiaDlssViewIdentityAllocator dynamic(true);
        const auto first = dynamic.acquire();
        const auto second = dynamic.acquire();
        ASSERT_TRUE(first.has_value());
        ASSERT_TRUE(second.has_value());
        EXPECT_EQ(*first, LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT);
        EXPECT_EQ(*second, LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT + 1);
        EXPECT_NE(*first, *second);
        dynamic.release(*first);
        EXPECT_FALSE(dynamic.owns(*first));
        const auto reused_after_retirement = dynamic.acquire();
        ASSERT_TRUE(reused_after_retirement.has_value());
        EXPECT_EQ(*reused_after_retirement, *first);

        NvidiaDlssViewIdentityAllocator legacy(false);
        const auto legacy_main = legacy.acquire();
        const auto legacy_left = legacy.acquire();
        const auto legacy_right = legacy.acquire();
        ASSERT_TRUE(legacy_main.has_value());
        ASSERT_TRUE(legacy_left.has_value());
        ASSERT_TRUE(legacy_right.has_value());
        EXPECT_EQ(*legacy_main, LFS_SCENE_UPSCALER_PLUGIN_VIEW_MAIN);
        EXPECT_EQ(*legacy_left, LFS_SCENE_UPSCALER_PLUGIN_VIEW_SPLIT_LEFT);
        EXPECT_EQ(*legacy_right, LFS_SCENE_UPSCALER_PLUGIN_VIEW_SPLIT_RIGHT);
        EXPECT_FALSE(legacy.acquire().has_value());

        NvidiaDlssViewIdentityAllocator pipelines(true);
        const auto pipeline_a = pipelines.acquire();
        const auto pipeline_b = pipelines.acquire();
        ASSERT_TRUE(pipeline_a.has_value());
        ASSERT_TRUE(pipeline_b.has_value());
        pipelines.release(*pipeline_a);
        EXPECT_FALSE(pipelines.owns(*pipeline_a));
        EXPECT_TRUE(pipelines.owns(*pipeline_b));
    }

    TEST(SceneUpscalerPluginApi, InvalidIdentityIsReservedAndRejectedByCapabilityContract) {
        EXPECT_EQ(LFS_SCENE_UPSCALER_PLUGIN_VIEW_INVALID, UINT32_MAX);
        EXPECT_NE(LFS_SCENE_UPSCALER_PLUGIN_VIEW_INVALID,
                  LFS_SCENE_UPSCALER_PLUGIN_VIEW_MAIN);
        EXPECT_EQ(lfs_scene_upscaler_plugin_view_id_valid(
                      LFS_SCENE_UPSCALER_PLUGIN_VIEW_INVALID),
                  0);
        EXPECT_EQ(lfs_scene_upscaler_plugin_view_id_valid(37), 1);
    }

} // namespace lfs::vis
