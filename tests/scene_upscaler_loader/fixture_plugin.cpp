/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/scene_upscaler_plugin_api.h"
#include <unordered_set>
namespace {
    struct Context {
        std::unordered_set<uint32_t> views;
    };
    void* create(const LfsSceneUpscalerBootstrapConfigV1* config) { return config ? new Context : nullptr; }
    void destroy(void* context) { delete static_cast<Context*>(context); }
    LfsSceneUpscalerPluginResult instanceExtensions(void*, const LfsSceneUpscalerExtensionSink*) { return LFS_SCENE_UPSCALER_PLUGIN_OK; }
    LfsSceneUpscalerPluginResult deviceExtensions(void*, VkInstance, VkPhysicalDevice, const LfsSceneUpscalerExtensionSink*) { return LFS_SCENE_UPSCALER_PLUGIN_OK; }
    LfsSceneUpscalerPluginResult initialize(void*, const LfsSceneUpscalerRuntimeConfigV1*) { return LFS_SCENE_UPSCALER_PLUGIN_OK; }
    LfsSceneUpscalerPluginResult optimal(void*, uint32_t width, uint32_t height, uint32_t, LfsSceneUpscalerOptimalSettingsV1* result) {
        *result = {.struct_size = sizeof(*result), .render_width = width / 2, .render_height = height / 2, .minimum_width = 1, .minimum_height = 1, .maximum_width = width, .maximum_height = height};
        return LFS_SCENE_UPSCALER_PLUGIN_OK;
    }
    LfsSceneUpscalerPluginResult feature(void* context, VkCommandBuffer, const LfsSceneUpscalerFeatureConfigV1* config) {
        static_cast<Context*>(context)->views.insert(config->view);
        return LFS_SCENE_UPSCALER_PLUGIN_OK;
    }
    LfsSceneUpscalerPluginResult evaluate(void* context, const LfsSceneUpscalerEvaluateV1* config) {
        return static_cast<Context*>(context)->views.contains(config->view) ? LFS_SCENE_UPSCALER_PLUGIN_OK : LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
    }
    void release(void* context, uint32_t view) { static_cast<Context*>(context)->views.erase(view); }
    void shutdown(void* context) { static_cast<Context*>(context)->views.clear(); }
    size_t error(void*, char*, size_t) { return 0; }
    const LfsSceneUpscalerPluginApiV1 api{
        sizeof(LfsSceneUpscalerPluginApiV1),
        TEST_PROVIDER_ABI,
        TEST_PROVIDER_ID,
        TEST_PROVIDER_NAME,
        create,
        destroy,
        instanceExtensions,
        deviceExtensions,
        initialize,
        optimal,
        feature,
        evaluate,
        release,
        shutdown,
        error,
        LFS_SCENE_UPSCALER_PLUGIN_CAPABILITY_DYNAMIC_VIEW_IDS,
    };
} // namespace
extern "C" LFS_SCENE_UPSCALER_PLUGIN_EXPORT const LfsSceneUpscalerPluginApiV1* lfs_scene_upscaler_plugin_get_api_v1() { return &api; }
