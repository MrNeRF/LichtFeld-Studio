/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/*
 * Stable C boundary between LichtFeld Studio and optional scene-reconstruction
 * plugins. The host must validate both abi_version and struct_size before using
 * an implementation. No STL, exceptions, ownership-bearing Vulkan objects, or
 * vendor SDK types cross this boundary.
 */

#include <stddef.h>
#include <stdint.h>
#include <wchar.h>

#include <vulkan/vulkan.h>

#define LFS_SCENE_UPSCALER_PLUGIN_ABI_V1   1u
#define LFS_SCENE_UPSCALER_PLUGIN_ENTRY_V1 "lfs_scene_upscaler_plugin_get_api_v1"

#if defined(_WIN32)
#if defined(LFS_SCENE_UPSCALER_PLUGIN_BUILD)
#define LFS_SCENE_UPSCALER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define LFS_SCENE_UPSCALER_PLUGIN_EXPORT
#endif
#else
#if defined(LFS_SCENE_UPSCALER_PLUGIN_BUILD)
#define LFS_SCENE_UPSCALER_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define LFS_SCENE_UPSCALER_PLUGIN_EXPORT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum LfsSceneUpscalerPluginResult {
    LFS_SCENE_UPSCALER_PLUGIN_OK = 0,
    LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT = 1,
    LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE = 2,
    LFS_SCENE_UPSCALER_PLUGIN_UNSUPPORTED_DEVICE = 3,
    LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR = 4,
} LfsSceneUpscalerPluginResult;

typedef enum LfsSceneUpscalerPluginQuality {
    LFS_SCENE_UPSCALER_PLUGIN_QUALITY = 0,
    LFS_SCENE_UPSCALER_PLUGIN_BALANCED = 1,
    LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE = 2,
} LfsSceneUpscalerPluginQuality;

typedef enum LfsSceneUpscalerPluginView {
    LFS_SCENE_UPSCALER_PLUGIN_VIEW_MAIN = 0,
    LFS_SCENE_UPSCALER_PLUGIN_VIEW_SPLIT_LEFT = 1,
    LFS_SCENE_UPSCALER_PLUGIN_VIEW_SPLIT_RIGHT = 2,
    LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT = 3,
} LfsSceneUpscalerPluginView;

typedef enum LfsSceneUpscalerPluginResetFlag {
    LFS_SCENE_UPSCALER_PLUGIN_RESET_NONE = 0,
    LFS_SCENE_UPSCALER_PLUGIN_RESET_CAMERA_CUT = 1u << 0u,
    LFS_SCENE_UPSCALER_PLUGIN_RESET_RENDER_SIZE = 1u << 1u,
    LFS_SCENE_UPSCALER_PLUGIN_RESET_OUTPUT_SIZE = 1u << 2u,
    LFS_SCENE_UPSCALER_PLUGIN_RESET_SCENE = 1u << 3u,
    LFS_SCENE_UPSCALER_PLUGIN_RESET_QUALITY = 1u << 4u,
    LFS_SCENE_UPSCALER_PLUGIN_RESET_REQUESTED = 1u << 5u,
    LFS_SCENE_UPSCALER_PLUGIN_RESET_RUNTIME = 1u << 6u,
} LfsSceneUpscalerPluginResetFlag;

typedef int (*LfsSceneUpscalerAppendExtensionFn)(void* user, const char* extension_name);

typedef struct LfsSceneUpscalerExtensionSink {
    uint32_t struct_size;
    void* user;
    LfsSceneUpscalerAppendExtensionFn append;
} LfsSceneUpscalerExtensionSink;

typedef struct LfsSceneUpscalerBootstrapConfigV1 {
    uint32_t struct_size;
    const char* project_id;
    const char* engine_version;
    const wchar_t* application_data_path;
    const wchar_t* plugin_directory;
} LfsSceneUpscalerBootstrapConfigV1;

typedef struct LfsSceneUpscalerRuntimeConfigV1 {
    uint32_t struct_size;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    PFN_vkGetDeviceProcAddr get_device_proc_addr;
} LfsSceneUpscalerRuntimeConfigV1;

typedef struct LfsSceneUpscalerOptimalSettingsV1 {
    uint32_t struct_size;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t minimum_width;
    uint32_t minimum_height;
    uint32_t maximum_width;
    uint32_t maximum_height;
    float sharpness;
} LfsSceneUpscalerOptimalSettingsV1;

typedef struct LfsSceneUpscalerFeatureConfigV1 {
    uint32_t struct_size;
    uint32_t view;
    uint32_t quality;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t motion_vectors_include_jitter;
} LfsSceneUpscalerFeatureConfigV1;

typedef struct LfsSceneUpscalerImageV1 {
    uint32_t struct_size;
    VkImage image;
    VkImageView view;
    VkImageLayout layout;
    VkFormat format;
    uint32_t allocation_width;
    uint32_t allocation_height;
    uint32_t valid_width;
    uint32_t valid_height;
    uint32_t writable;
} LfsSceneUpscalerImageV1;

typedef struct LfsSceneUpscalerEvaluateV1 {
    uint32_t struct_size;
    uint32_t view;
    VkCommandBuffer command_buffer;
    /* Perceptually encoded LDR color in the top-left valid subrectangle. */
    LfsSceneUpscalerImageV1 color;
    /* Non-inverted raster depth: near=0, far=1, aligned with color. */
    LfsSceneUpscalerImageV1 depth;
    /* Current-to-previous displacement in top-left render pixels. */
    LfsSceneUpscalerImageV1 motion;
    LfsSceneUpscalerImageV1 output;
    /* Exact projection jitter in render-pixel space, same axes as motion. */
    float jitter_x_pixels;
    float jitter_y_pixels;
    float motion_scale_x;
    float motion_scale_y;
    float pre_exposure;
    float frame_time_milliseconds;
    uint32_t reset_flags;
} LfsSceneUpscalerEvaluateV1;

typedef struct LfsSceneUpscalerPluginApiV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* plugin_id;
    const char* display_name;

    void* (*create)(const LfsSceneUpscalerBootstrapConfigV1* config);
    void (*destroy)(void* plugin);
    LfsSceneUpscalerPluginResult (*required_instance_extensions)(
        void* plugin, const LfsSceneUpscalerExtensionSink* sink);
    LfsSceneUpscalerPluginResult (*required_device_extensions)(
        void* plugin,
        VkInstance instance,
        VkPhysicalDevice physical_device,
        const LfsSceneUpscalerExtensionSink* sink);
    LfsSceneUpscalerPluginResult (*initialize_runtime)(
        void* plugin, const LfsSceneUpscalerRuntimeConfigV1* config);
    LfsSceneUpscalerPluginResult (*optimal_settings)(
        void* plugin,
        uint32_t output_width,
        uint32_t output_height,
        uint32_t quality,
        LfsSceneUpscalerOptimalSettingsV1* settings);
    LfsSceneUpscalerPluginResult (*create_feature)(
        void* plugin,
        VkCommandBuffer command_buffer,
        const LfsSceneUpscalerFeatureConfigV1* config);
    LfsSceneUpscalerPluginResult (*evaluate)(
        void* plugin, const LfsSceneUpscalerEvaluateV1* evaluation);
    void (*release_feature)(void* plugin, uint32_t view);
    void (*shutdown_runtime)(void* plugin);
    size_t (*last_error)(void* plugin, char* destination, size_t capacity);
} LfsSceneUpscalerPluginApiV1;

/*
 * Validate the complete v1 function table before the host dereferences any
 * plugin-owned metadata or callback. This helper is header-only so the same
 * contract can be exercised without loading a vendor module.
 */
static inline int lfs_scene_upscaler_plugin_api_v1_complete(
    const LfsSceneUpscalerPluginApiV1* api) {
    const size_t minimum_size =
        offsetof(LfsSceneUpscalerPluginApiV1, last_error) + sizeof(api->last_error);
    return api != NULL && api->struct_size >= minimum_size &&
           api->abi_version == LFS_SCENE_UPSCALER_PLUGIN_ABI_V1 &&
           api->plugin_id != NULL && api->display_name != NULL &&
           api->create != NULL && api->destroy != NULL &&
           api->required_instance_extensions != NULL &&
           api->required_device_extensions != NULL &&
           api->initialize_runtime != NULL && api->optimal_settings != NULL &&
           api->create_feature != NULL && api->evaluate != NULL &&
           api->release_feature != NULL && api->shutdown_runtime != NULL &&
           api->last_error != NULL;
}

typedef const LfsSceneUpscalerPluginApiV1* (*LfsSceneUpscalerGetPluginApiV1Fn)(void);

LFS_SCENE_UPSCALER_PLUGIN_EXPORT const LfsSceneUpscalerPluginApiV1*
lfs_scene_upscaler_plugin_get_api_v1(void);

#ifdef __cplusplus
}
#endif
