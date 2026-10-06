/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_upscaler_registry.hpp"

#include "rendering/scene_upscaler_plugin.hpp"

#include <algorithm>
#include <array>

namespace lfs::vis {
    namespace {
        constexpr std::array NATIVE_PRESETS{
            SceneUpscalerPreset{
                .id = "native",
                .label_key = "preferences.scene_reconstruction_off",
                .input_scale = 1.0f,
            },
        };
        constexpr std::array SPATIAL_PRESETS{
            SceneUpscalerPreset{
                .id = "quality",
                .label_key = "preferences.scene_reconstruction_quality",
                .input_scale = 0.75f,
            },
            SceneUpscalerPreset{
                .id = "balanced",
                .label_key = "preferences.scene_reconstruction_balanced",
                .input_scale = 0.67f,
            },
            SceneUpscalerPreset{
                .id = "performance",
                .label_key = "preferences.scene_reconstruction_performance",
                .input_scale = 0.50f,
            },
        };
        constexpr std::array TEMPORAL_PRESETS{
            SceneUpscalerPreset{
                .id = "quality",
                .label_key = "preferences.scene_reconstruction_quality",
                .input_scale = 0.75f,
            },
            SceneUpscalerPreset{
                .id = "balanced",
                .label_key = "preferences.scene_reconstruction_balanced",
                .input_scale = 0.67f,
            },
            SceneUpscalerPreset{
                .id = "performance",
                .label_key = "preferences.scene_reconstruction_performance",
                .input_scale = 0.50f,
            },
        };
        constexpr std::array METALFX_PRESETS{
            SceneUpscalerPreset{"quality", "preferences.scene_reconstruction_quality", 2.0f / 3.0f},
            SceneUpscalerPreset{"balanced", "preferences.scene_reconstruction_balanced", 1.0f / 1.7f},
            SceneUpscalerPreset{"performance", "preferences.scene_reconstruction_performance", 0.5f},
        };
        const std::array DESCRIPTORS{
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Native,
                .id = "native",
                .label_key = "preferences.scene_reconstruction_off",
                .presets = NATIVE_PRESETS,
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Spatial,
                .id = "spatial",
                .label_key = "preferences.scene_reconstruction_spatial",
                .presets = SPATIAL_PRESETS,
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Temporal,
                .id = "temporal",
                .label_key = "preferences.scene_reconstruction_temporal",
                .presets = TEMPORAL_PRESETS,
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::MetalFxSpatial,
                .id = "metalfx_spatial",
                .presets = METALFX_PRESETS,
                .display_name = "Apple MetalFX Spatial",
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::MetalFxTemporal,
                .id = "metalfx_temporal",
                .presets = METALFX_PRESETS,
                .display_name = "Apple MetalFX Temporal",
            },
        };

    } // namespace

    std::vector<SceneUpscalerDescriptor> sceneUpscalerDescriptors() {
        std::vector<SceneUpscalerDescriptor> available;
        for (const auto& descriptor : DESCRIPTORS) {
            if (isMetalFxBackend(descriptor.backend) && !metalFxBackendAvailable(descriptor.backend))
                continue;
            auto* const plugin = sceneUpscalerPlugin(descriptor.backend);
            if (plugin == nullptr || plugin->available())
                available.push_back(descriptor);
        }
        for (auto* const plugin : sceneUpscalerPlugins()) {
            if (plugin->available())
                available.push_back(sceneUpscalerDescriptor(plugin->info().backend));
        }
        return available;
    }

    SceneUpscalerDescriptor sceneUpscalerDescriptor(const SceneUpscalerBackend backend) {
        const auto found =
            std::ranges::find(DESCRIPTORS, backend, &SceneUpscalerDescriptor::backend);
        if (found != DESCRIPTORS.end())
            return *found;
        if (auto* const plugin = sceneUpscalerPlugin(backend)) {
            return {
                .backend = backend,
                .id = plugin->info().id,
                .label_key = "",
                .presets = plugin->info().presets,
                .display_name = plugin->displayName(),
            };
        }
        return DESCRIPTORS.front();
    }

    std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(const std::string_view id) {
        const auto found = std::ranges::find(DESCRIPTORS, id, &SceneUpscalerDescriptor::id);
        if (found != DESCRIPTORS.end())
            return found->backend;
        for (const auto* const plugin : sceneUpscalerPlugins()) {
            if (plugin->info().id == id)
                return plugin->info().backend;
        }
        return std::nullopt;
    }

    bool sceneUpscalerBackendAvailable(const SceneUpscalerBackend backend) {
        return std::ranges::contains(
            sceneUpscalerDescriptors(), backend, &SceneUpscalerDescriptor::backend);
    }

    std::string_view sceneUpscalerBackendId(const SceneUpscalerBackend backend) {
        return sceneUpscalerDescriptor(backend).id;
    }

    std::optional<SceneUpscalerPreset> sceneUpscalerPreset(
        const SceneUpscalerBackend backend,
        const std::string_view preset_id) {
        const auto presets = sceneUpscalerDescriptor(backend).presets;
        const auto found = std::ranges::find(presets, preset_id, &SceneUpscalerPreset::id);
        if (found == presets.end())
            return std::nullopt;
        return *found;
    }

    SceneUpscalerPreset defaultSceneUpscalerPreset(const SceneUpscalerBackend backend) {
        const auto presets = sceneUpscalerDescriptor(backend).presets;
        return presets.empty() ? SceneUpscalerPreset{
                                     .id = "native",
                                     .label_key = "preferences.scene_reconstruction_off",
                                     .input_scale = 1.0f,
                                 }
                               : presets.front();
    }

    std::optional<SceneUpscalerPreset> resolveSceneUpscalerPresetUpdate(
        const SceneUpscalerBackend backend,
        const std::optional<std::string_view> explicit_preset_id,
        const std::string_view remembered_preset_id) {
        if (explicit_preset_id) {
            return sceneUpscalerPreset(backend, *explicit_preset_id);
        }
        return sceneUpscalerPreset(backend, remembered_preset_id)
            .value_or(defaultSceneUpscalerPreset(backend));
    }

    SceneUpscalerSelection resolveSceneUpscalerSelection(
        const SceneUpscalerBackend requested,
        const bool runtime_available,
        const SceneUpscalerFallback fallback) {
        if (requested == SceneUpscalerBackend::Native || runtime_available) {
            return {
                .requested = requested,
                .effective = requested,
                .fallback = SceneUpscalerFallback::None,
            };
        }
        return {
            .requested = requested,
            .effective = SceneUpscalerBackend::Native,
            .fallback = fallback,
        };
    }

    std::string_view sceneUpscalerFallbackId(const SceneUpscalerFallback fallback) noexcept {
        switch (fallback) {
        case SceneUpscalerFallback::None:
            return "none";
        case SceneUpscalerFallback::RuntimeUnavailable:
            return "runtime_unavailable";
        case SceneUpscalerFallback::UnsupportedMode:
            return "unsupported_mode";
        }
        return "unknown";
    }

} // namespace lfs::vis

#if !defined(LFS_TENSOR_METAL) || defined(LFS_GRAPHICS_VULKAN)
namespace lfs::vis {
    bool metalFxBackendAvailable(SceneUpscalerBackend) { return false; }
}
#endif
