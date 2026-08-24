/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_upscaler_registry.hpp"

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
        constexpr std::array DESCRIPTORS{
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
        };
    } // namespace

    std::span<const SceneUpscalerDescriptor> sceneUpscalerDescriptors() {
        return DESCRIPTORS;
    }

    const SceneUpscalerDescriptor& sceneUpscalerDescriptor(const SceneUpscalerBackend backend) {
        const auto found = std::ranges::find(DESCRIPTORS, backend, &SceneUpscalerDescriptor::backend);
        return found != DESCRIPTORS.end() ? *found : DESCRIPTORS.front();
    }

    std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(const std::string_view id) {
        const auto found = std::ranges::find(DESCRIPTORS, id, &SceneUpscalerDescriptor::id);
        if (found == DESCRIPTORS.end())
            return std::nullopt;
        return found->backend;
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
        const bool runtime_available) {
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
            .fallback = SceneUpscalerFallback::RuntimeUnavailable,
        };
    }

    std::string_view sceneUpscalerFallbackId(const SceneUpscalerFallback fallback) noexcept {
        switch (fallback) {
        case SceneUpscalerFallback::None:
            return "none";
        case SceneUpscalerFallback::RuntimeUnavailable:
            return "runtime_unavailable";
        }
        return "unknown";
    }

} // namespace lfs::vis
