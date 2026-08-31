/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_upscaler_registry.hpp"

#include "rendering/amd_fsr3_plugin.hpp"
#include "rendering/nvidia_dlss_plugin.hpp"

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
        constexpr std::array NVIDIA_DLSS_PRESETS{
            SceneUpscalerPreset{
                .id = "quality",
                .label_key = "preferences.scene_reconstruction_quality",
                .input_scale = 2.0f / 3.0f,
            },
            SceneUpscalerPreset{
                .id = "balanced",
                .label_key = "preferences.scene_reconstruction_balanced",
                .input_scale = 0.58f,
            },
            SceneUpscalerPreset{
                .id = "performance",
                .label_key = "preferences.scene_reconstruction_performance",
                .input_scale = 0.50f,
            },
        };
        constexpr std::array AMD_FSR3_PRESETS{
            SceneUpscalerPreset{
                .id = "quality",
                .label_key = "preferences.scene_reconstruction_quality",
                .input_scale = 2.0f / 3.0f,
            },
            SceneUpscalerPreset{
                .id = "balanced",
                .label_key = "preferences.scene_reconstruction_balanced",
                .input_scale = 1.0f / 1.7f,
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
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::NvidiaDlss,
                .id = "nvidia-dlss",
                .label_key = "preferences.scene_reconstruction_nvidia_dlss",
                .presets = NVIDIA_DLSS_PRESETS,
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::AmdFsr3,
                .id = "amd-fsr3",
                .label_key = "preferences.scene_reconstruction_amd_fsr3",
                .presets = AMD_FSR3_PRESETS,
            },
        };

        template <bool IncludeNvidiaDlss, bool IncludeAmdFsr3>
        [[nodiscard]] constexpr auto makeAvailableDescriptors() {
            constexpr std::size_t count =
                DESCRIPTORS.size() - (IncludeNvidiaDlss ? 0u : 1u) -
                (IncludeAmdFsr3 ? 0u : 1u);
            std::array<SceneUpscalerDescriptor, count> filtered{};
            std::size_t index = 0;
            for (const auto& descriptor : DESCRIPTORS) {
                if ((!IncludeNvidiaDlss &&
                     descriptor.backend == SceneUpscalerBackend::NvidiaDlss) ||
                    (!IncludeAmdFsr3 &&
                     descriptor.backend == SceneUpscalerBackend::AmdFsr3)) {
                    continue;
                }
                filtered[index++] = descriptor;
            }
            return filtered;
        }

        constexpr auto CORE_DESCRIPTORS = makeAvailableDescriptors<false, false>();
        constexpr auto DESCRIPTORS_WITH_NVIDIA_DLSS =
            makeAvailableDescriptors<true, false>();
        constexpr auto DESCRIPTORS_WITH_AMD_FSR3 =
            makeAvailableDescriptors<false, true>();
    } // namespace

    std::span<const SceneUpscalerDescriptor> sceneUpscalerDescriptors() {
        const bool nvidia_dlss_available = nvidiaDlssPluginAvailable();
        const bool amd_fsr3_available = amdFsr3PluginAvailable();
        if (nvidia_dlss_available && amd_fsr3_available)
            return DESCRIPTORS;
        if (nvidia_dlss_available)
            return DESCRIPTORS_WITH_NVIDIA_DLSS;
        if (amd_fsr3_available)
            return DESCRIPTORS_WITH_AMD_FSR3;
        return CORE_DESCRIPTORS;
    }

    const SceneUpscalerDescriptor& sceneUpscalerDescriptor(const SceneUpscalerBackend backend) {
        const auto found =
            std::ranges::find(DESCRIPTORS, backend, &SceneUpscalerDescriptor::backend);
        return found != DESCRIPTORS.end() ? *found : DESCRIPTORS.front();
    }

    std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(const std::string_view id) {
        const auto found = std::ranges::find(DESCRIPTORS, id, &SceneUpscalerDescriptor::id);
        if (found == DESCRIPTORS.end())
            return std::nullopt;
        return found->backend;
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
