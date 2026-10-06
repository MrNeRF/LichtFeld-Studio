/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis {

    enum class SceneUpscalerBackend : std::uint32_t {
        Native = 0,
        Spatial,
        Temporal,
        FirstExternal,
        // Keep dynamically allocated Vulkan plugin identities ABI-stable.
        MetalFxSpatial = 0xfffffff0u,
        MetalFxTemporal,
    };

    enum class SceneUpscalerFallback : std::uint8_t {
        None = 0,
        RuntimeUnavailable,
        UnsupportedMode,
    };

    struct SceneUpscalerPreset {
        std::string_view id;
        std::string_view label_key;
        float input_scale = 1.0f;
    };

    struct SceneUpscalerDescriptor {
        SceneUpscalerBackend backend = SceneUpscalerBackend::Native;
        std::string_view id;
        std::string_view label_key;
        std::span<const SceneUpscalerPreset> presets;
        std::string display_name;
    };

    struct SceneUpscalerSelection {
        SceneUpscalerBackend requested = SceneUpscalerBackend::Native;
        SceneUpscalerBackend effective = SceneUpscalerBackend::Native;
        SceneUpscalerFallback fallback = SceneUpscalerFallback::None;

        [[nodiscard]] constexpr bool fellBack() const noexcept {
            return fallback != SceneUpscalerFallback::None;
        }

        constexpr bool operator==(const SceneUpscalerSelection&) const = default;
    };

    [[nodiscard]] constexpr bool isMetalFxBackend(SceneUpscalerBackend backend) noexcept {
        return backend == SceneUpscalerBackend::MetalFxSpatial ||
               backend == SceneUpscalerBackend::MetalFxTemporal;
    }
    [[nodiscard]] constexpr bool isTemporalSceneUpscaler(SceneUpscalerBackend backend) noexcept {
        return backend == SceneUpscalerBackend::Temporal ||
               backend == SceneUpscalerBackend::MetalFxTemporal;
    }
    [[nodiscard]] LFS_VIS_API bool metalFxBackendAvailable(SceneUpscalerBackend backend);

    // Built-in backends plus every optional plugin that is installed.
    [[nodiscard]] LFS_VIS_API std::vector<SceneUpscalerDescriptor> sceneUpscalerDescriptors();
    [[nodiscard]] LFS_VIS_API SceneUpscalerDescriptor sceneUpscalerDescriptor(
        SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(
        std::string_view id);
    [[nodiscard]] LFS_VIS_API bool sceneUpscalerBackendAvailable(SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::string_view sceneUpscalerBackendId(SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::optional<SceneUpscalerPreset> sceneUpscalerPreset(
        SceneUpscalerBackend backend, std::string_view preset_id);
    [[nodiscard]] LFS_VIS_API SceneUpscalerPreset defaultSceneUpscalerPreset(
        SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::optional<SceneUpscalerPreset> resolveSceneUpscalerPresetUpdate(
        SceneUpscalerBackend backend,
        std::optional<std::string_view> explicit_preset_id,
        std::string_view remembered_preset_id);
    [[nodiscard]] LFS_VIS_API SceneUpscalerSelection resolveSceneUpscalerSelection(
        SceneUpscalerBackend requested,
        bool runtime_available,
        SceneUpscalerFallback fallback = SceneUpscalerFallback::RuntimeUnavailable);
    [[nodiscard]] LFS_VIS_API std::string_view sceneUpscalerFallbackId(
        SceneUpscalerFallback fallback) noexcept;

} // namespace lfs::vis
