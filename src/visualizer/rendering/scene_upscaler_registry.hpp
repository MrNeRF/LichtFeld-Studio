/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace lfs::vis {

    enum class SceneUpscalerBackend : std::uint8_t {
        Native = 0,
        Spatial,
    };

    enum class SceneUpscalerFallback : std::uint8_t {
        None = 0,
        RuntimeUnavailable,
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

    [[nodiscard]] LFS_VIS_API std::span<const SceneUpscalerDescriptor> sceneUpscalerDescriptors();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& sceneUpscalerDescriptor(
        SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(
        std::string_view id);
    [[nodiscard]] LFS_VIS_API std::string_view sceneUpscalerBackendId(SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::optional<SceneUpscalerPreset> sceneUpscalerPreset(
        SceneUpscalerBackend backend, std::string_view preset_id);
    [[nodiscard]] LFS_VIS_API SceneUpscalerPreset defaultSceneUpscalerPreset(
        SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API SceneUpscalerSelection resolveSceneUpscalerSelection(
        SceneUpscalerBackend requested, bool runtime_available);

} // namespace lfs::vis
