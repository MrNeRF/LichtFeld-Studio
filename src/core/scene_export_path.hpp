/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/events.hpp"
#include "core/file_extensions.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <string_view>

namespace lfs::core {
    struct SceneExporterInfo {
        ExportFormat format;
        std::array<std::string_view, 3> extensions;
        bool unique_splat_suffix;
    };

    inline constexpr std::array kSceneExporterRegistry = {
        SceneExporterInfo{ExportFormat::PLY, {".ply"}, true},
        SceneExporterInfo{ExportFormat::SOG, {".sog"}, true},
        SceneExporterInfo{ExportFormat::SPZ, {".spz"}, true},
        SceneExporterInfo{ExportFormat::HTML_VIEWER, {".html"}, false},
        SceneExporterInfo{ExportFormat::USD, {".usd", ".usda", ".usdc"}, true},
        SceneExporterInfo{ExportFormat::NUREC_USDZ, {".usdz"}, true},
        SceneExporterInfo{ExportFormat::RAD, {".rad"}, true},
        SceneExporterInfo{ExportFormat::COLMAP, {}, false},
        SceneExporterInfo{ExportFormat::SSOG, {".ssog"}, true},
        SceneExporterInfo{ExportFormat::GALLERY_SCENE, {}, false},
        SceneExporterInfo{ExportFormat::GALLERY_SOG, {}, false},
        SceneExporterInfo{ExportFormat::GALLERY_SSOG, {}, false},
        SceneExporterInfo{ExportFormat::GALLERY_SPZ, {}, false},
        SceneExporterInfo{ExportFormat::GLB, {".glb"}, false},
    };

    static_assert([] {
        if (kSceneExporterRegistry.size() != static_cast<std::size_t>(ExportFormat::COUNT))
            return false;
        for (std::size_t i = 0; i < kSceneExporterRegistry.size(); ++i) {
            if (static_cast<std::size_t>(kSceneExporterRegistry[i].format) != i)
                return false;
        }
        return true;
    }());

    [[nodiscard]] inline std::string_view required_scene_export_extension(const ExportFormat format) {
        switch (format) {
        case ExportFormat::PLY:
        case ExportFormat::SOG:
        case ExportFormat::SPZ:
            return kSceneExporterRegistry[static_cast<std::size_t>(format)].extensions[0];
        default: return {};
        }
    }

    enum class SceneExportPathStatus { Match,
                                       Warn,
                                       Conflict };

    [[nodiscard]] inline SceneExportPathStatus scene_export_path_status(const ExportFormat format,
                                                                        const std::filesystem::path& path) {
        const auto extension = required_scene_export_extension(format);
        if (extension.empty() || has_extension(path, extension))
            return SceneExportPathStatus::Match;
        for (const auto& exporter : kSceneExporterRegistry) {
            if (!exporter.unique_splat_suffix)
                continue;
            for (const auto supported : exporter.extensions) {
                if (!supported.empty() && has_extension(path, supported))
                    return SceneExportPathStatus::Conflict;
            }
        }
        return SceneExportPathStatus::Warn;
    }

    [[nodiscard]] inline std::string_view scene_export_format_name(const ExportFormat format) {
        switch (format) {
        case ExportFormat::PLY: return "PLY";
        case ExportFormat::SOG: return "SOG";
        case ExportFormat::SPZ: return "SPZ";
        default: return {};
        }
    }
} // namespace lfs::core
