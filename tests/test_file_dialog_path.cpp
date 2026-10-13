/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene_export_path.hpp"
#include "gui/utils/file_dialog_path.hpp"
#include <cctype>
#include <gtest/gtest.h>

namespace {
    using lfs::vis::gui::detail::appendRequiredExtension;
    using lfs::vis::gui::detail::saveDialogDefaultName;

    TEST(FileDialogPathTest, NormalizesSupportedExtensionCase) {
        for (const auto* extension : {".ply", ".sog", ".spz", ".ssog", ".rad", ".licht", ".json", ".glb"}) {
            std::string upper = extension;
            for (char& c : upper)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            const auto path = std::filesystem::path("folder.with.dots") / ("cloud-context" + upper);
            const auto expected = std::filesystem::path("folder.with.dots") / ("cloud-context" + std::string(extension));
            EXPECT_EQ(appendRequiredExtension(path, extension), expected);
            EXPECT_EQ(appendRequiredExtension(path, extension).extension(), extension);
            EXPECT_EQ(saveDialogDefaultName(path.string(), extension), "cloud-context");
        }
        EXPECT_EQ(appendRequiredExtension("cloud.PlY", ".ply"), "cloud.ply");
    }

    TEST(FileDialogPathTest, PreservesExistingSaveBehavior) {
        for (const auto* extension : {".ply", ".sog", ".spz", ".ssog", ".rad", ".licht", ".json", ".glb",
                                      ".las", ".laz", ".png", ".jpg", ".txt", ".usd", ".usdz", ".html", ".mp4", ".py"}) {
            const std::string name = "cloud-context";
            const std::string suffixed = name + extension;
            EXPECT_EQ(appendRequiredExtension(suffixed, extension), suffixed);
            EXPECT_EQ(appendRequiredExtension(name, extension), suffixed);
            EXPECT_EQ(appendRequiredExtension(name + ".unsupported", extension), name + ".unsupported" + extension);
            EXPECT_EQ(appendRequiredExtension("", extension), "");
            EXPECT_EQ(saveDialogDefaultName("folder/" + suffixed, extension), name);
            EXPECT_EQ(saveDialogDefaultName("folder/" + name, extension), name);
            EXPECT_EQ(saveDialogDefaultName("", extension), "");
        }
        EXPECT_EQ(appendRequiredExtension("cloud", ""), "cloud");
        EXPECT_EQ(appendRequiredExtension("cloud.sog", ".ply"), "cloud.sog.ply");
        EXPECT_EQ(appendRequiredExtension("cloud.ply.backup", ".ply"), "cloud.ply.backup.ply");
        EXPECT_EQ(appendRequiredExtension(".hidden", ".ply"), ".hidden.ply");
    }

    TEST(SceneExportPathTest, RejectsOnlyConflictingSupportedExtensions) {
        using lfs::core::ExportFormat;
        using lfs::core::scene_export_path_status;
        using lfs::core::SceneExportPathStatus;

        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.spz"), SceneExportPathStatus::Conflict);
        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.SpZ"), SceneExportPathStatus::Conflict);
        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.ssog"), SceneExportPathStatus::Conflict);
        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.rad"), SceneExportPathStatus::Conflict);
        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.glb"), SceneExportPathStatus::Warn);
        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.html"), SceneExportPathStatus::Warn);
        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.usdz"), SceneExportPathStatus::Conflict);
        EXPECT_EQ(scene_export_path_status(ExportFormat::SOG, "cloud.ply"), SceneExportPathStatus::Conflict);
        EXPECT_EQ(scene_export_path_status(ExportFormat::SPZ, "cloud.sog"), SceneExportPathStatus::Conflict);
        for (const auto format : {ExportFormat::PLY, ExportFormat::SOG, ExportFormat::SPZ}) {
            for (const auto* path : {"cloud", "cloud.bak", "cloud.tmp", "cloud.data", "cloud.ply.backup"}) {
                EXPECT_EQ(scene_export_path_status(format, path), SceneExportPathStatus::Warn)
                    << path;
            }
        }
    }

    TEST(SceneExportPathTest, PreservesMatchingExtensionsAndOtherFormats) {
        using lfs::core::ExportFormat;
        using lfs::core::scene_export_path_status;
        using lfs::core::SceneExportPathStatus;

        EXPECT_EQ(scene_export_path_status(ExportFormat::PLY, "cloud.PLY"), SceneExportPathStatus::Match);
        EXPECT_EQ(scene_export_path_status(ExportFormat::SOG, "cloud.SoG"), SceneExportPathStatus::Match);
        EXPECT_EQ(scene_export_path_status(ExportFormat::SPZ, "cloud.SPZ"), SceneExportPathStatus::Match);
        EXPECT_EQ(scene_export_path_status(ExportFormat::USD, "cloud.usda"), SceneExportPathStatus::Match);
        EXPECT_EQ(scene_export_path_status(ExportFormat::SSOG, "bundle"), SceneExportPathStatus::Match);
    }
} // namespace
