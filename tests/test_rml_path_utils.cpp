/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_path_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"

#include <gtest/gtest.h>

TEST(RmlPathUtilsTest, FileUriRoundTripsPlatformPath) {
#ifdef _WIN32
    const auto path = std::filesystem::path(L"C:\\Program Files\\LichtFeld Studio\\icon atlas.png");
#else
    const auto path = std::filesystem::path("/tmp/lichtfeld studio/icon atlas.png");
#endif

    const std::string uri = lfs::vis::gui::rml_paths::filesystemPathToFileUri(path);
    const auto decoded = lfs::vis::gui::rml_paths::fileUriToPath(uri);

    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(lfs::vis::gui::rml_paths::normalizeFilesystemPath(*decoded),
              lfs::vis::gui::rml_paths::normalizeFilesystemPath(path));
}

TEST(RmlPathUtilsTest, PathToRmlImageSourceEncodesSpaces) {
#ifdef _WIN32
    const auto path = std::filesystem::path(L"C:\\Program Files\\LichtFeld Studio\\dropdown arrow.png");
    const std::string source = lfs::vis::gui::rml_theme::pathToRmlImageSource(path);

    EXPECT_EQ(source,
              "file:///C:/Program%20Files/LichtFeld%20Studio/dropdown%20arrow.png");
#else
    const auto path = std::filesystem::path("/tmp/lichtfeld studio/dropdown arrow.png");
    const std::string source = lfs::vis::gui::rml_theme::pathToRmlImageSource(path);

    EXPECT_EQ(source, "/tmp/lichtfeld%20studio/dropdown%20arrow.png");
#endif
}

#ifdef _WIN32
TEST(RmlSystemInterfaceTest, JoinsWindowsDrivePathsWithoutTreatingDriveAsUriScheme) {
    lfs::vis::gui::RmlSystemInterface system_interface(nullptr);
    Rml::String translated_path;

    system_interface.JoinPath(
        translated_path,
        "C:/ProgramPortable/LSF-NB/share/LichtFeld-Studio/assets/rmlui/scene_tree.rml",
        "../icon/scene/search.png");

    EXPECT_EQ(translated_path,
              "C:/ProgramPortable/LSF-NB/share/LichtFeld-Studio/assets/icon/scene/search.png");
}
#endif
