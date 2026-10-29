/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_path_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"

#include <chrono>
#include <format>
#include <fstream>
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

TEST(RmlDocumentUtilsTest, RewritesRelativeImageSourcesToResolvedImageSource) {
    const auto test_root = std::filesystem::temp_directory_path() /
                           std::format("lichtfeld-rml-{}", std::chrono::steady_clock::now().time_since_epoch().count());
    const auto document_path = test_root / "assets/rmlui/training.rml";
    const auto icon_path = test_root / "assets/icon/settings.png";

    std::filesystem::create_directories(document_path.parent_path());
    std::filesystem::create_directories(icon_path.parent_path());
    std::ofstream(icon_path, std::ios::binary) << "png";

    const std::string rewritten = lfs::vis::gui::rml_documents::rewriteRelativeImageSources(
        R"(<img class="training-panel-title-icon" src="../icon/settings.png" />)",
        document_path);

    EXPECT_NE(rewritten.find(lfs::vis::gui::rml_theme::pathToRmlImageSource(icon_path)),
              std::string::npos);

    std::filesystem::remove_all(test_root);
}

TEST(RmlDocumentUtilsTest, LeavesNonFileImageSourcesUnchanged) {
    const auto document_path = std::filesystem::path("/tmp/lichtfeld/rmlui/test.rml");
    const std::string source =
        R"(<img src="preview://cam=1&amp;path=%2Ftmp%2Fimage.png" /><img src="file:///tmp/icon.png" />)";

    const std::string rewritten =
        lfs::vis::gui::rml_documents::rewriteRelativeImageSources(source, document_path);

    EXPECT_EQ(rewritten, source);
}

// RmlUi passes every image source through JoinPath before LoadTexture opens it,
// so an encoded source must come back as the original path (e.g. a mounted
// "/Volumes/LichtFeld Studio" disk image).
TEST(RmlSystemInterfaceTest, JoinPathDecodesImageSourcesWithSpaces) {
#ifdef _WIN32
    const auto path = std::filesystem::path(L"C:\\Program Files\\LichtFeld Studio\\splash logo.png");
    const Rml::String document_path = "C:/Program Files/LichtFeld Studio/assets/rmlui/startup.rml";
#else
    const auto path = std::filesystem::path("/Volumes/LichtFeld Studio/LichtFeld-Studio.app/splash logo.png");
    const Rml::String document_path = "/Volumes/LichtFeld Studio/LichtFeld-Studio.app/assets/rmlui/startup.rml";
#endif
    lfs::vis::gui::RmlSystemInterface system_interface(nullptr);
    Rml::String translated_path;

    system_interface.JoinPath(translated_path, document_path,
                              lfs::vis::gui::rml_theme::pathToRmlImageSource(path));

    EXPECT_EQ(translated_path, lfs::vis::gui::rml_paths::normalizeFilesystemPath(path));
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
