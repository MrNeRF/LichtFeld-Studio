/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "gui/rmlui/rml_document_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

    std::filesystem::path projectRoot() {
#ifdef PROJECT_ROOT_PATH
        return std::filesystem::path(PROJECT_ROOT_PATH);
#else
        return std::filesystem::current_path();
#endif
    }

    std::string readText(const std::filesystem::path& path) {
        std::ifstream file(path);
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    std::filesystem::path rmlResourceDir() {
        return projectRoot() / "src/visualizer/gui/rmlui/resources";
    }

    bool containsPath(const std::vector<std::filesystem::path>& paths,
                      const std::filesystem::path& expected_path) {
        const auto normalized_expected = expected_path.lexically_normal();
        return std::any_of(paths.begin(), paths.end(), [&](const auto& path) {
            return path.lexically_normal() == normalized_expected;
        });
    }

} // namespace

TEST(RmlStaticStyleBoundaries, ImportDialogsShareOneStaticStylesheet) {
    const auto resource_dir = rmlResourceDir();
    const auto import_dialog_rcss = resource_dir / "import_dialog.rcss";

    EXPECT_TRUE(std::filesystem::exists(import_dialog_rcss));
    EXPECT_FALSE(std::filesystem::exists(resource_dir / "dataset_import_panel.rcss"));
    EXPECT_FALSE(std::filesystem::exists(resource_dir / "resume_checkpoint_panel.rcss"));

    const std::string dataset_rml = readText(resource_dir / "dataset_import_panel.rml");
    const std::string resume_rml = readText(resource_dir / "resume_checkpoint_panel.rml");

    EXPECT_NE(dataset_rml.find("import_dialog.rcss"), std::string::npos);
    EXPECT_NE(resume_rml.find("import_dialog.rcss"), std::string::npos);

    const auto dataset_stylesheets =
        lfs::vis::gui::rml_documents::loadLinkedStylesheetPaths(
            resource_dir / "dataset_import_panel.rml");
    const auto resume_stylesheets =
        lfs::vis::gui::rml_documents::loadLinkedStylesheetPaths(
            resource_dir / "resume_checkpoint_panel.rml");

    EXPECT_TRUE(containsPath(dataset_stylesheets, import_dialog_rcss));
    EXPECT_TRUE(containsPath(resume_stylesheets, import_dialog_rcss));
}

TEST(RmlStaticStyleBoundaries, ImmediateModeStaticStylesStayInRcss) {
    const std::string source = readText(projectRoot() / "src/python/lfs/rml_im_mode_layout.cpp");
    const std::string rcss = readText(rmlResourceDir() / "im_mode_panel.rcss");

    EXPECT_EQ(source.find("SetProperty(\"text-align\""), std::string::npos);
    EXPECT_EQ(source.find("SetProperty(\"width\", \"100%\""), std::string::npos);
    EXPECT_EQ(source.find("SetProperty(\"height\", \"8dp\""), std::string::npos);

    EXPECT_NE(rcss.find(".im-label--centered"), std::string::npos);
    EXPECT_NE(rcss.find(".im-spacing"), std::string::npos);
    EXPECT_NE(rcss.find(".im-control--fill"), std::string::npos);
    EXPECT_NE(rcss.find(".im-table-cell--fill"), std::string::npos);
}
