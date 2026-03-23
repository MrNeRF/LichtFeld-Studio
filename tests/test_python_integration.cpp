/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/logger.hpp"
#include "python/runner.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class PythonIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::python::ensure_initialized();
    }

    std::filesystem::path createTempScript(const std::string& content) {
        auto temp_dir = std::filesystem::temp_directory_path();
        auto script_path = temp_dir / "test_script.py";
        std::ofstream ofs(script_path);
        ofs << content;
        ofs.close();
        return script_path;
    }
};

namespace {
    bool formatterUnavailable(const lfs::python::FormatResult& result) {
        return !result.success &&
               (result.error.find("uv not found") != std::string::npos ||
                result.error.find("Failed to create venv for black") != std::string::npos ||
                result.error.find("ImportError") != std::string::npos);
    }
} // namespace

TEST_F(PythonIntegrationTest, InitializationSucceeds) {
    // Just verify that initialization doesn't throw
    EXPECT_NO_THROW(lfs::python::ensure_initialized());
}

TEST_F(PythonIntegrationTest, OutputCallbackCanBeSet) {
    bool callback_set = false;
    lfs::python::set_output_callback([&](const std::string&, bool) { callback_set = true; });
    EXPECT_TRUE(true); // If we got here, setting the callback didn't crash
}

TEST_F(PythonIntegrationTest, OutputRedirectCanBeInstalled) {
    // This should not throw
    EXPECT_NO_THROW(lfs::python::install_output_redirect());
}

TEST_F(PythonIntegrationTest, EmptyScriptListSucceeds) {
    auto result = lfs::python::run_scripts({});
    EXPECT_TRUE(result.has_value()) << "Empty script list should succeed";
}

TEST_F(PythonIntegrationTest, FormatPythonCodePreservesValidBlockIndentation) {
    const auto result = lfs::python::format_python_code("if True:\n    print('x')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code, "if True:\n    print(\"x\")\n");
}

TEST_F(PythonIntegrationTest, FormatPythonCodeDedentsIndentedSnippet) {
    const auto result = lfs::python::format_python_code("    if True:\n        print('x')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code, "if True:\n    print(\"x\")\n");
}

TEST_F(PythonIntegrationTest, FormatPythonCodeRepairsUnexpectedTopLevelIndent) {
    const auto result = lfs::python::format_python_code(
        "import lichtfeld as lf\n    scene = lf.get_scene()\nprint('hello world')\n");

    if (formatterUnavailable(result)) {
        GTEST_SKIP() << result.error;
    }
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.code.find("\n    scene = lf.get_scene()"), std::string::npos);
    EXPECT_NE(result.code.find("scene = lf.get_scene()"), std::string::npos);
    EXPECT_NE(result.code.find("print(\"hello world\")"), std::string::npos);
}

// NOTE: Tests that actually execute Python scripts require the lichtfeld module
// to be importable, which depends on the CommandCenter and training infrastructure.
// These are better tested via integration tests (running training with --python-script).
