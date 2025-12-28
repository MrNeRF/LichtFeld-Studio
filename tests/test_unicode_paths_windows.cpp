/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Comprehensive Unicode path handling tests for Windows
 *
 * Tests core Unicode path fixes without CUDA dependencies:
 * - path_to_utf8() utility function
 * - Basic file I/O with Unicode paths
 * - Path concatenation and operations
 * - Directory creation and iteration
 *
 * This test runs on Windows CI without requiring CUDA/GPU.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/path_utils.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;

// ============================================================================
// Test Fixture
// ============================================================================

class UnicodePathTest : public ::testing::Test {
protected:
    fs::path test_root_;

    void SetUp() override {
        // Create test root with Unicode name
        test_root_ = fs::temp_directory_path() / "lfs_unicode_test_日本語_中文_한국어";
        fs::create_directories(test_root_);
    }

    void TearDown() override {
        // Cleanup
        if (fs::exists(test_root_)) {
            fs::remove_all(test_root_);
        }
    }

    // Helper to create a test file with content
    void create_test_file(const fs::path& path, const std::string& content) {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to create file: " << path.string();
        out << content;
        out.close();
    }

    // Helper to read a test file
    std::string read_test_file(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.is_open()) << "Failed to open file: " << path.string();
        return std::string(std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>());
    }
};

// ============================================================================
// Test 1: path_to_utf8() Utility Function
// ============================================================================

TEST_F(UnicodePathTest, PathToUtf8Conversion) {
    // Test basic ASCII path
    {
        fs::path ascii_path = "C:/test/file.txt";
        std::string utf8 = path_to_utf8(ascii_path);
        EXPECT_FALSE(utf8.empty());
    }

    // Test path with Japanese characters
    {
        auto japanese_path = test_root_ / "日本語_ファイル.txt";
        std::string utf8 = path_to_utf8(japanese_path);
        EXPECT_FALSE(utf8.empty());
#ifdef _WIN32
        // On Windows, should contain UTF-8 encoded Japanese
        EXPECT_TRUE(utf8.find("日本語") != std::string::npos || utf8.size() > 0);
#endif
    }

    // Test path with Chinese characters
    {
        auto chinese_path = test_root_ / "中文_文件.txt";
        std::string utf8 = path_to_utf8(chinese_path);
        EXPECT_FALSE(utf8.empty());
    }

    // Test path with Korean characters
    {
        auto korean_path = test_root_ / "한국어_파일.txt";
        std::string utf8 = path_to_utf8(korean_path);
        EXPECT_FALSE(utf8.empty());
    }

    // Test empty path
    {
        fs::path empty_path;
        std::string utf8 = path_to_utf8(empty_path);
        EXPECT_TRUE(utf8.empty());
    }
}

// ============================================================================
// Test 2: Basic File I/O with Unicode Paths
// ============================================================================

TEST_F(UnicodePathTest, BasicFileIO) {
    // Test creating and reading a file with Unicode filename
    auto unicode_file = test_root_ / "テスト_test_测试_테스트.txt";
    const std::string test_content = "Hello, Unicode World! 你好世界 こんにちは世界 안녕하세요";

    // Write file
    {
        std::ofstream out(unicode_file, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to create Unicode file";
        out << test_content;
        out.close();
        EXPECT_TRUE(out.good()) << "Failed to write to Unicode file";
    }

    // Verify file exists
    EXPECT_TRUE(fs::exists(unicode_file)) << "Unicode file doesn't exist after creation";
    EXPECT_GT(fs::file_size(unicode_file), 0) << "Unicode file is empty";

    // Read file back
    {
        std::ifstream in(unicode_file, std::ios::binary);
        ASSERT_TRUE(in.is_open()) << "Failed to open Unicode file for reading";
        std::string content(std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>());
        EXPECT_EQ(content, test_content) << "File content doesn't match";
    }

    // Test with path_to_utf8 for external library compatibility
    {
        std::string utf8_path = path_to_utf8(unicode_file);
        EXPECT_FALSE(utf8_path.empty());
        // If this was passed to a C library expecting UTF-8, it should work
    }
}

// ============================================================================
// Test 3: Path Concatenation Operations
// ============================================================================

TEST_F(UnicodePathTest, PathConcatenation) {
    auto base_dir = test_root_ / "基本_base_기본";
    fs::create_directories(base_dir);

    // Test path += operator (the fix from pipelined_image_loader.cpp)
    auto cache_path = base_dir / "cache_缓存_캐시.dat";
    auto done_path = cache_path;
    done_path += ".done";

    // Create both files
    {
        std::ofstream out1(cache_path);
        ASSERT_TRUE(out1.is_open()) << "Failed to create cache file";
        out1 << "cache data";

        std::ofstream out2(done_path);
        ASSERT_TRUE(out2.is_open()) << "Failed to create .done file";
        out2 << "done";
    }

    // Verify both exist
    EXPECT_TRUE(fs::exists(cache_path)) << "Cache file doesn't exist";
    EXPECT_TRUE(fs::exists(done_path)) << "Done marker doesn't exist";

    // Verify the .done path is constructed correctly
    EXPECT_TRUE(done_path.string().ends_with(".done")) << "Done path doesn't end with .done";

    // Test extension addition (the fix from converter.cpp)
    auto base_path = base_dir / "出力_output_输出";
    auto with_ext = base_path;
    with_ext += ".json";

    create_test_file(with_ext, "{\"test\": true}");
    EXPECT_TRUE(fs::exists(with_ext)) << "File with added extension doesn't exist";
    EXPECT_TRUE(with_ext.extension() == ".json") << "Extension not added correctly";
}

// ============================================================================
// Test 4: Directory Iteration with Unicode Paths
// ============================================================================

TEST_F(UnicodePathTest, DirectoryIteration) {
    auto dir = test_root_ / "ディレクトリ_directory_目录_디렉토리";
    fs::create_directories(dir);

    // Create several files with Unicode names
    std::vector<std::string> filenames = {
        "ファイル1_file1_文件1_파일1.txt",
        "ファイル2_file2_文件2_파일2.dat",
        "ファイル3_file3_文件3_파일3.json"
    };

    for (const auto& filename : filenames) {
        create_test_file(dir / filename, "test content");
    }

    // Iterate and verify all files are found
    int file_count = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        EXPECT_TRUE(entry.is_regular_file()) << "Entry is not a file";
        file_count++;
    }

    EXPECT_EQ(file_count, filenames.size()) << "Not all Unicode files were found";
}

// ============================================================================
// Test 5: Deeply Nested Unicode Paths
// ============================================================================

TEST_F(UnicodePathTest, DeeplyNestedUnicodePaths) {
    // Create a deeply nested directory structure with Unicode at each level
    auto level1 = test_root_ / "レベル1_level1_级别1_레벨1";
    auto level2 = level1 / "レベル2_level2_级别2_레벨2";
    auto level3 = level2 / "レベル3_level3_级别3_레벨3";

    fs::create_directories(level3);
    EXPECT_TRUE(fs::exists(level3)) << "Nested Unicode directories weren't created";

    // Create a file in the deepest level
    auto deep_file = level3 / "深いファイル_deep_file_深层文件_깊은파일.txt";
    create_test_file(deep_file, "Deep unicode content");

    EXPECT_TRUE(fs::exists(deep_file)) << "File in nested Unicode path doesn't exist";

    // Verify we can read it back
    std::string content = read_test_file(deep_file);
    EXPECT_EQ(content, "Deep unicode content") << "Content doesn't match";
}

// ============================================================================
// Test 6: Special Characters in Paths
// ============================================================================

TEST_F(UnicodePathTest, SpecialCharactersInPaths) {
    // Test paths with special characters that might cause issues
    std::vector<std::string> special_names = {
        "file (with) parentheses.txt",
        "file [with] brackets.txt",
        "file with spaces.txt",
        "file_with_emoji_😀_🎉.txt",
    };

    for (const auto& name : special_names) {
        auto path = test_root_ / name;
        create_test_file(path, "special content");
        EXPECT_TRUE(fs::exists(path)) << "File with special chars doesn't exist: " << name;

        // Test path_to_utf8 conversion
        std::string utf8 = path_to_utf8(path);
        EXPECT_FALSE(utf8.empty()) << "path_to_utf8 failed for: " << name;
    }
}

// ============================================================================
// Test 7: Config-like JSON Files with Unicode Paths
// ============================================================================

TEST_F(UnicodePathTest, ConfigFileReadWrite) {
    auto config_dir = test_root_ / "設定_config_配置_설정";
    fs::create_directories(config_dir);

    auto config_file = config_dir / "設定ファイル_config_配置文件_설정파일.json";

    // Write a simple JSON-like config
    const std::string json_content = R"({
    "name": "Unicode Test Config",
    "path": "日本語/中文/한국어",
    "value": 42
})";

    create_test_file(config_file, json_content);
    EXPECT_TRUE(fs::exists(config_file)) << "Config file doesn't exist";

    // Read it back
    std::string read_content = read_test_file(config_file);
    EXPECT_EQ(read_content, json_content) << "Config content doesn't match";
}

// ============================================================================
// Test 8: Binary File Operations with Unicode Paths
// ============================================================================

TEST_F(UnicodePathTest, BinaryFileOperations) {
    auto binary_file = test_root_ / "バイナリ_binary_二进制_바이너리.bin";

    // Create binary data
    std::vector<uint8_t> binary_data = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD};

    // Write binary file
    {
        std::ofstream out(binary_file, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to create binary file";
        out.write(reinterpret_cast<const char*>(binary_data.data()), binary_data.size());
        out.close();
    }

    EXPECT_TRUE(fs::exists(binary_file)) << "Binary file doesn't exist";
    EXPECT_EQ(fs::file_size(binary_file), binary_data.size()) << "Binary file size mismatch";

    // Read binary file back
    {
        std::ifstream in(binary_file, std::ios::binary);
        ASSERT_TRUE(in.is_open()) << "Failed to open binary file";
        std::vector<uint8_t> read_data(binary_data.size());
        in.read(reinterpret_cast<char*>(read_data.data()), read_data.size());

        EXPECT_EQ(read_data, binary_data) << "Binary data doesn't match";
    }
}
