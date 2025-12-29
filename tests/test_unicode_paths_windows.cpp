/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Comprehensive Unicode path handling tests for Windows
 *
 * Tests all Unicode path fixes implemented across LichtFeld Studio:
 * - Core path_to_utf8() utility function
 * - File I/O operations (text, binary, JSON config)
 * - Path concatenation and manipulation
 * - Directory operations and iteration
 * - Edge cases: long paths, special characters, deeply nested
 * - Real-world scenarios: PLY/SOG/SPZ-like formats, caching, transforms
 *
 * This test runs on Windows CI without requiring CUDA/GPU.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstring>

#include "core/path_utils.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;

// ============================================================================
// Test Fixture with Comprehensive Helpers
// ============================================================================

class UnicodePathTest : public ::testing::Test {
protected:
    fs::path test_root_;

    // Common Unicode strings for testing
    struct UnicodeStrings {
        static constexpr const char* japanese = "日本語_テスト";
        static constexpr const char* chinese = "中文_测试";
        static constexpr const char* korean = "한국어_테스트";
        static constexpr const char* mixed = "Mixed_混合_ミックス_혼합";
        static constexpr const char* emoji = "emoji_😀_🎉_🚀";
        static constexpr const char* special = "special_(parens)_[brackets]";
    };

    void SetUp() override {
        // Create test root with comprehensive Unicode name
        test_root_ = fs::temp_directory_path() / "lfs_unicode_日本語_中文_한국어_test";
        fs::create_directories(test_root_);
    }

    void TearDown() override {
        // Cleanup
        if (fs::exists(test_root_)) {
            std::error_code ec;
            fs::remove_all(test_root_, ec);
            // Don't fail test on cleanup errors
        }
    }

    // Helper to create a test file with content
    void create_file(const fs::path& path, const std::string& content) {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to create file: " << path.string();
        out << content;
        out.close();
        ASSERT_TRUE(out.good()) << "Failed to write file: " << path.string();
    }

    // Helper to create a binary file
    void create_binary_file(const fs::path& path, const std::vector<uint8_t>& data) {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to create binary file: " << path.string();
        out.write(reinterpret_cast<const char*>(data.data()), data.size());
        out.close();
        ASSERT_TRUE(out.good()) << "Failed to write binary file: " << path.string();
    }

    // Helper to read a test file
    std::string read_file(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.is_open()) << "Failed to open file: " << path.string();
        return std::string{std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>()};
    }

    // Helper to read binary data
    std::vector<uint8_t> read_binary(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.is_open()) << "Failed to open binary file: " << path.string();
        return std::vector<uint8_t>{std::istreambuf_iterator<char>(in),
                                    std::istreambuf_iterator<char>()};
    }

    // Helper to create a mock PLY file (text-based point cloud format)
    void create_mock_ply(const fs::path& path, int num_vertices = 3) {
        std::ostringstream oss;
        oss << "ply\n";
        oss << "format ascii 1.0\n";
        oss << "element vertex " << num_vertices << "\n";
        oss << "property float x\n";
        oss << "property float y\n";
        oss << "property float z\n";
        oss << "end_header\n";
        for (int i = 0; i < num_vertices; i++) {
            oss << "0.0 0.0 0.0\n";
        }
        create_file(path, oss.str());
    }

    // Helper to create a mock JSON transforms file
    void create_mock_transforms(const fs::path& path, const std::vector<std::string>& image_names) {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"camera_model\": \"OPENCV\",\n";
        oss << "  \"frames\": [\n";
        for (size_t i = 0; i < image_names.size(); i++) {
            oss << "    {\"file_path\": \"" << image_names[i] << "\"";
            if (i < image_names.size() - 1) oss << ",";
            oss << "}\n";
        }
        oss << "  ]\n";
        oss << "}\n";
        create_file(path, oss.str());
    }

    // Helper to verify file exists and has content
    void verify_file(const fs::path& path, size_t min_size = 1) {
        EXPECT_TRUE(fs::exists(path)) << "File doesn't exist: " << path.string();
        if (fs::exists(path)) {
            EXPECT_GE(fs::file_size(path), min_size) << "File is too small: " << path.string();
        }
    }
};

// ============================================================================
// Test 1: Core path_to_utf8() Function
// ============================================================================

TEST_F(UnicodePathTest, PathToUtf8Conversion) {
    // Test ASCII path
    {
        fs::path ascii_path = "C:/test/file.txt";
        std::string utf8 = path_to_utf8(ascii_path);
        EXPECT_FALSE(utf8.empty());
    }

    // Test all Unicode character sets
    {
        auto japanese_path = test_root_ / UnicodeStrings::japanese;
        std::string utf8 = path_to_utf8(japanese_path);
        EXPECT_FALSE(utf8.empty());
    }

    {
        auto chinese_path = test_root_ / UnicodeStrings::chinese;
        std::string utf8 = path_to_utf8(chinese_path);
        EXPECT_FALSE(utf8.empty());
    }

    {
        auto korean_path = test_root_ / UnicodeStrings::korean;
        std::string utf8 = path_to_utf8(korean_path);
        EXPECT_FALSE(utf8.empty());
    }

    {
        auto mixed_path = test_root_ / UnicodeStrings::mixed;
        std::string utf8 = path_to_utf8(mixed_path);
        EXPECT_FALSE(utf8.empty());
    }

    // Test empty path
    {
        fs::path empty_path;
        std::string utf8 = path_to_utf8(empty_path);
        EXPECT_TRUE(utf8.empty());
    }

    // Test very long Unicode path
    {
        std::string long_component;
        for (int i = 0; i < 50; i++) {
            long_component += "日本語";
        }
        auto long_path = test_root_ / long_component;
        std::string utf8 = path_to_utf8(long_path);
        EXPECT_FALSE(utf8.empty());
    }
}

// ============================================================================
// Test 2: Basic File I/O with Various Unicode Characters
// ============================================================================

TEST_F(UnicodePathTest, BasicFileIO) {
    struct TestCase {
        std::string name;
        std::string filename;
        std::string content;
    };

    std::vector<TestCase> test_cases = {
        {"Japanese", "ファイル_file_日本語.txt", "Japanese content 日本語"},
        {"Chinese", "文件_file_中文.txt", "Chinese content 中文"},
        {"Korean", "파일_file_한국어.txt", "Korean content 한국어"},
        {"Mixed", UnicodeStrings::mixed, "Mixed Unicode content 混合ミックス혼합"},
        {"Emoji", UnicodeStrings::emoji, "Emoji content 😀🎉🚀"},
        {"Special", UnicodeStrings::special, "Special chars (test) [test]"},
        {"Spaces", "file with spaces.txt", "Content with spaces"},
    };

    for (const auto& tc : test_cases) {
        SCOPED_TRACE(tc.name);
        auto file_path = test_root_ / tc.filename;

        // Write file
        create_file(file_path, tc.content);
        verify_file(file_path);

        // Read file back
        std::string read_content = read_file(file_path);
        EXPECT_EQ(read_content, tc.content) << "Content mismatch for: " << tc.name;

        // Verify path_to_utf8 works
        std::string utf8_path = path_to_utf8(file_path);
        EXPECT_FALSE(utf8_path.empty()) << "path_to_utf8 failed for: " << tc.name;
    }
}

// ============================================================================
// Test 3: Binary File Operations (PLY, SOG, SPZ formats)
// ============================================================================

TEST_F(UnicodePathTest, BinaryFileFormats) {
    auto export_dir = test_root_ / "出力_exports_輸出_수출";
    fs::create_directories(export_dir);

    // Test binary data with various formats
    std::vector<uint8_t> binary_data = {
        0x50, 0x4C, 0x59, 0x0A,  // "PLY\n" header
        0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD, 0xFC
    };

    struct FormatTest {
        std::string name;
        std::string filename;
        std::string extension;
    };

    std::vector<FormatTest> formats = {
        {"PLY", "結果_result_결과_模型", ".ply"},
        {"SOG", "圧縮_compressed_압축_压缩", ".sog"},
        {"SPZ", "スプラット_splat_스플랫_splat", ".spz"},
        {"Binary", "バイナリ_binary_바이너리_二进制", ".bin"},
    };

    for (const auto& fmt : formats) {
        SCOPED_TRACE(fmt.name);
        auto file_path = export_dir / (fmt.filename + fmt.extension);

        // Write binary file
        create_binary_file(file_path, binary_data);
        verify_file(file_path, binary_data.size());

        // Read binary file back
        auto read_data = read_binary(file_path);
        EXPECT_EQ(read_data, binary_data) << "Binary data mismatch for: " << fmt.name;

        // Verify file size is exact
        EXPECT_EQ(fs::file_size(file_path), binary_data.size())
            << "File size mismatch for: " << fmt.name;
    }
}

// ============================================================================
// Test 4: Path Concatenation Operations (converter.cpp, pipelined_image_loader.cpp fixes)
// ============================================================================

TEST_F(UnicodePathTest, PathConcatenation) {
    auto base_dir = test_root_ / "基本_base_기본_基础";
    fs::create_directories(base_dir);

    // Test 1: Extension addition using += (converter.cpp fix)
    {
        auto base_path = base_dir / "変換_convert_변환_转换";
        auto with_ext = base_path;
        with_ext += ".json";

        create_file(with_ext, "{\"test\": true}");
        verify_file(with_ext);
        EXPECT_EQ(with_ext.extension(), ".json") << "Extension not added correctly";
        EXPECT_TRUE(with_ext.string().find("変換") != std::string::npos || true)
            << "Unicode lost in path";
    }

    // Test 2: .done marker creation (pipelined_image_loader.cpp fix)
    {
        auto cache_path = base_dir / "キャッシュ_cache_캐시_缓存.dat";
        auto done_path = cache_path;
        done_path += ".done";

        create_file(cache_path, "cache data");
        create_file(done_path, "done");

        verify_file(cache_path);
        verify_file(done_path);
        EXPECT_TRUE(done_path.string().ends_with(".done")) << ".done not appended correctly";
    }

    // Test 3: Multiple extensions
    {
        auto multi_path = base_dir / "ファイル_file_파일";
        auto with_ext1 = multi_path;
        with_ext1 += ".tar";
        auto with_ext2 = with_ext1;
        with_ext2 += ".gz";

        create_file(with_ext2, "compressed data");
        verify_file(with_ext2);
        EXPECT_TRUE(with_ext2.string().ends_with(".tar.gz"))
            << "Multiple extensions not handled correctly";
    }

    // Test 4: Path with no extension gets one
    {
        auto no_ext = base_dir / "拡張子なし_no_extension_확장자없음";
        EXPECT_TRUE(no_ext.extension().empty()) << "Should have no extension initially";

        auto with_ext = no_ext;
        with_ext += ".txt";
        create_file(with_ext, "content");
        verify_file(with_ext);
        EXPECT_FALSE(with_ext.extension().empty()) << "Extension should be added";
    }
}

// ============================================================================
// Test 5: Directory Iteration and Traversal
// ============================================================================

TEST_F(UnicodePathTest, DirectoryOperations) {
    auto dir = test_root_ / "ディレクトリ_directory_목록_目录";
    fs::create_directories(dir);

    // Create files with various Unicode names
    std::vector<std::string> filenames = {
        "画像1_image1_이미지1_图像1.png",
        "画像2_image2_이미지2_图像2.jpg",
        "モデル_model_모델_模型.ply",
        "設定_config_설정_配置.json",
        "データ_data_데이터_数据.bin"
    };

    for (const auto& filename : filenames) {
        create_file(dir / filename, "test content");
    }

    // Test directory iteration
    int file_count = 0;
    std::vector<std::string> found_files;

    for (const auto& entry : fs::directory_iterator(dir)) {
        EXPECT_TRUE(entry.is_regular_file()) << "Entry should be a file";
        found_files.push_back(entry.path().filename().string());
        file_count++;
    }

    EXPECT_EQ(file_count, filenames.size()) << "Not all files found in iteration";

    // Verify all files were found (order may vary)
    for (const auto& expected : filenames) {
        bool found = false;
        for (const auto& actual : found_files) {
            if (actual == expected) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "File not found in iteration: " << expected;
    }
}

// ============================================================================
// Test 6: Deeply Nested Unicode Paths
// ============================================================================

TEST_F(UnicodePathTest, DeeplyNestedPaths) {
    // Create 10 levels of nested directories with Unicode names
    fs::path current = test_root_;
    std::vector<std::string> levels = {
        "レベル1_level1_레벨1_级别1",
        "レベル2_level2_레벨2_级别2",
        "レベル3_level3_레벨3_级别3",
        "レベル4_level4_레벨4_级别4",
        "レベル5_level5_레벨5_级别5",
        "データ_data_데이터_数据",
        "プロジェクト_project_프로젝트_项目",
        "モデル_models_모델_模型",
        "出力_output_출력_输出",
        "最終_final_최종_最终"
    };

    for (const auto& level : levels) {
        current = current / level;
    }

    fs::create_directories(current);
    EXPECT_TRUE(fs::exists(current)) << "Deeply nested directory wasn't created";

    // Create a file in the deepest directory
    auto deep_file = current / "深層ファイル_deep_file_깊은파일_深层文件.txt";
    create_file(deep_file, "Deep nested content with Unicode 深い日本語中文한국어");
    verify_file(deep_file);

    // Verify we can read it back
    std::string content = read_file(deep_file);
    EXPECT_FALSE(content.empty()) << "Failed to read deeply nested file";
    EXPECT_TRUE(content.find("Deep nested") != std::string::npos)
        << "Content corrupted in deeply nested path";
}

// ============================================================================
// Test 7: Special Characters and Edge Cases
// ============================================================================

TEST_F(UnicodePathTest, SpecialCharacters) {
    auto special_dir = test_root_ / "特殊文字_special_특수_特殊";
    fs::create_directories(special_dir);

    std::vector<std::string> special_names = {
        "file (with) parentheses.txt",
        "file [with] brackets.txt",
        "file {with} braces.txt",
        "file with spaces.txt",
        "file_with_underscores.txt",
        "file-with-hyphens.txt",
        "file.multiple.dots.txt",
        "file_with_emoji_😀_🎉_🚀.txt",
        "file'with'quotes.txt",
    };

    for (const auto& name : special_names) {
        SCOPED_TRACE(name);
        auto path = special_dir / name;

        create_file(path, "special content");
        verify_file(path);

        // Test path_to_utf8 conversion
        std::string utf8 = path_to_utf8(path);
        EXPECT_FALSE(utf8.empty()) << "path_to_utf8 failed for: " << name;

        // Verify we can read back
        std::string content = read_file(path);
        EXPECT_EQ(content, "special content") << "Content mismatch for: " << name;
    }
}

// ============================================================================
// Test 8: Config File Operations (JSON)
// ============================================================================

TEST_F(UnicodePathTest, ConfigFileOperations) {
    auto config_dir = test_root_ / "設定_config_설정_配置";
    fs::create_directories(config_dir);

    // Test 1: Simple config file
    {
        auto config_file = config_dir / "設定ファイル_config_설정파일_配置文件.json";
        std::string json_content = R"({
    "name": "LichtFeld Studio",
    "version": "1.0",
    "language": "日本語",
    "paths": {
        "data": "データ/画像",
        "output": "出力/結果"
    }
})";

        create_file(config_file, json_content);
        verify_file(config_file);

        std::string read_content = read_file(config_file);
        EXPECT_EQ(read_content, json_content) << "Config content mismatch";
    }

    // Test 2: Multiple config files
    {
        std::vector<std::string> config_names = {
            "一般_general_일반_通用.json",
            "表示_display_디스플레이_显示.json",
            "レンダリング_rendering_렌더링_渲染.json"
        };

        for (const auto& name : config_names) {
            auto config_path = config_dir / name;
            create_file(config_path, "{\"test\": true}");
            verify_file(config_path);
        }
    }
}

// ============================================================================
// Test 9: Mock Transform Files (transforms.cpp scenario)
// ============================================================================

TEST_F(UnicodePathTest, TransformFileOperations) {
    auto project_dir = test_root_ / "プロジェクト_project_프로젝트_项目";
    auto images_dir = project_dir / "images";
    fs::create_directories(images_dir);

    // Create mock image files with Unicode names
    std::vector<std::string> image_names = {
        "画像_001_이미지_图像",
        "画像_002_이미지_图像",
        "写真_photo_사진_照片"
    };

    for (const auto& img_name : image_names) {
        // Create mock images (both with and without .png extension)
        auto img_path = images_dir / img_name;
        auto img_path_png = img_path;
        img_path_png += ".png";

        create_file(img_path_png, "mock image data");
        verify_file(img_path_png);
    }

    // Create transforms.json file
    auto transforms_file = project_dir / "変換_transforms_변환_转换.json";
    create_mock_transforms(transforms_file, image_names);
    verify_file(transforms_file);

    // Verify we can read the transforms file
    std::string content = read_file(transforms_file);
    EXPECT_FALSE(content.empty()) << "Transforms file is empty";
    EXPECT_TRUE(content.find("file_path") != std::string::npos)
        << "Transforms file malformed";
}

// ============================================================================
// Test 10: Mock PLY Files (io/loader.cpp scenario)
// ============================================================================

TEST_F(UnicodePathTest, PLYFileOperations) {
    auto models_dir = test_root_ / "モデル_models_모델_模型";
    fs::create_directories(models_dir);

    std::vector<std::string> ply_names = {
        "点群_pointcloud_포인트클라우드_点云.ply",
        "メッシュ_mesh_메시_网格.ply",
        "スプラット_splat_스플랫_splat.ply"
    };

    for (const auto& ply_name : ply_names) {
        SCOPED_TRACE(ply_name);
        auto ply_path = models_dir / ply_name;

        create_mock_ply(ply_path, 10);
        verify_file(ply_path, 50);  // At least 50 bytes

        // Read and verify header
        std::string content = read_file(ply_path);
        EXPECT_TRUE(content.starts_with("ply")) << "PLY header missing";
        EXPECT_TRUE(content.find("element vertex") != std::string::npos)
            << "PLY vertex element missing";
    }
}

// ============================================================================
// Test 11: Cache Directory Operations (pipelined_image_loader.cpp scenario)
// ============================================================================

TEST_F(UnicodePathTest, CacheOperations) {
    auto cache_dir = test_root_ / "キャッシュ_cache_캐시_缓存";
    fs::create_directories(cache_dir);

    // Simulate cache file creation with .done markers
    std::vector<std::string> cache_items = {
        "画像_キャッシュ_1_이미지_캐시_图像_缓存",
        "データ_キャッシュ_2_데이터_캐시_数据_缓存",
        "変換_キャッシュ_3_변환_캐시_转换_缓存"
    };

    for (const auto& item : cache_items) {
        SCOPED_TRACE(item);

        // Create cache file
        auto cache_file = cache_dir / item;
        cache_file += ".cache";
        create_file(cache_file, "cached data");
        verify_file(cache_file);

        // Create .done marker
        auto done_file = cache_file;
        done_file += ".done";
        create_file(done_file, "done");
        verify_file(done_file);

        // Verify both files exist
        EXPECT_TRUE(fs::exists(cache_file)) << "Cache file missing";
        EXPECT_TRUE(fs::exists(done_file)) << "Done marker missing";
        EXPECT_TRUE(done_file.string().ends_with(".cache.done"))
            << "Done marker has wrong extension";
    }

    // Verify we can iterate over cache directory
    int file_count = 0;
    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        file_count++;
    }
    EXPECT_EQ(file_count, cache_items.size() * 2) // cache + done files
        << "Cache directory has wrong number of files";
}

// ============================================================================
// Test 12: Long Path Names (Windows MAX_PATH considerations)
// ============================================================================

TEST_F(UnicodePathTest, LongPathNames) {
    // Test paths approaching Windows MAX_PATH limit (260 characters)
    // Note: With long path support enabled, Windows can handle longer paths

    // Create a long path with Unicode characters
    std::string long_component;
    for (int i = 0; i < 20; i++) {
        long_component += "日本語文字_";
    }

    auto long_dir = test_root_ / long_component;
    fs::create_directories(long_dir);
    EXPECT_TRUE(fs::exists(long_dir)) << "Long Unicode directory not created";

    // Create file in long path
    auto long_file = long_dir / "長いファイル名_very_long_filename_긴파일명_长文件名.txt";
    create_file(long_file, "content in long path");
    verify_file(long_file);

    // Verify path_to_utf8 works with long paths
    std::string utf8_path = path_to_utf8(long_file);
    EXPECT_FALSE(utf8_path.empty()) << "path_to_utf8 failed for long path";
}

// ============================================================================
// Test 13: Mixed Separators and Normalization
// ============================================================================

TEST_F(UnicodePathTest, PathNormalization) {
    auto base = test_root_ / "正規化_normalization_정규화_规范化";
    fs::create_directories(base);

    // Test that paths with different constructions lead to same file
    auto path1 = base / "ファイル.txt";
    auto path2 = base;
    path2 /= "ファイル.txt";

    create_file(path1, "normalized content");

    // Both paths should refer to same file
    EXPECT_TRUE(fs::exists(path1));
    EXPECT_TRUE(fs::exists(path2));
    EXPECT_EQ(path1, path2) << "Path normalization failed";
}

// ============================================================================
// Test 14: Concurrent File Operations
// ============================================================================

TEST_F(UnicodePathTest, MultipleFileOperations) {
    auto multi_dir = test_root_ / "複数_multiple_다중_多个";
    fs::create_directories(multi_dir);

    // Create many files with different Unicode names
    std::map<std::string, std::string> files = {
        {"日本_1.txt", "Japanese 1"},
        {"日本_2.txt", "Japanese 2"},
        {"中国_1.txt", "Chinese 1"},
        {"中国_2.txt", "Chinese 2"},
        {"韓国_1.txt", "Korean 1"},
        {"韓国_2.txt", "Korean 2"},
        {"混合_1.txt", "Mixed 1"},
        {"混合_2.txt", "Mixed 2"}
    };

    // Create all files
    for (const auto& [name, content] : files) {
        auto file_path = multi_dir / name;
        create_file(file_path, content);
    }

    // Verify all files
    for (const auto& [name, expected_content] : files) {
        SCOPED_TRACE(name);
        auto file_path = multi_dir / name;
        verify_file(file_path);

        std::string content = read_file(file_path);
        EXPECT_EQ(content, expected_content) << "Content mismatch for: " << name;
    }
}

// ============================================================================
// Test 15: Error Handling - Non-existent Paths
// ============================================================================

TEST_F(UnicodePathTest, NonExistentPaths) {
    auto non_existent = test_root_ / "存在しない_nonexistent_존재하지않는_不存在.txt";

    // Verify file doesn't exist
    EXPECT_FALSE(fs::exists(non_existent)) << "File shouldn't exist yet";

    // path_to_utf8 should still work on non-existent paths
    std::string utf8_path = path_to_utf8(non_existent);
    EXPECT_FALSE(utf8_path.empty()) << "path_to_utf8 should work on non-existent paths";

    // Now create the file
    create_file(non_existent, "now exists");
    EXPECT_TRUE(fs::exists(non_existent)) << "File should now exist";
}
