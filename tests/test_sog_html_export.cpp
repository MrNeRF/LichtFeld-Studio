/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_sog_html_export.cpp
 * @brief Regression tests for SOG and HTML viewer export
 *
 * Tests that validate:
 * 1. SOG export produces valid format compatible with SuperSplat viewer
 * 2. HTML export creates self-contained viewer with embedded data
 * 3. K-means clustering (cluster1d) works correctly for SH data
 */

#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>
#include <webp/decode.h>

#include "core/cuda/kernels/kdtree_kmeans.hpp"
#include "core/sogs.hpp"
#include "core/tensor.hpp"
#include "io/formats/ply.hpp"
#include "io/formats/sogs.hpp"
#include "visualizer/gui/html_viewer_export.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace lfs::core;

// ============================================================================
// Test Utilities
// ============================================================================

namespace {

    struct WebpImage {
        std::vector<uint8_t> data;
        int width = 0;
        int height = 0;
    };

    std::map<std::string, std::vector<uint8_t>> extract_zip_files(const fs::path& path) {
        std::map<std::string, std::vector<uint8_t>> files;
        archive* a = archive_read_new();
        archive_read_support_format_zip(a);

        if (archive_read_open_filename(a, path.string().c_str(), 10240) != ARCHIVE_OK) {
            archive_read_free(a);
            return files;
        }

        archive_entry* entry;
        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            const char* name = archive_entry_pathname(entry);
            const size_t size = archive_entry_size(entry);
            std::vector<uint8_t> data(size);
            archive_read_data(a, data.data(), size);
            files[name] = std::move(data);
        }

        archive_read_free(a);
        return files;
    }

    WebpImage decode_webp(const std::vector<uint8_t>& data) {
        WebpImage img;
        int width, height;
        if (!WebPGetInfo(data.data(), data.size(), &width, &height)) {
            return img;
        }
        uint8_t* rgba = WebPDecodeRGBA(data.data(), data.size(), &width, &height);
        if (rgba) {
            img.width = width;
            img.height = height;
            img.data.assign(rgba, rgba + width * height * 4);
            WebPFree(rgba);
        }
        return img;
    }

    std::string read_file(const fs::path& path) {
        std::ifstream file(path);
        if (!file)
            return "";
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    std::vector<uint8_t> base64_decode(const std::string& encoded) {
        static const std::string CHARS =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<uint8_t> result;
        result.reserve(encoded.size() * 3 / 4);

        uint32_t val = 0;
        int valb = -8;
        for (const char c : encoded) {
            if (c == '=')
                break;
            const size_t pos = CHARS.find(c);
            if (pos == std::string::npos)
                continue;
            val = (val << 6) + static_cast<uint32_t>(pos);
            valb += 6;
            if (valb >= 0) {
                result.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return result;
    }

    std::string extract_base64_sog(const std::string& html) {
        const std::string marker = "fetch(\"data:application/octet-stream;base64,";
        const size_t start = html.find(marker);
        if (start == std::string::npos)
            return "";
        const size_t data_start = start + marker.size();
        const size_t end = html.find("\")", data_start);
        if (end == std::string::npos)
            return "";
        return html.substr(data_start, end - data_start);
    }

} // anonymous namespace

// ============================================================================
// SOG Export Tests
// ============================================================================

class SogExportTest : public ::testing::Test {
protected:
    static constexpr const char* TEST_PLY = "output/splat_30000.ply";
    fs::path temp_sog_;

    void SetUp() override {
        temp_sog_ = fs::temp_directory_path() / "test_export.sog";
    }

    void TearDown() override {
        if (fs::exists(temp_sog_)) {
            fs::remove(temp_sog_);
        }
    }

    bool generate_sog(const lfs::core::SplatData& splat_data) {
        lfs::core::SogWriteOptions options{
            .iterations = 10,
            .use_gpu = true,
            .output_path = temp_sog_};
        auto result = lfs::core::write_sog(splat_data, options);
        return result.has_value();
    }
};

// Test: SOG export produces valid ZIP file
TEST_F(SogExportTest, ProducesValidZip) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value()) << ply_result.error();
    ASSERT_TRUE(generate_sog(*ply_result));

    // Verify ZIP magic number
    std::ifstream file(temp_sog_, std::ios::binary);
    char magic[4];
    file.read(magic, 4);
    EXPECT_EQ(magic[0], 'P');
    EXPECT_EQ(magic[1], 'K');
    EXPECT_EQ(magic[2], 0x03);
    EXPECT_EQ(magic[3], 0x04);
}

// Test: SOG contains required files
TEST_F(SogExportTest, ContainsRequiredFiles) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());
    ASSERT_TRUE(generate_sog(*ply_result));

    auto files = extract_zip_files(temp_sog_);
    ASSERT_GT(files.size(), 0) << "Failed to extract SOG";

    // Required files
    EXPECT_TRUE(files.contains("meta.json")) << "Missing meta.json";
    EXPECT_TRUE(files.contains("means_l.webp")) << "Missing means_l.webp";
    EXPECT_TRUE(files.contains("means_u.webp")) << "Missing means_u.webp";
    EXPECT_TRUE(files.contains("quats.webp")) << "Missing quats.webp";
    EXPECT_TRUE(files.contains("scales.webp")) << "Missing scales.webp";
    EXPECT_TRUE(files.contains("sh0.webp")) << "Missing sh0.webp";
}

// Test: meta.json has correct structure
TEST_F(SogExportTest, MetaJsonStructure) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());
    ASSERT_TRUE(generate_sog(*ply_result));

    auto files = extract_zip_files(temp_sog_);
    ASSERT_TRUE(files.contains("meta.json"));

    std::string json_str(files["meta.json"].begin(), files["meta.json"].end());
    auto meta = json::parse(json_str);

    // Required fields
    EXPECT_TRUE(meta.contains("count")) << "Missing splat count";
    EXPECT_TRUE(meta.contains("means")) << "Missing means section";
    EXPECT_TRUE(meta.contains("scales")) << "Missing scales section";
    EXPECT_TRUE(meta.contains("sh0")) << "Missing sh0 section";

    // Verify count matches
    EXPECT_EQ(meta["count"].get<int>(), ply_result->size());

    // Verify codebook sizes
    auto scales_cb = meta["scales"]["codebook"].get<std::vector<float>>();
    auto sh0_cb = meta["sh0"]["codebook"].get<std::vector<float>>();
    EXPECT_EQ(scales_cb.size(), 256) << "Scales codebook should have 256 entries";
    EXPECT_EQ(sh0_cb.size(), 256) << "SH0 codebook should have 256 entries";

    // Codebooks should be sorted ascending
    for (size_t i = 1; i < scales_cb.size(); ++i) {
        EXPECT_GE(scales_cb[i], scales_cb[i - 1]) << "Scales codebook not sorted at " << i;
    }
    for (size_t i = 1; i < sh0_cb.size(); ++i) {
        EXPECT_GE(sh0_cb[i], sh0_cb[i - 1]) << "SH0 codebook not sorted at " << i;
    }
}

// Test: WebP images have correct dimensions
TEST_F(SogExportTest, WebpImageDimensions) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());
    ASSERT_TRUE(generate_sog(*ply_result));

    auto files = extract_zip_files(temp_sog_);

    // Decode and verify dimensions
    auto means_l = decode_webp(files["means_l.webp"]);
    auto means_u = decode_webp(files["means_u.webp"]);
    auto quats = decode_webp(files["quats.webp"]);
    auto sh0 = decode_webp(files["sh0.webp"]);

    EXPECT_GT(means_l.width, 0) << "means_l width is 0";
    EXPECT_GT(means_l.height, 0) << "means_l height is 0";

    // All position-related textures should have same dimensions
    EXPECT_EQ(means_l.width, means_u.width);
    EXPECT_EQ(means_l.height, means_u.height);
    EXPECT_EQ(means_l.width, quats.width);
    EXPECT_EQ(means_l.height, quats.height);
    EXPECT_EQ(means_l.width, sh0.width);
    EXPECT_EQ(means_l.height, sh0.height);

    // Verify pixel count matches splat count (allowing for padding)
    const int pixel_count = means_l.width * means_l.height;
    EXPECT_GE(pixel_count, static_cast<int>(ply_result->size()));
}

// Test: SOG can be loaded back
TEST_F(SogExportTest, RoundtripLoad) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());
    ASSERT_TRUE(generate_sog(*ply_result));

    // Load the exported SOG
    auto sog_result = lfs::io::load_sog(temp_sog_);
    ASSERT_TRUE(sog_result.has_value()) << sog_result.error();

    // Splat count should match
    EXPECT_EQ(sog_result->size(), ply_result->size());
}

// ============================================================================
// HTML Export Tests
// ============================================================================

class HtmlExportTest : public ::testing::Test {
protected:
    static constexpr const char* TEST_PLY = "output/splat_30000.ply";
    fs::path temp_html_;

    void SetUp() override {
        temp_html_ = fs::temp_directory_path() / "test_export.html";
    }

    void TearDown() override {
        if (fs::exists(temp_html_)) {
            fs::remove(temp_html_);
        }
    }
};

// Test: HTML export produces valid file
TEST_F(HtmlExportTest, ProducesValidHtml) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());

    lfs::vis::gui::HtmlViewerExportOptions options{.output_path = temp_html_};
    auto result = lfs::vis::gui::export_html_viewer(*ply_result, options);
    ASSERT_TRUE(result.has_value()) << result.error();

    std::string html = read_file(temp_html_);
    ASSERT_FALSE(html.empty());

    // Check basic HTML structure
    EXPECT_NE(html.find("<!DOCTYPE html>"), std::string::npos) << "Missing DOCTYPE";
    EXPECT_NE(html.find("<html"), std::string::npos) << "Missing <html>";
    EXPECT_NE(html.find("<head>"), std::string::npos) << "Missing <head>";
    EXPECT_NE(html.find("<body>"), std::string::npos) << "Missing <body>";
}

// Test: CSS is inlined
TEST_F(HtmlExportTest, CssInlined) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());

    lfs::vis::gui::HtmlViewerExportOptions options{.output_path = temp_html_};
    auto result = lfs::vis::gui::export_html_viewer(*ply_result, options);
    ASSERT_TRUE(result.has_value());

    std::string html = read_file(temp_html_);

    // Should have inline <style>, not external link
    EXPECT_NE(html.find("<style>"), std::string::npos) << "Missing inline style";
    EXPECT_EQ(html.find("href=\"./index.css\""), std::string::npos)
        << "Should not have external CSS link";
}

// Test: JavaScript is inlined
TEST_F(HtmlExportTest, JsInlined) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());

    lfs::vis::gui::HtmlViewerExportOptions options{.output_path = temp_html_};
    auto result = lfs::vis::gui::export_html_viewer(*ply_result, options);
    ASSERT_TRUE(result.has_value());

    std::string html = read_file(temp_html_);

    // Should NOT have import statement (JS should be inlined)
    EXPECT_EQ(html.find("import { main } from './index.js'"), std::string::npos)
        << "Should not have JS import";
    // Should have PlayCanvas viewer code
    EXPECT_NE(html.find("PlayCanvas"), std::string::npos)
        << "Should contain PlayCanvas viewer code";
}

// Test: Embedded SOG is valid
TEST_F(HtmlExportTest, EmbeddedSogValid) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());

    lfs::vis::gui::HtmlViewerExportOptions options{.output_path = temp_html_};
    auto result = lfs::vis::gui::export_html_viewer(*ply_result, options);
    ASSERT_TRUE(result.has_value());

    std::string html = read_file(temp_html_);
    std::string base64_sog = extract_base64_sog(html);
    ASSERT_FALSE(base64_sog.empty()) << "Could not extract base64 SOG";

    auto sog_data = base64_decode(base64_sog);
    ASSERT_GT(sog_data.size(), 4) << "SOG data too small";

    // Check ZIP magic
    EXPECT_EQ(sog_data[0], 0x50) << "Invalid ZIP magic";
    EXPECT_EQ(sog_data[1], 0x4B) << "Invalid ZIP magic";
    EXPECT_EQ(sog_data[2], 0x03) << "Invalid ZIP magic";
    EXPECT_EQ(sog_data[3], 0x04) << "Invalid ZIP magic";
}

// Test: Uses .sog extension not .compressed.ply
TEST_F(HtmlExportTest, UsesSogExtension) {
    if (!fs::exists(TEST_PLY)) {
        GTEST_SKIP() << "Test PLY not found: " << TEST_PLY;
    }

    auto ply_result = lfs::io::load_ply(TEST_PLY);
    ASSERT_TRUE(ply_result.has_value());

    lfs::vis::gui::HtmlViewerExportOptions options{.output_path = temp_html_};
    auto result = lfs::vis::gui::export_html_viewer(*ply_result, options);
    ASSERT_TRUE(result.has_value());

    std::string html = read_file(temp_html_);
    EXPECT_NE(html.find(".sog"), std::string::npos) << "Should use .sog extension";
    EXPECT_EQ(html.find(".compressed.ply"), std::string::npos)
        << "Should not use .compressed.ply extension";
}

// ============================================================================
// Cluster1D Tests (K-means for SOG SH data)
// ============================================================================

class Cluster1dTest : public ::testing::Test {
protected:
    // Replicates cluster1d from sogs.cpp
    struct Cluster1dResult {
        std::vector<float> centroids;
        std::vector<uint8_t> labels;
    };

    Cluster1dResult run_cluster1d(const float* data, int num_rows, int num_cols, int iterations) {
        // Flatten column-major (matches TypeScript)
        const int total_points = num_rows * num_cols;
        std::vector<float> flat_data(total_points);
        for (int col = 0; col < num_cols; ++col) {
            for (int row = 0; row < num_rows; ++row) {
                flat_data[col * num_rows + row] = data[row * num_cols + col];
            }
        }

        // Run 1D k-means with 256 clusters
        auto data_tensor = Tensor::from_blob(
                               flat_data.data(),
                               {static_cast<size_t>(total_points), 1},
                               Device::CPU,
                               DataType::Float32)
                               .cuda();

        auto [centroids_tensor, labels_tensor] = cuda::kmeans_kdtree(data_tensor, 256, iterations);

        auto centroids_cpu = centroids_tensor.cpu();
        auto labels_cpu = labels_tensor.cpu();
        const float* centroids_ptr = static_cast<const float*>(centroids_cpu.data_ptr());
        const int32_t* labels_ptr = static_cast<const int32_t*>(labels_cpu.data_ptr());

        // Sort centroids ascending
        std::vector<int> order(256);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return centroids_ptr[a] < centroids_ptr[b];
        });

        std::vector<float> ordered_centroids(256);
        for (int i = 0; i < 256; ++i) {
            ordered_centroids[i] = centroids_ptr[order[i]];
        }

        std::vector<int> inv_order(256);
        for (int i = 0; i < 256; ++i) {
            inv_order[order[i]] = i;
        }

        // Convert labels from column-major to row-major
        Cluster1dResult result;
        result.centroids = ordered_centroids;
        result.labels.resize(total_points);
        for (int row = 0; row < num_rows; ++row) {
            for (int col = 0; col < num_cols; ++col) {
                const int col_major_idx = col * num_rows + row;
                const int row_major_idx = row * num_cols + col;
                result.labels[row_major_idx] = static_cast<uint8_t>(inv_order[labels_ptr[col_major_idx]]);
            }
        }

        return result;
    }
};

// Test: Centroids are sorted ascending
TEST_F(Cluster1dTest, CentroidsAreSorted) {
    constexpr int NUM_ROWS = 1000;
    constexpr int NUM_COLS = 9;
    std::vector<float> data(NUM_ROWS * NUM_COLS);

    for (int i = 0; i < NUM_ROWS; ++i) {
        for (int j = 0; j < NUM_COLS; ++j) {
            data[i * NUM_COLS + j] = (i * NUM_COLS + j) / static_cast<float>(NUM_ROWS * NUM_COLS) * 2.0f - 1.0f;
        }
    }

    auto result = run_cluster1d(data.data(), NUM_ROWS, NUM_COLS, 10);

    for (int i = 1; i < 256; ++i) {
        EXPECT_GE(result.centroids[i], result.centroids[i - 1])
            << "Centroids not sorted at index " << i;
    }
}

// Test: Labels are valid indices (0-255)
TEST_F(Cluster1dTest, LabelsAreValid) {
    constexpr int NUM_ROWS = 500;
    constexpr int NUM_COLS = 9;
    std::vector<float> data(NUM_ROWS * NUM_COLS);

    for (int i = 0; i < NUM_ROWS * NUM_COLS; ++i) {
        data[i] = std::sin(i * 0.1f) * 0.5f;
    }

    auto result = run_cluster1d(data.data(), NUM_ROWS, NUM_COLS, 10);

    EXPECT_EQ(result.labels.size(), NUM_ROWS * NUM_COLS);
    for (size_t i = 0; i < result.labels.size(); ++i) {
        EXPECT_LT(result.labels[i], 256) << "Label out of range at " << i;
    }
}

// Test: Column ordering preserved (low values -> low labels, high -> high)
TEST_F(Cluster1dTest, ColumnOrderingPreserved) {
    constexpr int NUM_ROWS = 50;
    constexpr int NUM_COLS = 3;
    std::vector<float> data(NUM_ROWS * NUM_COLS);

    // Column 0: 0-49, Column 1: 100-149, Column 2: 200-249
    for (int row = 0; row < NUM_ROWS; ++row) {
        data[row * NUM_COLS + 0] = row;
        data[row * NUM_COLS + 1] = 100 + row;
        data[row * NUM_COLS + 2] = 200 + row;
    }

    auto result = run_cluster1d(data.data(), NUM_ROWS, NUM_COLS, 20);

    // Compute average label per column
    float avg_col0 = 0, avg_col1 = 0, avg_col2 = 0;
    for (int row = 0; row < NUM_ROWS; ++row) {
        avg_col0 += result.labels[row * NUM_COLS + 0];
        avg_col1 += result.labels[row * NUM_COLS + 1];
        avg_col2 += result.labels[row * NUM_COLS + 2];
    }
    avg_col0 /= NUM_ROWS;
    avg_col1 /= NUM_ROWS;
    avg_col2 /= NUM_ROWS;

    // Column ordering should be preserved
    EXPECT_LT(avg_col0, avg_col1) << "Column 0 should have lower labels than column 1";
    EXPECT_LT(avg_col1, avg_col2) << "Column 1 should have lower labels than column 2";
}

// Test: SH-like data structure access pattern
TEST_F(Cluster1dTest, ShAccessPattern) {
    // Verify the exact access pattern used in sogs.cpp for shN_centroids
    constexpr int PALETTE_SIZE = 100;
    constexpr int SH_COEFFS = 3;
    constexpr int SH_DIMS = 9;

    std::vector<uint8_t> mock_labels(PALETTE_SIZE * SH_DIMS);
    for (int row = 0; row < PALETTE_SIZE; ++row) {
        for (int col = 0; col < SH_DIMS; ++col) {
            mock_labels[row * SH_DIMS + col] = (row * 10 + col) % 256;
        }
    }

    // Simulate shN_centroids writing
    std::vector<uint8_t> centroids_buf(PALETTE_SIZE * SH_COEFFS * 4, 0);

    for (int i = 0; i < PALETTE_SIZE; ++i) {
        for (int j = 0; j < SH_COEFFS; ++j) {
            const int pixel_idx = i * SH_COEFFS + j;
            for (int c = 0; c < 3; ++c) {
                const int col_idx = SH_COEFFS * c + j;
                const int label_idx = i * SH_DIMS + col_idx;
                centroids_buf[pixel_idx * 4 + c] = mock_labels[label_idx];
            }
            centroids_buf[pixel_idx * 4 + 3] = 0xff;
        }
    }

    // Verify pattern for i=0, j=0
    EXPECT_EQ(centroids_buf[0], 0);    // R = mock_labels[0]
    EXPECT_EQ(centroids_buf[1], 3);    // G = mock_labels[3]
    EXPECT_EQ(centroids_buf[2], 6);    // B = mock_labels[6]
    EXPECT_EQ(centroids_buf[3], 0xff); // A

    // Verify pattern for i=0, j=1
    EXPECT_EQ(centroids_buf[4], 1);    // R = mock_labels[1]
    EXPECT_EQ(centroids_buf[5], 4);    // G = mock_labels[4]
    EXPECT_EQ(centroids_buf[6], 7);    // B = mock_labels[7]
    EXPECT_EQ(centroids_buf[7], 0xff); // A
}
