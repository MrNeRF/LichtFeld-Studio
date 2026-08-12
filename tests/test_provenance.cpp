/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <archive.h>
#include <archive_entry.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/error.hpp"
#include "core/image_io.hpp"
#include "core/provenance.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "load-spz.h"
#include "provenance-lichtfeld.h"

#ifdef _WIN32
using ssize_t = std::ptrdiff_t;
#endif

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;

namespace {

    constexpr std::string_view kProvenanceKey = "lichtfeld_provenance";

    const std::regex kUuidRe{
        R"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"};
    const std::regex kExportedAtRe{R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)"};
    const std::regex kHexIshRe{R"([0-9a-fA-F]+)"};

    std::string read_file_bytes(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.good()) << "Failed to open " << path;
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    bool contains_text(const std::string& haystack, const std::string_view needle) {
        return haystack.find(needle) != std::string::npos;
    }

    // SOG writer uses libarchive ZIP without zip:compression=store, so entries
    // are deflated. Extract meta.json the same way the SOG loader does.
    std::optional<std::string> extract_sog_meta_json(const fs::path& path) {
        struct ArchiveReadDeleter {
            void operator()(struct archive* value) const {
                if (value) {
                    archive_read_free(value);
                }
            }
        };

        std::unique_ptr<struct archive, ArchiveReadDeleter> archive(archive_read_new());
        if (!archive) {
            return std::nullopt;
        }
        if (archive_read_support_format_zip(archive.get()) != ARCHIVE_OK ||
            archive_read_support_filter_all(archive.get()) != ARCHIVE_OK) {
            return std::nullopt;
        }
#ifdef _WIN32
        if (archive_read_open_filename_w(archive.get(), path.wstring().c_str(), 10240) != ARCHIVE_OK) {
            return std::nullopt;
        }
#else
        if (archive_read_open_filename(archive.get(), path.c_str(), 10240) != ARCHIVE_OK) {
            return std::nullopt;
        }
#endif

        struct archive_entry* entry = nullptr;
        while (archive_read_next_header(archive.get(), &entry) == ARCHIVE_OK) {
            const char* const pathname = archive_entry_pathname(entry);
            if (!pathname || std::string_view(pathname) != "meta.json") {
                continue;
            }
            if (!archive_entry_size_is_set(entry)) {
                return std::nullopt;
            }
            const la_int64_t signed_size = archive_entry_size(entry);
            if (signed_size < 0) {
                return std::nullopt;
            }
            std::string data(static_cast<size_t>(signed_size), '\0');
            size_t offset = 0;
            while (offset < data.size()) {
                const ssize_t n = archive_read_data(
                    archive.get(), data.data() + offset, data.size() - offset);
                if (n <= 0) {
                    return std::nullopt;
                }
                offset += static_cast<size_t>(n);
            }
            return data;
        }
        return std::nullopt;
    }

    void expect_stamp_in_text(const std::string& text, const ProvenanceStamp& stamp) {
        EXPECT_TRUE(contains_text(text, kProvenanceKey)) << "missing lichtfeld_provenance";
        EXPECT_TRUE(contains_text(text, stamp.export_id)) << "missing export_id " << stamp.export_id;
    }

    void expect_stamp_absent(const std::string& text, const std::string_view export_id = {}) {
        EXPECT_FALSE(contains_text(text, kProvenanceKey));
        if (!export_id.empty()) {
            EXPECT_FALSE(contains_text(text, export_id));
        }
    }

    SplatData create_test_splat(size_t num_points, int sh_degree) {
        constexpr int SH_COEFFS[] = {0, 3, 8, 15};
        const size_t sh_coeffs = sh_degree > 0 ? SH_COEFFS[sh_degree] : 0;

        auto means = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
        auto sh0 = Tensor::empty({num_points, 1, 3}, Device::CPU, DataType::Float32);
        auto scaling = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
        auto rotation = Tensor::empty({num_points, 4}, Device::CPU, DataType::Float32);
        auto opacity = Tensor::empty({num_points, 1}, Device::CPU, DataType::Float32);

        Tensor shN;
        if (sh_coeffs > 0) {
            shN = Tensor::empty({num_points, sh_coeffs, 3}, Device::CPU, DataType::Float32);
        }

        auto* means_ptr = static_cast<float*>(means.data_ptr());
        auto* sh0_ptr = static_cast<float*>(sh0.data_ptr());
        auto* scaling_ptr = static_cast<float*>(scaling.data_ptr());
        auto* rotation_ptr = static_cast<float*>(rotation.data_ptr());
        auto* opacity_ptr = static_cast<float*>(opacity.data_ptr());

        for (size_t i = 0; i < num_points; ++i) {
            means_ptr[i * 3 + 0] = static_cast<float>(i % 10);
            means_ptr[i * 3 + 1] = static_cast<float>((i / 10) % 10);
            means_ptr[i * 3 + 2] = static_cast<float>(i / 100);

            sh0_ptr[i * 3 + 0] = 0.5f + 0.1f * static_cast<float>(i % 5);
            sh0_ptr[i * 3 + 1] = 0.3f + 0.1f * static_cast<float>((i + 1) % 5);
            sh0_ptr[i * 3 + 2] = 0.4f + 0.1f * static_cast<float>((i + 2) % 5);

            scaling_ptr[i * 3 + 0] = -3.0f + 0.01f * static_cast<float>(i % 100);
            scaling_ptr[i * 3 + 1] = -3.0f + 0.01f * static_cast<float>((i + 1) % 100);
            scaling_ptr[i * 3 + 2] = -3.0f + 0.01f * static_cast<float>((i + 2) % 100);

            constexpr float inv_sqrt_two = 0.70710678118f;
            const std::array<std::array<float, 4>, 4> rotations = {{
                {1.0f, 0.0f, 0.0f, 0.0f},
                {inv_sqrt_two, inv_sqrt_two, 0.0f, 0.0f},
                {inv_sqrt_two, 0.0f, inv_sqrt_two, 0.0f},
                {inv_sqrt_two, 0.0f, 0.0f, inv_sqrt_two},
            }};
            std::copy(rotations[i % rotations.size()].begin(),
                      rotations[i % rotations.size()].end(),
                      rotation_ptr + i * 4);

            opacity_ptr[i] = -2.0f + 0.04f * static_cast<float>(i % 100);
        }

        if (sh_coeffs > 0) {
            auto* shN_ptr = static_cast<float*>(shN.data_ptr());
            for (size_t i = 0; i < num_points * sh_coeffs * 3; ++i) {
                shN_ptr[i] = 0.1f * static_cast<float>(static_cast<int>(i % 10) - 5);
            }
        }

        return SplatData(
            sh_degree,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            0.5f);
    }

} // namespace

class ProvenanceTest : public ::testing::Test {
protected:
    const fs::path temp_dir = fs::temp_directory_path() / "lfs_provenance_test";

    void SetUp() override {
        fs::create_directories(temp_dir);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(temp_dir, ec);
    }
};

TEST_F(ProvenanceTest, JsonParsesAsSingleLineStamp) {
    const auto stamp = make_provenance_stamp();
    const std::string text = provenance_to_json(stamp);

    EXPECT_EQ(text.find('\n'), std::string::npos);
    EXPECT_EQ(text.find('\r'), std::string::npos);

    const auto json = nlohmann::json::parse(text);
    ASSERT_TRUE(json.is_object());
    ASSERT_TRUE(json.contains("lichtfeld_provenance"));
    EXPECT_EQ(json["lichtfeld_provenance"].get<int>(), 1);

    ASSERT_TRUE(json.contains("export_id"));
    EXPECT_TRUE(std::regex_match(json["export_id"].get<std::string>(), kUuidRe));
    EXPECT_EQ(json["export_id"].get<std::string>(), stamp.export_id);

    ASSERT_TRUE(json.contains("app_version"));
    EXPECT_FALSE(json["app_version"].get<std::string>().empty());

    ASSERT_TRUE(json.contains("build_commit"));
    EXPECT_EQ(json["build_commit"].get<std::string>(), stamp.build_commit);
    EXPECT_FALSE(json["build_commit"].get<std::string>().empty());
    EXPECT_TRUE(std::regex_match(json["build_commit"].get<std::string>(), kHexIshRe));

    ASSERT_TRUE(json.contains("exported_at"));
    EXPECT_TRUE(std::regex_match(json["exported_at"].get<std::string>(), kExportedAtRe));
}

TEST_F(ProvenanceTest, JsonOmitsDefaultConstructedFields) {
    const auto json = nlohmann::json::parse(provenance_to_json(ProvenanceStamp{}));
    ASSERT_TRUE(json.is_object());
    EXPECT_EQ(json["lichtfeld_provenance"].get<int>(), 1);
    EXPECT_FALSE(json.contains("export_id"));
    EXPECT_FALSE(json.contains("iteration"));
    EXPECT_FALSE(json.contains("strategy"));
    EXPECT_FALSE(json.contains("app_version"));
    EXPECT_FALSE(json.contains("build_commit"));
    EXPECT_FALSE(json.contains("exported_at"));
    EXPECT_FALSE(json.contains("project"));
    EXPECT_FALSE(json.contains("commit"));
    EXPECT_FALSE(json.contains("node"));
    EXPECT_FALSE(json.contains("dataset"));
}

TEST_F(ProvenanceTest, MakeStampYieldsUniqueExportIds) {
    const auto a = make_provenance_stamp();
    const auto b = make_provenance_stamp();
    EXPECT_FALSE(a.export_id.empty());
    EXPECT_FALSE(b.export_id.empty());
    EXPECT_NE(a.export_id, b.export_id);
}

TEST_F(ProvenanceTest, MinimalStampJsonContainsExactlyThreeKeys) {
    const auto stamp = make_minimal_provenance_stamp();
    EXPECT_TRUE(stamp.export_id.empty());
    EXPECT_TRUE(stamp.exported_at.empty());
    EXPECT_EQ(stamp.iteration, -1);
    EXPECT_TRUE(stamp.strategy.empty());

    const auto json = nlohmann::json::parse(provenance_to_json(stamp));
    ASSERT_TRUE(json.is_object());
    EXPECT_EQ(json.size(), 3u);
    ASSERT_TRUE(json.contains("lichtfeld_provenance"));
    EXPECT_EQ(json["lichtfeld_provenance"].get<int>(), 1);
    ASSERT_TRUE(json.contains("app_version"));
    EXPECT_FALSE(json["app_version"].get<std::string>().empty());
    ASSERT_TRUE(json.contains("build_commit"));
    const auto commit = json["build_commit"].get<std::string>();
    EXPECT_FALSE(commit.empty());
    EXPECT_TRUE(std::regex_match(commit, kHexIshRe));
    EXPECT_FALSE(json.contains("export_id"));
    EXPECT_FALSE(json.contains("exported_at"));
}

TEST_F(ProvenanceTest, FullStampContainsBuildCommit) {
    const auto stamp = make_provenance_stamp();
    EXPECT_FALSE(stamp.build_commit.empty());
    EXPECT_TRUE(std::regex_match(stamp.build_commit, kHexIshRe));

    const auto json = nlohmann::json::parse(provenance_to_json(stamp));
    ASSERT_TRUE(json.contains("build_commit"));
    EXPECT_EQ(json["build_commit"].get<std::string>(), stamp.build_commit);
}

TEST_F(ProvenanceTest, PlyBinaryEmbedsStampAndReloads) {
    const auto splat = create_test_splat(8, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "binary.ply";

    const auto saved = save_ply(splat, {.output_path = path, .binary = true, .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();

    expect_stamp_in_text(read_file_bytes(path), stamp);

    const auto loaded = load_ply(path);
    ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
    EXPECT_EQ(loaded->value.size(), splat.size());
}

TEST_F(ProvenanceTest, PlyAsciiEmbedsStamp) {
    const auto splat = create_test_splat(8, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "ascii.ply";

    const auto saved = save_ply(splat, {.output_path = path, .binary = false, .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();

    const auto bytes = read_file_bytes(path);
    EXPECT_TRUE(bytes.starts_with("ply"));
    EXPECT_TRUE(contains_text(bytes, "format ascii 1.0"));
    expect_stamp_in_text(bytes, stamp);
}

TEST_F(ProvenanceTest, PlyStripWritesMinimalStamp) {
    const auto splat = create_test_splat(8, 0);
    const fs::path path = temp_dir / "minimal_stamp.ply";

    const auto saved = save_ply(splat, {.output_path = path, .binary = true, .provenance = make_minimal_provenance_stamp()});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();

    const auto bytes = read_file_bytes(path);
    EXPECT_TRUE(contains_text(bytes, kProvenanceKey));
    EXPECT_TRUE(contains_text(bytes, "build_commit"));
    EXPECT_FALSE(contains_text(bytes, "export_id"));
    EXPECT_FALSE(contains_text(bytes, "exported_at"));
}

TEST_F(ProvenanceTest, PlyDefaultOptionsWritesMinimalStamp) {
    const auto splat = create_test_splat(8, 0);
    const fs::path path = temp_dir / "default_options.ply";

    const auto saved = save_ply(splat, {.output_path = path, .binary = true});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();

    const auto bytes = read_file_bytes(path);
    EXPECT_TRUE(contains_text(bytes, kProvenanceKey));
    EXPECT_TRUE(contains_text(bytes, "build_commit"));
    EXPECT_FALSE(contains_text(bytes, "export_id"));
    EXPECT_FALSE(contains_text(bytes, "exported_at"));
}

TEST_F(ProvenanceTest, SogEmbedsStampInDeflatedMetaJson) {
    const auto splat = create_test_splat(16, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "stamped.sog";

    const auto saved = save_sog(splat, {.output_path = path,
                                        .kmeans_iterations = 1,
                                        .use_gpu = false,
                                        .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();

    const auto meta = extract_sog_meta_json(path);
    ASSERT_TRUE(meta.has_value()) << "failed to extract meta.json from SOG zip";
    expect_stamp_in_text(*meta, stamp);
}

TEST_F(ProvenanceTest, RadEmbedsStampInPlaintextMeta) {
    const auto splat = create_test_splat(16, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "stamped.rad";

    const auto saved = save_rad(splat, {.output_path = path, .compression_level = 1, .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();
    expect_stamp_in_text(read_file_bytes(path), stamp);
}

TEST_F(ProvenanceTest, UsdaEmbedsStamp) {
    const auto splat = create_test_splat(8, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "stamped.usda";

    const auto saved = save_usd(splat, {.output_path = path, .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();
    expect_stamp_in_text(read_file_bytes(path), stamp);
}

TEST_F(ProvenanceTest, NurecUsdzEmbedsStamp) {
    const auto splat = create_test_splat(8, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "stamped.usdz";

    const auto saved = save_nurec_usdz(splat, {.output_path = path, .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();
    expect_stamp_in_text(read_file_bytes(path), stamp);
}

TEST_F(ProvenanceTest, HtmlEmbedsStamp) {
    const auto splat = create_test_splat(8, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "stamped.html";

    const auto saved = export_html(splat, {.output_path = path,
                                           .kmeans_iterations = 1,
                                           .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();

    const auto text = read_file_bytes(path);
    EXPECT_TRUE(contains_text(text, R"(id="lichtfeld-provenance")"));
    EXPECT_TRUE(contains_text(text, stamp.export_id));
}

TEST_F(ProvenanceTest, SpzV4EmbedsStampInPlaintextExtensionZone) {
    const auto splat = create_test_splat(8, 0);
    const auto stamp = make_provenance_stamp();
    const std::string expected_json = provenance_to_json(stamp);
    const fs::path path = temp_dir / "stamped_v4.spz";

    const auto saved = save_spz(splat, {.output_path = path, .version = 4, .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();
    expect_stamp_in_text(read_file_bytes(path), stamp);

    const auto packed = spz::loadSpzPacked(path.string());
    ASSERT_GT(packed.numPoints, 0);
    const auto ext = spz::findExtensionByType<spz::SpzExtensionProvenanceLichtFeld>(packed.extensions);
    ASSERT_NE(ext, nullptr);
    EXPECT_EQ(ext->json, expected_json);
}

TEST_F(ProvenanceTest, SpzV3OmitsStampAndStillLoads) {
    const auto splat = create_test_splat(8, 0);
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "stamped_v3.spz";

    const auto saved = save_spz(splat, {.output_path = path, .version = 3, .provenance = stamp});
    ASSERT_TRUE(saved.has_value()) << saved.error().format();
    expect_stamp_absent(read_file_bytes(path), stamp.export_id);

    const auto packed = spz::loadSpzPacked(path.string());
    EXPECT_EQ(packed.numPoints, static_cast<int32_t>(splat.size()));
    EXPECT_EQ(spz::findExtensionByType<spz::SpzExtensionProvenanceLichtFeld>(packed.extensions), nullptr);
}

TEST_F(ProvenanceTest, PngCommentEmbedsStamp) {
    const auto stamp = make_provenance_stamp();
    const fs::path path = temp_dir / "stamped.png";
    auto image = Tensor::zeros({4, 4, 3}, Device::CPU, DataType::UInt8);

    save_image_u8(path, std::move(image), 95, provenance_to_json(stamp));
    ASSERT_TRUE(fs::exists(path));
    expect_stamp_in_text(read_file_bytes(path), stamp);
}
