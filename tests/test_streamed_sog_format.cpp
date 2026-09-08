/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../src/io/cuda/morton_encoding.hpp"
#include "core/argument_parser.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/sogs.hpp"
#include "io/formats/streamed_sog.hpp"
#include "io/formats/streamed_sog_morton.hpp"
#include "io/loader.hpp"
#include "io/splat_path.hpp"
#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <webp/decode.h>

namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    using namespace lfs::core;
    using namespace lfs::io;
    struct ScopedStreamedSogDirectory {
        fs::path path;
        ScopedStreamedSogDirectory() {
            static std::atomic_uint64_t sequence{0};
            path = fs::temp_directory_path() / std::format("lichtfeld_ssog_{}_{}", std::chrono::steady_clock::now().time_since_epoch().count(), sequence++);
            fs::create_directories(path);
        }
        ~ScopedStreamedSogDirectory() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    };
    Json read(const fs::path& p) {
        std::ifstream f(p);
        return Json::parse(f);
    }
    void write(const fs::path& p, const Json& j) {
        std::ofstream f(p);
        f << j.dump();
    }
    SplatData synthetic(size_t n, int degree = 1) {
        std::mt19937 rng(27);
        std::normal_distribution<float> normal(0, 1);
        const size_t k = (degree + 1) * (degree + 1) - 1;
        std::vector<float> means(n * 3), scales(n * 3), quats(n * 4), sh0(n * 3), shN(n * k * 3), opacity(n);
        for (size_t i = 0; i < n; ++i) {
            for (int a = 0; a < 3; ++a) {
                means[i * 3 + a] = normal(rng) * 1.5f + float((i % 8) >> a & 1) * 12;
                scales[i * 3 + a] = -2 + normal(rng) * 0.15f;
                sh0[i * 3 + a] = normal(rng) * 0.15f;
            }
            for (int a = 0; a < 4; ++a)
                quats[i * 4 + a] = normal(rng);
            opacity[i] = normal(rng) * 0.4f;
        }
        for (auto& v : shN)
            v = normal(rng) * 0.08f;
        return SplatData(degree, Tensor::from_vector(means, {n, 3}), Tensor::from_vector(sh0, {n, 1, 3}), Tensor::from_vector(shN, {n, k, 3}), Tensor::from_vector(scales, {n, 3}), Tensor::from_vector(quats, {n, 4}), Tensor::from_vector(opacity, {n, 1}), 1);
    }
    StreamedSogSaveOptions options(const fs::path& path, int levels = 1) {
        StreamedSogSaveOptions o;
        o.output_path = path;
        o.lod_levels = levels;
        o.chunk_count_k = 16;
        o.chunk_extent = 2;
        o.chunk_min_k = 1;
        return o;
    }
    void assert_structure(const fs::path& path, const Json& m) {
        ASSERT_EQ(m["version"], 1);
        ASSERT_EQ(m["lodErrors"], false);
        EXPECT_TRUE(m["asset"].contains("lichtfeld_provenance"));
        const auto files = m["filenames"].get<std::vector<std::string>>();
        std::vector<Tensor> means;
        std::vector<size_t> sizes;
        std::vector<std::vector<std::pair<size_t, size_t>>> ranges(files.size());
        for (const auto& f : files) {
            auto unit = load_sog(path / f);
            ASSERT_TRUE(unit) << unit.error();
            sizes.push_back(unit->size());
            means.push_back(unit->means().cpu());
            EXPECT_TRUE(read(path / f)["asset"].contains("lichtfeld_provenance"));
        }
        std::vector<size_t> counts(m["lodLevels"].get<int>());
        const auto visit = [&](auto&& self, const Json& node) -> void {
            if (node.contains("children")) {
                ASSERT_FALSE(node.contains("lods"));
                ASSERT_EQ(node["children"].size(), 2);
                for (const auto& child : node["children"]) {
                    for (int a = 0; a < 3; ++a) {
                        EXPECT_LE(node["bound"]["min"][a].get<double>(), child["bound"]["min"][a].get<double>());
                        EXPECT_GE(node["bound"]["max"][a].get<double>(), child["bound"]["max"][a].get<double>());
                    }
                    self(self, child);
                }
            } else
                for (const auto& [key, r] : node["lods"].items()) {
                    const size_t f = r["file"], off = r["offset"], n = r["count"];
                    ASSERT_LT(f, sizes.size());
                    ASSERT_LE(off + n, sizes[f]);
                    ranges[f].emplace_back(off, n);
                    counts[std::stoi(key)] += n;
                    const float* p = means[f].ptr<float>();
                    for (size_t i = off; i < off + n; ++i)
                        for (int a = 0; a < 3; ++a) {
                            ASSERT_GE(p[i * 3 + a], node["bound"]["min"][a].get<double>() - 0.01);
                            ASSERT_LE(p[i * 3 + a], node["bound"]["max"][a].get<double>() + 0.01);
                        }
                }
        };
        visit(visit, m["tree"]);
        for (size_t f = 0; f < files.size(); ++f) {
            std::sort(ranges[f].begin(), ranges[f].end());
            size_t end = 0;
            for (auto [off, n] : ranges[f]) {
                EXPECT_EQ(off, end);
                end += n;
            }
            EXPECT_EQ(end, sizes[f]);
        }
        EXPECT_EQ(m["counts"], Json(counts));
        EXPECT_EQ(m["count"], std::accumulate(counts.begin(), counts.end(), size_t{0}));
    }
    void compare(const SplatData& decoded, const SplatData& source) {
        ASSERT_EQ(decoded.size(), source.size());
        const size_t n = source.size();
        auto a = decoded.means().cpu(), b = source.means().cpu();
        auto as = decoded.get_scaling().cpu(), bs = source.get_scaling().cpu();
        auto ao = decoded.get_opacity().cpu(), bo = source.get_opacity().cpu();
        auto ac = decoded.sh0().cpu(), bc = source.sh0().cpu();
        auto ah = decoded.shN_canonical_cpu(), bh = source.shN_canonical_cpu();
        auto aq = decoded.get_rotation().cpu(), bq = source.get_rotation().cpu();
        std::vector<size_t> sorted(n);
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(), [&](size_t i, size_t j) { return b.ptr<float>()[i * 3] < b.ptr<float>()[j * 3]; });
        std::vector<bool> used(n);
        // Search the x-sorted reference in the position quantization window, then
        // match in 3D; simple lexicographic sorting is unstable after quantization.
        for (size_t i = 0; i < n; ++i) {
            const auto* p = a.ptr<float>() + i * 3;
            auto it = std::lower_bound(sorted.begin(), sorted.end(), p[0] - 0.015f, [&](size_t j, float x) { return b.ptr<float>()[j * 3] < x; });
            size_t best = n;
            double distance = INFINITY;
            for (; it != sorted.end() && b.ptr<float>()[*it * 3] <= p[0] + 0.015f; ++it) {
                if (used[*it])
                    continue;
                double d = 0;
                for (int axis = 0; axis < 3; ++axis)
                    d += std::pow(p[axis] - b.ptr<float>()[*it * 3 + axis], 2);
                if (d < distance) {
                    distance = d;
                    best = *it;
                }
            }
            ASSERT_LT(best, n);
            ASSERT_LT(distance, 0.015 * 0.015);
            used[best] = true;
            for (int axis = 0; axis < 3; ++axis) {
                EXPECT_NEAR(as.ptr<float>()[i * 3 + axis], bs.ptr<float>()[best * 3 + axis], 1.0f);
                EXPECT_NEAR(ac.ptr<float>()[i * 3 + axis], bc.ptr<float>()[best * 3 + axis], 1.0f);
            }
            EXPECT_NEAR(ao.ptr<float>()[i], bo.ptr<float>()[best], 0.02f);
            float dot = 0;
            for (int c = 0; c < 4; ++c)
                dot += aq.ptr<float>()[i * 4 + c] * bq.ptr<float>()[best * 4 + c];
            EXPECT_GT(std::abs(dot), 0.99f);
            const size_t k = source.max_sh_coeffs_rest() * 3;
            for (size_t c = 0; c < k; ++c)
                EXPECT_NEAR(ah.ptr<float>()[i * k + c], bh.ptr<float>()[best * k + c], 1.0f);
        }
    }
    std::string shell_quote(const std::string& s) {
        std::string q = "'";
        for (char c : s)
            q += c == '\'' ? "'\\''" : std::string(1, c);
        return q + "'";
    }
} // namespace
TEST(StreamedSogFormat, WriteReadRoundtripSyntheticSh1) {
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(60000);
    auto o = options(dir.path, 3);
    auto result = save_streamed_sog(s, o);
    ASSERT_TRUE(result) << result.error().format();
    const auto m = read(dir.path / "lod-meta.json");
    ASSERT_EQ(m["lodLevels"], 3);
    EXPECT_EQ(m["counts"], Json::array({60000, 30000, 15000}));
    EXPECT_GT(m["filenames"].size(), 3);
    assert_structure(dir.path, m);
    auto loaded = load_streamed_sog(dir.path);
    ASSERT_TRUE(loaded) << loaded.error();
    compare(*loaded, s);
    auto coarse = load_streamed_sog(dir.path / "lod-meta.json", {.lod_level = -1});
    ASSERT_TRUE(coarse) << coarse.error();
    EXPECT_EQ(coarse->size(), m["counts"].back().get<size_t>());
}
TEST(StreamedSogFormat, SingleLevelNoDecimation) {
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(3000, 0);
    auto result = save_streamed_sog(s, options(dir.path));
    ASSERT_TRUE(result) << result.error().format();
    const auto m = read(dir.path / "lod-meta.json");
    EXPECT_EQ(m["counts"], Json::array({3000}));
    assert_structure(dir.path, m);
    auto loader = Loader::create();
    EXPECT_TRUE(loader->canLoad(dir.path));
    EXPECT_FALSE(Loader::isDatasetPath(dir.path));
    LoadOptions validate;
    validate.validate_only = true;
    auto loaded = loader->load(dir.path, validate);
    ASSERT_TRUE(loaded) << loaded.error().format();
    EXPECT_EQ(loaded->loader_used, "Streamed SOG");
}
TEST(StreamedSogFormat, RejectsInvalidManifest) {
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(300, 0);
    auto saved = save_streamed_sog(s, options(dir.path));
    ASSERT_TRUE(saved) << saved.error().format();
    const auto original = read(dir.path / "lod-meta.json");
    std::vector<Json> invalid;
    auto m = original;
    m["version"] = 2;
    invalid.push_back(m);
    m = original;
    m["counts"][0] = 301;
    invalid.push_back(m);
    m = original;
    m["tree"]["lods"]["0"]["file"] = 100;
    invalid.push_back(m);
    m = original;
    m["tree"]["lods"]["0"]["offset"] = 1;
    invalid.push_back(m);
    m = original;
    m["lodLevels"] = 0;
    invalid.push_back(m);
    m = original;
    m["filenames"][0] = "../escape/meta.json";
    invalid.push_back(m);
    for (const auto& bad : invalid) {
        write(dir.path / "lod-meta.json", bad);
        EXPECT_FALSE(validate_streamed_sog(dir.path));
        EXPECT_FALSE(load_streamed_sog(dir.path));
    }
    m = original;
    m.erase("version");
    m.erase("counts");
    m.erase("count");
    write(dir.path / "lod-meta.json", m);
    EXPECT_TRUE(validate_streamed_sog(dir.path));
    EXPECT_TRUE(load_streamed_sog(dir.path));
}
TEST(StreamedSogFormat, ReplacesPreviousExport) {
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(2000, 0);
    auto first = save_streamed_sog(s, options(dir.path, 2));
    ASSERT_TRUE(first) << first.error().format();
    fs::create_directory(dir.path / "env");
    fs::create_directory(dir.path / "9_8");
    write(dir.path / "notes.json", {{"keep", true}});
    auto second = save_streamed_sog(s, options(dir.path / "lod-meta.json", 1));
    ASSERT_TRUE(second) << second.error().format();
    const auto m = read(dir.path / "lod-meta.json");
    EXPECT_EQ(m["lodLevels"], 1);
    EXPECT_FALSE(fs::exists(dir.path / "env"));
    EXPECT_FALSE(fs::exists(dir.path / "9_8"));
    EXPECT_FALSE(fs::exists(dir.path / "1_0"));
    EXPECT_TRUE(fs::exists(dir.path / "notes.json"));
    EXPECT_TRUE(load_streamed_sog(dir.path));
}
TEST(StreamedSogFormat, CancellationPreservesPreviousExport) {
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(500, 0);
    auto o = options(dir.path);
    ASSERT_TRUE(save_streamed_sog(s, o));
    const auto original = read(dir.path / "lod-meta.json");
    for (const float threshold : {0.0f, 0.5f, 1.0f}) {
        o.progress_callback = [=](float p, const std::string&) { return p < threshold; };
        auto result = save_streamed_sog(s, o);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
        EXPECT_EQ(read(dir.path / "lod-meta.json"), original);
        EXPECT_TRUE(load_streamed_sog(dir.path));
    }
}
TEST(StreamedSogFormat, VisibleRowsAndEnvironment) {
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(1000, 0);
    std::vector<bool> deleted(1000);
    std::fill_n(deleted.begin(), 100, true);
    s.deleted() = Tensor::from_vector(deleted, {1000});
    auto saved = save_streamed_sog(s, options(dir.path));
    ASSERT_TRUE(saved) << saved.error().format();
    auto m = read(dir.path / "lod-meta.json");
    EXPECT_EQ(m["counts"][0], 900);
    auto env = synthetic(100, 1);
    SogEncodeOptions eo;
    eo.output_path = dir.path / "env";
    auto encoded = encode_sog_directory(env, eo);
    ASSERT_TRUE(encoded) << encoded.error().format();
    m["environment"] = "env/meta.json";
    write(dir.path / "lod-meta.json", m);
    auto loaded = load_streamed_sog(dir.path);
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->size(), 1000);
    EXPECT_EQ(loaded->get_max_sh_degree(), 1);
    auto rest = loaded->shN_canonical_cpu();
    for (size_t i = 0; i < 900 * 9; ++i)
        ASSERT_EQ(rest.ptr<float>()[i], 0);
}
TEST(StreamedSogFormat, ReadsReferenceStreamedSog) {
    const char* reference = std::getenv("LFS_SSOG_REFERENCE");
    if (!reference)
        GTEST_SKIP() << "Set LFS_SSOG_REFERENCE to lod-meta.json";
    const auto m = read(reference);
    for (int l : {0, 3}) {
        auto result = load_streamed_sog(reference, {.lod_level = l});
        ASSERT_TRUE(result) << result.error();
        EXPECT_EQ(result->size(), m["counts"][l].get<size_t>());
        for (auto t : {result->means().cpu(), result->scaling_raw().cpu(), result->rotation_raw().cpu(), result->opacity_raw().cpu(), result->sh0().cpu(), result->shN_canonical_cpu()})
            for (size_t i = 0; i < t.numel(); ++i)
                ASSERT_TRUE(std::isfinite(t.ptr<float>()[i]));
    }
}
TEST(StreamedSogFormat, ReferenceReadsOurs) {
    const char* cli = std::getenv("LFS_SPLAT_TRANSFORM");
    if (!cli || std::system("node --version > /dev/null 2>&1") != 0)
        GTEST_SKIP() << "Set LFS_SPLAT_TRANSFORM and install node";
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(3000, 1);
    const auto out = dir.path / "ssog";
    auto result = save_streamed_sog(s, options(out, 3));
    ASSERT_TRUE(result) << result.error().format();
    const auto m = read(out / "lod-meta.json");
    const std::string base = "node " + shell_quote(cli) + " -g cpu --max-workers 0 " + shell_quote((out / "lod-meta.json").string());
    const auto ply = dir.path / "back.ply";
    const auto log = dir.path / "reference.log";
    ASSERT_EQ(std::system((base + " --select-lod 0 " + shell_quote(ply.string()) + " > " + shell_quote(log.string()) + " 2>&1").c_str()), 0) << std::ifstream(log).rdbuf();
    std::ifstream file(ply, std::ios::binary);
    std::string line;
    size_t vertices = 0;
    while (std::getline(file, line) && line != "end_header")
        if (line.starts_with("element vertex "))
            vertices = std::stoull(line.substr(15));
    EXPECT_EQ(vertices, m["counts"][0].get<size_t>());
    const auto info = dir.path / "info.json";
    ASSERT_EQ(std::system((base + " --info json null > " + shell_quote(info.string()) + " 2> " + shell_quote(log.string())).c_str()), 0) << std::ifstream(log).rdbuf();
    const auto info_json = read(info);
    EXPECT_EQ(info_json.at("lodCounts"), m["counts"]);
    EXPECT_EQ(info_json.at("numLods"), m["lodLevels"]);
}

TEST(StreamedSogFormat, CliOptionsAndAliases) {
    ScopedStreamedSogDirectory dir;
    const auto input = (dir.path / "input.ply").string();
    std::ofstream(input).put('\n');
    for (const char* alias : {"ssog", "streamed-sog", "lod-meta.json"}) {
        const char* argv[] = {"LichtFeld-Studio", "convert", input.c_str(), "-f", alias,
                              "--lod-levels", "2", "--lod-ratio", "0.25", "--lod-chunk-count", "32",
                              "--lod-chunk-extent", "8", "--lod-chunk-min", "2", "-o", "result_ssog"};
        auto parsed = lfs::core::args::parse_args(std::size(argv), argv);
        ASSERT_TRUE(parsed) << parsed.error();
        const auto* mode = std::get_if<lfs::core::args::ConvertMode>(&*parsed);
        ASSERT_NE(mode, nullptr);
        EXPECT_EQ(mode->params.format, lfs::core::param::OutputFormat::STREAMED_SOG);
        EXPECT_EQ(mode->params.output_path, fs::path("result_ssog"));
        EXPECT_EQ(mode->params.lod_levels, 2);
        EXPECT_FLOAT_EQ(mode->params.lod_ratio, 0.25f);
        EXPECT_EQ(mode->params.lod_chunk_count, 32);
        EXPECT_FLOAT_EQ(mode->params.lod_chunk_extent, 8);
        EXPECT_EQ(mode->params.lod_chunk_min, 2);
    }
    const char* bad[] = {"LichtFeld-Studio", "convert", input.c_str(), "-f", "ssog", "--lod-ratio", "1"};
    EXPECT_FALSE(lfs::core::args::parse_args(std::size(bad), bad));
}

TEST(StreamedSogFormat, BundleAndDirectoryPayloadsMatch) {
    ScopedStreamedSogDirectory dir;
    auto splats = synthetic(2048);
    const auto stamp = make_minimal_provenance_stamp();
    const auto bundle = dir.path / "bundle.sog";
    auto saved = save_sog(splats, {.output_path = bundle, .provenance = stamp});
    ASSERT_TRUE(saved) << saved.error().format();
    SogEncodeOptions o;
    o.output_path = dir.path / "directory";
    o.provenance = stamp;
    auto encoded = encode_sog_directory(splats, o);
    ASSERT_TRUE(encoded) << encoded.error().format();
    auto fast = o;
    fast.output_path = dir.path / "fast_directory";
    fast.fast_webp = true;
    ASSERT_TRUE(encode_sog_directory(splats, fast));
    std::unique_ptr<archive, decltype(&archive_read_free)> input(archive_read_new(), archive_read_free);
    ASSERT_EQ(archive_read_support_format_zip(input.get()), ARCHIVE_OK);
#ifdef _WIN32
    ASSERT_EQ(archive_read_open_filename_w(input.get(), bundle.wstring().c_str(), 10240), ARCHIVE_OK);
#else
    ASSERT_EQ(archive_read_open_filename(input.get(), bundle.c_str(), 10240), ARCHIVE_OK);
#endif
    archive_entry* entry = nullptr;
    std::vector<std::string> names;
    while (archive_read_next_header(input.get(), &entry) == ARCHIVE_OK) {
        const std::string name = archive_entry_pathname(entry);
        names.push_back(name);
        std::string bytes(static_cast<size_t>(archive_entry_size(entry)), '\0');
        ASSERT_EQ(archive_read_data(input.get(), bytes.data(), bytes.size()), static_cast<la_ssize_t>(bytes.size()));
        std::ifstream file(o.output_path / name, std::ios::binary);
        const std::string other((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        EXPECT_EQ(bytes, other) << name;
        std::ifstream fast_file(fast.output_path / name, std::ios::binary);
        const std::string fast_bytes((std::istreambuf_iterator<char>(fast_file)), std::istreambuf_iterator<char>());
        if (name.ends_with(".webp")) {
            int w = 0, h = 0, fw = 0, fh = 0;
            std::unique_ptr<uint8_t, decltype(&WebPFree)> pixels(WebPDecodeRGBA(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), &w, &h), WebPFree);
            std::unique_ptr<uint8_t, decltype(&WebPFree)> fast_pixels(WebPDecodeRGBA(reinterpret_cast<const uint8_t*>(fast_bytes.data()), fast_bytes.size(), &fw, &fh), WebPFree);
            ASSERT_TRUE(pixels);
            ASSERT_TRUE(fast_pixels);
            ASSERT_EQ(w, fw);
            ASSERT_EQ(h, fh);
            EXPECT_TRUE(std::equal(pixels.get(), pixels.get() + size_t(w) * h * 4, fast_pixels.get())) << name;
        } else {
            EXPECT_EQ(bytes, fast_bytes) << name;
        }
    }
    EXPECT_EQ(names, (std::vector<std::string>{"means_l.webp", "means_u.webp", "quats.webp", "scales.webp", "sh0.webp", "shN_centroids.webp", "shN_labels.webp", "meta.json"}));
}

TEST(StreamedSogFormat, TinyInputHasEmptyCoarsestLevel) {
    ScopedStreamedSogDirectory dir;
    auto splats = synthetic(3, 0);
    auto saved = save_streamed_sog(splats, options(dir.path, 4));
    ASSERT_TRUE(saved) << saved.error().format();
    EXPECT_EQ(read(dir.path / "lod-meta.json")["counts"], Json::array({3, 2, 1, 0}));
    auto loaded = load_streamed_sog(dir.path, {.lod_level = -1});
    ASSERT_TRUE(loaded) << loaded.error();
    EXPECT_EQ(loaded->size(), 0);
}

TEST(StreamedSogFormat, ImportNamesUseAssetDirectories) {
    ScopedStreamedSogDirectory dir;
    const auto asset = dir.path / "garden.mcp";
    fs::create_directories(asset / "1_0");
    std::ofstream(asset / "lod-meta.json") << "{}";
    std::ofstream(asset / "1_0" / "meta.json") << "{}";
    EXPECT_EQ(splat_import_name(asset / "lod-meta.json"), "garden.mcp");
    EXPECT_EQ(splat_import_name(asset), "garden.mcp");
    EXPECT_EQ(splat_import_name(asset / ""), "garden.mcp");
    EXPECT_EQ(splat_import_name(asset / "1_0" / "meta.json"), "garden.mcp_1_0");
    EXPECT_EQ(splat_import_name(asset / "1_0"), "garden.mcp_1_0");
    EXPECT_EQ(splat_import_name(dir.path / "garden.sog"), "garden");
    EXPECT_EQ(splat_import_name(dir.path / "garden.ply"), "garden");
    EXPECT_EQ(splat_import_name(dir.path / "garden.spz"), "garden");
}

TEST(StreamedSogFormat, WorkerProgressIsSerializedMonotoneAndCancellable) {
    ScopedStreamedSogDirectory dir;
    auto s = synthetic(12000, 1);
    auto o = options(dir.path, 3);
    float last = -1;
    std::atomic_int callbacks{0};
    bool saw_worker = false;
    const auto caller = std::this_thread::get_id();
    o.progress_callback = [&](float p, const std::string&) {
        EXPECT_EQ(callbacks.fetch_add(1), 0);
        EXPECT_GE(p, last);
        EXPECT_LE(p, 1.0f);
        last = p;
        saw_worker |= std::this_thread::get_id() != caller;
        callbacks.fetch_sub(1);
        return true;
    };
    ASSERT_TRUE(save_streamed_sog(s, o));
    EXPECT_EQ(last, 1.0f);
    if (!std::getenv("LFS_SSOG_UNIT_WORKERS"))
        EXPECT_TRUE(saw_worker);
    const auto original = read(dir.path / "lod-meta.json");
    last = -1;
    o.progress_callback = [&](float p, const std::string&) {
        EXPECT_GE(p, last);
        last = p;
        return p < 0.65f;
    };
    auto cancelled = save_streamed_sog(s, o);
    ASSERT_FALSE(cancelled);
    EXPECT_EQ(cancelled.error().code, ErrorCode::CANCELLED);
    EXPECT_EQ(read(dir.path / "lod-meta.json"), original);
}

TEST(StreamedSogFormat, CpuLeafMortonMatchesCudaIncludingStableTies) {
    for (const size_t n : {1u, 17u, 4096u, 30001u}) {
        auto s = synthetic(n, 0);
        auto positions = s.means().cpu();
        for (int mode = 0; mode < 3; ++mode) {
            if (mode == 1)
                for (size_t i = 0; i < n; ++i)
                    positions.ptr<float>()[i * 3 + 2] = 0;
            if (mode == 2)
                for (size_t i = 0; i < n; ++i)
                    std::copy_n(positions.ptr<float>(), 3, positions.ptr<float>() + i * 3);
            std::vector<int> rows(n);
            std::iota(rows.begin(), rows.end(), 0);
            sort_streamed_sog_leaf(positions.ptr<float>(), rows);
            auto gpu = morton_sort_indices_for_positions(positions.cuda()).cpu();
            ASSERT_TRUE(gpu.is_valid());
            EXPECT_TRUE(std::equal(rows.begin(), rows.end(), gpu.ptr<int>())) << "n=" << n << " mode=" << mode;
        }
    }
}
