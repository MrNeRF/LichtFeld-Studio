/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../src/io/cuda/morton_encoding.hpp"
#include "app/include/app/converter.hpp"
#include "core/argument_parser.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/sogs.hpp"
#include "io/formats/ssog.hpp"
#include "io/formats/ssog_morton.hpp"
#include "io/loader.hpp"
#include "io/loaders/ssog_loader.hpp"
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

namespace {
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    using namespace lfs::core;
    using namespace lfs::io;
    struct ScopedSsogDirectory {
        fs::path path;
        ScopedSsogDirectory() {
            static std::atomic_uint64_t sequence{0};
            path = fs::temp_directory_path() / std::format("lichtfeld_ssog_{}_{}", std::chrono::steady_clock::now().time_since_epoch().count(), sequence++);
            fs::create_directories(path);
        }
        ~ScopedSsogDirectory() {
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
    SsogSaveOptions options(const fs::path& path, int levels = 1) {
        SsogSaveOptions o;
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
            ASSERT_TRUE(unit) << unit.error().message;
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
TEST(SsogFormat, WriteReadRoundtripSyntheticSh1) {
    ScopedSsogDirectory dir;
    auto s = synthetic(60000);
    auto o = options(dir.path, 3);
    auto result = save_ssog(s, o);
    ASSERT_TRUE(result) << result.error().format();
    const auto m = read(dir.path / "lod-meta.json");
    ASSERT_EQ(m["lodLevels"], 3);
    EXPECT_EQ(m["counts"], Json::array({60000, 30000, 15000}));
    EXPECT_GT(m["filenames"].size(), 3);
    assert_structure(dir.path, m);
    auto loaded = load_ssog(dir.path);
    ASSERT_TRUE(loaded) << loaded.error().message;
    compare(*loaded, s);
    auto coarse = load_ssog(dir.path / "lod-meta.json", {.lod_level = -1});
    ASSERT_TRUE(coarse) << coarse.error().message;
    EXPECT_EQ(coarse->size(), m["counts"].back().get<size_t>());
}
TEST(SsogFormat, SingleLevelNoDecimation) {
    ScopedSsogDirectory dir;
    auto s = synthetic(3000, 0);
    auto result = save_ssog(s, options(dir.path));
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
    EXPECT_EQ(loaded->loader_used, "SSOG");
}
TEST(SsogFormat, RejectsInvalidManifest) {
    ScopedSsogDirectory dir;
    auto s = synthetic(300, 0);
    auto saved = save_ssog(s, options(dir.path));
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
        EXPECT_FALSE(validate_ssog(dir.path));
        EXPECT_FALSE(load_ssog(dir.path));
    }
    m = original;
    m.erase("version");
    m.erase("counts");
    m.erase("count");
    write(dir.path / "lod-meta.json", m);
    EXPECT_TRUE(validate_ssog(dir.path));
    EXPECT_TRUE(load_ssog(dir.path));
}

TEST(SsogFormat, VisibleRowsAndEnvironment) {
    ScopedSsogDirectory dir;
    auto s = synthetic(1000, 0);
    std::vector<bool> deleted(1000);
    std::fill_n(deleted.begin(), 100, true);
    s.deleted() = Tensor::from_vector(deleted, {1000});
    auto saved = save_ssog(s, options(dir.path));
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
    auto loaded = load_ssog(dir.path);
    ASSERT_TRUE(loaded) << loaded.error().message;
    EXPECT_EQ(loaded->size(), 1000);
    EXPECT_EQ(loaded->get_max_sh_degree(), 1);
    auto rest = loaded->shN_canonical_cpu();
    for (size_t i = 0; i < 900 * 9; ++i)
        ASSERT_EQ(rest.ptr<float>()[i], 0);
}
TEST(SsogFormat, ReadsReferenceSsog) {
    const char* reference = std::getenv("LFS_SSOG_REFERENCE");
    if (!reference)
        GTEST_SKIP() << "Set LFS_SSOG_REFERENCE to lod-meta.json";
    const auto m = read(reference);
    for (int l : {0, 3}) {
        auto result = load_ssog(reference, {.lod_level = l});
        ASSERT_TRUE(result) << result.error().message;
        EXPECT_EQ(result->size(), m["counts"][l].get<size_t>());
        for (auto t : {result->means().cpu(), result->scaling_raw().cpu(), result->rotation_raw().cpu(), result->opacity_raw().cpu(), result->sh0().cpu(), result->shN_canonical_cpu()})
            for (size_t i = 0; i < t.numel(); ++i)
                ASSERT_TRUE(std::isfinite(t.ptr<float>()[i]));
    }
}
TEST(SsogFormat, ReferenceReadsOurs) {
    const char* cli = std::getenv("LFS_SPLAT_TRANSFORM");
    if (!cli || std::system("node --version > /dev/null 2>&1") != 0)
        GTEST_SKIP() << "Set LFS_SPLAT_TRANSFORM and install node";
    ScopedSsogDirectory dir;
    auto s = synthetic(3000, 1);
    const auto out = dir.path / "ssog";
    auto result = save_ssog(s, options(dir.path / "scene.ssog", 3));
    ASSERT_TRUE(result) << result.error().format();
    ASSERT_EQ(std::system(("python3 -E -m zipfile -e " + shell_quote((dir.path / "scene.ssog").string()) + " " + shell_quote(out.string())).c_str()), 0);
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

TEST(SsogFormat, CliOptions) {
    ScopedSsogDirectory dir;
    const auto input = (dir.path / "input.ply").string();
    std::ofstream(input).put('\n');
    const char* argv[] = {"LichtFeld-Studio", "convert", input.c_str(), "-f", "ssog",
                          "--lod-levels", "2", "--lod-ratio", "0.25", "--lod-chunk-count", "32",
                          "--lod-chunk-extent", "8", "--lod-chunk-min", "2", "-o", "result_ssog"};
    auto parsed = lfs::core::args::parse_args(std::size(argv), argv);
    ASSERT_TRUE(parsed) << parsed.error();
    const auto* mode = std::get_if<lfs::core::args::ConvertMode>(&*parsed);
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(mode->params.format, lfs::core::param::OutputFormat::SSOG);
    EXPECT_EQ(mode->params.output_path, fs::path("result_ssog"));
    EXPECT_EQ(mode->params.lod_levels, 2);
    EXPECT_FLOAT_EQ(mode->params.lod_ratio, 0.25f);
    EXPECT_EQ(mode->params.lod_chunk_count, 32);
    EXPECT_FLOAT_EQ(mode->params.lod_chunk_extent, 8);
    EXPECT_EQ(mode->params.lod_chunk_min, 2);
    const char* bad[] = {"LichtFeld-Studio", "convert", input.c_str(), "-f", "ssog", "--lod-ratio", "1"};
    EXPECT_FALSE(lfs::core::args::parse_args(std::size(bad), bad));
    for (const auto* levels : {"0", "1", "8", "9"}) {
        const char* args[] = {"LichtFeld-Studio", "convert", input.c_str(), "-f", ".ssog", "--lod-levels", levels};
        auto result = lfs::core::args::parse_args(std::size(args), args);
        EXPECT_EQ(result.has_value(), std::string_view(levels) == "1" || std::string_view(levels) == "8");
        auto o = options(dir.path, std::stoi(levels));
        EXPECT_EQ(o.validate(), result.has_value());
    }
}

TEST(SsogFormat, TinyInputHasEmptyCoarsestLevel) {
    ScopedSsogDirectory dir;
    auto splats = synthetic(3, 0);
    auto saved = save_ssog(splats, options(dir.path, 4));
    ASSERT_TRUE(saved) << saved.error().format();
    EXPECT_EQ(read(dir.path / "lod-meta.json")["counts"], Json::array({3, 2, 1, 0}));
    auto loaded = load_ssog(dir.path, {.lod_level = -1});
    ASSERT_TRUE(loaded) << loaded.error().message;
    EXPECT_EQ(loaded->size(), 0);
}

TEST(SsogFormat, ImportNamesUseAssetDirectories) {
    ScopedSsogDirectory dir;
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

TEST(SsogFormat, CancellationPreservesPreviousExport) {
    ScopedSsogDirectory dir;
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
    ASSERT_TRUE(save_ssog(s, o));
    EXPECT_EQ(last, 1.0f);
    EXPECT_TRUE(saw_worker);
    const auto original = read(dir.path / "lod-meta.json");
    for (const float threshold : {0.0f, 0.65f, 1.0f}) {
        last = -1;
        o.progress_callback = [&](float p, const std::string&) {
            EXPECT_EQ(callbacks.fetch_add(1), 0);
            EXPECT_GE(p, last);
            last = p;
            callbacks.fetch_sub(1);
            return p < threshold;
        };
        auto cancelled = save_ssog(s, o);
        ASSERT_FALSE(cancelled);
        EXPECT_EQ(cancelled.error().code, ErrorCode::CANCELLED);
        EXPECT_EQ(read(dir.path / "lod-meta.json"), original);
        EXPECT_TRUE(load_ssog(dir.path));
    }
    fs::create_directory(dir.path / "env");
    fs::create_directory(dir.path / "9_8");
    write(dir.path / "notes.json", {{"keep", true}});
    ASSERT_TRUE(save_ssog(s, options(dir.path / "lod-meta.json", 1)));
    EXPECT_EQ(read(dir.path / "lod-meta.json")["lodLevels"], 1);
    EXPECT_FALSE(fs::exists(dir.path / "env"));
    EXPECT_FALSE(fs::exists(dir.path / "9_8"));
    EXPECT_FALSE(fs::exists(dir.path / "1_0"));
    EXPECT_TRUE(fs::exists(dir.path / "notes.json"));
    EXPECT_TRUE(load_ssog(dir.path));
}

TEST(SsogFormat, CpuLeafMortonMatchesCudaIncludingStableTies) {
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
            sort_ssog_leaf(positions.ptr<float>(), rows);
            auto gpu = morton_sort_indices_for_positions(positions.cuda()).cpu();
            ASSERT_TRUE(gpu.is_valid());
            EXPECT_TRUE(std::equal(rows.begin(), rows.end(), gpu.ptr<int>())) << "n=" << n << " mode=" << mode;
        }
    }
}

TEST(SsogFormat, BundleEntriesRoundtripAndRegistry) {
    ScopedSsogDirectory dir;
    const auto bundle = dir.path / "scene.ssog";
    const auto unpacked = dir.path / "unpacked.ssog";
    auto saved = save_ssog(synthetic(2048, 1), options(bundle, 3));
    ASSERT_TRUE(saved) << saved.error().format();
    ASSERT_TRUE(fs::is_regular_file(bundle));
    std::unique_ptr<archive, decltype(&archive_read_free)> zip(archive_read_new(), archive_read_free);
    ASSERT_EQ(archive_read_support_format_zip(zip.get()), ARCHIVE_OK);
#ifdef _WIN32
    ASSERT_EQ(archive_read_open_filename_w(zip.get(), bundle.wstring().c_str(), 10240), ARCHIVE_OK);
#else
    ASSERT_EQ(archive_read_open_filename(zip.get(), bundle.c_str(), 10240), ARCHIVE_OK);
#endif
    archive_entry* entry = nullptr;
    std::set<std::string> names;
    std::string last;
    int status;
    while ((status = archive_read_next_header(zip.get(), &entry)) == ARCHIVE_OK) {
        last = archive_entry_pathname(entry);
        ASSERT_TRUE(names.insert(last).second);
        const auto destination = unpacked / last;
        fs::create_directories(destination.parent_path());
        std::ofstream output(destination, std::ios::binary);
        std::array<char, 65536> bytes;
        la_ssize_t n;
        while ((n = archive_read_data(zip.get(), bytes.data(), bytes.size())) > 0)
            output.write(bytes.data(), n);
        ASSERT_EQ(n, 0);
        ASSERT_TRUE(output);
    }
    ASSERT_EQ(status, ARCHIVE_EOF);
    EXPECT_EQ(last, "lod-meta.json");
    const auto manifest = read(unpacked / "lod-meta.json");
    for (const auto& name : manifest["filenames"]) {
        const fs::path unit = name.get<std::string>();
        for (const auto* file : {"meta.json", "means_l.webp", "means_u.webp", "scales.webp", "quats.webp", "sh0.webp", "shN_centroids.webp", "shN_labels.webp"})
            EXPECT_TRUE(names.contains((unit.parent_path() / file).generic_string()));
    }
    for (int level : {0, 1, 2, -1}) {
        auto bundled = load_ssog(bundle, {.lod_level = level});
        auto directory = load_ssog(unpacked, {.lod_level = level});
        ASSERT_TRUE(bundled) << bundled.error().format();
        ASSERT_TRUE(directory) << directory.error().format();
        ASSERT_EQ(bundled->size(), directory->size());
        // Identical manifest traversal guarantees row order too, a stronger check than sorting.
        EXPECT_EQ(bundled->means().cpu().to_vector(), directory->means().cpu().to_vector());
        EXPECT_EQ(bundled->scaling_raw().cpu().to_vector(), directory->scaling_raw().cpu().to_vector());
        EXPECT_EQ(bundled->opacity_raw().cpu().to_vector(), directory->opacity_raw().cpu().to_vector());
        EXPECT_EQ(bundled->rotation_raw().cpu().to_vector(), directory->rotation_raw().cpu().to_vector());
        EXPECT_EQ(bundled->shN_canonical_cpu().to_vector(), directory->shN_canonical_cpu().to_vector());
    }
    EXPECT_TRUE(is_ssog_path(bundle));
    SsogLoader format_loader;
    EXPECT_EQ(format_loader.supportedExtensions(), std::vector<std::string>{".ssog"});
    EXPECT_TRUE(format_loader.canLoad(bundle));
    EXPECT_FALSE(Loader::isDatasetPath(bundle));
    EXPECT_EQ(Loader::getDatasetType(bundle), DatasetType::Unknown);
    auto registry = Loader::create();
    EXPECT_TRUE(registry->canLoad(bundle));
    auto loaded = registry->load(bundle);
    ASSERT_TRUE(loaded) << loaded.error().format();
    EXPECT_EQ(std::get<std::shared_ptr<SplatData>>(loaded->data)->size(), 2048);
}

TEST(SsogFormat, BundleCancellationPreservesDestination) {
    ScopedSsogDirectory dir;
    const auto bundle = dir.path / "scene.ssog";
    std::ofstream(bundle) << "existing destination";
    auto o = options(bundle, 2);
    o.progress_callback = [](float p, const std::string&) { return p < 1; };
    auto saved = save_ssog(synthetic(512), o);
    ASSERT_FALSE(saved);
    EXPECT_EQ(saved.error().code, ErrorCode::CANCELLED);
    std::ifstream file(bundle);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(file), {}), "existing destination");
    for (const auto& entry : fs::directory_iterator(dir.path))
        EXPECT_EQ(entry.path(), bundle);
    fs::remove(bundle);
    saved = save_ssog(synthetic(512), o);
    EXPECT_FALSE(saved);
    EXPECT_TRUE(fs::is_empty(dir.path));
}

TEST(SsogFormat, RejectsInvalidBundle) {
    ScopedSsogDirectory dir;
    const auto bundle = dir.path / "bad.ssog";
    std::ofstream(bundle) << "not a ZIP archive";
    EXPECT_FALSE(validate_ssog(bundle));
    EXPECT_FALSE(load_ssog(bundle));
    {
        std::ofstream file(bundle, std::ios::binary);
        file << "PK\x03\x04";
    }
    EXPECT_FALSE(validate_ssog(bundle));
}

TEST(SsogFormat, GalleryWrapperAndBundledUnitsPreserveDecodedSplats) {
    ScopedSsogDirectory dir;
    const auto source = dir.path / "source";
    ASSERT_TRUE(save_ssog(synthetic(256, 0), options(source)));
    auto expected = load_ssog(source);
    ASSERT_TRUE(expected) << expected.error().format();
    for (const bool bundled_units : {false, true}) {
        auto manifest = read(source / "lod-meta.json");
        const auto output = dir.path / (bundled_units ? "bundled.ssog" : "wrapped.ssog");
        auto outer = make_sog_archive(output);
        ASSERT_TRUE(outer->open());
        if (bundled_units) {
            for (size_t i = 0; i < manifest["filenames"].size(); ++i) {
                const fs::path relative = manifest["filenames"][i].get<std::string>();
                const auto unit_path = dir.path / std::format("unit{}.sog", i);
                auto unit = make_sog_archive(unit_path);
                ASSERT_TRUE(unit->open());
                for (const auto& entry : fs::directory_iterator(source / relative.parent_path())) {
                    if (!entry.is_regular_file())
                        continue;
                    std::ifstream file(entry.path(), std::ios::binary);
                    const std::string bytes(std::istreambuf_iterator<char>{file}, {});
                    ASSERT_TRUE(unit->add_file(entry.path().filename().generic_string(), bytes.data(), bytes.size()));
                }
                ASSERT_TRUE(unit->close());
                std::ifstream file(unit_path, std::ios::binary);
                const std::string bytes(std::istreambuf_iterator<char>{file}, {});
                const auto name = unit_path.filename().generic_string();
                ASSERT_TRUE(outer->add_file("scene/" + name, bytes.data(), bytes.size()));
                manifest["filenames"][i] = name;
            }
        } else {
            for (const auto& entry : fs::recursive_directory_iterator(source)) {
                if (!entry.is_regular_file() || entry.path().filename() == "lod-meta.json")
                    continue;
                std::ifstream file(entry.path(), std::ios::binary);
                const std::string bytes(std::istreambuf_iterator<char>{file}, {});
                ASSERT_TRUE(outer->add_file("scene/" + entry.path().lexically_relative(source).generic_string(), bytes.data(), bytes.size()));
            }
        }
        const auto metadata = manifest.dump();
        ASSERT_TRUE(outer->add_file("scene/lod-meta.json", metadata.data(), metadata.size()));
        ASSERT_TRUE(outer->close());
        ASSERT_TRUE(validate_ssog(output));
        auto loaded = load_ssog(output);
        ASSERT_TRUE(loaded) << loaded.error().format();
        EXPECT_EQ(loaded->size(), expected->size());
        EXPECT_EQ(loaded->means().cpu().to_vector(), expected->means().cpu().to_vector());
        EXPECT_EQ(loaded->scaling_raw().cpu().to_vector(), expected->scaling_raw().cpu().to_vector());
        EXPECT_EQ(loaded->sh0().cpu().to_vector(), expected->sh0().cpu().to_vector());
    }
}

TEST(SsogFormat, GalleryBundlesRejectAmbiguousRootsAndUnsafeNestedEntries) {
    ScopedSsogDirectory dir;
    const std::string metadata = R"({"version":1,"lodLevels":1,"filenames":["unit.sog"]})";
    {
        const auto path = dir.path / "ambiguous.ssog";
        auto outer = make_sog_archive(path);
        ASSERT_TRUE(outer->open());
        for (const auto* name : {"lod-meta.json", "other/lod-meta.json"})
            ASSERT_TRUE(outer->add_file(name, metadata.data(), metadata.size()));
        ASSERT_TRUE(outer->close());
        auto valid = validate_ssog(path);
        ASSERT_FALSE(valid);
        EXPECT_NE(valid.error().message.find("exactly one"), std::string::npos);
    }
    for (const bool bomb : {false, true}) {
        const auto unit_path = dir.path / "unit.sog";
        auto unit = make_sog_archive(unit_path);
        ASSERT_TRUE(unit->open());
        const std::string meta = R"({"count":1})";
        ASSERT_TRUE(unit->add_file("meta.json", meta.data(), meta.size()));
        const std::string data(bomb ? 17 * 1024 * 1024 : 8, 'x');
        ASSERT_TRUE(unit->add_file(bomb ? "payload.bin" : "../escape.json", data.data(), data.size()));
        ASSERT_TRUE(unit->close());
        std::ifstream file(unit_path, std::ios::binary);
        const std::string bytes(std::istreambuf_iterator<char>{file}, {});
        const auto path = dir.path / (bomb ? "bomb.ssog" : "traversal.ssog");
        auto outer = make_sog_archive(path);
        ASSERT_TRUE(outer->open());
        ASSERT_TRUE(outer->add_file("lod-meta.json", metadata.data(), metadata.size()));
        ASSERT_TRUE(outer->add_file("unit.sog", bytes.data(), bytes.size()));
        ASSERT_TRUE(outer->close());
        auto valid = validate_ssog(path);
        ASSERT_FALSE(valid);
        EXPECT_NE(valid.error().message.find(bomb ? "size limit" : "escapes archive root"), std::string::npos)
            << valid.error().format();
        EXPECT_FALSE(fs::exists(dir.path / "escape.json"));
    }
}

TEST(SsogFormat, NestedChunksCannotMultiplyOuterExpansionAllowance) {
    ScopedSsogDirectory dir;
    const auto unit_path = dir.path / "unit.sog";
    auto unit = make_sog_archive(unit_path);
    ASSERT_TRUE(unit->open());
    const std::string meta = R"({"count":1})";
    ASSERT_TRUE(unit->add_file("meta.json", meta.data(), meta.size()));
    // Repeated independently compressed entries compress again in the outer ZIP.
    // Each layer alone has a modest ratio, but together they exceed its allowance.
    std::mt19937 random(42);
    std::string payload(8192, '\0');
    for (size_t i = 0; i < 4096; ++i)
        payload[i] = static_cast<char>(random() & 255);
    for (int i = 0; i < 2300; ++i)
        ASSERT_TRUE(unit->add_file(std::format("extra{}.bin", i), payload.data(), payload.size()));
    ASSERT_TRUE(unit->close());
    std::ifstream file(unit_path, std::ios::binary);
    const std::string bytes(std::istreambuf_iterator<char>{file}, {});
    ASSERT_LT(bytes.size(), 16 * 1024 * 1024);
    const auto path = dir.path / "nested.ssog";
    auto outer = make_sog_archive(path);
    ASSERT_TRUE(outer->open());
    const std::string manifest = R"({"version":1,"lodLevels":1,"filenames":["unit.sog"]})";
    ASSERT_TRUE(outer->add_file("lod-meta.json", manifest.data(), manifest.size()));
    ASSERT_TRUE(outer->add_file("unit.sog", bytes.data(), bytes.size()));
    ASSERT_TRUE(outer->close());
    ASSERT_LT(fs::file_size(path) * 20, 16 * 1024 * 1024);
    auto valid = validate_ssog(path);
    ASSERT_FALSE(valid);
    EXPECT_NE(valid.error().message.find("size limit"), std::string::npos) << valid.error().format();
}

TEST(SsogFormat, ConvertBundleDirectoryDefaultAndBack) {
    ScopedSsogDirectory dir;
    const auto input = dir.path / "input.ply";
    ASSERT_TRUE(save_ply(synthetic(128, 0), {.output_path = input}));
    for (const auto& destination : {dir.path / "out.ssog", dir.path / "outdir", fs::path{}}) {
        const auto input_string = input.string();
        const auto output_string = destination.string();
        std::vector<const char*> argv{"LichtFeld-Studio", "convert", input_string.c_str(), "-f", "ssog", "--lod-levels", "2", "-y"};
        if (!destination.empty()) {
            argv.push_back("-o");
            argv.push_back(output_string.c_str());
        }
        auto parsed = lfs::core::args::parse_args(static_cast<int>(argv.size()), argv.data());
        ASSERT_TRUE(parsed) << parsed.error();
        const auto* mode = std::get_if<lfs::core::args::ConvertMode>(&*parsed);
        ASSERT_NE(mode, nullptr);
        ASSERT_EQ(lfs::app::run_converter(mode->params), 0);
        const auto actual = destination.empty() ? dir.path / "input.ssog" : destination;
        auto loaded = load_ssog(actual);
        ASSERT_TRUE(loaded) << loaded.error().format();
        EXPECT_EQ(loaded->size(), 128);
        EXPECT_EQ(fs::is_regular_file(actual), actual.extension() == ".ssog");
    }
    lfs::core::param::ConvertParameters back;
    back.input_path = dir.path / "out.ssog";
    back.output_path = dir.path / "back.ply";
    back.format = lfs::core::param::OutputFormat::PLY;
    back.overwrite = true;
    ASSERT_EQ(lfs::app::run_converter(back), 0);
    auto loaded = Loader::create()->load(back.output_path);
    ASSERT_TRUE(loaded) << loaded.error().format();
    EXPECT_EQ(std::get<std::shared_ptr<SplatData>>(loaded->data)->size(), 128);
}
