/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// A large model through save, load, export and import, every row checked at every stage:
//   build in memory -> save .licht -> load -> export PLY (whole, and through a crop box) -> import PLY
//   -> save .licht -> load.
// The model has SH degree 3, held in Q16 as training holds it, and LFS_TEST_LARGE_ROUNDTRIP_ROWS rows (100,000,000 by
// default; 150,000,000 needs an exclusive hold of the GPU lock on a shared machine). Row r stores its id exactly in
// its position, x = r / 4096 and y = r % 4096, with z = 0.5 and rotation (1, 0, 0, 0); every other value is a hash of
// (r, column), a multiple of 2^-24 in [-0.5, 0.5), which fp32 holds exactly (the scheme of tools/ply_roundtrip.py and
// tools/ply_check.py, so their checks can read the same files). Rows can come back in any order: the id comes from
// the position. A project loads the way the project export reads a visible splat (the document's SPLT chunk,
// hydrated); the PLY goes through the Loader.
//
// What must hold: positions, colours, opacities, scales and rotations bit for bit at every stage. SH: the project
// stores the values the Q16 codes decode to and loads them bit for bit, as fp32; the PLY holds exactly what the CPU
// decodes from those codes; an import quantizes again, so its decoded values stay within one quantization step of
// its own block's bounds.
// LFS_TEST_LARGE_ROUNDTRIP_KEEP=1 keeps the exported PLY in the test's folder for tools/ply_check.py.
//
// Runs with LFS_TEST_HUGE_SH=1: about 52 GB of device and 90 GB of host memory at 100M rows (on unified memory the
// peak footprint was 15.4 GB at 10M rows, about 1.4 KB a row), and about 45 GB of disk in TMPDIR at its peak.
// RowsNotAMultipleOfFourSurvive runs the same at 100,000,002 rows.

#include "large_model_test_support.hpp"

#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "core/uuid.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "io/splat_chapter.hpp"
#include "lfs/training/sh_value_codec.hpp"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace project = lfs::io::project;

    // The PLY columns of tools/ply_roundtrip.py, in its order.
    constexpr size_t kColumns = 62;
    constexpr size_t kDc = 6, kRest = 9, kOpacity = 54, kScale = 55, kRot = 58;

    std::string column_name(const size_t c) {
        static constexpr std::array<const char*, 6> head{"x", "y", "z", "nx", "ny", "nz"};
        if (c < 6)
            return head[c];
        if (c < kRest)
            return std::format("f_dc_{}", c - kDc);
        if (c < kOpacity)
            return std::format("f_rest_{}", c - kRest);
        if (c == kOpacity)
            return "opacity";
        if (c < kRot)
            return std::format("scale_{}", c - kScale);
        return std::format("rot_{}", c - kRot);
    }

    float hashed(const uint64_t r, const uint64_t column) {
        const uint64_t h = (r * 2654435761ull + 40503ull * (column + 1)) & 0xFFFFFFFFull;
        return static_cast<float>(static_cast<double>(h >> 8) / 16777216.0 - 0.5);
    }

    // The value of column c of row r.
    float expected(const uint64_t r, const size_t c) {
        if (c == 0)
            return static_cast<float>(r / 4096);
        if (c == 1)
            return static_cast<float>(r % 4096);
        if (c == 2)
            return 0.5f;
        if (c < 6)
            return 0.0f;
        if (c >= kRot)
            return c == kRot ? 1.0f : 0.0f;
        return hashed(r, c);
    }

    // The canonical SH cell (coefficient k, channel ch) is the PLY's f_rest_{ch * 15 + k}.
    size_t rest_column(const uint32_t cell) { return kRest + (cell % 3) * 15 + cell / 3; }

    size_t rows_from_environment() {
        if (const char* value = std::getenv("LFS_TEST_LARGE_ROUNDTRIP_ROWS"))
            return static_cast<size_t>(std::strtoull(value, nullptr, 10));
        return 100'000'000;
    }

    // A host [rows, width] tensor of columns first..first + width.
    Tensor columns(const size_t rows, const size_t first, const size_t width, const lfs::core::TensorShape& shape) {
        Tensor host = Tensor::empty(shape, Device::CPU, DataType::Float32);
        auto* const data = host.ptr<float>();
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < width; ++c)
                data[r * width + c] = expected(r, first + c);
        return host;
    }

    // The Q16 codes and bounds of a model, on the host.
    struct HostQ16 {
        std::vector<uint16_t> codes;
        std::vector<float> bounds;
        [[nodiscard]] float decode(const size_t r, const uint32_t cell) const {
            const uint16_t code = codes[(r / 32) * (lm::kCells * 32) + cell * 32 + r % 32];
            const float lo = bounds[r / 256 * 2], hi = bounds[r / 256 * 2 + 1];
            return std::fmaf(hi - lo, static_cast<float>(code) * (1.0f / 65535.0f), lo);
        }
        [[nodiscard]] float step(const size_t r) const { return (bounds[r / 256 * 2 + 1] - bounds[r / 256 * 2]) / 65535.0f; }
    };

    HostQ16 host_q16(const SplatData& model) {
        EXPECT_TRUE(model.shN_value_quantized());
        return {lm::host_copy<uint16_t>(model.shN()), lm::host_copy<float>(model.shN_value_bounds())};
    }

    // Counts of values that differ, per check.
    struct Differences {
        size_t wrong = 0;
        std::string first;
        void note(const std::string& what) {
            if (wrong++ == 0)
                first = what;
        }
    };

    // Positions, colours, opacities, scales and rotations of a model in row order, bit for bit.
    void expect_attributes(const SplatData& model, const size_t rows, const std::string& stage) {
        const auto check = [&](const Tensor& tensor, const size_t first, const size_t width, const char* name) {
            const auto values = lm::host_copy<float>(tensor.contiguous());
            ASSERT_EQ(values.size(), rows * width) << stage << ", " << name;
            Differences d;
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < width; ++c) {
                    const float want = expected(r, first + c), got = values[r * width + c];
                    if (std::memcmp(&got, &want, 4) != 0)
                        d.note(std::format("row {}, {}: {} instead of {}", r, column_name(first + c), got, want));
                }
            EXPECT_EQ(d.wrong, 0u) << stage << ", " << name << ": " << d.wrong << " values differ; first " << d.first;
        };
        check(model.means(), 0, 3, "positions");
        check(model.sh0(), kDc, 3, "colours");
        check(model.opacity_raw(), kOpacity, 1, "opacities");
        check(model.scaling_raw(), kScale, 3, "scales");
        check(model.rotation_raw(), kRot, 4, "rotations");
    }

    // The model's SH against the reference. A Q16 model: codes and bounds bit for bit when `exact`, otherwise every
    // decoded value within one quantization step of the model's own block bounds. An fp32 model (a project stores the
    // values its Q16 codes decode to and loads them as fp32): every value bit for bit as the reference decodes it.
    void expect_sh(const SplatData& model, const HostQ16& reference, const size_t rows, const bool exact,
                   const std::string& stage) {
        Differences d;
        if (!model.shN_value_quantized()) {
            const Tensor canonical = model.shN_canonical_cpu().contiguous();
            ASSERT_EQ(static_cast<size_t>(canonical.numel()), rows * lm::kCells) << stage;
            ASSERT_EQ(canonical.dtype(), DataType::Float32) << stage;
            const float* const values = canonical.ptr<float>();
            for (size_t r = 0; r < rows; ++r)
                for (uint32_t c = 0; c < lm::kCells; ++c) {
                    const float want = reference.decode(r, c), got = values[r * lm::kCells + c];
                    if (std::memcmp(&got, &want, 4) != 0)
                        d.note(std::format("row {}, cell {}: {} instead of {}", r, c, got, want));
                }
            EXPECT_EQ(d.wrong, 0u) << stage << ", SH: " << d.wrong << " differ; first " << d.first;
            return;
        }
        const HostQ16 q = host_q16(model);
        ASSERT_EQ(q.codes.size(), reference.codes.size()) << stage;
        if (exact) {
            for (size_t i = 0; i < q.codes.size(); ++i)
                if (q.codes[i] != reference.codes[i])
                    d.note(std::format("code {} (row {})", i, i / (lm::kCells * 32) * 32 + i % 32));
            for (size_t i = 0; i < q.bounds.size(); ++i)
                if (std::memcmp(&q.bounds[i], &reference.bounds[i], 4) != 0)
                    d.note(std::format("bound {} (rows {}-)", i, i / 2 * 256));
        } else {
            for (size_t r = 0; r < rows; ++r)
                for (uint32_t c = 0; c < lm::kCells; ++c)
                    if (std::fabs(q.decode(r, c) - reference.decode(r, c)) > q.step(r) * 1.0001f)
                        d.note(std::format("row {}, cell {}: {} instead of {} (step {})", r, c, q.decode(r, c),
                                           reference.decode(r, c), q.step(r)));
        }
        EXPECT_EQ(d.wrong, 0u) << stage << ", SH: " << d.wrong << " differ; first " << d.first;
    }

    // Every row of a binary PLY: each id once if `ids` is empty, else exactly the ids `ids` accepts; every column as
    // expected(), f_rest as the reference decodes it.
    template <class Accept>
    void expect_ply(const fs::path& path, const HostQ16& reference, const size_t rows, Accept&& accept,
                    const size_t accepted_rows, const std::string& stage) {
        std::ifstream file(path, std::ios::binary);
        ASSERT_TRUE(file) << stage << ": can't open " << path;
        std::string line;
        size_t vertices = 0;
        std::vector<std::string> properties;
        while (std::getline(file, line)) {
            if (line.rfind("element vertex ", 0) == 0)
                vertices = std::stoull(line.substr(15));
            else if (line.rfind("property ", 0) == 0) {
                ASSERT_EQ(line.rfind("property float ", 0), 0u) << stage << ": " << line;
                properties.push_back(line.substr(15));
            } else if (line == "end_header")
                break;
        }
        ASSERT_EQ(vertices, accepted_rows) << stage << ": vertex count";
        std::array<size_t, kColumns> at{};
        for (size_t c = 0; c < kColumns; ++c) {
            const auto found = std::find(properties.begin(), properties.end(), column_name(c));
            if (c >= 3 && c < 6 && found == properties.end()) {
                at[c] = SIZE_MAX; // normals are optional
                continue;
            }
            ASSERT_NE(found, properties.end()) << stage << ": no property " << column_name(c);
            at[c] = static_cast<size_t>(found - properties.begin());
        }
        const size_t width = properties.size();
        std::vector<uint8_t> seen(rows, 0);
        std::vector<float> chunk;
        constexpr size_t kChunk = 1 << 20;
        Differences d;
        size_t bad_ids = 0, repeated = 0;
        for (size_t done = 0; done < vertices; done += kChunk) {
            const size_t n = std::min(kChunk, vertices - done);
            chunk.resize(n * width);
            file.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(n * width * sizeof(float)));
            ASSERT_TRUE(file) << stage << ": the file ends inside row " << done;
            for (size_t i = 0; i < n; ++i) {
                const float* const v = chunk.data() + i * width;
                const float x = v[at[0]], y = v[at[1]];
                const uint64_t r = static_cast<uint64_t>(x) * 4096 + static_cast<uint64_t>(y);
                if (x != std::floor(x) || y != std::floor(y) || y < 0 || y >= 4096 || r >= rows || !accept(r)) {
                    ++bad_ids;
                    continue;
                }
                repeated += seen[r]++ ? 1 : 0;
                for (size_t c = 2; c < kColumns; ++c) {
                    if (at[c] == SIZE_MAX)
                        continue;
                    const bool sh = c >= kRest && c < kOpacity;
                    const float want = sh ? reference.decode(r, static_cast<uint32_t>(((c - kRest) % 15) * 3 + (c - kRest) / 15))
                                          : expected(r, c);
                    const float got = v[at[c]];
                    if (std::memcmp(&got, &want, 4) != 0)
                        d.note(std::format("row {}, {}: {} instead of {}", r, column_name(c), got, want));
                }
            }
        }
        size_t missing = 0;
        for (size_t r = 0; r < rows; ++r)
            missing += accept(r) && !seen[r] ? 1 : 0;
        EXPECT_EQ(bad_ids, 0u) << stage << ": rows with an id outside the model or the crop";
        EXPECT_EQ(repeated, 0u) << stage << ": ids seen more than once";
        EXPECT_EQ(missing, 0u) << stage << ": ids missing";
        EXPECT_EQ(d.wrong, 0u) << stage << ": " << d.wrong << " values differ; first " << d.first;
    }

    lfs::core::Uuid uuid(const uint64_t tag) {
        return *lfs::core::Uuid::from_string(std::format("4202{:04x}-0000-4000-8000-{:012x}", 0x7e57, tag));
    }

    // Saves `model` as the one splat of a project at `path`.
    void save_project(const SplatData& model, const fs::path& path, const uint64_t tag) {
        auto document = project::ProjectDocument::create(uuid(tag), 100);
        ASSERT_TRUE(document) << lfs::format_for_developer(document.error());
        const auto node = uuid(tag + 1);
        ASSERT_TRUE(document->edit_scene_graph().upsert_node(project::SceneNodeRecord{
            .uuid = node,
            .type = "splat",
            .name = "Model",
            .payload = project::PayloadBinding{.fourcc = "SPLT", .instance_uuid = node, .source_kind = "ply"},
        }));
        auto captured = project::SplatChapterPayload::capture(model, project::SplatSourceKind::ImportedPly, false);
        ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
        ASSERT_TRUE(document->set_splat(node, std::move(*captured)));
        ASSERT_TRUE(document->edit_project().upsert_embed_decision(project::EmbedDecision{
            .uuid = node, .node_uuid = node, .payload_fourcc = "SPLT", .decision = "embedded",
            .reason = "large model round trip"}));
        // An embedded payload needs the provenance of its import (the save fills in its content hash).
        project::ReferenceFingerprint fingerprint;
        fingerprint.size = 1000 + tag;
        fingerprint.mtime_unix_ns = 2000 + static_cast<int64_t>(tag);
        fingerprint.head_xxh3.bytes.fill(1);
        fingerprint.tail_xxh3.bytes.fill(2);
        ASSERT_TRUE(document->edit_project().upsert_embedded_payload_provenance(project::EmbeddedPayloadProvenance{
            .uuid = node,
            .node_uuid = node,
            .fourcc = "SPLT",
            .import_locator = {.preferred = "assets/model.ply", .base = project::LocatorBase::Project},
            .import_fingerprint = fingerprint}));
        const auto saved = document->save(
            path, {.commit = {.kind = project::CommitKind::Explicit, .commit_uuid = uuid(tag + 2),
                              .snapshot_uuid = uuid(tag + 3), .wallclock_unix_ns = 200},
                   .file_uuid = uuid(tag + 4)});
        ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
    }

    // Loads the one splat of the project at `path` (saved by save_project with `tag`) the way the project export
    // reads a visible splat: its SPLT chunk read whole, then hydrated.
    std::shared_ptr<SplatData> load_project(const fs::path& path, const uint64_t tag) {
        auto document = project::ProjectDocument::open(path);
        EXPECT_TRUE(document) << (document ? std::string{} : lfs::format_for_developer(document.error()));
        if (!document)
            return nullptr;
        const auto* const reader = document->source_reader();
        const auto* const chunk = reader ? reader->find(project::FOURCC_SPLT, uuid(tag + 1)) : nullptr;
        EXPECT_NE(chunk, nullptr) << "no splat chunk in " << path;
        if (!chunk)
            return nullptr;
        auto bytes = reader->read_chunk(*chunk);
        EXPECT_TRUE(bytes) << (bytes ? std::string{} : lfs::format_for_developer(bytes.error()));
        if (!bytes)
            return nullptr;
        auto payload = project::SplatChapterPayload::from_lfsp(std::move(*bytes));
        EXPECT_TRUE(payload) << (payload ? std::string{} : lfs::format_for_developer(payload.error()));
        if (!payload)
            return nullptr;
        auto data = payload->hydrate();
        EXPECT_TRUE(data) << (data ? std::string{} : lfs::format_for_developer(data.error()));
        if (!data)
            return nullptr;
        return std::shared_ptr<SplatData>(std::move(*data));
    }

    std::shared_ptr<SplatData> load_model(const fs::path& path) {
        auto loaded = lfs::io::Loader::create()->load(path);
        EXPECT_TRUE(loaded) << (loaded ? std::string{} : loaded.error().format());
        if (!loaded)
            return nullptr;
        const auto* const splat = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
        return splat ? *splat : nullptr;
    }

    void round_trip(const size_t rows) {
        ASSERT_GE(rows, size_t{1024}) << "the round trip needs at least 1,024 rows";
        if (const auto reason = lm::skip_reason(rows * 518, rows * 900))
            GTEST_SKIP() << *reason;
        lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);
        struct Restore {
            ~Restore() { lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt); }
        } restore;
        static std::atomic<int> counter{0};
        const fs::path folder =
            fs::temp_directory_path() /
            std::format("lfs_large_model_roundtrip_{}_{}", std::chrono::steady_clock::now().time_since_epoch().count(),
                        counter++);
        fs::create_directories(folder);
        struct Remove {
            fs::path folder;
            ~Remove() {
                if (!lm::env_is_one("LFS_TEST_LARGE_ROUNDTRIP_KEEP")) {
                    std::error_code ignored;
                    fs::remove_all(folder, ignored);
                }
            }
        } remove{folder};

        // The model, SH in Q16.
        SplatData model;
        {
            Tensor sh = Tensor::empty({rows, size_t{lm::kRest}, size_t{3}}, Device::CPU, DataType::Float32);
            auto* const data = sh.ptr<float>();
            for (size_t r = 0; r < rows; ++r)
                for (uint32_t cell = 0; cell < lm::kCells; ++cell)
                    data[r * lm::kCells + cell] = expected(r, rest_column(cell));
            model = SplatData(3, columns(rows, 0, 3, {rows, 3}).gpu(), columns(rows, kDc, 3, {rows, 1, 3}).gpu(),
                              sh.gpu(), columns(rows, kScale, 3, {rows, 3}).gpu(),
                              columns(rows, kRot, 4, {rows, 4}).gpu(), columns(rows, kOpacity, 1, {rows, 1}).gpu(),
                              1.0f);
        }
        ASSERT_TRUE(model.apply_shN_value_quant());
        HostQ16 reference = host_q16(model);
        {
            // The quantization itself: every decoded value within one step of its original.
            Differences d;
            for (size_t r = 0; r < rows; ++r)
                for (uint32_t c = 0; c < lm::kCells; ++c)
                    if (std::fabs(reference.decode(r, c) - expected(r, rest_column(c))) > reference.step(r) * 1.0001f)
                        d.note(std::format("row {}, cell {}", r, c));
            EXPECT_EQ(d.wrong, 0u) << "quantized model: " << d.wrong << " SH values off by more than a step; first "
                                   << d.first;
        }
        expect_attributes(model, rows, "the model");

        // Save, load.
        const fs::path first_project = folder / "first.licht";
        save_project(model, first_project, 10);
        {
            const auto loaded = load_project(first_project, 10);
            ASSERT_TRUE(loaded);
            ASSERT_EQ(static_cast<size_t>(loaded->size()), rows);
            expect_attributes(*loaded, rows, "project, loaded");
            expect_sh(*loaded, reference, rows, true, "project, loaded");
        }
        fs::remove(first_project);

        // Export everything, and through a crop box over the rows around both SH wraps (or the upper half of a
        // smaller model).
        const fs::path ply = folder / "export.ply";
        {
            const auto saved = lfs::io::save_ply(model, {.output_path = ply});
            ASSERT_TRUE(saved) << saved.error().format();
        }
        expect_ply(ply, reference, rows, [](uint64_t) { return true; }, rows, "PLY export");
        {
            const float x0 = rows > 96'000'000 ? 89'000'000.0f / 4096 : std::floor(rows / 2 / 4096.0f);
            const float x1 = rows > 96'000'000 ? 96'000'000.0f / 4096 : std::floor((rows - 1) / 4096.0f);
            const uint64_t first_row = static_cast<uint64_t>(std::ceil(x0)) * 4096;
            const uint64_t end_row = std::min<uint64_t>(rows, (static_cast<uint64_t>(std::floor(x1)) + 1) * 4096);
            const Tensor mask = lfs::core::compute_cropbox_mask(model.means(), {x0 - 0.25f, -1.0f, 0.0f},
                                                                {x1 + 0.25f, 4097.0f, 1.0f}, glm::mat4(1.0f));
            const SplatData cropped = lfs::core::extract_by_mask(model, mask);
            ASSERT_EQ(static_cast<uint64_t>(cropped.size()), end_row - first_row);
            const fs::path cropped_ply = folder / "cropped.ply";
            const auto saved = lfs::io::save_ply(cropped, {.output_path = cropped_ply});
            ASSERT_TRUE(saved) << saved.error().format();
            expect_ply(cropped_ply, reference, rows,
                       [&](const uint64_t r) { return r >= first_row && r < end_row; }, end_row - first_row,
                       "PLY export through a crop box");
            fs::remove(cropped_ply);
        }
        model = SplatData{}; // the rest needs only the exported file and the reference

        // Import the whole export, save it as a project and load it again.
        std::shared_ptr<SplatData> imported = load_model(ply);
        ASSERT_TRUE(imported);
        ASSERT_EQ(static_cast<size_t>(imported->size()), rows);
        if (!imported->shN_value_quantized())
            ASSERT_TRUE(imported->apply_shN_value_quant());
        expect_attributes(*imported, rows, "PLY import");
        expect_sh(*imported, reference, rows, false, "PLY import");
        const HostQ16 imported_q16 = host_q16(*imported);
        reference = HostQ16{};
        if (!lm::env_is_one("LFS_TEST_LARGE_ROUNDTRIP_KEEP"))
            fs::remove(ply);
        const fs::path second_project = folder / "second.licht";
        save_project(*imported, second_project, 20);
        imported.reset();
        {
            const auto loaded = load_project(second_project, 20);
            ASSERT_TRUE(loaded);
            ASSERT_EQ(static_cast<size_t>(loaded->size()), rows);
            expect_attributes(*loaded, rows, "project of the import, loaded");
            expect_sh(*loaded, imported_q16, rows, true, "project of the import, loaded");
        }
    }

    TEST(LargeModelRoundTrip, EveryRowSurvivesSaveLoadExportImport) { round_trip(rows_from_environment()); }

    // The same with a row count that is not a multiple of 4 (100,000,002 by default, or
    // LFS_TEST_LARGE_ROUNDTRIP_ODD_ROWS): an fp32 canonical SH tensor of such a model is only 4-byte aligned, so any
    // whole copy of one moves 4-byte elements, 2^32 of them from 95,443,718 rows.
    TEST(LargeModelRoundTrip, RowsNotAMultipleOfFourSurvive) {
        const char* value = std::getenv("LFS_TEST_LARGE_ROUNDTRIP_ODD_ROWS");
        const size_t rows = value ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : size_t{100'000'002};
        ASSERT_NE(rows % 4, 0u) << "LFS_TEST_LARGE_ROUNDTRIP_ODD_ROWS must not be a multiple of 4";
        round_trip(rows);
    }

} // namespace
